"""
Main Market-Maker Runtime - Integration loop.

Integrates all layers:
- Feeds (WebSocket streams)
- Risk (portfolio engine, market state)
- Strategy (QS, market-maker, markout, parity, satellite)
- Execution (CLOB executor)
- Reporting (truth report)
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List

from config import (
    ACTIVE_QUOTE_COUNT,
    PAPER_MODE,
    QS_BOOK_STALENESS_S_PAPER,
    QS_BOOK_STALENESS_S_PROD,
    QS_MIN_LIQUIDITY,
    RECONCILE_INTERVAL_SECONDS,
    WS_STALENESS_THRESHOLD_MS,
)
from execution.clob_executor import CLOBExecutor
from execution.order_state_store import OrderStateStore
from execution.paper_simulator import PaperTradingSimulator
from execution.wallet import preflight_checks
from feeds.market_ws import MarketWebSocketFeed
from feeds.user_ws import UserWebSocketFeed
from models.types import Fill, Market, MarketState, OrderBook
from reporting.truth_report import TruthReportBuilder, write_daily_report
from risk import market_state as market_state_module
from risk.portfolio_engine import PortfolioRiskEngine
from scanner.event_refresher import EventRefresher
from scanner.market_fetcher_v2 import fetch_markets_with_metadata
from strategy.market_maker import create_maker_orders
from strategy.markout_tracker import MarkoutTracker
from strategy.parity_scanner import scan_all_parity
from strategy.quoteability_scorer import (
    compute_qs,
    get_qs_veto_summary,
    reset_qs_veto_counters,
    select_active_set,
    should_mutate_quotes,
)
from strategy.resolution_risk_scorer import compute_rrs
from strategy.satellite_filter import load_astra_predictions, scan_satellite_opportunities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class MarketMakerRuntime:
    """
    Main runtime loop for allocator-grade market-maker.

    ChatGPT: "This ordering prevents the classic 'bot runs but bleeds silently' failure."
    """

    def __init__(self, clob_client):
        """
        Initialize runtime.

        Args:
            clob_client: py-clob-client instance
        """
        self.clob_client = clob_client

        # Core components
        self.order_store = OrderStateStore()
        self.executor = CLOBExecutor(clob_client, self.order_store)
        self.risk_engine = PortfolioRiskEngine()
        self.markout_tracker = MarkoutTracker()
        self.truth_report_builder = TruthReportBuilder(date=datetime.now().strftime("%Y-%m-%d"))
        self.event_refresher = EventRefresher()

        # Paper trading simulator (if in paper mode)
        self.paper_simulator = PaperTradingSimulator() if PAPER_MODE else None

        # Feeds
        self.market_feed = MarketWebSocketFeed(on_stale_callback=self._on_feed_stale)
        self.user_feed = UserWebSocketFeed(
            api_key=None,  # TODO: Load from config
            on_fill_callback=self._on_fill,
            on_order_update_callback=self._on_order_update,
            on_disconnect_callback=self._on_feed_disconnect,
        )

        # State
        self.markets: Dict[str, Market] = {}
        self.qs_scores: Dict[str, float] = {}
        self.active_set: List[Market] = []
        self.running = False
        self.unsafe_mode = False  # Circuit breaker flag
        self.current_date = datetime.now().strftime("%Y-%m-%d")

        # Reseed tracking
        self.zero_active_cycles = 0  # Consecutive cycles with Active set = 0
        self.last_reseed_time = time.time()  # FIX: Initialize to NOW, not 0 (enforces rate limit from startup)
        self.skip_fetch_cycles = 0  # Skip fetch for N cycles (after reseed)
        self.cached_markets: List[Market] = []  # Cached markets from reseed

        # Reseed circuit breaker
        self.reseed_consecutive_failures = 0  # Track consecutive reseed failures
        self.reseed_degraded_until = 0.0  # Unix timestamp when reseed unpaused

        # Astra V2 integration
        self.astra_predictions = {}

    async def startup(self):
        """
        Startup sequence.

        1. Preflight checks
        2. Reconciliation
        3. WebSocket connections
        4. Load predictions
        """
        logger.info("=" * 80)
        logger.info("MARKET-MAKER STARTUP")
        logger.info("=" * 80)

        # Preflight checks (skip in paper mode)
        if not PAPER_MODE:
            logger.info("Running wallet preflight checks...")
            try:
                preflight_checks(self.clob_client)
            except Exception as e:
                logger.error(f"Preflight checks failed: {e}")
                raise

            # Reconciliation
            logger.info("Reconciling order state with CLOB...")
            try:
                clob_orders = []  # type: ignore[var-annotated]  # TODO: Fetch from CLOB
                self.order_store.reconcile_with_clob(clob_orders)
            except Exception as e:
                logger.error(f"Reconciliation failed: {e}")
        else:
            logger.info("PAPER_MODE: Skipping wallet checks and reconciliation")

        # Connect WebSocket feeds
        logger.info("Connecting WebSocket feeds...")
        await self.market_feed.connect()

        # Start market feed listener task
        asyncio.create_task(self.market_feed.listen())

        # Give listener a moment to start
        await asyncio.sleep(0.5)

        if not PAPER_MODE:
            await self.user_feed.connect()
            await self.user_feed.subscribe()
        else:
            logger.info("PAPER_MODE: Skipping user feed connection")

        # Load Astra predictions
        logger.info("Loading Astra V2 predictions...")
        from config import PREDICTIONS_FILE

        self.astra_predictions = load_astra_predictions(PREDICTIONS_FILE)  # type: ignore[arg-type]

        logger.info("Startup complete. Beginning main loop...")
        self.running = True

    async def main_loop(self):
        """
        Main event loop.

        Cycle:
        1. Fetch markets
        2. Subscribe to order books
        3. Compute QS → select active set
        4. Generate maker quotes
        5. Scan parity arb
        6. Scan satellite opportunities
        7. Submit orders
        8. Monitor for mutations
        """
        cycle_count = 0

        while self.running:
            cycle_start = time.time()
            cycle_count += 1

            logger.info(f"--- Cycle {cycle_count} ---")

            try:
                # Ensure WS connection alive at start of each cycle
                await self.market_feed.ensure_connected()

                # Check WS degraded mode
                if self.market_feed.degraded_mode:
                    logger.warning("WS in degraded mode, skipping cycle operations")
                    await asyncio.sleep(60)
                    continue

                # Check unsafe mode (staleness detected)
                if self.unsafe_mode:
                    logger.warning("UNSAFE MODE: Skipping quoting cycle")
                    await asyncio.sleep(5)
                    continue

                # Check reseed degraded mode
                if self.reseed_degraded_until > time.time():
                    logger.debug(f"RESEED circuit breaker active until {self.reseed_degraded_until:.0f}")
                    # Still allow WS reconnection, just skip reseeding
                    await asyncio.sleep(60)
                    continue

                # 1. Fetch markets (or use cached from reseed)
                if self.skip_fetch_cycles > 0:
                    logger.info(f"SKIP_FETCH_AFTER_RESEED: using cached {len(self.cached_markets)} markets")
                    markets = self.cached_markets
                    self.skip_fetch_cycles -= 1
                else:
                    markets = await self._fetch_markets()
                    logger.info(f"Fetched {len(markets)} markets")

                # 1b. Refresh event metadata (GAP #7 integration)
                if cycle_count % 60 == 0:  # Every 60 cycles (~1 hour with 1min cycles)
                    try:
                        events = await self.event_refresher.refresh_events(markets)
                        negRisk_markets = self.event_refresher.get_negRisk_markets(markets)
                        if negRisk_markets:
                            logger.warning(
                                f"🚨 {len(negRisk_markets)} negRisk markets detected. "
                                "Parity arb will be disabled for these markets."
                            )
                    except Exception as e:
                        logger.error(f"Event refresh failed: {e}", exc_info=True)

                # 2. Subscribe to order books
                await self._subscribe_books(markets)

                # Warmup: wait for initial book snapshots (Cycle 1 only)
                if cycle_count == 1:
                    from config import WS_WARMUP_MIN_BOOKS, WS_WARMUP_TIMEOUT_S

                    logger.info(
                        f"Cycle 1 warmup: waiting up to {WS_WARMUP_TIMEOUT_S}s for min {WS_WARMUP_MIN_BOOKS} books..."
                    )

                    warmup_start = time.time()
                    warmup_elapsed = 0.0

                    while warmup_elapsed < WS_WARMUP_TIMEOUT_S:
                        feed_health = self.market_feed.get_feed_health()
                        books_received = feed_health["unique_assets_with_book"]

                        if books_received >= WS_WARMUP_MIN_BOOKS:
                            logger.info(f"Warmup complete: {books_received} books received in {warmup_elapsed:.1f}s")
                            break

                        await asyncio.sleep(0.5)  # Check every 500ms
                        warmup_elapsed = time.time() - warmup_start

                    # Log final warmup stats
                    feed_health = self.market_feed.get_feed_health()
                    logger.info(
                        f"Warmup finished: warmup_s={warmup_elapsed:.1f} "
                        f"books_received={feed_health['unique_assets_with_book']} "
                        f"msg_types={feed_health['msg_type_counts']}"
                    )

                # 3. Update market states
                for market in markets:
                    market.state = market_state_module.update_market_state(market)

                # 4. Compute QS with filter reason tracking
                reset_qs_veto_counters()  # Reset counters for this cycle

                self.qs_scores = {}
                filter_reasons = {
                    "total": len(markets),
                    "no_book": 0,
                    "negRisk": 0,
                    "qs_zero": 0,
                    "qs_nonzero": 0,
                }

                # Sample debug for first 3 markets with books
                sample_count = 0

                for market in markets:
                    book = self.market_feed.get_book(market.yes_token_id)
                    if not book:
                        filter_reasons["no_book"] += 1
                        continue

                    # Compute RRS (GAP #7 integration)
                    rrs = compute_rrs(market, market.raw_metadata)

                    qs = compute_qs(market, book, rrs, market.state)

                    # Override with toxicity
                    cluster_id = self.risk_engine.assign_cluster(market)
                    qs = self.markout_tracker.override_quoteability(market.condition_id, cluster_id, qs)

                    self.qs_scores[market.condition_id] = qs

                    # Sample debug for first 5 markets with books (enhanced logging)
                    if sample_count < 5:
                        spread = book.best_ask - book.best_bid if book.best_bid and book.best_ask else 0
                        spread_bps = (
                            spread / ((book.best_ask + book.best_bid) / 2) * 10000
                            if book.best_bid and book.best_ask
                            else 0
                        )

                        # Determine primary veto reason
                        primary_veto = "none"
                        if qs == 0:
                            # Check veto reasons in order of priority
                            if rrs > 0.35:
                                primary_veto = "rrs"
                            elif market.state in (
                                MarketState.CLOSE_WINDOW,
                                MarketState.POST_CLOSE,
                                MarketState.PROPOSED,
                                MarketState.CHALLENGE_WINDOW,
                            ):
                                primary_veto = "state"
                            elif spread < market.tick_size:
                                primary_veto = "crossed"
                            elif book.timestamp_age_ms > (
                                (QS_BOOK_STALENESS_S_PAPER if PAPER_MODE else QS_BOOK_STALENESS_S_PROD) * 1000
                            ):
                                primary_veto = "stale"
                            elif market.liquidity < QS_MIN_LIQUIDITY:
                                primary_veto = "liquidity"
                            else:
                                primary_veto = "other"

                        logger.info(
                            f"QS_SAMPLE cond={market.condition_id[:12]} yes_asset={market.yes_token_id[:12]} "
                            f"has_book=True book_age_s={book.timestamp_age_ms / 1000:.1f} "
                            f"bid={book.best_bid:.3f} ask={book.best_ask:.3f} spread={spread:.4f} tick={market.tick_size:.2f} "
                            f"rrs={rrs:.2f} state={market.state.value} qs={qs:.3f} veto={primary_veto}"
                        )
                        sample_count += 1

                    # Track filter reasons
                    if "negRisk" in cluster_id:
                        filter_reasons["negRisk"] += 1
                    if qs == 0:
                        filter_reasons["qs_zero"] += 1
                    else:
                        filter_reasons["qs_nonzero"] += 1

                logger.info(
                    f"Filter summary: total={filter_reasons['total']} "
                    f"no_book={filter_reasons['no_book']} "
                    f"negRisk={filter_reasons['negRisk']} "
                    f"qs_zero={filter_reasons['qs_zero']} "
                    f"qs_nonzero={filter_reasons['qs_nonzero']}"
                )

                # Log QS veto summary
                from strategy.quoteability_scorer import get_qs_veto_summary

                logger.info(f"QS veto summary: {get_qs_veto_summary()}")

                # Log top 10 markets by QS (for debugging)
                if self.qs_scores:
                    top_10 = sorted(self.qs_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    logger.info("Top 10 QS: " + ", ".join([f"{cid[:8]}={qs:.3f}" for cid, qs in top_10]))

                # Log feed health
                feed_health = self.market_feed.get_feed_health()
                logger.info(
                    f"FEED_HEALTH ws_msgs={feed_health['ws_msgs']} "
                    f"book_msgs={feed_health['book_msgs']} "
                    f"unique_assets_with_book={feed_health['unique_assets_with_book']} "
                    f"parse_err={feed_health['parse_err']} "
                    f"max_age_s={feed_health['max_age_seconds']:.1f} "
                    f"msg_types={feed_health['msg_type_counts']}"
                )

                # 5. Select active set
                cluster_assignments = {m.condition_id: self.risk_engine.assign_cluster(m) for m in markets}

                self.active_set = select_active_set(markets, self.qs_scores, cluster_assignments)

                logger.info(f"Active set: {len(self.active_set)} markets")

                # Track zero-active cycles for reseed triggering
                if len(self.active_set) == 0:
                    self.zero_active_cycles += 1
                else:
                    self.zero_active_cycles = 0  # Reset on non-zero active set

                # Reseed check: trigger if persistently zero active
                from config import RESEED_MIN_INTERVAL_SECONDS, RESEED_TRIGGER_ZERO_ACTIVE_CYCLES

                now = time.time()
                time_since_last_reseed = now - self.last_reseed_time

                if (
                    self.zero_active_cycles >= RESEED_TRIGGER_ZERO_ACTIVE_CYCLES
                    and time_since_last_reseed >= RESEED_MIN_INTERVAL_SECONDS
                ):
                    logger.warning(
                        f"RESEED TRIGGER: {self.zero_active_cycles} consecutive zero-active cycles. "
                        f"Reseeding universe..."
                    )

                    # Wrap reseed in try/except with circuit breaker
                    try:
                        await self._reseed_universe()
                        self.zero_active_cycles = 0
                        self.last_reseed_time = now
                        self.reseed_consecutive_failures = 0  # Reset on success
                        # Skip rest of this cycle to allow new subscriptions to warm up
                        await asyncio.sleep(10)
                        continue
                    except Exception as e:
                        self.reseed_consecutive_failures += 1
                        logger.error(
                            f"RESEED failed ({self.reseed_consecutive_failures} consecutive): {e}", exc_info=True
                        )

                        if self.reseed_consecutive_failures >= 5:
                            # Enter degraded mode: pause reseeding for 10 minutes
                            self.reseed_degraded_until = time.time() + 600
                            logger.error(
                                f"RESEED circuit breaker triggered after 5 failures. "
                                f"Pausing reseeds until {self.reseed_degraded_until:.0f} (10 minutes)"
                            )

                # 6. Generate maker quotes
                maker_intents = []
                for market in self.active_set:
                    book = self.market_feed.get_book(market.yes_token_id)
                    if not book:
                        continue

                    # Get inventory
                    inventory_usd = self.risk_engine.get_market_exposure(market.condition_id)

                    # Get RRS (GAP #7 integration)
                    rrs = compute_rrs(market, market.raw_metadata)

                    # Create orders
                    bid_order, ask_order = create_maker_orders(market, book, inventory_usd, rrs, market.state)

                    if bid_order:
                        maker_intents.append(bid_order)
                    if ask_order:
                        maker_intents.append(ask_order)

                logger.info(f"Generated {len(maker_intents)} maker orders")

                # 7. Scan parity arb
                yes_books = {m.condition_id: self.market_feed.get_book(m.yes_token_id) for m in markets}
                no_books = {m.condition_id: self.market_feed.get_book(m.no_token_id) for m in markets}

                parity_opps = scan_all_parity(markets, yes_books, no_books, self.risk_engine)

                logger.info(f"Parity scan: {len(parity_opps)} opportunities")

                # 8. Scan satellite opportunities
                satellite_exposure = self.risk_engine.exposure.satellite_risk_used_usd

                satellite_recs = scan_satellite_opportunities(markets, self.astra_predictions, satellite_exposure)

                logger.info(f"Satellite scan: {len(satellite_recs)} opportunities")

                # 9. Submit orders (batched)
                if maker_intents:
                    batches = self.executor.slice_batch(maker_intents)
                    for batch in batches:
                        result = await self.executor.submit_batch_orders(batch)
                        logger.info(f"Batch submitted: {result['submitted']} orders, {result['failed']} failed")
                        # Record quote events
                        self.truth_report_builder.record_quote_event("quote", count=result["submitted"])

                # 10. Monitor staleness (skip in PAPER_MODE for burn-in)
                if not PAPER_MODE:
                    self.market_feed.check_staleness()
                    self.order_store.cancel_stale_orders(WS_STALENESS_THRESHOLD_MS)

                # 11. Paper trading: simulate fills (GAP #8 integration)
                if self.paper_simulator:
                    await self._simulate_paper_fills(markets)

                # 12. Check for date change and generate daily report
                self._check_daily_report()

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

            # Sleep until next cycle
            cycle_duration = time.time() - cycle_start
            sleep_time = max(1.0, 60.0 - cycle_duration)  # 1-minute cycles
            await asyncio.sleep(sleep_time)

    async def _fetch_markets(self) -> List[Market]:
        """
        Fetch markets from Gamma API with full metadata (GAP #6 FIX).

        Returns markets with:
        - tick_size, fee_rate_bps
        - time_to_close
        - event association + negRisk flags

        PAPER MODE: Fetches more markets (500 vs 200) for better NORMAL state coverage.
        """
        try:
            from config import MARKET_FETCH_LIMIT_PAPER, MARKET_FETCH_LIMIT_PROD, PAPER_MODE

            fetch_limit = MARKET_FETCH_LIMIT_PAPER if PAPER_MODE else MARKET_FETCH_LIMIT_PROD

            markets = await fetch_markets_with_metadata(
                limit=fetch_limit,
                min_liquidity=500.0,
                active_only=True,
            )

            logger.info(
                f"Fetched {len(markets)} markets (mode={'PAPER' if PAPER_MODE else 'PROD'}, limit={fetch_limit})"
            )

            # Register markets with executor (GAP #1 FIX)
            for market in markets:
                self.executor.register_market(market)
                self.markets[market.condition_id] = market

            return markets

        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}", exc_info=True)
            return []

    async def _subscribe_books(self, markets: List[Market]):
        """
        Subscribe to order books for markets with prefiltering.

        PAPER MODE OPTIMIZATION:
        - Excludes close_window markets (time_to_close < 24h) to improve book quality
        - Ranks by liquidity proxy and subscribes to top MAX_SUBSCRIBE_MARKETS
        - Batches subscriptions to avoid server rejection
        """
        from config import (
            MAX_SUBSCRIBE_MARKETS,
            PAPER_MODE,
            PREFILTER_EXCLUDE_CLOSE_WINDOW,
            STATE_CLOSE_WINDOW_THRESHOLD_HOURS,
        )
        from risk.market_state import update_market_state

        # Prefilter markets
        filtered_markets = []
        filter_stats = {"total": len(markets), "excluded_close_window": 0, "excluded_negRisk": 0}

        for market in markets:
            # Update market state
            market.state = update_market_state(market, {})

            # Exclude close_window markets in paper mode if configured
            if PAPER_MODE and PREFILTER_EXCLUDE_CLOSE_WINDOW:
                if market.state == MarketState.CLOSE_WINDOW:
                    filter_stats["excluded_close_window"] += 1
                    continue

            # Exclude negRisk markets (use risk engine method)
            cluster_id = self.risk_engine.assign_cluster(market)
            if "negRisk" in cluster_id:
                filter_stats["excluded_negRisk"] += 1
                continue

            filtered_markets.append(market)

        logger.info(
            f"Prefilter: {len(filtered_markets)}/{filter_stats['total']} markets "
            f"(excluded: close_window={filter_stats['excluded_close_window']}, "
            f"negRisk={filter_stats['excluded_negRisk']})"
        )

        # Rank by activity score and limit to MAX_SUBSCRIBE_MARKETS
        from scanner.market_fetcher_v2 import compute_activity_score

        for market in filtered_markets:
            market.activity_score = compute_activity_score(market)

        filtered_markets.sort(key=lambda m: m.activity_score, reverse=True)
        subscribe_markets = filtered_markets[:MAX_SUBSCRIBE_MARKETS]

        # Log top 5 activity scores for observability
        if subscribe_markets:
            top_5_scores = [
                (m.condition_id[:12], m.activity_score, m.volume_24h, m.liquidity) for m in subscribe_markets[:5]
            ]
            logger.info(f"Top 5 activity scores: {top_5_scores}")

        logger.info(f"Subscribing to top {len(subscribe_markets)} markets by activity score")

        # Build asset ID list
        asset_ids = []
        for market in subscribe_markets:
            asset_ids.append(market.yes_token_id)
            asset_ids.append(market.no_token_id)

        # Subscribe to new assets in batches of 100
        new_assets = [a for a in asset_ids if a not in self.market_feed.subscribed_assets]
        if new_assets:
            batch_size = 100
            for i in range(0, len(new_assets), batch_size):
                batch = new_assets[i : i + batch_size]
                await self.market_feed.subscribe(batch)
                await asyncio.sleep(0.1)  # Small delay between batches

    async def _reseed_universe(self):
        """
        Reseed market universe when Active set goes persistently zero.

        Triggered when:
        - Active set = 0 for RESEED_TRIGGER_ZERO_ACTIVE_CYCLES consecutive cycles
        - Min RESEED_MIN_INTERVAL_SECONDS has passed since last reseed

        Actions:
        1. Unsubscribe from all current assets (best effort, skip if WS dead)
        2. Ensure WS connection alive
        3. Fetch fresh market data
        4. Refilter and resubscribe to new active universe
        5. Warmup to collect fresh books
        """
        logger.info("=== RESEED UNIVERSE ===")

        # 1. Unsubscribe from current assets (BEST EFFORT)
        if self.market_feed.is_connected():
            current_assets = list(self.market_feed.subscribed_assets)
            if current_assets:
                logger.info(f"Unsubscribing from {len(current_assets)} stale assets...")
                try:
                    await self.market_feed.unsubscribe(current_assets)
                except Exception as e:
                    logger.warning(f"Unsubscribe failed (non-fatal): {e}")
        else:
            logger.warning("WS disconnected, skipping unsubscribe (will reconnect fresh)")

        # Clear local tracking (in case unsubscribe skipped)
        self.market_feed.subscribed_assets.clear()
        self.market_feed.active_asset_ids.clear()

        # 2. Ensure WS connection alive before proceeding
        await self.market_feed.ensure_connected()

        # 3. Fetch fresh markets
        markets = await self._fetch_markets()
        logger.info(f"Reseed: fetched {len(markets)} fresh markets")

        # Cache markets and skip fetch for next cycle
        self.cached_markets = markets
        self.skip_fetch_cycles = 1
        logger.info("Reseed: cached markets for next cycle (skip_fetch_cycles=1)")

        # 4. Resubscribe to new active universe
        await self._subscribe_books(markets)

        # 5. Warmup to collect fresh books
        from config import WS_WARMUP_MIN_BOOKS, WS_WARMUP_TIMEOUT_S

        logger.info(f"Reseed warmup: waiting up to {WS_WARMUP_TIMEOUT_S}s for {WS_WARMUP_MIN_BOOKS} books...")

        warmup_start = time.time()
        warmup_elapsed = 0.0

        while warmup_elapsed < WS_WARMUP_TIMEOUT_S:
            feed_health = self.market_feed.get_feed_health()
            books_received = feed_health["unique_assets_with_book"]

            if books_received >= WS_WARMUP_MIN_BOOKS:
                logger.info(f"Reseed warmup complete: {books_received} books in {warmup_elapsed:.1f}s")
                break

            await asyncio.sleep(0.5)
            warmup_elapsed = time.time() - warmup_start

        feed_health = self.market_feed.get_feed_health()
        logger.info(
            f"Reseed complete: books_received={feed_health['unique_assets_with_book']} warmup_s={warmup_elapsed:.1f}"
        )

    async def _simulate_paper_fills(self, markets: List[Market]):
        """
        Simulate fills for paper trading (GAP #8 integration).

        CRITICAL: Uses conservative fill probabilities to avoid fake Sharpe.
        """
        # Record current book snapshots
        for market in markets:
            book = self.market_feed.get_book(market.yes_token_id)
            if book:
                self.paper_simulator.record_book_snapshot(market.condition_id, book)  # type: ignore[union-attr]

        # Check each live order for potential fill
        live_orders = self.order_store.get_live_orders()

        for order in live_orders:
            # Get current book
            book = self.market_feed.get_book(order.token_id)
            if not book:
                continue

            # Calculate time in market
            from datetime import datetime

            placed_time = datetime.fromisoformat(order.placed_at)
            time_in_market = (datetime.now() - placed_time).total_seconds()

            # Simulate fill probability
            filled, size_filled, fill_price = self.paper_simulator.simulate_fill_probability(  # type: ignore[union-attr]
                order,
                time_in_market,
                book,
            )

            if filled and size_filled > 0:
                # Create simulated Fill
                from execution.mid import compute_mid

                mid = compute_mid(book.best_bid, book.best_ask, book.last_mid, book.timestamp_age_ms)

                fill = self.paper_simulator.create_simulated_fill(  # type: ignore[union-attr]
                    order,
                    size_filled,
                    fill_price,  # type: ignore[arg-type]
                    mid or 0.5,
                )

                # Route through the same fill handler
                self._on_fill(fill)

                logger.info(f"Paper fill simulated: {order.order_id} {size_filled:.2f} tokens @ {fill_price:.4f}")

    def _on_fill(self, fill: Fill):
        """
        Handle fill event (GAP #3 + #4 + markout integration).

        CRITICAL: This is the single truth path for both WS and paper fills.
        """
        logger.info(
            f"Fill: {fill.fill_id} {fill.side} {fill.size_tokens:.2f} @ {fill.price:.4f} "
            f"({'MAKER' if fill.maker else 'TAKER'})"
        )

        # GAP #3: Update order state with partial fill
        if fill.order_id in self.order_store.orders:
            order = self.order_store.orders[fill.order_id]

            # Update partial fill in order store
            self.order_store.update_partial_fill(
                fill.order_id,
                fill.size_tokens,
                datetime.now().isoformat(),
            )

            # GAP #4: Update reservation (release filled portion)
            order_dict = {
                "order_id": order.order_id,
                "condition_id": order.condition_id,
                "token_id": order.token_id,
                "side": order.side,
                "price": order.price,
                "size_in_tokens": fill.size_tokens,  # Size filled
            }
            self.risk_engine.update_reservation_partial_fill(
                fill.order_id,
                fill.size_tokens,
                order_dict,
            )

            # GAP #2: Classify maker/taker
            from execution.mid import compute_mid

            book = self.market_feed.get_book(order.token_id)
            book_mid = None
            if book:
                book_mid = compute_mid(
                    book.best_bid,
                    book.best_ask,
                    book.last_mid,
                    book.timestamp_age_ms,
                )

            fill.classify_maker_taker(order, book_mid)

        # Markout tracking
        if fill.maker:
            # Record book snapshot for markout calculation
            book = self.market_feed.get_book(fill.token_id)
            if book:
                self.markout_tracker.record_fill_for_markout(
                    fill,
                    book,
                )

        # Record in truth report (NEW: TruthReportBuilder)
        market = self.markets.get(fill.condition_id)
        cluster_id = self.risk_engine.assign_cluster(market) if market else "unknown"

        pnl = 0.0  # TODO: Compute realized P&L from position tracking

        # Get markout values if available
        markout_data = self.markout_tracker.get_markout(fill.fill_id) if fill.maker else {}  # type: ignore[attr-defined]

        self.truth_report_builder.record_fill(
            fill=fill,
            cluster_id=cluster_id,
            pnl=pnl,
            is_maker=fill.maker,
            realized_spread=markout_data.get("realized_spread"),
            markout_30s=markout_data.get("markout_30s"),
            markout_2m=markout_data.get("markout_2m"),
            markout_10m=markout_data.get("markout_10m"),
        )

    def _on_order_update(self, order_id: str, status, timestamp_iso: str):
        """Handle order status update."""
        self.order_store.update_order_status(order_id, status, timestamp_iso)

        # Record cancel/replace events
        if status == "CANCELED":
            self.truth_report_builder.record_quote_event("cancel")
        elif status == "REPLACED":
            self.truth_report_builder.record_quote_event("replace")

    def _check_daily_report(self):
        """
        Check if date has changed and generate daily report.

        Called at end of each main loop cycle.
        """
        current_date = datetime.now().strftime("%Y-%m-%d")

        if current_date != self.current_date:
            # Date changed - finalize and write report for previous day
            logger.info(f"Date changed from {self.current_date} to {current_date}. Generating daily report.")

            # Compute portfolio snapshot (placeholder values for now)
            portfolio_snapshot = {
                "daily_return": 0.0,
                "weekly_return": 0.0,
                "monthly_return": 0.0,
                "sharpe_90d": 0.0,
                "calmar_90d": 0.0,
                "max_drawdown": 0.0,
                "cluster_exposures": dict(self.risk_engine.exposure.cluster_exposure_usd),  # type: ignore[attr-defined]
                "aggregate_exposure": sum(self.risk_engine.exposure.cluster_exposure_usd.values()),  # type: ignore[attr-defined]
                "max_market_inventory": max(self.risk_engine.exposure.market_exposure_usd.values(), default=0.0),  # type: ignore[attr-defined]
            }

            self.truth_report_builder.set_portfolio_snapshot(**portfolio_snapshot)  # type: ignore[arg-type]

            # Record health metrics
            # TODO: Track actual uptime from feed connections
            self.truth_report_builder.record_health(
                ws_market_uptime_seconds=86000.0,  # Placeholder
                ws_user_uptime_seconds=85000.0,  # Placeholder
            )

            # Finalize and write report
            report = self.truth_report_builder.finalize()
            report_path = write_daily_report(report)
            logger.info(f"Daily report written: {report_path}")

            # Start new builder for new date
            self.current_date = current_date
            self.truth_report_builder = TruthReportBuilder(date=current_date)

    def _on_feed_stale(self, asset_id: str, age_ms: int):
        """Handle stale feed event (CIRCUIT BREAKER)."""
        logger.error(
            f"CIRCUIT BREAKER: Feed stale for {asset_id} (age={age_ms}ms). "
            "Entering unsafe mode and canceling all orders."
        )

        self.unsafe_mode = True

        # Cancel all orders
        asyncio.create_task(self.executor.cancel_all(reason="Stale feed"))

    def _on_feed_disconnect(self):
        """Handle user feed disconnect (CIRCUIT BREAKER)."""
        logger.error("CIRCUIT BREAKER: User feed disconnected. Entering unsafe mode and canceling all orders.")

        self.unsafe_mode = True
        asyncio.create_task(self.executor.cancel_all(reason="Feed disconnect"))

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down market-maker...")

        self.running = False

        # Cancel all orders
        await self.executor.cancel_all(reason="Shutdown")

        # Stop feeds
        self.market_feed.stop()
        self.user_feed.stop()

        logger.info("Shutdown complete")


async def main():
    """Entry point."""
    # TODO: Initialize py-clob-client
    clob_client = None

    runtime = MarketMakerRuntime(clob_client)

    try:
        await runtime.startup()
        await runtime.main_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
