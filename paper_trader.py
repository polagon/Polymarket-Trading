"""
Paper Trading Loop — continuous simulation for learning and improvement.

This is the test harness that:
1. Scans for opportunities every 10 minutes
2. Simulates position entry at current market prices
3. Tracks positions until markets resolve
4. Records outcomes and P&L
5. Feeds results to the learning agent to improve future estimates

Run: python paper_trader.py
     python paper_trader.py --fast    (1-minute scan interval for testing)

The "AI brain" (learning_agent.py) observes every outcome and updates
strategy guidance that flows back into Claude's estimation prompts.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import report
from config import (
    ALERT_COOLDOWN_SECONDS,
    ALERT_WEBHOOK_URL,
    ALLOW_NEW_DB,
    ANTHROPIC_API_KEY,
    ASTRA_HEALTH_PATH,
    BANKROLL,
    BURN_IN_MAX_FAILED_MONITORS,
    BURN_IN_MONITOR_CYCLES,
    CLAUDE_MODEL,
    DEPTH_PROXY_FLOOR_USD,
    DISK_FAIL_MB,
    DISK_WARN_MB,
    ERRORS_FAIL,
    ERRORS_WARN,
    EV_BASE_ADVERSE_BUFFER,
    EV_BASE_SLIPPAGE,
    EV_EPS,
    EV_K_DEPTH,
    EV_K_SPREAD,
    EV_MAKER_THRESHOLD,
    FEED_FAIL_AGE_S,
    FEED_WARN_AGE_S,
    FLOW_TOXICITY_MIN_SAMPLES,
    FLOW_TOXICITY_THRESHOLD,
    FLOW_TOXICITY_WINDOW,
    FRED_API_KEY,
    GATE_FEED_MAX_AGE_S,
    GATE_MAX_CUMULATIVE_DD_PCT,
    GATE_MAX_DAILY_LOSS_PCT,
    GATE_MAX_ERRORS_PER_CYCLE,
    GATE_MAX_MEMORY_MB,
    GATE_MIN_DISK_FREE_MB,
    JSON_LOGGING,
    MAX_DAILY_LOSS_PCT,
    MEMORY_FAIL_MB,
    MEMORY_WARN_MB,
    ODDS_API_KEY,
    ORDER_DEFAULT_TTL_S,
    ORDER_MAX_CHASE_TICKS,
    ORDER_STALE_THRESHOLD_S,
    ORDER_TICK_SIZE,
    PAPER_MODE,
    RISK_CATEGORY_CAPS,
    RISK_DAILY_LOSS_HALT_PCT,
    RISK_DRAWDOWN_HALT_PCT,
    RISK_HALT_COOLDOWN_S,
    SCAN_INTERVAL_SECONDS,
    compute_config_hash,
    get_canonical_config_dict,
    get_git_sha,
)
from data_sources import crypto as crypto_source
from data_sources.economic_calendar import check_markets_for_events, format_calendar_context, get_todays_events
from data_sources.signals import fetch_all_signals
from data_sources.sports import get_sports_estimates
from data_sources.weather import fetch_forecast, parse_weather_question
from data_sources.whale_tracker import format_whale_context, track_volume_and_detect_whales

# ── Loop 4: Allocator-grade modules ──────────────────────────────────────────
from definitions.registry import DefinitionRegistry
from execution.order_manager import OrderManager
from metrics.drawdown import DrawdownTracker
from metrics.performance import PerformanceEngine, trade_record_from_paper_position
from ops.alerts import AlertManager
from ops.artifacts import ArtifactWriter
from ops.gate_engine import GateContext, GateEngine
from ops.health import _get_disk_free_mb, _get_memory_mb, write_heartbeat
from ops.logging_setup import setup_logging
from ops.run_context import RunContext, set_cycle_context
from ops.startup_checklist import all_passed, run_startup_checklist
from risk.risk_engine import RiskEngine
from scanner.kelly_sizer import size_position
from scanner.learning_agent import LearningAgent, Prediction
from scanner.longshot_screener import scan_for_arbitrage, screen_longshot_markets, summarize_longshot_stats
from scanner.market_fetcher import fetch_active_markets
from scanner.mispricing_detector import find_opportunities
from scanner.probability_estimator import PROMPT_BUNDLE_HASH, PROMPT_REGISTRY, estimate_markets
from scanner.semantic_clusters import SemanticClusterEngine
from scanner.trade_logger import CURRENT_SCHEMA_VERSION, backup_db, init_db
from signals.flow_toxicity import FlowToxicityAnalyzer
from strategies.crypto_threshold import CryptoThresholdStrategy
from telemetry.trade_telemetry import TradeTelemetry

POSITIONS_FILE = Path("memory/paper_positions.json")
PNL_FILE = Path("memory/paper_pnl.json")

# Heartbeat tracking — module-level so it persists across scan calls
_last_successful_fetch: Optional[datetime] = None


@dataclass
class PaperPosition:
    condition_id: str
    question: str
    category: str
    direction: str  # "BUY YES" or "BUY NO"
    entry_price: float  # Price we "bought" at
    position_size: float  # USD amount
    our_probability: float
    market_price_at_entry: float
    timestamp: str
    resolved: bool = False
    outcome: Optional[bool] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    resolution_time: Optional[str] = None
    vix_kelly_mult: Optional[float] = None  # VIX dampening applied at entry (for learning agent)
    metadata: Optional[dict] = None  # Misc tracking (e.g. dual-confirmation timestamps)


class PaperPortfolio:
    def __init__(self):
        Path("memory").mkdir(exist_ok=True)
        self.positions: list[PaperPosition] = self._load()
        self.cash = BANKROLL
        self.invested = 0.0
        self._monitor_only: bool = False

    # -- Monitor-only API (Activity 17: belt-and-suspenders) ---------------

    def set_monitor_only(self, enabled: bool) -> None:
        """Enable or disable monitor-only mode. When enabled, open_position() is blocked."""
        self._monitor_only = enabled

    @property
    def monitor_only(self) -> bool:
        """True if monitor-only mode is active (positions cannot be opened)."""
        return self._monitor_only

    def _load(self) -> list[PaperPosition]:
        if POSITIONS_FILE.exists():
            try:
                data = json.loads(POSITIONS_FILE.read_text())
                return [PaperPosition(**p) for p in data]
            except Exception as e:
                logging.getLogger("astra.portfolio").error("Failed to load positions from %s: %s", POSITIONS_FILE, e)
                return []
        return []

    def _save(self):
        POSITIONS_FILE.write_text(json.dumps([asdict(p) for p in self.positions], indent=2))

    def open_position(self, opp, sizing: dict, vix_kelly_mult: float = 1.0) -> Optional[PaperPosition]:
        """Open a paper position for an opportunity."""
        # Activity 17: Belt-and-suspenders — block positions in monitor-only mode
        if self._monitor_only:
            logging.getLogger("astra.portfolio").warning("open_position blocked: monitor-only mode active")
            return None

        pos_size = sizing["position_dollars"]
        if pos_size <= 0:
            return None

        # Check we have enough cash
        if pos_size > self.cash:
            pos_size = min(self.cash * 0.5, pos_size)

        if pos_size < 1.0:
            return None

        # Don't double up on same market
        open_ids = {p.condition_id for p in self.positions if not p.resolved}
        if opp.market.condition_id in open_ids:
            return None

        entry_price = opp.market_price if opp.direction == "BUY YES" else (1.0 - opp.market_price)

        pos = PaperPosition(
            condition_id=opp.market.condition_id,
            question=opp.market.question,
            category=opp.market.category,
            direction=opp.direction,
            entry_price=entry_price,
            position_size=pos_size,
            our_probability=opp.our_estimate,
            market_price_at_entry=opp.market_price,
            timestamp=datetime.now(timezone.utc).isoformat(),
            vix_kelly_mult=vix_kelly_mult,  # Record for learning agent correlation analysis
        )

        self.cash -= pos_size
        self.invested += pos_size
        self.positions.append(pos)
        self._save()
        return pos

    def check_resolutions(self, current_markets: list) -> list[PaperPosition]:
        """
        Check if any open positions have resolved.

        A market is considered RESOLVED only if:
          1. It has disappeared from the active markets list AND was entered ≥2 hours ago
             (brief disappearances can happen due to API lag — the 2h buffer avoids false hits)
          2. OR its hours_to_expiry has passed (end date is in the past)

        We do NOT use price proximity (0.02 / 0.98) as a resolution signal — a market
        can trade at 1–2% for days and then recover. Low price ≠ resolved.
        """
        market_map = {m.condition_id: m for m in current_markets}
        newly_resolved = []
        now = datetime.now(timezone.utc)

        for pos in self.positions:
            if pos.resolved:
                continue

            market = market_map.get(pos.condition_id)

            if market is not None:
                # Market still active — only resolve if end date has passed
                if market.hours_to_expiry > 0:
                    continue  # Still live, nothing to do
                # hours_to_expiry <= 0: end date passed.
                # DUAL-CONFIRMATION GUARD: require expiry to have been <= 0 for
                # at least 2 consecutive scans before resolving.
                # This prevents false resolution from stale/cached hours_to_expiry.
                last_seen_expired = pos.metadata.get("first_seen_expired_at") if pos.metadata else None
                if last_seen_expired is None:
                    # First time we've seen this market expired — record timestamp, don't resolve yet
                    if pos.metadata is None:
                        pos.metadata = {}
                    pos.metadata["first_seen_expired_at"] = now.isoformat()
                    self._save()
                    continue
                # Check if we've seen it expired for at least 15 minutes (2 scan intervals)
                try:
                    first_expired = datetime.fromisoformat(last_seen_expired.replace("Z", "+00:00"))
                    minutes_expired = (now - first_expired).total_seconds() / 60
                except Exception:
                    minutes_expired = 999
                if minutes_expired < 15:
                    continue  # Not confirmed expired yet
                # Dual-confirmed expired — use final market price for outcome
                resolved_yes = market.yes_price >= 0.5
            else:
                # Market not in active list — check entry age before marking resolved
                # (avoids false resolution from transient API gaps)
                try:
                    entry_time = datetime.fromisoformat(pos.timestamp.replace("Z", "+00:00"))
                    age_hours = (now - entry_time).total_seconds() / 3600
                except Exception:
                    age_hours = 999  # Unknown age — treat as old enough

                if age_hours < 4.0:
                    # Too recent — API gap more likely than genuine resolution (raised 2h → 4h)
                    continue

                # Market gone and position is old enough — treat as resolved
                # We don't know the outcome, so mark with outcome=None (no P&L)
                pos.resolved = True
                pos.resolution_time = now.isoformat()
                pos.pnl = None  # Unknown outcome — won't poison calibration data
                self._save()
                logger.warning(
                    "Position %s marked resolved-unknown: market disappeared after %.1fh (no outcome data)",
                    pos.condition_id[:16],
                    age_hours,
                )
                continue  # Don't append to newly_resolved (no P&L to record)

            # We have a definitive resolution
            pos.resolved = True
            pos.outcome = resolved_yes
            pos.resolution_time = now.isoformat()

            # Calculate P&L
            if pos.direction == "BUY YES":
                if resolved_yes:
                    exit_price = 1.0
                    shares = pos.position_size / pos.entry_price
                    pos.pnl = shares - pos.position_size
                else:
                    pos.pnl = -pos.position_size
                    exit_price = 0.0
            else:  # BUY NO
                if not resolved_yes:
                    exit_price = 1.0
                    shares = pos.position_size / pos.entry_price
                    pos.pnl = shares - pos.position_size
                else:
                    pos.pnl = -pos.position_size
                    exit_price = 0.0

            pos.exit_price = exit_price
            self.cash += pos.position_size + (pos.pnl or 0)
            self.invested -= pos.position_size
            newly_resolved.append(pos)

        if newly_resolved:
            self._save()

        return newly_resolved

    def get_stats(self) -> dict:
        resolved = [p for p in self.positions if p.resolved and p.pnl is not None]
        open_pos = [p for p in self.positions if not p.resolved]
        total_pnl = sum(p.pnl for p in resolved)  # type: ignore[misc]
        wins = [p for p in resolved if p.pnl > 0]  # type: ignore[operator]

        # Sharpe and Sortino ratios (requires ≥10 resolved positions)
        sharpe_ratio = 0.0
        sortino_ratio = 0.0

        if len(resolved) >= 10:
            # Calculate returns as P&L / entry size
            returns = np.array([p.pnl / p.position_size for p in resolved if p.position_size > 0])  # type: ignore[operator]

            if len(returns) > 0:
                avg_return = np.mean(returns)
                std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0

                # Sharpe = (avg_return - risk_free_rate) / std_dev(returns)
                # Assuming risk-free rate = 0 for simplicity
                if std_return > 1e-9:
                    sharpe_ratio = avg_return / std_return

                # Sortino = (avg_return - risk_free_rate) / std_dev(negative_returns)
                # Only penalize downside volatility
                downside_returns = returns[returns < 0]
                if len(downside_returns) >= 3:
                    downside_std = np.std(downside_returns, ddof=1)
                    if downside_std > 1e-9:
                        sortino_ratio = avg_return / downside_std

        return {
            "total_positions": len(self.positions),
            "open": len(open_pos),
            "resolved": len(resolved),
            "wins": len(wins),
            "losses": len(resolved) - len(wins),
            "win_rate": len(wins) / len(resolved) if resolved else 0.0,
            "total_pnl": round(total_pnl, 2),
            "cash": round(self.cash, 2),
            "invested": round(max(0, self.invested), 2),
            "portfolio_value": round(self.cash + max(0, self.invested), 2),
            "return_pct": round((self.cash + max(0, self.invested) - BANKROLL) / BANKROLL * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
        }


async def run_paper_scan(
    portfolio: PaperPortfolio,
    learning_agent: LearningAgent,
    cluster_engine: SemanticClusterEngine,
    scan_number: int,
    allow_new_positions: bool = True,
):
    global _last_successful_fetch
    t_start = time.time()

    # Fetch markets with focus on shorter-duration markets
    markets = await fetch_active_markets(limit=400)

    # ── Heartbeat check ─────────────────────────────────────────────────────
    if not markets:
        report.console.print("[bold red]⚠ HEARTBEAT FAIL: No markets returned from Gamma API[/bold red]")
        if _last_successful_fetch is not None:
            silent_mins = (datetime.now(timezone.utc) - _last_successful_fetch).total_seconds() / 60
            report.console.print(f"[red]  Last successful fetch: {silent_mins:.0f} min ago — check connectivity[/red]")
        return
    _last_successful_fetch = datetime.now(timezone.utc)

    # Check for resolved positions
    resolved = portfolio.check_resolutions(markets)
    for pos in resolved:
        learning_agent.update_outcome(pos.condition_id, pos.outcome or False, pos.pnl or 0)
        color = "green" if (pos.pnl or 0) > 0 else "red"
        report.console.print(
            f"[{color}]RESOLVED: {pos.question[:50]} → "
            f"{'YES' if pos.outcome else 'NO'} | P&L: ${pos.pnl:+.2f}[/{color}]"
        )

    # ── Circuit breaker: daily loss limit ───────────────────────────────────
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_pnl = sum(
        p.pnl
        for p in portfolio.positions
        if p.resolved and p.pnl is not None and (p.resolution_time or "").startswith(today_str)
    )
    breaker_limit = BANKROLL * MAX_DAILY_LOSS_PCT
    if today_pnl < -breaker_limit:
        report.console.print(
            f"[bold red]⛔ CIRCUIT BREAKER TRIGGERED[/bold red]\n"
            f"   Today P&L: [red]${today_pnl:+.2f}[/red]  "
            f"Limit: [white]-${breaker_limit:.2f}[/white]  "
            f"({MAX_DAILY_LOSS_PCT:.0%} of ${BANKROLL:.0f} bankroll)\n"
            f"   No new positions will be opened for the rest of today."
        )
        # Still run reporting / resolution but skip all new position logic
        _run_scan_reporting_only(portfolio, learning_agent, cluster_engine, scan_number, markets, t_start)
        return

    # Get external data
    crypto_markets = [m for m in markets if m.category == "crypto"]
    weather_markets = [m for m in markets if m.category == "weather"]
    sports_markets = [m for m in markets if m.category == "sports"]

    coin_ids = set()
    for m in crypto_markets:
        p = crypto_source.parse_crypto_question(m.question)
        if p:
            coin_ids.add(p["coin_id"])

    price_data = await crypto_source.fetch_prices(list(coin_ids)) if coin_ids else {}

    locations = set()
    for m in weather_markets:
        p = parse_weather_question(m.question)
        if p and p.get("location"):
            locations.add(p["location"])

    forecasts = {}
    if locations:
        results = await asyncio.gather(*[fetch_forecast(loc) for loc in locations], return_exceptions=True)
        for loc, result in zip(locations, results):
            if not isinstance(result, Exception) and result:
                forecasts[loc] = result

    # Sports odds
    sports_estimates = {}
    if ODDS_API_KEY and sports_markets:
        try:
            sports_estimates = await get_sports_estimates([m.question for m in sports_markets])
        except Exception as e:
            logger.warning("Sports odds fetch failed: %s: %s", type(e).__name__, e)

    # Market signals (Fear/Greed, macro overlay, VIX)
    vix_kelly_mult = 1.0  # default — no VIX dampening
    try:
        market_context = await fetch_all_signals()
        signals_summary = market_context.summary()
        # Extract VIX Kelly multiplier (reduces position sizes in high-vol regimes)
        if market_context.macro is not None:
            vix_kelly_mult = market_context.macro.vix_kelly_multiplier
            if vix_kelly_mult < 1.0:
                vix_val = market_context.macro.vix
                label = market_context.macro.vix_label
                report.console.print(
                    f"[yellow]⚡ VIX={vix_val:.1f} [{label}] — Kelly multiplier: {vix_kelly_mult:.2f}×[/yellow]"
                )
    except Exception as e:
        logger.warning("Market signals fetch failed: %s: %s", type(e).__name__, e)
        signals_summary = ""

    # ── Whale tracker: detect volume spikes (Tier B evidence) ───────────────
    whale_signals = track_volume_and_detect_whales(markets)
    whale_context = format_whale_context(whale_signals)
    if whale_signals:
        report.console.print(f"[dim]🐋 Whale signals: {len(whale_signals)} volume spike(s) detected[/dim]")

    # ── Economic calendar: flag markets near FOMC/CPI/NFP ───────────────────
    calendar_events = check_markets_for_events(markets, window_hours=48.0)
    calendar_context = format_calendar_context(calendar_events, markets)
    if calendar_events:
        report.console.print(f"[dim]📅 Calendar: {len(calendar_events)} market(s) near high-impact events[/dim]")

    # Estimate probabilities — inject all context sources
    learning_context = learning_agent.get_strategy_context()
    context_parts = [p for p in [signals_summary, whale_context, calendar_context, learning_context] if p]
    learning_context = "\n\n".join(context_parts)

    estimates = await estimate_markets(markets, price_data, forecasts, learning_context, sports_estimates)

    # Per-scan AI estimate tracking (Loop 1 monitoring — Loop 2 SWOT)
    ai_estimates = [e for e in estimates if e.source == "astra_v2"]
    algo_estimates = [e for e in estimates if e.source != "astra_v2"]
    if estimates:
        logger.info(
            "Scan %d: %d estimates produced (%d AI adversarial, %d algo). Markets: %d",
            scan_number,
            len(estimates),
            len(ai_estimates),
            len(algo_estimates),
            len(markets),
        )
    else:
        logger.warning(
            "Scan %d: ZERO estimates produced from %d markets — "
            "check ANTHROPIC_API_KEY credits and Tier 1 data sources",
            scan_number,
            len(markets),
        )

    # Find opportunities (Astra V2 core) — pass whale signals for score boosting
    opportunities = find_opportunities(markets, estimates, whale_signals=whale_signals if whale_signals else [])  # type: ignore[arg-type]

    # Longshot bias screener (structural edge, no Astra call needed)
    longshot_signals = screen_longshot_markets(markets)
    arb_signals = scan_for_arbitrage(markets)
    if longshot_signals:
        report.console.print(summarize_longshot_stats(longshot_signals))

    # Semantic cluster: check for leader-follower signals
    try:
        follower_signals = cluster_engine.check_leader_resolutions(markets)
        if follower_signals:
            for sig in follower_signals:
                report.console.print(
                    f"[bold magenta]FOLLOWER SIGNAL: {sig.direction} | "
                    f"{sig.follower_question[:50]} | "
                    f"conf={sig.confidence:.2f} | {sig.cluster_label}[/bold magenta]"
                )
    except Exception as e:
        logger.warning("Semantic cluster check failed: %s: %s", type(e).__name__, e)
        follower_signals = []

    # Periodically re-cluster (every 10 scans)
    if scan_number % 10 == 1:
        try:
            await cluster_engine.discover_relationships(markets)
            cluster_stats = cluster_engine.get_stats()
            if cluster_stats["total_relationships"] > 0:
                report.console.print(
                    f"[dim]Semantic clusters: {cluster_stats['total_relationships']} pairs | "
                    f"accuracy={cluster_stats['accuracy']:.1%}[/dim]"
                    if cluster_stats["accuracy"]
                    else f"[dim]Semantic clusters: {cluster_stats['total_relationships']} pairs discovered[/dim]"
                )
        except Exception as e:
            logger.debug("Cluster discovery failed (scan %d): %s", scan_number, e)

    # Print arb signals
    if arb_signals:
        for arb in arb_signals[:3]:
            report.console.print(
                f"[bold yellow]ARB SIGNAL: {arb.market.question[:50]} | "
                f"YES={arb.yes_price:.3f} + NO={arb.no_price:.3f} = {arb.combined_price:.3f} | "
                f"profit≈{arb.guaranteed_profit_pct:.1%}[/bold yellow]"
            )

    # Open paper positions for top opportunities
    # Use Astra V2's kelly_pct directly (already risk-adjusted)
    new_positions = []

    if not allow_new_positions:
        logger.info("Monitor-only mode: skipping position opening (scan %d)", scan_number)
    else:
        # S3: Portfolio correlation check — build category exposure map of open positions
        open_positions = [p for p in portfolio.positions if not p.resolved]
        open_category_counts: dict[str, int] = {}
        for p in open_positions:  # type: ignore[assignment]
            open_category_counts[p.category] = open_category_counts.get(p.category, 0) + 1  # type: ignore[union-attr]

    if allow_new_positions:
        for opp in opportunities[:5]:  # max 5 new positions per scan
            # Astra V2 provides kelly_position_pct — apply VIX dampening on top
            kelly_pct = opp.kelly_pct if opp.kelly_pct > 0 else 0.005  # fallback 0.5%
            kelly_pct = kelly_pct * vix_kelly_mult  # reduce sizing in high-vol regimes

            # S3: Correlation penalty — if 3+ positions in same category, halve Kelly
            # (prevents single news event from wiping correlated positions simultaneously)
            cat = opp.market.category
            same_cat_open = open_category_counts.get(cat, 0)
            if same_cat_open >= 3:
                kelly_pct = kelly_pct * 0.5
                logger.info(
                    "Correlation penalty: %d open %s positions → Kelly halved for %s",
                    same_cat_open,
                    cat,
                    opp.market.question[:40],
                )
            elif same_cat_open >= 1:
                kelly_pct = kelly_pct * 0.75  # Mild reduction for 1-2 same-category positions

            sizing = {
                "position_dollars": round(portfolio.cash * kelly_pct, 2),
                "position_pct": kelly_pct,
            }
            pos = portfolio.open_position(opp, sizing, vix_kelly_mult=vix_kelly_mult)  # type: ignore[assignment]
            if pos:
                new_positions.append((pos, opp))
                open_category_counts[cat] = open_category_counts.get(cat, 0) + 1  # Track for next iteration
                # Record for Astra V2 learning (with full audit fields)
                pred = Prediction(
                    market_condition_id=opp.market.condition_id,
                    question=opp.market.question,
                    category=opp.market.category,
                    our_probability=opp.our_estimate,
                    probability_low=opp.estimate.probability_low,
                    probability_high=opp.estimate.probability_high,
                    market_price=opp.market_price,
                    direction=opp.direction,
                    source=opp.estimate.source,
                    truth_state=opp.estimate.truth_state,
                    reasoning=opp.estimate.reasoning,
                    key_unknowns=opp.estimate.key_evidence_needed,
                    no_trade=False,
                    timestamp=pos.timestamp,
                )
                learning_agent.record_prediction(pred)

    elapsed = time.time() - t_start
    stats = portfolio.get_stats()
    learn_stats = learning_agent.get_stats()

    # Print report
    _print_paper_header(scan_number, len(markets), stats, elapsed)
    report.print_opportunities(opportunities)

    if new_positions:
        report.console.print(f"[bold green]Opened {len(new_positions)} new paper positions:[/bold green]")
        for pos, opp in new_positions:
            report.console.print(
                f"  {opp.direction} ${pos.position_size:.2f} @ {pos.entry_price:.3f} | "
                f"edge={opp.edge:+.1%} | {pos.question[:50]}"
            )
        report.console.print()

    # Periodically trigger learning evolution
    if learn_stats.get("resolved", 0) >= 5 and learn_stats["resolved"] % 5 == 0:
        report.console.print("[dim]Running AI learning cycle...[/dim]")
        await learning_agent.evolve()
        report.console.print("[green]AI brain updated.[/green]")


def _run_scan_reporting_only(
    portfolio: PaperPortfolio,
    learning_agent: LearningAgent,
    cluster_engine: SemanticClusterEngine,
    scan_number: int,
    markets: list,
    t_start: float,
):
    """Minimal scan pass used when circuit breaker is active — no new positions."""
    elapsed = time.time() - t_start
    stats = portfolio.get_stats()
    _print_paper_header(scan_number, len(markets), stats, elapsed)


def _print_paper_header(scan_number: int, n_markets: int, stats: dict, elapsed: float):
    from rich.panel import Panel

    pnl_color = "green" if stats["total_pnl"] >= 0 else "red"
    ret_color = "green" if stats["return_pct"] >= 0 else "red"

    report.console.print(
        Panel(
            f"[bold cyan]Paper Trading[/bold cyan] — Scan #{scan_number}  |  "
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
            f"Portfolio: [white]${stats['portfolio_value']:.2f}[/white]  "
            f"Cash: [white]${stats['cash']:.2f}[/white]  "
            f"Invested: [white]${stats['invested']:.2f}[/white]  "
            f"Return: [{ret_color}]{stats['return_pct']:+.1f}%[/{ret_color}]\n"
            f"Trades: [white]{stats['resolved']}[/white] resolved  "
            f"Win rate: [white]{stats['win_rate']:.0%}[/white]  "
            f"Total P&L: [{pnl_color}]${stats['total_pnl']:+.2f}[/{pnl_color}]  "
            f"Open positions: [white]{stats['open']}[/white]",
            border_style="cyan",
        )
    )


def _setup_logging():
    """DEPRECATED: Use ops.logging_setup.setup_logging() instead.
    Kept for backward compatibility if called from tests.
    """
    from logging.handlers import RotatingFileHandler

    Path("memory").mkdir(exist_ok=True)
    root = logging.getLogger("astra")
    root.setLevel(logging.DEBUG)

    fh = RotatingFileHandler("memory/astra.log", maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(ch)

    root.info("Astra logging initialized — log file: memory/astra.log")


logger = logging.getLogger("astra.paper_trader")


async def _shutdown_paper(
    run_ctx: RunContext,
    artifact_writer: ArtifactWriter,
    alert_mgr: AlertManager,
    portfolio: Optional["PaperPortfolio"] = None,
    exit_reason: str = "normal_shutdown",
):
    """Graceful shutdown: flush artifacts, update manifest, log summary."""
    logger.info(f"Shutdown initiated: {exit_reason}")

    # 1. Write final gate status
    try:
        artifact_writer.write_gate_status(run_ctx.cycle_id, {"status": "shutdown", "gates": [], "reason": exit_reason})
    except Exception as e:
        logger.error(f"Shutdown: failed to write final gate_status: {e}")

    # 2. Write final metrics + drawdown if portfolio available
    try:
        if portfolio:
            snap = portfolio._metrics_engine.compute("all_time")  # type: ignore[attr-defined]
            artifact_writer.write_metrics(
                run_ctx.cycle_id, snap._asdict() if hasattr(snap, "_asdict") else {"status": "final"}
            )
            dd_state = portfolio._drawdown_tracker.get_state()  # type: ignore[attr-defined]
            artifact_writer.write_drawdown_state(
                run_ctx.cycle_id, dd_state._asdict() if hasattr(dd_state, "_asdict") else {"status": "final"}
            )
    except Exception as e:
        logger.error(f"Shutdown: failed to write final metrics: {e}")

    # 3. Update manifest (always attempt, even if earlier steps failed)
    try:
        artifact_writer.write_manifest_end(exit_reason)
    except Exception as e:
        logger.error(f"Shutdown: failed to update manifest: {e}")

    # 4. Alert
    alert_mgr.send_alert("Astra Shutdown", f"Run {run_ctx.run_id[:8]} stopped: {exit_reason}", "info", "shutdown")
    logger.info(f"Shutdown complete for run {run_ctx.run_id}")


async def main():
    args = sys.argv[1:]
    fast_mode = "--fast" in args
    interval = 60 if fast_mode else SCAN_INTERVAL_SECONDS

    # ── Activity 1: Run Identity ──────────────────────────────────────────────
    Path("memory").mkdir(exist_ok=True)
    git_sha = get_git_sha()
    config_dict = get_canonical_config_dict()
    config_hash = compute_config_hash(config_dict)

    run_ctx = RunContext(
        git_sha=git_sha,
        paper_mode=True,
        config_hash=config_hash,
        prompt_bundle_hash=PROMPT_BUNDLE_HASH,
    )

    # ── Activity 11: Structured Logging ───────────────────────────────────────
    setup_logging(run_id=run_ctx.run_id, json_mode=JSON_LOGGING)

    # ── Activity 4: DB Schema Versioning ──────────────────────────────────────
    try:
        db_conn = init_db()
        run_ctx.schema_version = db_conn.execute("PRAGMA user_version").fetchone()[0]
    except RuntimeError as e:
        logger.critical(f"DB initialization failed: {e}")
        raise SystemExit(1)

    # ── Activity 5: DB Backup ─────────────────────────────────────────────────
    try:
        backup_db()
        logger.info("DB backup created on startup")
    except Exception as e:
        logger.warning(f"DB backup failed (non-fatal in paper mode): {e}")

    # ── Activity 2: Config Snapshot ───────────────────────────────────────────
    import json as _json

    config_snapshot_id = f"snap_{run_ctx.run_id[:8]}_0"
    try:
        db_conn.execute(
            "INSERT OR IGNORE INTO config_snapshots (snapshot_id, run_id, timestamp, config_hash, config_json) VALUES (?, ?, ?, ?, ?)",
            (
                config_snapshot_id,
                run_ctx.run_id,
                run_ctx.started_at,
                config_hash,
                _json.dumps(config_dict, sort_keys=True, default=str),
            ),
        )
        db_conn.commit()
        logger.info(f"Config snapshot written: {config_hash[:16]}...")
    except Exception as e:
        logger.warning(f"Config snapshot write failed: {e}")
    _last_config_hash = config_hash

    # ── Activity 6: Artifacts ─────────────────────────────────────────────────
    artifact_writer = ArtifactWriter(run_ctx.run_id)
    try:
        artifact_writer.ensure_writable()
    except RuntimeError as e:
        logger.critical(f"Artifacts directory not writable: {e}")
        raise SystemExit(1)

    # Write initial manifest
    artifact_writer.write_manifest_start(run_ctx.to_manifest_dict())

    # ── Activity 3: Prompt Registry ───────────────────────────────────────────
    artifact_writer.write_prompt_registry(
        {
            "prompt_bundle_hash": PROMPT_BUNDLE_HASH,
            **PROMPT_REGISTRY,
        }
    )

    # ── Activity 10: Alerting ─────────────────────────────────────────────────
    alert_mgr = AlertManager(webhook_url=ALERT_WEBHOOK_URL, cooldown_seconds=ALERT_COOLDOWN_SECONDS)

    # ── Activity 7: Gate Engine ───────────────────────────────────────────────
    gate_engine = GateEngine(
        feed_max_age_s=GATE_FEED_MAX_AGE_S,
        max_daily_loss_pct=GATE_MAX_DAILY_LOSS_PCT,
        max_cumulative_dd_pct=GATE_MAX_CUMULATIVE_DD_PCT,
        max_memory_mb=GATE_MAX_MEMORY_MB,
        min_disk_free_mb=GATE_MIN_DISK_FREE_MB,
        max_errors_per_cycle=GATE_MAX_ERRORS_PER_CYCLE,
        feed_warn_age_s=FEED_WARN_AGE_S,
        feed_fail_age_s=FEED_FAIL_AGE_S,
        memory_warn_mb=MEMORY_WARN_MB,
        memory_fail_mb=MEMORY_FAIL_MB,
        disk_warn_mb=DISK_WARN_MB,
        disk_fail_mb=DISK_FAIL_MB,
        errors_warn=ERRORS_WARN,
        errors_fail=ERRORS_FAIL,
    )

    # ── Activity 8: Signal Handlers ───────────────────────────────────────────
    shutdown_requested = False
    portfolio = None  # Will be set below

    def _request_shutdown():
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Shutdown requested via signal")

    loop = asyncio.get_event_loop()
    signal_handlers_registered = False
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
            signal_handlers_registered = True
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # ── Initialize trading components ─────────────────────────────────────────
    portfolio = PaperPortfolio()
    learning_agent_obj = LearningAgent()
    cluster_engine = SemanticClusterEngine()

    # ── Loop 4: Allocator-grade components ─────────────────────────────────────
    definition_registry = DefinitionRegistry()
    toxicity_analyzer = FlowToxicityAnalyzer(
        window_size=FLOW_TOXICITY_WINDOW,
        threshold=FLOW_TOXICITY_THRESHOLD,
        min_samples=FLOW_TOXICITY_MIN_SAMPLES,
    )
    risk_engine = RiskEngine(
        daily_loss_halt_pct=RISK_DAILY_LOSS_HALT_PCT,
        drawdown_halt_pct=RISK_DRAWDOWN_HALT_PCT,
        category_caps=RISK_CATEGORY_CAPS,
        cooldown_seconds=RISK_HALT_COOLDOWN_S,
        initial_equity=BANKROLL,
    )
    trade_telemetry = TradeTelemetry(
        artifacts_dir=Path("artifacts"),
        run_id=run_ctx.run_id,
    )
    order_manager = OrderManager(
        default_ttl=ORDER_DEFAULT_TTL_S,
        stale_threshold=ORDER_STALE_THRESHOLD_S,
        max_chase_ticks=ORDER_MAX_CHASE_TICKS,
        tick_size=ORDER_TICK_SIZE,
    )
    crypto_strategy = CryptoThresholdStrategy(
        registry=definition_registry,
        ev_gate_config={
            "fees_pct": 0.02,
            "base_slip": EV_BASE_SLIPPAGE,
            "k_spread": EV_K_SPREAD,
            "k_depth": EV_K_DEPTH,
            "base_buffer": EV_BASE_ADVERSE_BUFFER,
            "maker_threshold": EV_MAKER_THRESHOLD,
            "eps": EV_EPS,
            "depth_floor_usd": DEPTH_PROXY_FLOOR_USD,
        },
        risk_engine=risk_engine,
        toxicity_analyzer=toxicity_analyzer,
        telemetry=trade_telemetry,
    )
    logger.info(
        "Loop 4 components initialized: registry=%s, risk=%s, strategy=%s",
        type(definition_registry).__name__,
        type(risk_engine).__name__,
        crypto_strategy.name,
    )

    report.console.print()
    report.console.print("[bold cyan]Polymarket Paper Trader[/bold cyan]")
    report.console.print(f"Run ID: [white]{run_ctx.run_id}[/white]")
    report.console.print(f"Git SHA: [white]{git_sha or 'unknown'}[/white]")
    report.console.print(f"Config hash: [white]{config_hash[:16]}...[/white]")
    report.console.print(f"Starting bankroll: [green]${BANKROLL:.2f}[/green]")
    report.console.print(
        f"Scan interval: [white]{interval}s[/white]{'  [yellow](fast mode)[/yellow]' if fast_mode else ''}"
    )
    report.console.print(f"Model: [cyan]{CLAUDE_MODEL}[/cyan]")

    # Startup capability check
    if not ANTHROPIC_API_KEY:
        report.console.print("[bold red]⚠ ANTHROPIC_API_KEY missing — AI estimation DISABLED (Tier 1 only)[/bold red]")
        report.console.print("[dim]  Set ANTHROPIC_API_KEY in .env (see .env.example)[/dim]")
        logger.error("STARTUP: ANTHROPIC_API_KEY not set — adversarial AI pipeline will not run")
    else:
        report.console.print(
            f"[green]✓ Anthropic API key loaded (len={len(ANTHROPIC_API_KEY)}, model={CLAUDE_MODEL})[/green]"
        )

    if not FRED_API_KEY:
        report.console.print(
            "[yellow]⚠ FRED_API_KEY missing — macro signals (Fed rate, CPI, unemployment) disabled[/yellow]"
        )
        report.console.print("[dim]  Get free key at fred.stlouisfed.org → add FRED_API_KEY to .env[/dim]")
        logger.warning("STARTUP: FRED_API_KEY not set — macro economic signals will be None")
    else:
        report.console.print("[green]✓ FRED API key loaded[/green]")

    if not ODDS_API_KEY:
        report.console.print("[yellow]⚠ ODDS_API_KEY missing — sports estimation disabled[/yellow]")
        logger.warning("STARTUP: ODDS_API_KEY not set — sports markets will be unestimated")

    # Show today's high-impact economic events on startup
    todays_events = get_todays_events(window_hours=24.0)
    if todays_events:
        report.console.print("[bold yellow]📅 Economic events in next 24h:[/bold yellow]")
        for ev in todays_events:
            report.console.print(f"  [yellow]• {ev.name} ({ev.event_type})[/yellow] [dim]{ev.date_utc[:16]} UTC[/dim]")
    report.console.print()

    alert_mgr.send_alert("Astra Started", f"Run {run_ctx.run_id[:8]} started (paper mode)", "info", "startup")

    # ── Activity 17: Startup Checklist ────────────────────────────────────────
    checklist_results = run_startup_checklist(
        run_ctx=run_ctx,
        db_conn=db_conn,
        artifact_writer=artifact_writer,
        gate_engine=gate_engine,
        alert_mgr=alert_mgr,
        config_hash=config_hash,
        prompt_bundle_hash=PROMPT_BUNDLE_HASH,
        api_key_set=bool(ANTHROPIC_API_KEY),
        paper_mode=PAPER_MODE,
        heartbeat_path=ASTRA_HEALTH_PATH,
        signal_handlers_registered=signal_handlers_registered,
    )
    if not all_passed(checklist_results):
        failed = [r for r in checklist_results if not r.passed and not r.warn_only]
        for r in failed:
            logger.critical("Startup check FAILED: %s: %s", r.name, r.message)
        raise SystemExit(1)
    for r in checklist_results:
        if r.warn_only and not r.passed:
            logger.warning("Startup check WARNING: %s: %s", r.name, r.message)
    logger.info("All %d startup checks passed", len(checklist_results))

    # ── Activity 17: Monitor-only Phase ───────────────────────────────────────
    monitor_only_remaining = BURN_IN_MONITOR_CYCLES
    monitor_failures = 0
    if BURN_IN_MONITOR_CYCLES > 0:
        portfolio.set_monitor_only(True)
        logger.info("Monitor-only phase: %d clean cycles required before trading", BURN_IN_MONITOR_CYCLES)
    else:
        portfolio.set_monitor_only(False)

    # ── Main Loop ─────────────────────────────────────────────────────────────
    while not shutdown_requested:
        cycle_start = time.monotonic()
        cycle_id = run_ctx.next_cycle()
        set_cycle_context(cycle_id, "paper_trader")
        error_count = 0

        # Activity 17: Determine if we're in monitor-only mode
        in_monitor_only = monitor_only_remaining > 0

        try:
            await run_paper_scan(
                portfolio,
                learning_agent_obj,
                cluster_engine,
                cycle_id,
                allow_new_positions=not in_monitor_only,
            )
        except KeyboardInterrupt:
            shutdown_requested = True
            break
        except Exception as e:
            error_count += 1
            report.console.print(f"[red]Error: {e}[/red]")
            import traceback

            report.console.print(f"[dim]{traceback.format_exc()}[/dim]")

        cycle_duration = time.monotonic() - cycle_start

        # ── Activity 7: Gate Evaluation ───────────────────────────────────────
        gate_ctx = GateContext(
            ws_connected=(_last_successful_fetch is not None),
            feed_age_s=(
                (datetime.now(timezone.utc) - _last_successful_fetch).total_seconds() if _last_successful_fetch else 999
            ),
            daily_pnl_pct=0.0,  # TODO: compute from portfolio
            cumulative_dd_pct=0.0,
            error_count=error_count,
            memory_mb=_get_memory_mb(),
            disk_free_mb=_get_disk_free_mb("memory"),
        )
        gate_status = gate_engine.evaluate(gate_ctx, run_id=run_ctx.run_id, cycle_id=cycle_id)

        # ── Activity 17: Monitor-only countdown ───────────────────────────────
        if in_monitor_only:
            if gate_status.status == "ok":
                monitor_only_remaining -= 1
                logger.info(
                    "Monitor-only: %d/%d clean cycles",
                    BURN_IN_MONITOR_CYCLES - monitor_only_remaining,
                    BURN_IN_MONITOR_CYCLES,
                )
                if monitor_only_remaining == 0:
                    portfolio.set_monitor_only(False)
                    logger.info("Monitor-only phase complete — positions now allowed")
            else:
                # Degraded/halted resets counter
                monitor_failures += 1
                monitor_only_remaining = BURN_IN_MONITOR_CYCLES  # reset
                logger.warning(
                    "Monitor-only reset: gate status %s (failure %d/%d)",
                    gate_status.status,
                    monitor_failures,
                    BURN_IN_MAX_FAILED_MONITORS,
                )
                if monitor_failures >= BURN_IN_MAX_FAILED_MONITORS:
                    logger.critical("Too many monitor failures (%d), halting", monitor_failures)
                    shutdown_requested = True

        # Alert on transitions
        if gate_status.transitions:
            for t in gate_status.transitions:
                severity = "critical" if t["to"] == "halted" else "warning"
                alert_mgr.send_alert(
                    f"Gate: {t['from']} → {t['to']}",
                    f"Cycle {cycle_id}",
                    severity,
                    f"gate_transition_{t['to']}",
                )

        # ── Activity 6: Write Cycle Artifacts ─────────────────────────────────
        # Include monitor-only state in artifacts (Activity 17)
        gate_dict = gate_status.to_dict()
        gate_dict["monitor_only"] = in_monitor_only
        gate_dict["monitor_remaining"] = monitor_only_remaining
        gate_dict["monitor_failures"] = monitor_failures

        decision_report = {
            "monitor_only": in_monitor_only,
            "monitor_remaining": monitor_only_remaining,
            "monitor_failures": monitor_failures,
            "positions_allowed": not in_monitor_only,
        }

        try:
            artifact_writer.write_all_cycle_artifacts(
                cycle_id=cycle_id,
                gate_status=gate_dict,
                decision_report=decision_report,
            )
        except Exception as e:
            logger.error(f"Artifact write failed: {e}")

        # ── Activity 9: Heartbeat ─────────────────────────────────────────────
        try:
            write_heartbeat(
                path=ASTRA_HEALTH_PATH,
                run_id=run_ctx.run_id,
                cycle_id=cycle_id,
                status=gate_status.status,
                cycle_duration_s=cycle_duration,
                ws_connected=gate_ctx.ws_connected,
                active_markets=len(portfolio.positions) if portfolio else 0,
                open_orders=0,
                daily_pnl_usd=0.0,
                memory_dir="memory",
            )
        except Exception as e:
            logger.error(f"Heartbeat write failed: {e}")

        # ── Activity 2: Config Hash Change Detection ──────────────────────────
        try:
            current_config_hash = compute_config_hash()
            if current_config_hash != _last_config_hash:
                new_snap_id = f"snap_{run_ctx.run_id[:8]}_{cycle_id}"
                new_config_dict = get_canonical_config_dict()
                db_conn.execute(
                    "INSERT OR IGNORE INTO config_snapshots (snapshot_id, run_id, timestamp, config_hash, config_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        new_snap_id,
                        run_ctx.run_id,
                        datetime.now(timezone.utc).isoformat(),
                        current_config_hash,
                        _json.dumps(new_config_dict, sort_keys=True, default=str),
                    ),
                )
                db_conn.commit()
                _last_config_hash = current_config_hash
                logger.info(f"Config changed mid-run: {current_config_hash[:16]}...")
                artifact_writer.append_anomaly(cycle_id, "config_change", {"new_hash": current_config_hash})
        except Exception as e:
            logger.warning(f"Config hash check failed: {e}")

        # Heartbeat stale warning
        if _last_successful_fetch is not None:
            silent_mins = (datetime.now(timezone.utc) - _last_successful_fetch).total_seconds() / 60
            if silent_mins > 5:
                report.console.print(f"[bold red]⚠ API SILENT {silent_mins:.0f}min — Gamma API may be down.[/bold red]")

        report.console.print(
            f"[dim]Cycle {cycle_id} ({cycle_duration:.1f}s) | Gate: {gate_status.status} | Next scan in {interval}s. Ctrl+C to stop.[/dim]\n"
        )

        if shutdown_requested:
            break

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break

    # ── Graceful Shutdown ─────────────────────────────────────────────────────
    exit_reason = "sigterm" if shutdown_requested else "normal_shutdown"
    await _shutdown_paper(run_ctx, artifact_writer, alert_mgr, portfolio, exit_reason)

    # Final summary
    if portfolio:
        stats = portfolio.get_stats()
        report.console.print("\n[bold]Final Paper Trading Summary[/bold]")
        for k, v in stats.items():
            report.console.print(f"  {k}: {v}")

    if db_conn:
        db_conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        report.console.print("\n[dim]Paper trading stopped.[/dim]")
