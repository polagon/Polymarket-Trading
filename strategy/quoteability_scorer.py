"""
Quoteability Score (QS) - Select best markets to quote from watchlist.

CRITICAL FIXES:
- #4: QS computation with hard vetoes
- #12: Uses time_to_close (NOT hours_to_expiry)
- #14: Debounced refresh (separate scoring vs mutation)
"""
import logging
from typing import List, Dict, Optional
from collections import defaultdict

from models.types import Market, OrderBook, MarketState
from risk import market_state
from config import (
    ACTIVE_QUOTE_COUNT,
    MAX_MARKETS_PER_CLUSTER_IN_ACTIVE_SET,
    QS_WEIGHT_DEPTH,
    QS_WEIGHT_CHURN,
    QS_WEIGHT_JUMP,
    QS_WEIGHT_RESOLUTION,
    QS_MIN_LIQUIDITY,
    QS_RECENT_JUMP_THRESHOLD,
    QS_BOOK_STALENESS_S_PROD,
    QS_BOOK_STALENESS_S_PAPER,
    PAPER_MODE,
)

logger = logging.getLogger(__name__)

# QS veto reason counters (reset per cycle)
QS_VETO_COUNTERS = defaultdict(int)


def reset_qs_veto_counters():
    """Reset QS veto counters for new cycle."""
    global QS_VETO_COUNTERS
    QS_VETO_COUNTERS.clear()


def get_qs_veto_summary() -> str:
    """Get QS veto summary string with mode-aware staleness threshold."""
    stale_threshold_s = QS_BOOK_STALENESS_S_PAPER if PAPER_MODE else QS_BOOK_STALENESS_S_PROD
    return (
        f"ok={QS_VETO_COUNTERS['qs_ok']} "
        f"no_book={QS_VETO_COUNTERS['veto_no_book']} "
        f"stale={QS_VETO_COUNTERS['veto_stale_book']} "
        f"crossed={QS_VETO_COUNTERS['veto_crossed']} "
        f"depth={QS_VETO_COUNTERS['veto_depth']} "
        f"rrs={QS_VETO_COUNTERS['veto_rrs']} "
        f"state={QS_VETO_COUNTERS['veto_market_state']} "
        f"liquidity={QS_VETO_COUNTERS['veto_liquidity']} "
        f"stale_threshold_s={stale_threshold_s}"
    )


def compute_qs(
    market: Market,
    book: OrderBook,
    rrs: float,
    state: MarketState,
    recent_jump_rate: float = 0.0,
) -> float:
    """
    Compute Quoteability Score (0-1).

    CRITICAL FIX #12: Uses time_to_close and MarketState, NOT hours_to_expiry.

    Hard vetoes (return 0):
    - RRS > 0.35
    - MarketState in (CLOSE_WINDOW, POST_CLOSE, PROPOSED, CHALLENGE_WINDOW)
    - Time-to-close < 48h AND recent jumps > 5%
    - Spread < tick (broken book)
    - Book data stale (> 5 seconds old)

    Args:
        market: Market instance
        book: OrderBook for YES token
        rrs: Resolution Risk Score
        state: MarketState
        recent_jump_rate: Recent price jump rate (1hr)

    Returns:
        QS score (0-1), 0 = vetoed
    """
    # Hard veto: RRS too high
    if rrs > 0.35:
        QS_VETO_COUNTERS['veto_rrs'] += 1
        logger.debug(f"QS veto: {market.condition_id} RRS={rrs:.2f} > 0.35")
        return 0.0

    # Hard veto: Market state unsafe
    unsafe_states = {
        MarketState.CLOSE_WINDOW,
        MarketState.POST_CLOSE,
        MarketState.PROPOSED,
        MarketState.CHALLENGE_WINDOW,
    }
    if state in unsafe_states:
        QS_VETO_COUNTERS['veto_market_state'] += 1
        logger.debug(f"QS veto: {market.condition_id} state={state.value}")
        return 0.0

    # Hard veto: Near close + volatile
    if market.time_to_close and market.time_to_close < 48 and recent_jump_rate > QS_RECENT_JUMP_THRESHOLD:
        QS_VETO_COUNTERS['veto_near_close_volatile'] += 1
        logger.debug(
            f"QS veto: {market.condition_id} near-close + jumpy "
            f"(ttc={market.time_to_close:.1f}h, jump={recent_jump_rate:.2%})"
        )
        return 0.0

    # Hard veto: Crossed/locked book (spread < tick)
    spread = book.best_ask - book.best_bid
    if spread < market.tick_size:
        QS_VETO_COUNTERS['veto_crossed'] += 1
        logger.debug(f"QS veto: {market.condition_id} spread={spread:.4f} < tick={market.tick_size}")
        return 0.0

    # Hard veto: Stale book (mode-aware threshold)
    stale_threshold_ms = (QS_BOOK_STALENESS_S_PAPER if PAPER_MODE else QS_BOOK_STALENESS_S_PROD) * 1000
    if book.timestamp_age_ms > stale_threshold_ms:
        QS_VETO_COUNTERS['veto_stale_book'] += 1
        logger.debug(
            f"QS veto: {market.condition_id} book age={book.timestamp_age_ms}ms > {stale_threshold_ms}ms"
        )
        return 0.0

    # Hard veto: Insufficient liquidity
    if market.liquidity < QS_MIN_LIQUIDITY:
        QS_VETO_COUNTERS['veto_liquidity'] += 1
        logger.debug(f"QS veto: {market.condition_id} liquidity=${market.liquidity:.0f} < ${QS_MIN_LIQUIDITY:.0f}")
        return 0.0

    # Compute scoring components
    depth_score = compute_depth_score(book)
    churn_score = compute_churn_score(book.churn_rate)
    jump_score = compute_jump_score(recent_jump_rate)
    resolution_score = compute_resolution_score(market.time_to_close)

    # Fee regime bonus (maker rebates if available)
    fee_bonus = 1.0  # TODO: Add rebate detection from market metadata

    qs = (
        QS_WEIGHT_DEPTH * depth_score
        + QS_WEIGHT_CHURN * churn_score
        + QS_WEIGHT_JUMP * jump_score
        + QS_WEIGHT_RESOLUTION * resolution_score
    ) * fee_bonus

    # Track success
    if qs > 0:
        QS_VETO_COUNTERS['qs_ok'] += 1

    return qs


def compute_depth_score(book: OrderBook, levels: List[float] = [0.01, 0.02, 0.03]) -> float:
    """
    Compute depth score at price levels.

    Args:
        book: OrderBook
        levels: Price levels to check (e.g., [1¢, 2¢, 3¢] from mid)

    Returns:
        Depth score (0-1)
    """
    if not book.bids or not book.asks:
        return 0.0

    mid = (book.best_bid + book.best_ask) / 2.0

    total_depth = 0.0

    # Check depth at each level
    for level in levels:
        bid_threshold = mid - level
        ask_threshold = mid + level

        # Sum size within threshold
        bid_depth = sum(size for price, size in book.bids if price >= bid_threshold)
        ask_depth = sum(size for price, size in book.asks if price <= ask_threshold)

        total_depth += (bid_depth + ask_depth)

    # Normalize (assume 1000 tokens total is "good")
    depth_score = min(1.0, total_depth / 1000.0)
    return depth_score


def compute_churn_score(churn_rate: float) -> float:
    """
    Compute churn score (lower churn = better).

    Args:
        churn_rate: Updates per second

    Returns:
        Churn score (0-1), higher = better
    """
    # Inverse relationship: low churn = good
    # Normalize: 0.5 updates/sec = 0.5 score
    churn_score = 1.0 / (1.0 + churn_rate)
    return churn_score


def compute_jump_score(jump_rate: float) -> float:
    """
    Compute jump score (lower jumps = better).

    Args:
        jump_rate: Recent price jump rate (0-1)

    Returns:
        Jump score (0-1), higher = better
    """
    # Inverse relationship: low jumps = good
    jump_score = 1.0 / (1.0 + jump_rate * 10.0)
    return jump_score


def compute_resolution_score(time_to_close: Optional[float]) -> float:
    """
    Compute resolution score (more time = better).

    CRITICAL FIX #12: Uses time_to_close.

    Args:
        time_to_close: Hours until trading closes

    Returns:
        Resolution score (0-1), higher = better
    """
    if time_to_close is None:
        return 0.0

    if time_to_close > 168:  # > 1 week
        return 1.0
    elif time_to_close > 72:  # > 3 days
        return 0.8
    elif time_to_close > 48:  # > 2 days
        return 0.6
    else:
        return 0.4


def select_active_set(
    markets: List[Market],
    qs_scores: Dict[str, float],
    cluster_assignments: Dict[str, str],
) -> List[Market]:
    """
    Select top ACTIVE_QUOTE_COUNT markets by QS with cluster diversity.

    CRITICAL: Enforces max markets per cluster in active set.

    Args:
        markets: List of all markets
        qs_scores: {condition_id: qs_score}
        cluster_assignments: {condition_id: cluster_id}

    Returns:
        List of markets for active quoting (≤ ACTIVE_QUOTE_COUNT)
    """
    # Filter out vetoed markets (QS = 0)
    candidates = [m for m in markets if qs_scores.get(m.condition_id, 0.0) > 0]

    # Sort by QS descending
    candidates.sort(key=lambda m: qs_scores.get(m.condition_id, 0.0), reverse=True)

    active_set = []
    cluster_counts = defaultdict(int)

    for market in candidates:
        cluster_id = cluster_assignments.get(market.condition_id, "unknown")

        # Check cluster diversity limit
        if cluster_counts[cluster_id] >= MAX_MARKETS_PER_CLUSTER_IN_ACTIVE_SET:
            logger.debug(
                f"Active set: skipping {market.condition_id} "
                f"(cluster {cluster_id} at limit {MAX_MARKETS_PER_CLUSTER_IN_ACTIVE_SET})"
            )
            continue

        active_set.append(market)
        cluster_counts[cluster_id] += 1

        if len(active_set) >= ACTIVE_QUOTE_COUNT:
            break

    logger.info(
        f"Active set selected: {len(active_set)} markets across {len(cluster_counts)} clusters"
    )

    return active_set


def should_mutate_quotes(
    market_id: str,
    old_bid: float,
    old_ask: float,
    new_bid: float,
    new_ask: float,
    state: MarketState,
    state_changed: bool,
    toxicity_flag: bool,
    approaching_expiry: bool,
    tick_size: float = 0.01,
) -> bool:
    """
    Decide if quotes need cancel/replace.

    CRITICAL FIX #14: Separate scoring cadence from mutation cadence.

    Triggers:
    - Price drift > MUTATION_MIN_DRIFT_TICKS
    - Market state changed (NORMAL → WATCH, etc.)
    - Toxicity detected
    - Approaching expiration (within 30s)

    Args:
        market_id: Market condition_id
        old_bid: Current bid price
        old_ask: Current ask price
        new_bid: New bid price
        new_ask: New ask price
        state: Current MarketState
        state_changed: True if state changed since last quote
        toxicity_flag: True if market flagged as toxic
        approaching_expiry: True if orders expiring soon
        tick_size: Tick size

    Returns:
        True if mutation needed, False otherwise
    """
    from config import MUTATION_MIN_DRIFT_TICKS

    # Check price drift
    bid_drift_ticks = abs(new_bid - old_bid) / tick_size
    ask_drift_ticks = abs(new_ask - old_ask) / tick_size

    if bid_drift_ticks > MUTATION_MIN_DRIFT_TICKS or ask_drift_ticks > MUTATION_MIN_DRIFT_TICKS:
        logger.debug(
            f"Mutation trigger: {market_id} price drift "
            f"(bid={bid_drift_ticks:.1f} ticks, ask={ask_drift_ticks:.1f} ticks)"
        )
        return True

    # State change
    if state_changed:
        logger.debug(f"Mutation trigger: {market_id} state changed to {state.value}")
        return True

    # Toxicity
    if toxicity_flag:
        logger.warning(f"Mutation trigger: {market_id} toxicity detected")
        return True

    # Approaching expiry
    if approaching_expiry:
        logger.debug(f"Mutation trigger: {market_id} approaching expiry")
        return True

    return False


def get_qs_stats(qs_scores: Dict[str, float]) -> dict:
    """Get QS statistics."""
    if not qs_scores:
        return {
            "total_markets": 0,
            "vetoed_count": 0,
            "quoteable_count": 0,
            "avg_qs": 0.0,
            "max_qs": 0.0,
        }

    vetoed = sum(1 for qs in qs_scores.values() if qs == 0.0)
    quoteable = sum(1 for qs in qs_scores.values() if qs > 0.0)
    avg_qs = sum(qs_scores.values()) / len(qs_scores)
    max_qs = max(qs_scores.values())

    return {
        "total_markets": len(qs_scores),
        "vetoed_count": vetoed,
        "quoteable_count": quoteable,
        "avg_qs": avg_qs,
        "max_qs": max_qs,
    }
