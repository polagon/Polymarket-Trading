"""
Market-Maker Core - Inventory-aware quoting with fair value bands.

CRITICAL FIXES:
- #4: Inventory skew CORRECTED (long → lower both bid/ask)
- #6: Tick rounding before clamping
- #12: Uses time_to_close
"""
import logging
from typing import Tuple, Optional

from models.types import Market, OrderBook, OrderIntent, MarketState
from execution import mid as mid_module, units, expiration
from risk import market_state
from config import (
    FV_BASE_HALF_WIDTH,
    FV_HIGH_CHURN_MULTIPLIER,
    FV_JUMP_RISK_MULTIPLIER,
    FV_NEAR_CLOSE_MULTIPLIER,
    BASE_QUOTE_SIZE_USD,
    INVENTORY_SIZE_REDUCTION_MAX,
    INVENTORY_SKEW_MAX_CENTS,
    MAX_MARKET_INVENTORY_PCT,
    BANKROLL,
    MIN_PRICE,
    MAX_PRICE,
    STANDARD_TICK_SIZE,
)

logger = logging.getLogger(__name__)


def compute_fair_value_band(
    market: Market,
    book: OrderBook,
    recent_jump_rate: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute fair value band [fv_low, fv_high] around midpoint.

    Band widens with:
    - Churn (more book updates = wider)
    - Jump risk (recent large moves = wider)
    - Time-to-close (CRITICAL FIX #12: uses time_to_close)

    Args:
        market: Market instance
        book: OrderBook
        recent_jump_rate: Recent max jump (1hr)

    Returns:
        (fv_low, fv_high) tuple
    """
    # Get mid price
    mid_price = mid_module.compute_mid(book.best_bid, book.best_ask, book.last_mid, book.timestamp_age_ms)

    if mid_price is None:
        logger.warning(f"Cannot compute FV band for {market.condition_id}: no mid price")
        return (0.50, 0.50)  # Fallback

    # Base half-width
    half_width = FV_BASE_HALF_WIDTH

    # Widen for high churn
    if book.churn_rate > 0.5:
        half_width *= FV_HIGH_CHURN_MULTIPLIER
        logger.debug(f"FV band widened for churn: {market.condition_id} churn={book.churn_rate:.2f}/s")

    # Widen for jump risk
    if recent_jump_rate > 0.03:
        half_width *= FV_JUMP_RISK_MULTIPLIER
        logger.debug(f"FV band widened for jumps: {market.condition_id} jump={recent_jump_rate:.2%}")

    # Widen approaching close (CRITICAL FIX #12)
    if market.time_to_close and market.time_to_close < 72:
        half_width *= FV_NEAR_CLOSE_MULTIPLIER
        logger.debug(f"FV band widened near close: {market.condition_id} ttc={market.time_to_close:.1f}h")

    # Compute band
    fv_low = max(0.01, mid_price - half_width)
    fv_high = min(0.99, mid_price + half_width)

    return (fv_low, fv_high)


def compute_quotes(
    market: Market,
    fv_low: float,
    fv_high: float,
    inventory_usd: float,
    margin: float = 0.005,
) -> Tuple[float, float]:
    """
    Compute bid/ask quotes with inventory skew.

    CRITICAL FIX #4: Inventory skew CORRECTED.
    - Long inventory (positive) → shift quotes DOWN (lower both bid and ask)
    - Short inventory (negative) → shift quotes UP (raise both bid and ask)

    This encourages selling when long, buying when short.

    CRITICAL FIX #6: Tick rounding BEFORE clamping.

    Args:
        market: Market instance
        fv_low: Fair value band lower bound
        fv_high: Fair value band upper bound
        inventory_usd: Current inventory in USD (mark-to-mid, signed)
        margin: Base margin outside fair band (default 0.5¢)

    Returns:
        (bid, ask) tuple

    Raises:
        ValueError: If bid >= ask (spread collapsed)
    """
    # Inventory as fraction of max position
    max_position = BANKROLL * MAX_MARKET_INVENTORY_PCT
    inventory_fraction = inventory_usd / max_position  # -1 to +1

    # Skew amount: ±1¢ for full inventory
    # CRITICAL: Note the NEGATIVE sign (long → lower prices)
    skew = -inventory_fraction * INVENTORY_SKEW_MAX_CENTS

    logger.debug(
        f"Inventory skew: {market.condition_id} inv=${inventory_usd:.2f} "
        f"({inventory_fraction:.1%} of max) → skew={skew:.4f}"
    )

    # Base quotes: margin outside fair band
    bid = fv_low - margin + skew
    ask = fv_high + margin + skew

    # CRITICAL FIX #6: Tick rounding BEFORE clamping
    tick_size = market.tick_size
    bid = round(bid / tick_size) * tick_size
    ask = round(ask / tick_size) * tick_size

    # Clamp to [0.01, 0.99]
    bid = max(MIN_PRICE, min(MAX_PRICE, bid))
    ask = max(MIN_PRICE, min(MAX_PRICE, ask))

    # Ensure bid < ask (enforce min tick)
    if ask - bid < tick_size:
        # Widen spread to at least one tick
        mid_point = (bid + ask) / 2.0
        bid = round((mid_point - tick_size / 2) / tick_size) * tick_size
        ask = round((mid_point + tick_size / 2) / tick_size) * tick_size

    # Hard veto: refuse if spread collapsed
    if bid >= ask:
        raise ValueError(
            f"Cannot place quotes: bid {bid:.4f} >= ask {ask:.4f}. "
            f"Spread collapsed for {market.condition_id}"
        )

    return (bid, ask)


def compute_quote_size(
    market: Market,
    inventory_usd: float,
    mid_price: float,
    side: str,
) -> float:
    """
    Compute quote size with inventory-aware reduction.

    CRITICAL FIX #10: Returns size in TOKENS.

    Reduces size as inventory approaches cap.

    Args:
        market: Market instance
        inventory_usd: Current inventory in USD (mark-to-mid)
        mid_price: Current mid price
        side: "BUY" or "SELL"

    Returns:
        Quote size in TOKENS
    """
    base_size_usd = BASE_QUOTE_SIZE_USD

    # Reduce size as inventory increases
    max_position = BANKROLL * MAX_MARKET_INVENTORY_PCT
    inventory_utilization = abs(inventory_usd) / max_position  # 0 to 1

    size_multiplier = 1.0 - (INVENTORY_SIZE_REDUCTION_MAX * inventory_utilization)

    target_usd = base_size_usd * size_multiplier

    # Convert to tokens
    size_tokens = units.usd_to_tokens(target_usd, mid_price, side)

    logger.debug(
        f"Quote size: {market.condition_id} {side} "
        f"base=${base_size_usd:.2f} inv_util={inventory_utilization:.1%} "
        f"→ ${target_usd:.2f} ({size_tokens:.2f} tokens)"
    )

    return size_tokens


def create_maker_orders(
    market: Market,
    book: OrderBook,
    inventory_usd: float,
    rrs: float,
    state: MarketState,
) -> Tuple[Optional[OrderIntent], Optional[OrderIntent]]:
    """
    Create two-sided maker orders for a market.

    Integrates:
    - Risk checks (via market_state)
    - Fair value band
    - Inventory-aware skew
    - Tick rounding

    Args:
        market: Market instance
        book: OrderBook (YES token)
        inventory_usd: Current inventory (mark-to-mid USD)
        rrs: Resolution Risk Score
        state: MarketState

    Returns:
        (bid_order, ask_order) tuple, either can be None if vetoed
    """
    # Check if allowed to quote
    allowed_actions = market_state.get_allowed_actions(state, rrs)

    if not allowed_actions["can_quote"]:
        logger.debug(f"Market-maker: {market.condition_id} cannot quote (state={state.value}, rrs={rrs:.2f})")
        return (None, None)

    # Compute fair value band
    fv_low, fv_high = compute_fair_value_band(market, book)

    # Compute quotes with inventory skew
    try:
        bid, ask = compute_quotes(market, fv_low, fv_high, inventory_usd)
    except ValueError as e:
        logger.error(f"Market-maker: {market.condition_id} quote computation failed: {e}")
        return (None, None)

    # Get mid price for sizing
    mid_price = mid_module.compute_mid(book.best_bid, book.best_ask, book.last_mid, book.timestamp_age_ms)
    if mid_price is None:
        logger.warning(f"Market-maker: {market.condition_id} no mid price, cannot size orders")
        return (None, None)

    # Compute sizes
    bid_size = compute_quote_size(market, inventory_usd, mid_price, "BUY")
    ask_size = compute_quote_size(market, inventory_usd, mid_price, "SELL")

    # Compute GTD expiration
    exp_timestamp = expiration.compute_gtd_expiration()

    # Create OrderIntent objects
    bid_order = OrderIntent(
        condition_id=market.condition_id,
        token_id=market.yes_token_id,
        side="BUY",
        price=bid,
        size_in_tokens=bid_size,
        order_type="GTD",
        post_only=True,
        expiration=exp_timestamp,
        fee_rate_bps=market.fee_rate_bps,
    )

    ask_order = OrderIntent(
        condition_id=market.condition_id,
        token_id=market.yes_token_id,
        side="SELL",
        price=ask,
        size_in_tokens=ask_size,
        order_type="GTD",
        post_only=True,
        expiration=exp_timestamp,
        fee_rate_bps=market.fee_rate_bps,
    )

    logger.info(
        f"Maker orders created: {market.condition_id} "
        f"bid={bid:.4f}@{bid_size:.2f} ask={ask:.4f}@{ask_size:.2f}"
    )

    return (bid_order, ask_order)
