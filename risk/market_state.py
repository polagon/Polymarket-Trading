"""
Market Time Model & State Machine.

CRITICAL FIX #3: Three-clock time model, NOT hours_to_expiry.
- time_to_close: Hours until trading ends
- time_to_proposal_expected: Hours until resolution proposal
- challenge_window_start: Unix timestamp when 2-hour challenge period begins
"""
import logging
from typing import Optional, Dict
from models.types import Market, MarketState
from config import (
    STATE_WATCH_THRESHOLD_HOURS,
    STATE_CLOSE_WINDOW_THRESHOLD_HOURS,
    CHALLENGE_WINDOW_DURATION_HOURS,
)

logger = logging.getLogger(__name__)


def update_market_state(market: Market, metadata: Optional[dict] = None) -> MarketState:
    """
    Update market state based on time clocks and metadata events.

    CRITICAL: State transitions determine allowed actions (can_quote, can_accumulate, etc.)

    Transitions triggered by:
    - time_to_close thresholds
    - Clarification posted (immediate CHALLENGE_WINDOW + cancel-all)
    - Resolution proposal detected

    Args:
        market: Market instance with time_to_close populated
        metadata: Optional metadata dict with clarification/proposal flags

    Returns:
        Updated MarketState

    State Machine:
        NORMAL → WATCH → CLOSE_WINDOW → POST_CLOSE → PROPOSED → CHALLENGE_WINDOW → RESOLVED
    """
    metadata = metadata or {}

    # Immediate transitions (override time-based)
    if metadata.get("clarification_posted"):
        logger.warning(f"Market {market.condition_id}: Clarification posted → CHALLENGE_WINDOW")
        return MarketState.CHALLENGE_WINDOW

    if metadata.get("resolution_proposed"):
        logger.info(f"Market {market.condition_id}: Resolution proposed → CHALLENGE_WINDOW")
        return MarketState.CHALLENGE_WINDOW

    # Time-based transitions
    if market.time_to_close is None:
        # No time_to_close means resolved or no data
        logger.warning(f"Market {market.condition_id}: No time_to_close → RESOLVED")
        return MarketState.RESOLVED

    if market.time_to_close < 0:
        # Trading has closed, awaiting proposal
        return MarketState.POST_CLOSE

    if market.time_to_close < STATE_CLOSE_WINDOW_THRESHOLD_HOURS:
        # Within 24h of close
        return MarketState.CLOSE_WINDOW

    if market.time_to_close < STATE_WATCH_THRESHOLD_HOURS:
        # Within 72h of close
        return MarketState.WATCH

    # Default: normal trading
    return MarketState.NORMAL


def get_allowed_actions(state: MarketState, rrs: float) -> Dict[str, any]:
    """
    Define allowed actions per state.

    CRITICAL: This is the enforcement point for market state gates.

    Returns:
        {
            "can_quote": bool,
            "can_accumulate": bool,
            "max_position_multiplier": float,  # Applied to normal caps
            "require_cancel_all": bool,
        }
    """
    if state == MarketState.NORMAL:
        return {
            "can_quote": True,
            "can_accumulate": True,
            "max_position_multiplier": 1.0,
            "require_cancel_all": False,
        }

    elif state == MarketState.WATCH:
        # Approaching close: reduce position limits, no new accumulation
        return {
            "can_quote": True,
            "can_accumulate": False,
            "max_position_multiplier": 0.8,
            "require_cancel_all": False,
        }

    elif state == MarketState.CLOSE_WINDOW:
        # No new inventory unless RRS extremely low
        can_quote = rrs < 0.15
        return {
            "can_quote": can_quote,
            "can_accumulate": False,
            "max_position_multiplier": 0.5,
            "require_cancel_all": False,
        }

    elif state in (MarketState.POST_CLOSE, MarketState.PROPOSED, MarketState.CHALLENGE_WINDOW):
        # Cancel-all, no new exposure
        return {
            "can_quote": False,
            "can_accumulate": False,
            "max_position_multiplier": 0.0,
            "require_cancel_all": True,
        }

    elif state == MarketState.RESOLVED:
        return {
            "can_quote": False,
            "can_accumulate": False,
            "max_position_multiplier": 0.0,
            "require_cancel_all": True,
        }

    else:
        # Unknown state: fail closed
        logger.error(f"Unknown market state: {state}. Failing closed.")
        return {
            "can_quote": False,
            "can_accumulate": False,
            "max_position_multiplier": 0.0,
            "require_cancel_all": True,
        }


def should_cancel_all_for_state(market: Market) -> bool:
    """
    Check if market state requires cancel-all.

    Use this in circuit breakers.

    Args:
        market: Market with updated state

    Returns:
        True if cancel-all required
    """
    actions = get_allowed_actions(market.state, rrs=0.0)  # RRS doesn't matter for cancel-all
    return actions["require_cancel_all"]


def compute_position_cap_multiplier(market: Market, rrs: float) -> float:
    """
    Get position cap multiplier for this market state.

    CRITICAL FIX #21: Near-close ratchet uses time_to_close, NOT hours_to_expiry.

    Args:
        market: Market with time_to_close and state
        rrs: Resolution Risk Score

    Returns:
        Multiplier to apply to normal position caps (0.0-1.0)
    """
    actions = get_allowed_actions(market.state, rrs)
    return actions["max_position_multiplier"]
