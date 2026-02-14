"""
Unit tests for risk layer (Market State Machine + Portfolio Risk Engine).

CRITICAL: These tests lock risk invariants that prevent blowups.
"""

import pytest

from config import BANKROLL, MAX_AGG_EXPOSURE_PCT, MAX_CLUSTER_EXPOSURE_PCT
from models.types import Event, Market, MarketState
from risk import market_state
from risk.portfolio_engine import PortfolioRiskEngine

# ============================================================================
# MARKET STATE MACHINE TESTS (CRITICAL FIX #3)
# ============================================================================


def test_state_normal():
    """Test NORMAL state for markets far from close."""
    market = Market(
        condition_id="test1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=100.0,  # 100 hours to close
    )

    state = market_state.update_market_state(market)
    assert state == MarketState.NORMAL


def test_state_watch():
    """Test WATCH state when < 72h to close."""
    market = Market(
        condition_id="test2",
        question="Test",
        description="",
        yes_token_id="yes2",
        no_token_id="no2",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=48.0,  # 48 hours (< 72h threshold)
    )

    state = market_state.update_market_state(market)
    assert state == MarketState.WATCH


def test_state_close_window():
    """Test CLOSE_WINDOW state when < 24h to close."""
    market = Market(
        condition_id="test3",
        question="Test",
        description="",
        yes_token_id="yes3",
        no_token_id="no3",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=12.0,  # 12 hours (< 24h threshold)
    )

    state = market_state.update_market_state(market)
    assert state == MarketState.CLOSE_WINDOW


def test_state_post_close():
    """Test POST_CLOSE state when trading ended."""
    market = Market(
        condition_id="test4",
        question="Test",
        description="",
        yes_token_id="yes4",
        no_token_id="no4",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=-5.0,  # Negative = closed
    )

    state = market_state.update_market_state(market)
    assert state == MarketState.POST_CLOSE


def test_state_clarification_immediate_challenge():
    """Test that clarification posted immediately triggers CHALLENGE_WINDOW."""
    market = Market(
        condition_id="test5",
        question="Test",
        description="",
        yes_token_id="yes5",
        no_token_id="no5",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=100.0,  # Still far from close
    )

    metadata = {"clarification_posted": True}
    state = market_state.update_market_state(market, metadata)

    assert state == MarketState.CHALLENGE_WINDOW, "Clarification should trigger immediate CHALLENGE_WINDOW"


def test_allowed_actions_normal():
    """Test allowed actions in NORMAL state."""
    actions = market_state.get_allowed_actions(MarketState.NORMAL, rrs=0.2)

    assert actions["can_quote"] is True
    assert actions["can_accumulate"] is True
    assert actions["max_position_multiplier"] == 1.0
    assert actions["require_cancel_all"] is False


def test_allowed_actions_close_window():
    """Test allowed actions in CLOSE_WINDOW state."""
    # High RRS → cannot quote
    actions = market_state.get_allowed_actions(MarketState.CLOSE_WINDOW, rrs=0.30)
    assert actions["can_quote"] is False

    # Low RRS → can quote
    actions = market_state.get_allowed_actions(MarketState.CLOSE_WINDOW, rrs=0.10)
    assert actions["can_quote"] is True

    # Always: no accumulation, reduced caps
    assert actions["can_accumulate"] is False
    assert actions["max_position_multiplier"] == 0.5


def test_allowed_actions_challenge_window():
    """Test that CHALLENGE_WINDOW requires cancel-all."""
    actions = market_state.get_allowed_actions(MarketState.CHALLENGE_WINDOW, rrs=0.0)

    assert actions["can_quote"] is False
    assert actions["require_cancel_all"] is True


# ============================================================================
# PORTFOLIO RISK ENGINE TESTS (CRITICAL FIX #5, #17, #18, #21)
# ============================================================================


def test_cluster_assignment_deterministic():
    """CRITICAL INVARIANT: Same market → same cluster_id across restarts."""
    engine = PortfolioRiskEngine()

    market = Market(
        condition_id="market1",
        question="Will BTC hit $100k?",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        category="crypto",
        resolution_source="coinmarketcap",
    )

    cluster1 = engine.assign_cluster(market)
    cluster2 = engine.assign_cluster(market)

    assert cluster1 == cluster2, "Cluster assignment must be deterministic"


def test_cluster_assignment_neg_risk():
    """CRITICAL FIX #17: negRisk events → single cluster."""
    engine = PortfolioRiskEngine()

    event = Event(
        event_id="event123",
        title="Negative Risk Event",
        neg_risk=True,
    )

    market1 = Market(
        condition_id="market1",
        question="Outcome A",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        event=event,
    )

    market2 = Market(
        condition_id="market2",
        question="Outcome B",
        description="",
        yes_token_id="yes2",
        no_token_id="no2",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        event=event,
    )

    cluster1 = engine.assign_cluster(market1)
    cluster2 = engine.assign_cluster(market2)

    assert cluster1 == cluster2, "negRisk markets in same event must share cluster"
    assert "negRisk_event_event123" in cluster1


def test_can_enter_position_cluster_cap():
    """Test cluster cap enforcement."""
    engine = PortfolioRiskEngine()

    market = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=100.0,
    )

    cluster_id = engine.assign_cluster(market)

    # First position within cap (use small position: 1% of bankroll)
    allowed, reason = engine.can_enter_position(market, BANKROLL * 0.01, cluster_id)
    assert allowed is True, f"Small position should pass: {reason}"

    # Simulate existing exposure at cluster cap
    engine.exposure.cluster_exposures[cluster_id] = BANKROLL * MAX_CLUSTER_EXPOSURE_PCT

    # Second position should violate cluster cap
    allowed, reason = engine.can_enter_position(market, 1.0, cluster_id)
    assert allowed is False
    assert "Cluster cap exceeded" in reason


def test_can_enter_position_aggregate_cap():
    """Test aggregate cap enforcement."""
    engine = PortfolioRiskEngine()

    market = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=100.0,
    )

    # Simulate high aggregate exposure
    engine.exposure.total_exposure_usd = BANKROLL * MAX_AGG_EXPOSURE_PCT

    allowed, reason = engine.can_enter_position(market, 1.0)
    assert allowed is False
    assert "Aggregate cap exceeded" in reason


def test_can_enter_position_near_close_ratchet():
    """CRITICAL FIX #21: Near-close ratchet uses time_to_close."""
    engine = PortfolioRiskEngine()

    market = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=30.0,  # < NEAR_RESOLUTION_HOURS (48)
    )

    # Near-close should tighten caps
    max_position = BANKROLL * 0.01  # 1% per market
    near_close_max = max_position * 0.5  # Halved

    # Position larger than near-close cap should fail
    allowed, reason = engine.can_enter_position(market, near_close_max + 1.0)
    assert allowed is False
    assert "Near-close cap" in reason


def test_balance_reservations():
    """CRITICAL FIX #18: Balance reservations from open orders."""
    engine = PortfolioRiskEngine()

    order_buy = {
        "condition_id": "market1",
        "token_id": "yes1",
        "side": "BUY",
        "price": 0.50,
        "size_in_tokens": 100.0,
    }

    order_sell = {
        "condition_id": "market1",
        "token_id": "yes1",
        "side": "SELL",
        "price": 0.50,
        "size_in_tokens": 50.0,
    }

    # Reserve for BUY
    engine.reserve_for_order(order_buy)
    assert engine.exposure.reserved_usdc_by_market["market1"] == 50.0  # 0.50 * 100

    # Reserve for SELL
    engine.reserve_for_order(order_sell)
    assert engine.exposure.reserved_tokens_by_token_id["yes1"] == 50.0

    # Release reservations
    engine.release_reservation(order_buy)
    assert engine.exposure.reserved_usdc_by_market["market1"] == 0.0

    engine.release_reservation(order_sell)
    assert engine.exposure.reserved_tokens_by_token_id["yes1"] == 0.0


def test_parity_arb_disabled_for_neg_risk():
    """CRITICAL FIX #17: Parity arb disabled for negRisk events."""
    engine = PortfolioRiskEngine()

    event = Event(event_id="event1", title="Test", neg_risk=True)

    market = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        event=event,
    )

    can_trade = engine.can_trade_parity_arb(market)
    assert can_trade is False, "Parity arb must be disabled for negRisk events"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
