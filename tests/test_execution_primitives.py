"""
Unit tests for execution layer primitives.

CRITICAL: These tests lock invariants that prevent production failures.
Tests MUST pass before any merge/deployment.
"""
import pytest
import time
from execution import fees, units, mid, expiration


# ============================================================================
# FEE TESTS (CRITICAL FIX #2, #11)
# ============================================================================

def test_effective_cost_buy_basic():
    """Test BUY cost calculation with standard 2% fee."""
    # Buy 100 tokens @ 50¢ with 2% fee
    cost = fees.effective_cost_buy(0.50, 100.0, 200)
    assert cost == 51.0, "Cost = 100*0.50 * 1.02 = $51"


def test_effective_proceeds_sell_basic():
    """Test SELL proceeds calculation with standard 2% fee."""
    # Sell 100 tokens @ 50¢ with 2% fee
    proceeds = fees.effective_proceeds_sell(0.50, 100.0, 200)
    assert proceeds == 49.0, "Proceeds = 100*0.50 * 0.98 = $49"


def test_net_parity_profit_profitable():
    """Test parity arb profit calculation when profitable."""
    # YES @ 48¢, NO @ 50¢, total cost = 98¢ + fees
    profit = fees.net_parity_profit(0.48, 0.50, 200, 1.0)

    yes_cost = 0.48 * 1.02
    no_cost = 0.50 * 1.02
    expected = 1.0 - yes_cost - no_cost

    assert abs(profit - expected) < 0.0001, f"Expected {expected:.4f}, got {profit:.4f}"


def test_net_parity_profit_unprofitable():
    """Test parity arb when unprofitable (total cost > $1)."""
    # YES @ 52¢, NO @ 52¢, total cost > $1
    profit = fees.net_parity_profit(0.52, 0.52, 200, 1.0)
    assert profit < 0, "Should be negative when total cost > $1"


def test_fees_never_hardcoded():
    """CRITICAL INVARIANT: Fees must come from fee_rate_bps parameter, never hardcoded."""
    # Test with various fee rates
    for fee_bps in [100, 200, 300]:  # 1%, 2%, 3%
        cost = fees.effective_cost_buy(0.50, 100.0, fee_bps)
        expected = 50.0 * (1.0 + fee_bps / 10000.0)
        assert abs(cost - expected) < 0.0001, f"Fee must come from parameter, not hardcoded"


# ============================================================================
# UNIT CONVERSION TESTS (CRITICAL FIX #10)
# ============================================================================

def test_tokens_to_usd_buy():
    """Test BUY position conversion to USD exposure."""
    exposure = units.tokens_to_usd(100.0, 0.50, "BUY")
    assert exposure == 50.0, "Buy 100 @ 50¢ = $50 exposure"


def test_tokens_to_usd_sell():
    """Test SELL (short) position conversion to USD exposure."""
    exposure = units.tokens_to_usd(100.0, 0.50, "SELL")
    assert exposure == 50.0, "Sell 100 @ 50¢ = $50 exposure to downside"


def test_usd_to_tokens_buy():
    """Test USD to tokens conversion for BUY."""
    tokens = units.usd_to_tokens(50.0, 0.50, "BUY")
    assert tokens == 100.0, "$50 / 50¢ = 100 tokens"


def test_usd_to_tokens_sell():
    """Test USD to tokens conversion for SELL."""
    tokens = units.usd_to_tokens(50.0, 0.50, "SELL")
    assert tokens == 100.0, "$50 / (1-0.50) = 100 tokens"


def test_usd_to_tokens_edge_cases():
    """Test edge cases for unit conversion."""
    # Mid price = 0 (BUY)
    tokens = units.usd_to_tokens(50.0, 0.0, "BUY")
    assert tokens == 0.0, "Zero mid price should return 0"

    # Mid price = 1 (SELL)
    tokens = units.usd_to_tokens(50.0, 1.0, "SELL")
    assert tokens == 0.0, "Mid price = 1 should return 0 for SELL"


def test_aggregate_exposure_mixed_positions():
    """Test portfolio exposure aggregation with long + short positions."""
    positions = {
        "YES_token_1": 100.0,   # Long 100 YES @ 50¢
        "NO_token_2": -50.0,    # Short 50 NO @ 40¢
    }
    mids = {
        "YES_token_1": 0.50,
        "NO_token_2": 0.40,
    }

    exposure = units.aggregate_exposure_usd(positions, mids)

    # Long YES: 100 * 0.50 = 50
    # Short NO: 50 * (1 - 0.40) = 30
    # Total = 80
    assert exposure == 80.0, f"Expected 80.0, got {exposure}"


def test_units_consistent_internal():
    """CRITICAL INVARIANT: Internal sizing must be consistent (tokens → USD → tokens)."""
    # Round-trip test
    original_usd = 50.0
    mid = 0.50

    tokens = units.usd_to_tokens(original_usd, mid, "BUY")
    usd_back = units.tokens_to_usd(tokens, mid, "BUY")

    assert abs(usd_back - original_usd) < 0.0001, "Round-trip must be consistent"


# ============================================================================
# MID CALCULATION TESTS (ChatGPT Final Fix)
# ============================================================================

def test_mid_standard_case():
    """Test standard mid calculation with both bid and ask."""
    m = mid.compute_mid(0.48, 0.52, last_mid=None, book_age_ms=0)
    assert m == 0.50, "Mid = (0.48 + 0.52) / 2 = 0.50"


def test_mid_one_sided_bid_only():
    """Test fallback when only bid available."""
    m = mid.compute_mid(0.50, None, last_mid=None, book_age_ms=0)
    assert m == 0.52, "Bid-only fallback: 0.50 + 0.02 offset = 0.52"


def test_mid_one_sided_ask_only():
    """Test fallback when only ask available."""
    m = mid.compute_mid(None, 0.52, last_mid=None, book_age_ms=0)
    assert m == 0.50, "Ask-only fallback: 0.52 - 0.02 offset = 0.50"


def test_mid_stale_book_fallback():
    """Test fallback to last_mid when book is stale."""
    # Book age > threshold, should use last_mid
    m = mid.compute_mid(0.48, 0.52, last_mid=0.51, book_age_ms=15000)
    assert m == 0.51, "Stale book should use last_mid"


def test_mid_crossed_book_refuses():
    """Test that crossed book (bid > ask) returns None."""
    m = mid.compute_mid(0.55, 0.50, last_mid=None, book_age_ms=0)
    assert m is None, "Crossed book should refuse to compute mid"


def test_mid_no_data_refuses():
    """Test that no data (no bid, no ask, no last_mid) returns None."""
    m = mid.compute_mid(None, None, last_mid=None, book_age_ms=0)
    assert m is None, "No data should refuse to compute mid"


def test_mid_reliability_check():
    """Test is_mid_reliable() function."""
    # Reliable: both sides, not stale
    assert mid.is_mid_reliable(0.48, 0.52, 0) is True

    # Unreliable: stale
    assert mid.is_mid_reliable(0.48, 0.52, 15000) is False

    # Unreliable: one-sided
    assert mid.is_mid_reliable(0.48, None, 0) is False

    # Unreliable: crossed
    assert mid.is_mid_reliable(0.55, 0.50, 0) is False


# ============================================================================
# GTD EXPIRATION TESTS (CRITICAL FIX #1, #9, #20)
# ============================================================================

def test_gtd_expiration_basic():
    """Test GTD expiration calculation."""
    now = int(time.time())
    exp = expiration.compute_gtd_expiration(120)

    # Should be now + buffer + desired (default: 60 + 120 = 180)
    delta = exp - now
    assert 175 <= delta <= 185, f"Expected ~180s delta, got {delta}s"


def test_gtd_near_expiry_detection():
    """Test near-expiry detection for cancel/replace logic."""
    # Expiring in 25 seconds
    exp_soon = int(time.time()) + 25
    assert expiration.is_gtd_near_expiry(exp_soon, 30) is True

    # Expiring in 35 seconds
    exp_later = int(time.time()) + 35
    assert expiration.is_gtd_near_expiry(exp_later, 30) is False


def test_gtd_never_uses_ttl_seconds():
    """CRITICAL INVARIANT: Must use unix timestamp, NEVER ttl_seconds."""
    exp = expiration.compute_gtd_expiration(120)

    # Unix timestamp should be ~10 digits
    assert exp > 1_000_000_000, "Must return unix timestamp, not relative seconds"

    # Should be in reasonable future (not more than 1 hour from now)
    now = int(time.time())
    assert exp - now < 3600, "Expiration should be within 1 hour"


# ============================================================================
# INTEGRATION SMOKE TEST
# ============================================================================

def test_full_order_flow_simulation():
    """
    Smoke test: Simulate full order lifecycle with execution primitives.

    This tests that all primitives work together correctly.
    """
    # Market setup
    bid = 0.48
    ask = 0.52
    fee_rate_bps = 200

    # Compute mid
    m = mid.compute_mid(bid, ask, last_mid=None, book_age_ms=0)
    assert m == 0.50

    # Decide to BUY 100 tokens at ask
    size_tokens = 100.0
    cost = fees.effective_cost_buy(ask, size_tokens, fee_rate_bps)
    assert cost == 53.04  # 100 * 0.52 * 1.02

    # Convert to USD exposure
    exposure = units.tokens_to_usd(size_tokens, m, "BUY")
    assert exposure == 50.0  # Mark-to-mid: 100 * 0.50

    # Compute GTD expiration
    exp = expiration.compute_gtd_expiration(120)
    assert exp > int(time.time())

    # All primitives worked together ✅


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
