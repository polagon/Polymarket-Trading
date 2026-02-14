"""
Unit tests for strategy layer (QS, Market-Maker, Markout/Toxicity).

CRITICAL: These tests lock strategy invariants.
"""

import time

import pytest

from config import ACTIVE_QUOTE_COUNT
from models.types import Fill, Market, MarketState, OrderBook
from strategy import market_maker, quoteability_scorer
from strategy.markout_tracker import MarkoutTracker

# ============================================================================
# QUOTEABILITY SCORE TESTS (CRITICAL FIX #4, #12)
# ============================================================================


def test_qs_veto_high_rrs():
    """Test that high RRS vetoes market."""
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

    book = OrderBook(
        token_id="yes1",
        best_bid=0.48,
        best_ask=0.52,
        bids=[(0.48, 100.0)],
        asks=[(0.52, 100.0)],
        timestamp_ms=int(time.time() * 1000),
        timestamp_age_ms=0,
        churn_rate=0.1,
    )

    # High RRS → veto
    qs = quoteability_scorer.compute_qs(
        market=market,
        book=book,
        rrs=0.40,  # > 0.35 threshold
        state=MarketState.NORMAL,
    )

    assert qs == 0.0, "High RRS must veto market"


def test_qs_veto_unsafe_state():
    """Test that unsafe states veto market."""
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
        time_to_close=5.0,  # Close soon
    )

    book = OrderBook(
        token_id="yes1",
        best_bid=0.48,
        best_ask=0.52,
        bids=[(0.48, 100.0)],
        asks=[(0.52, 100.0)],
        timestamp_ms=int(time.time() * 1000),
        timestamp_age_ms=0,
    )

    # CLOSE_WINDOW state → veto
    qs = quoteability_scorer.compute_qs(
        market=market,
        book=book,
        rrs=0.20,
        state=MarketState.CLOSE_WINDOW,
    )

    assert qs == 0.0, "CLOSE_WINDOW state must veto market"


def test_qs_uses_time_to_close():
    """CRITICAL FIX #12: QS must use time_to_close, not hours_to_expiry."""
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
        time_to_close=200.0,  # Far from close
        liquidity=1000.0,  # Added
    )

    book = OrderBook(
        token_id="yes1",
        best_bid=0.48,
        best_ask=0.52,
        bids=[(0.48, 100.0)],
        asks=[(0.52, 100.0)],
        timestamp_ms=int(time.time() * 1000),
        timestamp_age_ms=0,
        churn_rate=0.1,  # Added
    )

    qs = quoteability_scorer.compute_qs(
        market=market,
        book=book,
        rrs=0.20,
        state=MarketState.NORMAL,
    )

    assert qs > 0.0, "Normal market should have positive QS"
    assert not hasattr(market, "hours_to_expiry"), "Must use time_to_close, not hours_to_expiry"


def test_active_set_cluster_diversity():
    """Test active set enforces cluster diversity."""
    markets = []
    qs_scores = {}
    cluster_assignments = {}

    # Create 20 markets in same cluster
    for i in range(20):
        market = Market(
            condition_id=f"market{i}",
            question=f"Test {i}",
            description="",
            yes_token_id=f"yes{i}",
            no_token_id=f"no{i}",
            yes_bid=0.48,
            yes_ask=0.52,
            no_bid=0.48,
            no_ask=0.52,
            time_to_close=100.0,
        )
        markets.append(market)
        qs_scores[market.condition_id] = 0.8  # All have same high QS
        cluster_assignments[market.condition_id] = "cluster_test"  # Same cluster

    # Select active set
    active_set = quoteability_scorer.select_active_set(markets, qs_scores, cluster_assignments)

    # Should limit markets from same cluster
    from config import MAX_MARKETS_PER_CLUSTER_IN_ACTIVE_SET

    assert len(active_set) <= MAX_MARKETS_PER_CLUSTER_IN_ACTIVE_SET, (
        f"Active set must respect cluster limit ({MAX_MARKETS_PER_CLUSTER_IN_ACTIVE_SET}), got {len(active_set)}"
    )


# ============================================================================
# MARKET-MAKER TESTS (CRITICAL FIX #4, #6)
# ============================================================================


def test_tick_rounding_in_quotes():
    """CRITICAL FIX #6: Tick rounding enforced in quote computation."""
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
        tick_size=0.01,
    )

    fv_low = 0.4923  # Not tick-rounded
    fv_high = 0.5134  # Not tick-rounded

    bid, ask = market_maker.compute_quotes(market, fv_low, fv_high, inventory_usd=0.0)

    # Check tick rounding (should be multiples of 0.01)
    assert abs(bid - round(bid / 0.01) * 0.01) < 1e-9, f"Bid {bid:.4f} not tick-rounded"
    assert abs(ask - round(ask / 0.01) * 0.01) < 1e-9, f"Ask {ask:.4f} not tick-rounded"


def test_fv_band_uses_time_to_close():
    """CRITICAL FIX #12: FV band must use time_to_close."""
    market_far = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=200.0,  # Far from close
    )

    market_near = Market(
        condition_id="market2",
        question="Test",
        description="",
        yes_token_id="yes2",
        no_token_id="no2",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=50.0,  # Near close
    )

    book = OrderBook(
        token_id="yes1",
        best_bid=0.48,
        best_ask=0.52,
        timestamp_ms=int(time.time() * 1000),
        timestamp_age_ms=0,
        churn_rate=0.1,
    )

    fv_far_low, fv_far_high = market_maker.compute_fair_value_band(market_far, book)
    fv_near_low, fv_near_high = market_maker.compute_fair_value_band(market_near, book)

    # Near-close should have wider band
    width_far = fv_far_high - fv_far_low
    width_near = fv_near_high - fv_near_low

    assert width_near > width_far, "Near-close band should be wider than far-close"


# ============================================================================
# MARKOUT/TOXICITY TESTS (CRITICAL FIX #4)
# ============================================================================


def test_markout_calculation():
    """Test markout calculation."""
    tracker = MarkoutTracker()

    fill = Fill(
        fill_id="fill1",
        order_id="order1",
        condition_id="market1",
        token_id="yes1",
        side="BUY",  # Bought
        price=0.50,
        size_tokens=100.0,
        timestamp=1000000,  # 1000 seconds (milliseconds)
    )

    # Book snapshots: mid price moved UP after buy (good markout)
    book_snapshots = {
        1000000 + 30000: 0.52,  # 30s after: +2¢
        1000000 + 120000: 0.53,  # 2m (120s) after: +3¢
        1000000 + 600000: 0.54,  # 10m (600s) after: +4¢
    }

    markouts = tracker.compute_markout(fill, book_snapshots, intervals=[30, 120, 600])

    # BUY at 0.50, mid went to 0.52 → markout = +1 * (0.52 - 0.50) = +0.02
    assert markouts["markout_30s"] == pytest.approx(0.02, abs=0.001)
    assert markouts["markout_120s"] == pytest.approx(0.03, abs=0.001)
    assert markouts["markout_600s"] == pytest.approx(0.04, abs=0.001)


def test_toxic_market_detection():
    """Test toxic market detection."""
    tracker = MarkoutTracker()

    market_id = "market1"
    cluster_id = "cluster1"

    # Track 25 fills with negative markout (toxic)
    for i in range(25):
        tracker.track_fill_markout(market_id, cluster_id, markout_2m=-0.003)

    # Should detect toxicity
    is_toxic = tracker.is_toxic_market(market_id, cluster_id)
    assert is_toxic is True, "Should detect toxic market with persistent negative markout"


def test_qs_override_for_toxicity():
    """CRITICAL: Toxicity must OVERRIDE QS."""
    tracker = MarkoutTracker()

    market_id = "market1"
    cluster_id = "cluster1"

    # Mark market as toxic
    for i in range(25):
        tracker.track_fill_markout(market_id, cluster_id, markout_2m=-0.003)

    # Original QS is high
    original_qs = 0.85

    # Override should veto
    adjusted_qs = tracker.override_quoteability(market_id, cluster_id, original_qs)

    assert adjusted_qs == 0.0, "Toxicity must override QS to 0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
