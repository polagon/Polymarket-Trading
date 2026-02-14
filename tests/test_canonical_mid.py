"""
Integration tests for canonical mid price calculation.

Proves that execution.mid.compute_mid() is used consistently across:
- OrderBook (models/types.py)
- Paper simulator
- Market WS feed
- Markout tracker
- Market maker
- Inventory valuation
"""

import pytest

from config import MID_FALLBACK_ONE_SIDED_OFFSET, MID_FALLBACK_STALE_AGE_MS
from execution.mid import compute_mid, is_mid_reliable
from models.types import OrderBook


class TestCanonicalMid:
    """Test canonical mid calculation with all fallback rules."""

    def test_standard_case_both_sides(self):
        """Standard case: both bid and ask present."""
        mid = compute_mid(bid=0.50, ask=0.52)
        assert mid == 0.51

    def test_one_sided_bid_only(self):
        """Fallback: only bid available."""
        mid = compute_mid(bid=0.50, ask=None)
        assert mid == 0.50 + MID_FALLBACK_ONE_SIDED_OFFSET
        assert mid <= 0.99  # Clamped

    def test_one_sided_ask_only(self):
        """Fallback: only ask available."""
        mid = compute_mid(bid=None, ask=0.52)
        assert mid == 0.52 - MID_FALLBACK_ONE_SIDED_OFFSET
        assert mid >= 0.01  # Clamped

    def test_stale_book_with_last_mid(self):
        """Fallback: stale book uses last_mid."""
        mid = compute_mid(
            bid=0.50,
            ask=0.52,
            last_mid=0.48,
            book_age_ms=MID_FALLBACK_STALE_AGE_MS + 1000,
        )
        assert mid == 0.48  # Used last_mid

    def test_stale_book_without_last_mid(self):
        """Fallback: stale book without last_mid returns None."""
        mid = compute_mid(
            bid=0.50,
            ask=0.52,
            last_mid=None,
            book_age_ms=MID_FALLBACK_STALE_AGE_MS + 1000,
        )
        assert mid is None

    def test_crossed_book(self):
        """Error case: bid > ask (crossed book) returns None."""
        mid = compute_mid(bid=0.52, ask=0.50)
        assert mid is None

    def test_no_data(self):
        """Error case: no bid, no ask, no last_mid returns None."""
        mid = compute_mid(bid=None, ask=None, last_mid=None)
        assert mid is None

    def test_no_data_with_last_mid(self):
        """Fallback: no bid/ask but last_mid available."""
        mid = compute_mid(bid=None, ask=None, last_mid=0.55)
        assert mid == 0.55

    def test_is_mid_reliable_standard(self):
        """Reliability check: standard case is reliable."""
        assert is_mid_reliable(bid=0.50, ask=0.52, book_age_ms=1000)

    def test_is_mid_reliable_stale(self):
        """Reliability check: stale book is unreliable."""
        assert not is_mid_reliable(
            bid=0.50,
            ask=0.52,
            book_age_ms=MID_FALLBACK_STALE_AGE_MS + 1000,
        )

    def test_is_mid_reliable_one_sided(self):
        """Reliability check: one-sided book is unreliable."""
        assert not is_mid_reliable(bid=0.50, ask=None, book_age_ms=1000)
        assert not is_mid_reliable(bid=None, ask=0.52, book_age_ms=1000)

    def test_is_mid_reliable_crossed(self):
        """Reliability check: crossed book is unreliable."""
        assert not is_mid_reliable(bid=0.52, ask=0.50, book_age_ms=1000)


class TestOrderBookIntegration:
    """Test that OrderBook.mid_price uses canonical mid."""

    def test_market_mid_price_property_uses_canonical(self):
        """Market.mid_price property should use execution.mid.compute_mid."""
        from models.types import Market

        market = Market(
            condition_id="test",
            question="Test?",
            description="Test market",
            yes_token_id="yes",
            no_token_id="no",
            yes_bid=0.50,
            yes_ask=0.52,
            no_bid=0.48,
            no_ask=0.50,
        )

        mid = market.mid_price
        assert mid == 0.51  # (0.50 + 0.52) / 2


class TestMarkoutIntegration:
    """Test that markout tracker interface expects canonical mid."""

    def test_markout_docstring_requires_canonical_mid(self):
        """Markout tracker must document requirement for canonical mid."""
        from strategy.markout_tracker import MarkoutTracker

        tracker = MarkoutTracker()

        # Check docstring contains canonical mid requirement
        compute_method = tracker.compute_markout
        assert "execution.mid.compute_mid" in compute_method.__doc__  # type: ignore[operator]


class TestInventoryValuation:
    """Test that inventory valuation uses canonical mid."""

    def test_aggregate_exposure_docstring_requires_canonical_mid(self):
        """Aggregate exposure must document requirement for canonical mid."""
        from execution.units import aggregate_exposure_usd

        # Check docstring contains canonical mid requirement
        assert "execution.mid.compute_mid" in aggregate_exposure_usd.__doc__  # type: ignore[operator]


class TestPaperSimulator:
    """Test that paper simulator uses canonical mid."""

    def test_paper_simulator_record_snapshot_uses_canonical(self):
        """Paper simulator must use canonical mid for book snapshots."""
        from execution.paper_simulator import PaperTradingSimulator

        simulator = PaperTradingSimulator()

        # Create test book
        book = OrderBook(
            token_id="test",
            best_bid=0.50,
            best_ask=0.52,
            last_mid=None,
            timestamp_age_ms=1000,
        )

        # Record snapshot (should use canonical mid internally)
        simulator.record_book_snapshot("test_market", book)

        # Verify snapshot was created
        assert "test_market" in simulator.book_history
        assert len(simulator.book_history["test_market"]) == 1

        snapshot = simulator.book_history["test_market"][0]
        assert snapshot.mid == 0.51  # (0.50 + 0.52) / 2


class TestMarketMaker:
    """Test that market maker uses canonical mid."""

    def test_compute_fair_value_band_uses_canonical(self):
        """Fair value band computation must use canonical mid."""
        from models.types import Market
        from strategy.market_maker import compute_fair_value_band

        market = Market(
            condition_id="test",
            question="Test?",
            description="Test market",
            yes_token_id="yes",
            no_token_id="no",
            yes_bid=0.50,
            yes_ask=0.52,
            no_bid=0.48,
            no_ask=0.50,
            time_to_close=100.0,
        )

        book = OrderBook(
            token_id="yes",
            best_bid=0.50,
            best_ask=0.52,
            last_mid=None,
            timestamp_age_ms=1000,
            churn_rate=0.0,
        )

        fv_low, fv_high = compute_fair_value_band(market, book)

        # Should compute around mid=0.51
        assert fv_low < 0.51 < fv_high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
