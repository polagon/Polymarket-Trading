"""
Tests for Flow Toxicity v1 — defensive composite regime filter.

Key invariants:
  - Toxicity is per market_id
  - Low samples (< min_samples) → multiplier = 0.0 (no fake toxicity)
  - BROKEN regime → toxicity_flag = True

Loop 4: Defensive only.
"""

from __future__ import annotations

import pytest

from signals.flow_toxicity import FlowToxicityAnalyzer, FlowToxicityState


def _feed_balanced_book(analyzer: FlowToxicityAnalyzer, market_id: str, n: int = 20) -> None:
    """Feed a balanced book with tight spreads (< 1%)."""
    for i in range(n):
        analyzer.update(
            market_id=market_id,
            best_bid=0.498,
            best_ask=0.502,  # spread = 0.004/0.50 = 0.8% → TIGHT
            trade_side="BUY" if i % 2 == 0 else "SELL",
            trade_size=10.0,
            timestamp=1000.0 + i,
        )


def _feed_wide_spread(analyzer: FlowToxicityAnalyzer, market_id: str, n: int = 20) -> None:
    """Feed a wide spread book (>3%)."""
    for i in range(n):
        analyzer.update(
            market_id=market_id,
            best_bid=0.45,
            best_ask=0.55,  # spread = 0.10/0.50 = 20% → BROKEN
            trade_side="BUY" if i % 2 == 0 else "SELL",
            trade_size=10.0,
            timestamp=1000.0 + i,
        )


def _feed_one_sided(analyzer: FlowToxicityAnalyzer, market_id: str, n: int = 20) -> None:
    """Feed a one-sided book (bid=0)."""
    for i in range(n):
        analyzer.update(
            market_id=market_id,
            best_bid=0.0,
            best_ask=0.55,
            trade_side="SELL",
            trade_size=10.0,
            timestamp=1000.0 + i,
        )


def _feed_heavy_buy(analyzer: FlowToxicityAnalyzer, market_id: str, n: int = 20) -> None:
    """Feed heavy buy imbalance."""
    for i in range(n):
        analyzer.update(
            market_id=market_id,
            best_bid=0.49,
            best_ask=0.51,
            trade_side="BUY",
            trade_size=100.0,
            timestamp=1000.0 + i,
        )


class TestSpreadRegime:
    """Tests for spread regime classification."""

    def test_balanced_book_normal_regime(self) -> None:
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_balanced_book(analyzer, "m1")
        state = analyzer.get_state("m1")
        assert state.spread_regime in ("TIGHT", "NORMAL")
        assert state.sample_count == 20

    def test_wide_spread_broken_regime(self) -> None:
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_wide_spread(analyzer, "m1")
        state = analyzer.get_state("m1")
        assert state.spread_regime == "BROKEN"  # 20% spread > 8%

    def test_one_sided_book_broken_regime(self) -> None:
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_one_sided(analyzer, "m1")
        state = analyzer.get_state("m1")
        assert state.spread_regime == "BROKEN"
        assert state.toxicity_flag is True


class TestImbalance:
    """Tests for order flow imbalance."""

    def test_balanced_low_imbalance(self) -> None:
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_balanced_book(analyzer, "m1")
        state = analyzer.get_state("m1")
        assert abs(state.imbalance_score) < 0.2

    def test_heavy_buy_positive_imbalance(self) -> None:
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_heavy_buy(analyzer, "m1")
        state = analyzer.get_state("m1")
        assert state.imbalance_score > 0.5
        assert state.toxicity_score > 0.0


class TestMinSamples:
    """Tests for minimum sample requirement."""

    def test_low_samples_multiplier_zero(self) -> None:
        """sample_count < min_samples → multiplier = 0.0."""
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        # Feed only 5 observations (< 10 min_samples)
        for i in range(5):
            analyzer.update("m1", 0.49, 0.51, "BUY", 100.0, 1000.0 + i)
        mult = analyzer.get_toxicity_multiplier("m1")
        assert mult == 0.0

    def test_low_samples_flag_false(self) -> None:
        """sample_count < min_samples → toxicity_flag = False."""
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        for i in range(5):
            analyzer.update("m1", 0.0, 0.55, "SELL", 100.0, 1000.0 + i)  # BROKEN-like
        state = analyzer.get_state("m1")
        assert state.toxicity_flag is False
        assert state.sample_count == 5

    def test_sufficient_samples_multiplier_nonzero(self) -> None:
        """With enough samples and toxic flow, multiplier > 0."""
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_heavy_buy(analyzer, "m1")
        mult = analyzer.get_toxicity_multiplier("m1")
        assert mult > 0.0


class TestEmptyWindow:
    """Tests for empty/unknown market handling."""

    def test_empty_window_neutral(self) -> None:
        analyzer = FlowToxicityAnalyzer()
        state = analyzer.get_state("unknown_market")
        assert state.spread_regime == "NORMAL"
        assert state.toxicity_score == 0.0
        assert state.toxicity_flag is False
        assert state.sample_count == 0

    def test_empty_window_multiplier_zero(self) -> None:
        analyzer = FlowToxicityAnalyzer()
        mult = analyzer.get_toxicity_multiplier("unknown_market")
        assert mult == 0.0


class TestPerMarketIsolation:
    """Tests that toxicity is per-market, not global."""

    def test_different_markets_independent(self) -> None:
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_heavy_buy(analyzer, "toxic_market")
        _feed_balanced_book(analyzer, "clean_market")

        toxic_state = analyzer.get_state("toxic_market")
        clean_state = analyzer.get_state("clean_market")

        assert toxic_state.toxicity_score > clean_state.toxicity_score


class TestToxicityMultiplierRange:
    """Tests that multiplier is bounded [0, 1]."""

    def test_multiplier_bounded(self) -> None:
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_heavy_buy(analyzer, "m1")
        mult = analyzer.get_toxicity_multiplier("m1")
        assert 0.0 <= mult <= 1.0

    def test_multiplier_clean_near_zero(self) -> None:
        analyzer = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        _feed_balanced_book(analyzer, "m1")
        mult = analyzer.get_toxicity_multiplier("m1")
        assert mult < 0.5
