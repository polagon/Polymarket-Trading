"""
Tests for Astra V2 Mispricing Detector — scanner/mispricing_detector.py

Deterministic. No network calls. No mocks needed (pure functions + data filtering).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MIN_CONFIDENCE, MIN_MARKET_LIQUIDITY, MISPRICING_THRESHOLD
from scanner.market_fetcher import Market
from scanner.mispricing_detector import (
    Opportunity,
    _apply_longshot_calibration,
    find_opportunities,
)
from scanner.probability_estimator import Estimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_market(**kw) -> Market:
    defaults = dict(
        condition_id="cond_1",
        question="Test market",
        end_date_iso="2026-06-01T00:00:00Z",
        category="politics",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        yes_price=0.50,
        no_price=0.50,
        liquidity=10000.0,
        volume=5000.0,
        hours_to_expiry=200.0,
    )
    defaults.update(kw)
    return Market(**defaults)  # type: ignore[arg-type]


def _make_estimate(**kw) -> Estimate:
    defaults = dict(
        market_condition_id="cond_1",
        question="Test market",
        category="politics",
        probability=0.70,
        probability_low=0.60,
        probability_high=0.80,
        confidence=0.80,
        market_type="Politics",
        modeling_approach="B",
        trap_flags=[],
        edge=0.20,
        ev_after_costs=0.195,
        robustness_score=4,
        kelly_position_pct=0.01,
        source="astra_v2",
        truth_state="Supported",
        reasoning="Strong thesis",
        key_evidence_needed="None",
        no_trade=False,
        no_trade_reason="",
        details={"test": True},
    )
    defaults.update(kw)
    return Estimate(**defaults)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# 1) TestLongshotCalibration
# ═══════════════════════════════════════════════════════════════════════════
class TestLongshotCalibration:
    def test_extreme_longshot_shaded_down(self):
        """Market price < 5%: estimate shaded down by 28%."""
        result = _apply_longshot_calibration(0.50, market_price=0.03)
        assert result == pytest.approx(0.50 * 0.72, abs=0.001)

    def test_strong_longshot_shaded_down(self):
        """Market price < 10%: estimate shaded down by 20%."""
        result = _apply_longshot_calibration(0.50, market_price=0.08)
        assert result == pytest.approx(0.50 * 0.80, abs=0.001)

    def test_strong_favorite_boosted(self):
        """Market price > 92%: estimate boosted by 6%, capped at 0.99."""
        result = _apply_longshot_calibration(0.95, market_price=0.95)
        expected = min(0.99, 0.95 * 1.06)
        assert result == pytest.approx(expected, abs=0.001)
        assert result <= 0.99

    def test_favorite_boosted(self):
        """Market price > 85%: estimate boosted by 4%, capped at 0.99."""
        result = _apply_longshot_calibration(0.90, market_price=0.88)
        expected = min(0.99, 0.90 * 1.04)
        assert result == pytest.approx(expected, abs=0.001)
        assert result <= 0.99

    def test_mid_range_no_adjustment(self):
        """Market price in 10%-85%: no adjustment."""
        result = _apply_longshot_calibration(0.60, market_price=0.50)
        assert result == pytest.approx(0.60, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════
# 2) TestFindOpportunities
# ═══════════════════════════════════════════════════════════════════════════
class TestFindOpportunities:
    def test_high_edge_included(self):
        market = _make_market(condition_id="opp1", yes_price=0.40)
        est = _make_estimate(
            market_condition_id="opp1",
            probability=0.70,
            confidence=0.85,
            edge=0.30,
            ev_after_costs=0.295,
            robustness_score=4,
            no_trade=False,
        )
        opps = find_opportunities([market], [est])
        assert len(opps) == 1
        assert opps[0].market.condition_id == "opp1"

    def test_no_trade_excluded(self):
        market = _make_market(condition_id="opp_nt")
        est = _make_estimate(
            market_condition_id="opp_nt",
            probability=0.70,
            edge=0.20,
            ev_after_costs=0.195,
            robustness_score=4,
            no_trade=True,
        )
        opps = find_opportunities([market], [est])
        assert len(opps) == 0

    def test_low_confidence_excluded(self):
        market = _make_market(condition_id="opp_lc")
        est = _make_estimate(
            market_condition_id="opp_lc",
            probability=0.70,
            confidence=MIN_CONFIDENCE - 0.01,
            edge=0.20,
            ev_after_costs=0.195,
            robustness_score=4,
        )
        opps = find_opportunities([market], [est])
        assert len(opps) == 0

    def test_dead_market_excluded(self):
        """Markets with yes_price < 2% or > 98% should be filtered out."""
        market = _make_market(condition_id="opp_dead", yes_price=0.01)
        est = _make_estimate(
            market_condition_id="opp_dead",
            probability=0.50,
            confidence=0.85,
            edge=0.49,
            ev_after_costs=0.485,
            robustness_score=5,
        )
        opps = find_opportunities([market], [est])
        assert len(opps) == 0

    def test_hallucination_guard_excludes_longshot(self):
        """Hallucination guard: calibrated_p > 5x market price on longshot → excluded."""
        market = _make_market(condition_id="opp_hall", yes_price=0.06)
        # Even though estimate says 0.50, after calibration (longshot region)
        # if calibrated_p > 0.06 * 5 = 0.30 → excluded
        est = _make_estimate(
            market_condition_id="opp_hall",
            probability=0.50,  # Will be calibrated
            confidence=0.85,
            edge=0.44,
            ev_after_costs=0.435,
            robustness_score=4,
        )
        opps = find_opportunities([market], [est])
        # Calibrated p at 6% market: 0.50 * 0.72 = 0.36 > 0.06 * 5 = 0.30 → filtered
        assert len(opps) == 0

    def test_whale_boost_increases_score(self):
        """Whale signals with sigma > 2.0 should boost opportunity score."""
        market = _make_market(condition_id="opp_whale", yes_price=0.40)
        est = _make_estimate(
            market_condition_id="opp_whale",
            probability=0.70,
            confidence=0.85,
            edge=0.30,
            ev_after_costs=0.295,
            robustness_score=4,
        )

        @dataclass
        class WhaleSignal:
            condition_id: str
            sigma: float

        # Without whale
        opps_no_whale = find_opportunities([market], [est], whale_signals=None)
        # With whale
        whale = WhaleSignal(condition_id="opp_whale", sigma=3.0)
        opps_with_whale = find_opportunities([market], [est], whale_signals=[whale])

        assert len(opps_no_whale) == 1
        assert len(opps_with_whale) == 1
        assert opps_with_whale[0].score > opps_no_whale[0].score

    def test_sort_order_descending_by_score(self):
        """Opportunities should be sorted by score descending."""
        markets = []
        estimates = []
        for i, edge_mult in enumerate([0.10, 0.30, 0.20]):
            cid = f"opp_sort_{i}"
            mkt = _make_market(condition_id=cid, yes_price=0.40)
            est = _make_estimate(
                market_condition_id=cid,
                probability=0.40 + edge_mult,
                confidence=0.85,
                edge=edge_mult,
                ev_after_costs=edge_mult - 0.005,
                robustness_score=4,
            )
            markets.append(mkt)
            estimates.append(est)

        opps = find_opportunities(markets, estimates)
        assert len(opps) >= 2
        for i in range(len(opps) - 1):
            assert opps[i].score >= opps[i + 1].score
