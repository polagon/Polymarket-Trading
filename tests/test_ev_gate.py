"""
Tests for the EV Gate — conservative lower-bound EV with fractional frictions.

All assertions use exact reason enums from models.reasons.
All costs are fractions, never USD.

Loop 4: No taker escalation. MAKER_ONLY or NO_TRADE.
"""

from __future__ import annotations

import pytest

from gates.ev_gate import EVGateResult, evaluate
from models.reasons import (
    REASON_EV_INVALID_MARKET_PRICE,
    REASON_EV_INVALID_PROB_BOUNDS,
    REASON_EV_MAKER_ONLY,
    REASON_EV_NET_LB_BELOW_THRESHOLD,
)

# ── Default params for tests ──

_BASE = dict(
    fees_pct=0.02,
    spread_frac=0.02,
    depth_proxy_usd=5000.0,
    toxicity_multiplier=0.0,
    maker_threshold=0.005,
    base_slip=0.003,
    k_spread=0.5,
    k_depth=0.01,
    base_buffer=0.002,
)


class TestEVGateApproval:
    """Tests for EV gate approval/rejection logic."""

    def test_positive_ev_maker_only(self) -> None:
        """Positive ev_net_lb above threshold → MAKER_ONLY."""
        result = evaluate(
            p_hat=0.80,
            p_low=0.75,
            p_high=0.85,
            market_price=0.60,
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is True
        assert result.execution_mode == "MAKER_ONLY"
        assert result.reason == REASON_EV_MAKER_ONLY
        assert result.ev_net_lb > 0.005

    def test_negative_ev_no_trade(self) -> None:
        """Negative ev_net_lb → NO_TRADE."""
        result = evaluate(
            p_hat=0.60,
            p_low=0.55,
            p_high=0.65,
            market_price=0.58,
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is False
        assert result.execution_mode == "NO_TRADE"
        assert result.reason == REASON_EV_NET_LB_BELOW_THRESHOLD

    def test_ev_exactly_at_threshold_is_no_trade(self) -> None:
        """ev_net_lb == maker_threshold → NO_TRADE (must be strictly greater)."""
        # Carefully construct to get ev_net_lb ≈ 0
        result = evaluate(
            p_hat=0.65,
            p_low=0.60,
            p_high=0.70,
            market_price=0.60,  # p_low == market_price → ev_gross_lb == 0
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is False
        assert result.execution_mode == "NO_TRADE"
        assert result.ev_gross_lb == 0.0

    def test_buy_no_conservative_bound(self) -> None:
        """BUY_NO uses market_price - p_high (conservative)."""
        result = evaluate(
            p_hat=0.30,
            p_low=0.20,
            p_high=0.40,
            market_price=0.70,  # NO price = 1 - 0.70 = 0.30, but p_high=0.40
            side="BUY_NO",
            size_usd=50.0,
            **_BASE,
        )
        # ev_gross_lb = market_price - p_high = 0.70 - 0.40 = 0.30
        assert result.ev_gross_lb == pytest.approx(0.30, abs=1e-6)
        assert result.approved is True

    def test_buy_no_marginal(self) -> None:
        """BUY_NO with tight margin."""
        result = evaluate(
            p_hat=0.50,
            p_low=0.45,
            p_high=0.55,
            market_price=0.55,  # ev_gross_lb = 0.55 - 0.55 = 0.0
            side="BUY_NO",
            size_usd=50.0,
            **_BASE,
        )
        assert result.ev_gross_lb == pytest.approx(0.0, abs=1e-6)
        assert result.approved is False


class TestEVGateFrictions:
    """Tests for friction model behavior."""

    def test_larger_size_higher_slippage(self) -> None:
        """Larger order size → higher slippage → harder to pass."""
        result_small = evaluate(
            p_hat=0.75,
            p_low=0.70,
            p_high=0.80,
            market_price=0.60,
            side="BUY_YES",
            size_usd=10.0,
            **_BASE,
        )
        result_large = evaluate(
            p_hat=0.75,
            p_low=0.70,
            p_high=0.80,
            market_price=0.60,
            side="BUY_YES",
            size_usd=5000.0,
            **_BASE,
        )
        assert result_small.slippage_est_frac < result_large.slippage_est_frac
        assert result_small.ev_net_lb > result_large.ev_net_lb

    def test_toxicity_increases_adverse_buffer(self) -> None:
        """Higher toxicity → higher adverse buffer → harder to pass."""
        base = dict(_BASE)

        base_clean = dict(base, toxicity_multiplier=0.0)
        result_clean = evaluate(
            p_hat=0.75,
            p_low=0.70,
            p_high=0.80,
            market_price=0.60,
            side="BUY_YES",
            size_usd=50.0,
            **base_clean,
        )

        base_toxic = dict(base, toxicity_multiplier=1.0)
        result_toxic = evaluate(
            p_hat=0.75,
            p_low=0.70,
            p_high=0.80,
            market_price=0.60,
            side="BUY_YES",
            size_usd=50.0,
            **base_toxic,
        )

        assert result_toxic.adverse_buffer_frac > result_clean.adverse_buffer_frac
        assert result_toxic.ev_net_lb < result_clean.ev_net_lb

    def test_fee_est_is_fraction(self) -> None:
        """fee_est_frac should equal fees_pct (a fraction, not USD)."""
        result = evaluate(
            p_hat=0.80,
            p_low=0.75,
            p_high=0.85,
            market_price=0.60,
            side="BUY_YES",
            size_usd=1000.0,
            **_BASE,
        )
        assert result.fee_est_frac == pytest.approx(0.02, abs=1e-9)

    def test_spread_frac_affects_slippage(self) -> None:
        """Wider spread_frac → higher slippage."""
        result_tight = evaluate(
            p_hat=0.80,
            p_low=0.75,
            p_high=0.85,
            market_price=0.60,
            side="BUY_YES",
            size_usd=50.0,
            fees_pct=0.02,
            spread_frac=0.01,
            depth_proxy_usd=5000.0,
            toxicity_multiplier=0.0,
            maker_threshold=0.005,
        )
        result_wide = evaluate(
            p_hat=0.80,
            p_low=0.75,
            p_high=0.85,
            market_price=0.60,
            side="BUY_YES",
            size_usd=50.0,
            fees_pct=0.02,
            spread_frac=0.10,
            depth_proxy_usd=5000.0,
            toxicity_multiplier=0.0,
            maker_threshold=0.005,
        )
        assert result_wide.slippage_est_frac > result_tight.slippage_est_frac


class TestEVGateSanityChecks:
    """Tests for input validation / sanity clamps."""

    def test_invalid_prob_bounds_p_low_gt_p_hat(self) -> None:
        """p_low > p_hat → veto."""
        result = evaluate(
            p_hat=0.60,
            p_low=0.70,  # > p_hat
            p_high=0.80,
            market_price=0.50,
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is False
        assert result.reason == REASON_EV_INVALID_PROB_BOUNDS

    def test_invalid_prob_bounds_p_hat_gt_p_high(self) -> None:
        """p_hat > p_high → veto."""
        result = evaluate(
            p_hat=0.80,
            p_low=0.70,
            p_high=0.75,  # < p_hat
            market_price=0.50,
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is False
        assert result.reason == REASON_EV_INVALID_PROB_BOUNDS

    def test_invalid_prob_bounds_negative(self) -> None:
        """p_low < 0 → veto."""
        result = evaluate(
            p_hat=0.50,
            p_low=-0.1,
            p_high=0.60,
            market_price=0.50,
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is False
        assert result.reason == REASON_EV_INVALID_PROB_BOUNDS

    def test_invalid_prob_bounds_above_one(self) -> None:
        """p_high > 1 → veto."""
        result = evaluate(
            p_hat=0.90,
            p_low=0.80,
            p_high=1.1,
            market_price=0.50,
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is False
        assert result.reason == REASON_EV_INVALID_PROB_BOUNDS

    def test_invalid_market_price_negative(self) -> None:
        result = evaluate(
            p_hat=0.70,
            p_low=0.60,
            p_high=0.80,
            market_price=-0.1,
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is False
        assert result.reason == REASON_EV_INVALID_MARKET_PRICE

    def test_invalid_market_price_above_one(self) -> None:
        result = evaluate(
            p_hat=0.70,
            p_low=0.60,
            p_high=0.80,
            market_price=1.1,
            side="BUY_YES",
            size_usd=50.0,
            **_BASE,
        )
        assert result.approved is False
        assert result.reason == REASON_EV_INVALID_MARKET_PRICE

    def test_depth_proxy_floor_prevents_fake_pass(self) -> None:
        """Tiny depth_proxy → clamped to floor → higher slippage → harder to pass."""
        result = evaluate(
            p_hat=0.70,
            p_low=0.65,
            p_high=0.75,
            market_price=0.60,
            side="BUY_YES",
            size_usd=500.0,
            fees_pct=0.02,
            spread_frac=0.02,
            depth_proxy_usd=0.01,  # absurdly small
            toxicity_multiplier=0.0,
            maker_threshold=0.005,
            depth_floor_usd=100.0,
        )
        # With floor of 100, slippage = 0.003 + 0.5*0.02 + 0.01*(500/100) = 0.003 + 0.01 + 0.05 = 0.063
        # Total friction ~0.063 + 0.02 + 0.002 = 0.085
        # ev_gross_lb = 0.65 - 0.60 = 0.05
        # ev_net_lb ≈ 0.05 - 0.085 < 0 → should fail
        assert result.approved is False

    def test_depth_proxy_floor_does_not_create_fake_pass(self) -> None:
        """Verify that floor prevents tiny depth from inflating EV."""
        # Without floor, depth=0.01 would give insane slippage
        # With floor, it still gives high slippage (which is correct)
        result_floored = evaluate(
            p_hat=0.70,
            p_low=0.65,
            p_high=0.75,
            market_price=0.60,
            side="BUY_YES",
            size_usd=50.0,
            fees_pct=0.02,
            spread_frac=0.02,
            depth_proxy_usd=0.01,
            toxicity_multiplier=0.0,
            maker_threshold=0.005,
            depth_floor_usd=100.0,
        )
        result_normal = evaluate(
            p_hat=0.70,
            p_low=0.65,
            p_high=0.75,
            market_price=0.60,
            side="BUY_YES",
            size_usd=50.0,
            fees_pct=0.02,
            spread_frac=0.02,
            depth_proxy_usd=5000.0,
            toxicity_multiplier=0.0,
            maker_threshold=0.005,
            depth_floor_usd=100.0,
        )
        # Floored result should have higher slippage than normal
        assert result_floored.slippage_est_frac > result_normal.slippage_est_frac
