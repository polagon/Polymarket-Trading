"""Tests for signals/crypto_estimator.py — lognormal probability model (no network)."""

import math

import pytest

from signals.crypto_estimator import (
    ProbabilityEstimate,
    _close_probability,
    _touch_probability,
    estimate_probability,
    get_default_vol,
    time_to_cutoff_years,
)


class TestTouchProbability:
    """Barrier hitting probability under GBM."""

    def test_already_above_strike(self) -> None:
        """Spot >= strike → already touched → P=1."""
        assert _touch_probability(spot=100000, strike=90000, vol=0.65, t=1.0) == 1.0

    def test_spot_equals_strike(self) -> None:
        """Spot == strike → already touched → P=1."""
        assert _touch_probability(spot=100000, strike=100000, vol=0.65, t=1.0) == 1.0

    def test_far_otm_low_vol(self) -> None:
        """BTC at $100k, strike $1M, low vol, 1 year → very low probability."""
        p = _touch_probability(spot=100000, strike=1000000, vol=0.30, t=1.0)
        assert p < 0.01  # < 1%

    def test_far_otm_high_vol_long_time(self) -> None:
        """BTC at $100k, strike $1M, high vol, 5 years → meaningful probability."""
        p = _touch_probability(spot=100000, strike=1000000, vol=0.80, t=5.0)
        assert p > 0.05  # > 5%

    def test_near_atm(self) -> None:
        """Strike 5% above spot → high probability."""
        p = _touch_probability(spot=100000, strike=105000, vol=0.65, t=1.0)
        assert p > 0.7

    def test_zero_vol(self) -> None:
        """Zero vol → won't move → P=0 for OTM."""
        assert _touch_probability(spot=100000, strike=200000, vol=0, t=1.0) == 0.0

    def test_zero_time(self) -> None:
        """Zero time → P=0 for OTM."""
        assert _touch_probability(spot=100000, strike=200000, vol=0.65, t=0) == 0.0

    def test_monotone_in_vol(self) -> None:
        """Higher vol → higher touch probability for OTM."""
        p_low = _touch_probability(spot=100000, strike=200000, vol=0.30, t=1.0)
        p_high = _touch_probability(spot=100000, strike=200000, vol=0.80, t=1.0)
        assert p_high > p_low

    def test_monotone_in_time(self) -> None:
        """Longer time → higher touch probability."""
        p_short = _touch_probability(spot=100000, strike=200000, vol=0.65, t=0.5)
        p_long = _touch_probability(spot=100000, strike=200000, vol=0.65, t=5.0)
        assert p_long > p_short


class TestCloseProbability:
    """Terminal distribution probability under GBM."""

    def test_deep_itm(self) -> None:
        """Spot far above strike → high close probability (vol drag lowers it)."""
        p = _close_probability(spot=200000, strike=100000, vol=0.65, t=1.0)
        assert p > 0.7

    def test_deep_otm(self) -> None:
        """Spot far below strike → low close probability."""
        p = _close_probability(spot=100000, strike=1000000, vol=0.65, t=1.0)
        assert p < 0.01

    def test_atm(self) -> None:
        """At the money → ~50% (slightly less due to vol drag)."""
        p = _close_probability(spot=100000, strike=100000, vol=0.65, t=1.0)
        assert 0.3 < p < 0.7

    def test_zero_strike(self) -> None:
        """Strike=0 → always above → P=1."""
        assert _close_probability(spot=100000, strike=0, vol=0.65, t=1.0) == 1.0


class TestEstimateProbability:
    """Full estimator with bounds."""

    def test_bounds_ordering(self) -> None:
        """p_low <= p_hat <= p_high always."""
        est = estimate_probability(
            spot=100000,
            strike=200000,
            time_years=2.0,
            vol=0.65,
            resolution_type="touch",
            op=">=",
        )
        assert est.is_valid
        assert est.p_low <= est.p_hat <= est.p_high

    def test_bounds_ordering_close(self) -> None:
        est = estimate_probability(
            spot=100000,
            strike=200000,
            time_years=2.0,
            vol=0.65,
            resolution_type="close",
            op=">=",
        )
        assert est.is_valid

    def test_flip_for_lte_operator(self) -> None:
        """<= operator flips probability."""
        est_gte = estimate_probability(
            spot=100000,
            strike=200000,
            time_years=1.0,
            vol=0.65,
            resolution_type="close",
            op=">=",
        )
        est_lte = estimate_probability(
            spot=100000,
            strike=200000,
            time_years=1.0,
            vol=0.65,
            resolution_type="close",
            op="<=",
        )
        # P(>= K) + P(< K) ≈ 1
        assert abs(est_gte.p_hat + est_lte.p_hat - 1.0) < 0.01

    def test_zero_spot(self) -> None:
        """Zero spot → p_hat=0."""
        est = estimate_probability(
            spot=0,
            strike=200000,
            time_years=1.0,
            vol=0.65,
            resolution_type="touch",
            op=">=",
        )
        assert est.p_hat == 0.0

    def test_btc_1m_touch(self) -> None:
        """BTC at ~$97k, strike $1M, ~4 years, vol=65% → small but nonzero."""
        est = estimate_probability(
            spot=97000,
            strike=1000000,
            time_years=4.0,
            vol=0.65,
            resolution_type="touch",
            op=">=",
        )
        assert 0.0 < est.p_hat < 0.5
        assert est.is_valid

    def test_vol_spread_creates_bounds(self) -> None:
        """Vol buffer should create meaningful spread between p_low and p_high."""
        est = estimate_probability(
            spot=100000,
            strike=200000,
            time_years=2.0,
            vol=0.65,
            resolution_type="touch",
            op=">=",
        )
        assert est.p_high - est.p_low > 0.01  # non-trivial spread


class TestDefaultVol:
    def test_btc(self) -> None:
        assert get_default_vol("BTC") == 0.65

    def test_eth(self) -> None:
        assert get_default_vol("ETH") == 0.80

    def test_unknown(self) -> None:
        assert get_default_vol("UNKNOWN_COIN") == 0.80  # default


class TestTimeToCutoff:
    def test_future_date(self) -> None:
        t = time_to_cutoff_years("2030-12-31T23:59:59+00:00")
        assert t > 3.0  # more than 3 years from now

    def test_past_date(self) -> None:
        t = time_to_cutoff_years("2020-01-01T00:00:00+00:00")
        assert t == 0.0  # clamped to 0

    def test_invalid_date(self) -> None:
        t = time_to_cutoff_years("not-a-date")
        assert t == 0.0
