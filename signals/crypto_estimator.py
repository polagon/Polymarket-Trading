"""
Crypto threshold probability estimator — lognormal model.

Loop 5: Produces calibrated p_hat / p_low / p_high for crypto_threshold markets.

Model:
  - Touch (hit level any time before cutoff):
    P(touch) = P(max(S) >= K) under GBM
    Closed-form: uses reflection principle for barrier hitting probability
    Conservative: uses realized vol with buffer, no drift
  - Close (above/below at close):
    P(S_T >= K) = N(-d2) under risk-neutral GBM
    Conservative: uses upper vol bound for p_low, lower for p_high

Inputs (from DefinitionContract + live data):
  - spot: current underlying price
  - strike: condition.level from definition
  - time_to_cutoff_years: (cutoff_ts_utc - now) in years
  - vol: annualized realized volatility (or heuristic)
  - op: ">=" or "<=" from condition

Output:
  - p_hat: central estimate
  - p_low: conservative lower bound (for EV gate)
  - p_high: conservative upper bound
  - All satisfy: 0 <= p_low <= p_hat <= p_high <= 1
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class ProbabilityEstimate:
    """Probability bounds for a crypto threshold market."""

    p_hat: float
    p_low: float
    p_high: float
    model: str = "lognormal_v1"
    spot: float = 0.0
    strike: float = 0.0
    vol: float = 0.0
    time_years: float = 0.0

    @property
    def is_valid(self) -> bool:
        return 0.0 <= self.p_low <= self.p_hat <= self.p_high <= 1.0


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (fast approximation)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _touch_probability(spot: float, strike: float, vol: float, t: float) -> float:
    """Probability that GBM hits strike at any time in [0, t].

    Uses reflection principle for a driftless GBM (conservative: mu=0).
    P(max S >= K) = 2 * N(-d) where d = (ln(K/S)) / (vol * sqrt(t))

    For strike below spot (already touched), returns 1.0.
    """
    if spot <= 0 or vol <= 0 or t <= 0:
        return 0.0
    if strike <= spot:
        return 1.0  # already above strike

    log_ratio = math.log(strike / spot)
    vol_sqrt_t = vol * math.sqrt(t)

    if vol_sqrt_t < _EPS:
        return 0.0  # zero vol, won't move

    d = log_ratio / vol_sqrt_t
    return 2.0 * _norm_cdf(-d)


def _close_probability(spot: float, strike: float, vol: float, t: float) -> float:
    """Probability that GBM is above strike at time t.

    P(S_t >= K) = N(-d2) where d2 = (ln(K/S) - 0.5*vol^2*t) / (vol*sqrt(t))
    Conservative: uses mu=0 (no drift), which is risk-neutral.
    """
    if spot <= 0 or vol <= 0 or t <= 0:
        return 0.0
    if strike <= 0:
        return 1.0

    log_ratio = math.log(strike / spot)
    vol_sqrt_t = vol * math.sqrt(t)

    if vol_sqrt_t < _EPS:
        # Zero vol: deterministic check
        return 1.0 if spot >= strike else 0.0

    # d2 = (ln(S/K) + (mu - 0.5*vol^2)*t) / (vol*sqrt(t)), with mu=0
    d2 = (-log_ratio - 0.5 * vol * vol * t) / vol_sqrt_t
    return _norm_cdf(d2)


# Default vol estimates by underlying (annualized)
# Conservative: these are heuristic starting points; realized vol from price series is better
_DEFAULT_VOLS: dict[str, float] = {
    "BTC": 0.65,
    "ETH": 0.80,
    "SOL": 1.00,
    "DOGE": 1.20,
    "MEGAETH": 1.50,  # new token, very high uncertainty
}

# Vol buffer for conservative bounds (p_low uses vol+buffer, p_high uses vol-buffer)
_VOL_BUFFER = 0.10  # ±10% absolute


def estimate_probability(
    spot: float,
    strike: float,
    time_years: float,
    vol: float,
    resolution_type: str,
    op: str,
) -> ProbabilityEstimate:
    """Estimate touch/close probability with conservative bounds.

    Args:
        spot: Current underlying price.
        strike: Threshold level from DefinitionContract.
        time_years: Time to cutoff in years.
        vol: Annualized realized volatility.
        resolution_type: "touch" or "close".
        op: Condition operator (">=" or "<=").

    Returns:
        ProbabilityEstimate with p_hat, p_low, p_high.
    """
    if spot <= 0 or strike <= 0 or time_years <= 0 or vol <= 0:
        return ProbabilityEstimate(
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            spot=spot,
            strike=strike,
            vol=vol,
            time_years=time_years,
        )

    # For "<=" operators: P(S <= K) = 1 - P(S >= K)
    # We compute P(S >= K) then flip if needed
    flip = op in ("<=", "<")
    effective_strike = strike

    if resolution_type == "touch":
        prob_fn = _touch_probability
    elif resolution_type == "close":
        prob_fn = _close_probability
    else:
        logger.warning("Unknown resolution_type %s, defaulting to close", resolution_type)
        prob_fn = _close_probability

    # Central estimate
    p_hat = prob_fn(spot, effective_strike, vol, time_years)

    # Conservative bounds: vary vol
    vol_high = vol + _VOL_BUFFER  # higher vol → higher touch/close prob for OTM
    vol_low = max(vol - _VOL_BUFFER, 0.05)  # lower vol → lower prob

    p_with_high_vol = prob_fn(spot, effective_strike, vol_high, time_years)
    p_with_low_vol = prob_fn(spot, effective_strike, vol_low, time_years)

    p_low = min(p_hat, p_with_low_vol, p_with_high_vol)
    p_high = max(p_hat, p_with_low_vol, p_with_high_vol)

    if flip:
        p_hat = 1.0 - p_hat
        old_low = p_low
        p_low = 1.0 - p_high
        p_high = 1.0 - old_low

    # Clamp to [0, 1]
    p_hat = max(0.0, min(1.0, p_hat))
    p_low = max(0.0, min(1.0, p_low))
    p_high = max(0.0, min(1.0, p_high))

    # Ensure ordering
    p_low = min(p_low, p_hat)
    p_high = max(p_high, p_hat)

    return ProbabilityEstimate(
        p_hat=p_hat,
        p_low=p_low,
        p_high=p_high,
        spot=spot,
        strike=strike,
        vol=vol,
        time_years=time_years,
    )


def get_default_vol(underlying: str) -> float:
    """Get default annualized volatility for an underlying."""
    return _DEFAULT_VOLS.get(underlying.upper(), 0.80)


def time_to_cutoff_years(cutoff_ts_utc: str) -> float:
    """Parse cutoff timestamp and return time to cutoff in years."""
    try:
        cutoff = datetime.fromisoformat(cutoff_ts_utc.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = cutoff - now
        years = delta.total_seconds() / (365.25 * 24 * 3600)
        return max(years, 0.0)
    except (ValueError, TypeError):
        return 0.0
