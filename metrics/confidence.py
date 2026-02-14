"""
Confidence Intervals & Sample Size Gates.

CRITICAL: Prevents premature conclusions from small samples.

Without this module, Astra could:
- Report Sharpe 5.0 from 8 trades (statistically meaningless)
- Pass Gate B after a lucky streak
- Scale capital based on noise rather than signal

GATES:
- Each metric has a minimum N before it's trusted
- Below the gate, the metric returns NaN (not 0)
- Consumers (gates, dashboard) check gate_passed before acting

CONFIDENCE INTERVALS:
- Sharpe: Mertens (2002) asymptotic SE
- Win rate: Wilson score (better for p near 0 or 1)
- P&L: Bootstrap percentile (for skewed distributions)
- Brier: Normal approximation (CLT applies for n > 30)
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("astra.metrics.confidence")


# ============================================================================
# SAMPLE SIZE GATES
# ============================================================================


@dataclass
class SampleGate:
    """
    Sample size gate for a specific metric.

    A gate blocks metric usage until sufficient data exists.
    """

    metric_name: str
    n_required: int
    n_actual: int
    passed: bool
    shortfall: int  # How many more trades needed

    @property
    def progress_pct(self) -> float:
        """Progress toward gate (0-100%)."""
        if self.n_required <= 0:
            return 100.0
        return min(100.0, (self.n_actual / self.n_required) * 100.0)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"NEED {self.shortfall}"
        return f"<Gate '{self.metric_name}': {self.n_actual}/{self.n_required} [{status}]>"


# Minimum sample sizes for each metric
# Based on statistical literature for reliable estimation
METRIC_GATES = {
    "sharpe_ratio": 30,  # Lo (2002): Sharpe requires ~30 for SE < 50%
    "sortino_ratio": 30,  # Same reasoning as Sharpe
    "calmar_ratio": 50,  # Needs enough history to observe meaningful drawdowns
    "win_rate": 20,  # Binomial CI width < 20% at n=20
    "profit_factor": 20,  # Needs both wins and losses
    "var_95": 40,  # Tail estimation needs reasonable sample
    "cvar_95": 40,
    "brier_score": 20,  # CLT approximation for mean
    "worst_month": 60,  # Need ~2+ months of trades
    "max_drawdown": 30,  # Peak-trough requires meaningful path
    "expectancy": 10,  # Mean P&L converges faster
}


def sample_size_gate(metric_name: str, n_actual: int) -> SampleGate:
    """
    Check if a metric has sufficient sample size.

    Args:
        metric_name: Name of the metric to gate
        n_actual: Number of trades/observations available

    Returns:
        SampleGate with pass/fail status
    """
    n_required = METRIC_GATES.get(metric_name, 20)
    passed = n_actual >= n_required
    shortfall = max(0, n_required - n_actual)

    return SampleGate(
        metric_name=metric_name,
        n_required=n_required,
        n_actual=n_actual,
        passed=passed,
        shortfall=shortfall,
    )


def check_all_gates(n_trades: int) -> dict:
    """
    Check all metric gates against a trade count.

    Returns:
        {"sharpe_ratio": SampleGate(...), "sortino_ratio": SampleGate(...), ...}
    """
    return {name: sample_size_gate(name, n_trades) for name in METRIC_GATES}


# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================


def confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    method: str = "normal",
) -> Tuple[float, float]:
    """
    Compute confidence interval for the mean of a distribution.

    Args:
        values: List of observations
        confidence: Confidence level (default 0.95)
        method: "normal" (CLT), "bootstrap", or "wilson" (for proportions)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if not values or len(values) < 2:
        return (float("nan"), float("nan"))

    if method == "normal":
        return _ci_normal(values, confidence)
    elif method == "bootstrap":
        return _ci_bootstrap(values, confidence)
    elif method == "wilson":
        # Wilson is for proportions — interpret values as 0/1
        n = len(values)
        p = sum(1 for v in values if v > 0) / n
        return _ci_wilson(p, n, confidence)
    else:
        logger.warning("Unknown CI method '%s', falling back to normal", method)
        return _ci_normal(values, confidence)


def _ci_normal(values: List[float], confidence: float) -> Tuple[float, float]:
    """Normal (CLT) confidence interval for the mean."""
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1)) / math.sqrt(n)

    # z-score for confidence level
    z = _z_score(confidence)

    return (mean - z * se, mean + z * se)


def _ci_bootstrap(
    values: List[float],
    confidence: float,
    n_resamples: int = 5000,
) -> Tuple[float, float]:
    """
    Bootstrap percentile confidence interval.

    Better for skewed distributions (like P&L).
    """
    arr = np.array(values)
    n = len(arr)
    rng = np.random.default_rng(42)  # Deterministic seed

    # Resample and compute means
    boot_means = np.array([float(np.mean(rng.choice(arr, size=n, replace=True))) for _ in range(n_resamples)])

    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return (lower, upper)


def _ci_wilson(p: float, n: int, confidence: float) -> Tuple[float, float]:
    """
    Wilson score interval for proportions (win rate).

    Superior to normal approximation when p is near 0 or 1,
    or when n is small.
    """
    if n <= 0:
        return (float("nan"), float("nan"))

    z = _z_score(confidence)
    z2 = z * z

    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (lower, upper)


def sharpe_confidence_interval(
    sharpe: float,
    n_trades: int,
    confidence: float = 0.95,
    annualization_factor: float = 1.0,
) -> Tuple[float, float]:
    """
    Confidence interval for Sharpe ratio using Mertens (2002).

    SE(Sharpe) = sqrt((1 + 0.5 * Sharpe^2) / n)

    For annualized Sharpe, both the estimate and SE are scaled by
    sqrt(trades_per_year).

    Args:
        sharpe: Estimated Sharpe ratio (per-trade or annualized)
        n_trades: Number of trades
        confidence: Confidence level
        annualization_factor: sqrt(trades_per_year) if annualized, 1.0 if per-trade

    Returns:
        (lower_bound, upper_bound)
    """
    if n_trades < 5 or math.isnan(sharpe):
        return (float("nan"), float("nan"))

    z = _z_score(confidence)

    # Per-trade SE
    se_per_trade = math.sqrt((1 + 0.5 * (sharpe / annualization_factor) ** 2) / n_trades)

    # Scale to match the Sharpe being reported
    se = se_per_trade * annualization_factor

    return (sharpe - z * se, sharpe + z * se)


def brier_confidence_interval(
    brier_scores: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Confidence interval for mean Brier score.

    Uses normal approximation (CLT) — valid for n > 20.
    """
    if len(brier_scores) < 5:
        return (float("nan"), float("nan"))

    return _ci_normal(brier_scores, confidence)


# ============================================================================
# HELPERS
# ============================================================================


def _z_score(confidence: float) -> float:
    """Get z-score for a given confidence level."""
    # Common values (avoid scipy dependency)
    z_table = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }
    # Use closest match
    closest = min(z_table.keys(), key=lambda k: abs(k - confidence))
    if abs(closest - confidence) < 0.001:
        return z_table[closest]

    # Fallback: approximate using inverse normal
    # For 95% CI, z = 1.96 is standard
    logger.debug("Non-standard confidence %.3f — using 1.96 fallback", confidence)
    return 1.96


def min_trades_for_sharpe_precision(
    target_precision: float = 0.5,
    expected_sharpe: float = 1.0,
    confidence: float = 0.95,
) -> int:
    """
    How many trades needed for Sharpe CI width < target_precision?

    Useful for answering: "How long until we can trust this Sharpe?"

    Formula: n = z^2 * (1 + 0.5 * SR^2) / (precision/2)^2

    Args:
        target_precision: CI half-width (e.g., 0.5 means +/- 0.5)
        expected_sharpe: Expected per-trade Sharpe
        confidence: Confidence level

    Returns:
        Minimum number of trades
    """
    z = _z_score(confidence)
    numerator = z**2 * (1 + 0.5 * expected_sharpe**2)
    denominator = target_precision**2
    return int(math.ceil(numerator / denominator))
