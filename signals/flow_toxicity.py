"""
Flow Toxicity v1 — defensive composite regime filter + buffer multiplier.

NOT "institutional-grade VPIN." A defensible v1 composite that:
  - Classifies spread regime (TIGHT/NORMAL/WIDE/BROKEN)
  - Measures order flow imbalance
  - Tracks quote instability (churn)
  - Computes a normalized toxicity score

Toxicity is per market_id, never global.
If sample_count < min_samples → toxicity_multiplier = 0.0 (no fake toxicity).

Loop 4: Defensive only — raises EV hurdle, does not generate signals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

_EPS = 1e-9


@dataclass
class FlowToxicityState:
    """Current toxicity state for a market.

    Attributes:
        spread_regime: "TIGHT" | "NORMAL" | "WIDE" | "BROKEN".
        spread_pct: (ask - bid) / max(mid, eps).
        imbalance_score: [-1, 1] buy/sell volume imbalance.
        churn_proxy: Mid changes per second (quote instability).
        toxicity_score: [0, 1] normalized composite.
        toxicity_flag: True if score > threshold OR BROKEN (and sufficient samples).
        sample_count: Number of observations in window.
    """

    spread_regime: str
    spread_pct: float
    imbalance_score: float
    churn_proxy: float
    toxicity_score: float
    toxicity_flag: bool
    sample_count: int


@dataclass
class _Observation:
    """Single observation in the sliding window."""

    best_bid: float
    best_ask: float
    mid: float
    trade_side: Optional[str]
    trade_size: float
    timestamp: float


def _neutral_state() -> FlowToxicityState:
    """Return neutral (non-toxic) state for insufficient data."""
    return FlowToxicityState(
        spread_regime="NORMAL",
        spread_pct=0.0,
        imbalance_score=0.0,
        churn_proxy=0.0,
        toxicity_score=0.0,
        toxicity_flag=False,
        sample_count=0,
    )


def _classify_spread(spread_pct: float, has_both_sides: bool) -> str:
    """Classify spread regime.

    TIGHT: <1%, NORMAL: 1-3%, WIDE: 3-8%, BROKEN: >8% or one-sided.
    """
    if not has_both_sides:
        return "BROKEN"
    if spread_pct > 0.08:
        return "BROKEN"
    if spread_pct > 0.03:
        return "WIDE"
    if spread_pct > 0.01:
        return "NORMAL"
    return "TIGHT"


class FlowToxicityAnalyzer:
    """Per-market flow toxicity analysis.

    Maintains a sliding window of observations per market_id.
    Computes spread regime, imbalance, churn, and composite toxicity.

    Args:
        window_size: Maximum observations per market (50).
        threshold: Toxicity score threshold for flagging (0.7).
        min_samples: Minimum observations before toxicity is meaningful (10).
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.7,
        min_samples: int = 10,
    ) -> None:
        self._window_size = window_size
        self._threshold = threshold
        self._min_samples = min_samples
        self._observations: dict[str, deque[_Observation]] = {}

    def update(
        self,
        market_id: str,
        best_bid: float,
        best_ask: float,
        trade_side: Optional[str],
        trade_size: float,
        timestamp: float,
    ) -> None:
        """Add an observation to the market's sliding window.

        Args:
            market_id: Market identifier.
            best_bid: Current best bid (0 if one-sided).
            best_ask: Current best ask (0 if one-sided).
            trade_side: "BUY" or "SELL" if a trade occurred, else None.
            trade_size: Trade size (0 if no trade).
            timestamp: Observation epoch timestamp.
        """
        if market_id not in self._observations:
            self._observations[market_id] = deque(maxlen=self._window_size)

        mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0.0

        obs = _Observation(
            best_bid=best_bid,
            best_ask=best_ask,
            mid=mid,
            trade_side=trade_side,
            trade_size=trade_size,
            timestamp=timestamp,
        )
        self._observations[market_id].append(obs)

    def get_state(self, market_id: str) -> FlowToxicityState:
        """Compute current toxicity state for a market.

        If sample_count < min_samples → returns neutral state (no fake toxicity).
        """
        window = self._observations.get(market_id)
        if not window:
            return _neutral_state()

        sample_count = len(window)

        # If insufficient samples, return neutral (no fake toxicity)
        if sample_count < self._min_samples:
            return FlowToxicityState(
                spread_regime="NORMAL",
                spread_pct=0.0,
                imbalance_score=0.0,
                churn_proxy=0.0,
                toxicity_score=0.0,
                toxicity_flag=False,
                sample_count=sample_count,
            )

        # ── Spread regime ──
        latest = window[-1]
        has_both_sides = latest.best_bid > 0 and latest.best_ask > 0
        spread_pct = (latest.best_ask - latest.best_bid) / max(latest.mid, _EPS) if has_both_sides else 1.0
        spread_regime = _classify_spread(spread_pct, has_both_sides)

        # ── Order flow imbalance ──
        buy_vol = 0.0
        sell_vol = 0.0
        for obs in window:
            if obs.trade_side == "BUY":
                buy_vol += obs.trade_size
            elif obs.trade_side == "SELL":
                sell_vol += obs.trade_size
        total_vol = buy_vol + sell_vol
        imbalance_score = (buy_vol - sell_vol) / max(total_vol, _EPS)

        # ── Churn proxy (quote instability) ──
        mid_changes = 0
        for i in range(1, len(window)):
            if abs(window[i].mid - window[i - 1].mid) > _EPS:
                mid_changes += 1
        time_span = max(window[-1].timestamp - window[0].timestamp, _EPS)
        churn_proxy = mid_changes / time_span

        # ── Composite toxicity score ──
        # Weighted average: |imbalance| 0.3, spread_width 0.4, churn 0.3
        spread_norm = min(spread_pct / 0.10, 1.0)  # normalize to [0, 1] where 10% = max
        churn_norm = min(churn_proxy / 5.0, 1.0)  # normalize: 5 changes/sec = max
        imbalance_norm = abs(imbalance_score)

        toxicity_score = 0.3 * imbalance_norm + 0.4 * spread_norm + 0.3 * churn_norm
        toxicity_score = min(max(toxicity_score, 0.0), 1.0)

        # Flag: score > threshold OR BROKEN regime
        toxicity_flag = toxicity_score > self._threshold or spread_regime == "BROKEN"

        return FlowToxicityState(
            spread_regime=spread_regime,
            spread_pct=spread_pct,
            imbalance_score=imbalance_score,
            churn_proxy=churn_proxy,
            toxicity_score=toxicity_score,
            toxicity_flag=toxicity_flag,
            sample_count=sample_count,
        )

    def get_toxicity_multiplier(self, market_id: str) -> float:
        """Get toxicity multiplier for EV gate adverse buffer scaling.

        Returns 0.0 (clean) to 1.0 (toxic).
        If sample_count < min_samples → returns 0.0 (no fake toxicity).
        """
        state = self.get_state(market_id)
        if state.sample_count < self._min_samples:
            return 0.0
        return state.toxicity_score
