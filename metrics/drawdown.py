"""
Drawdown Tracker — Real-time drawdown monitoring.

Used by:
- main_maker.py: Circuit breaker on max drawdown
- paper_trader.py: Daily loss limit
- truth_report.py: Portfolio snapshot drawdown field

DESIGN:
- Append-only equity curve (new equity points pushed on each fill/resolution)
- O(1) peak tracking (running maximum)
- Calendar-month grouping for worst-month computation
- Stateless recalculation from equity curve (no hidden accumulators)
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("astra.metrics.drawdown")


@dataclass
class DrawdownState:
    """
    Current drawdown state snapshot.

    All percentage fields are fractions (0.05 = 5%), not basis points.
    """

    # Current
    current_equity: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_usd: float = 0.0
    current_drawdown_pct: float = 0.0

    # Historical max
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0

    # Time-based
    peak_timestamp: Optional[datetime] = None
    time_since_peak_hours: float = 0.0
    time_under_water_days: float = 0.0

    # Monthly
    worst_month_pnl: float = 0.0
    worst_month_label: str = ""  # "2026-01"
    current_month_pnl: float = 0.0

    # Recovery
    recovery_needed_pct: float = 0.0  # % gain needed to reach peak

    def is_breaching(self, max_dd_pct: float) -> bool:
        """Check if current drawdown exceeds threshold."""
        return self.current_drawdown_pct < max_dd_pct  # DD is negative

    def to_dict(self) -> dict:
        return {
            "current_equity": round(self.current_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "current_drawdown_usd": round(self.current_drawdown_usd, 2),
            "current_drawdown_pct": round(self.current_drawdown_pct, 4),
            "max_drawdown_usd": round(self.max_drawdown_usd, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "peak_timestamp": self.peak_timestamp.isoformat() if self.peak_timestamp else None,
            "time_since_peak_hours": round(self.time_since_peak_hours, 1),
            "time_under_water_days": round(self.time_under_water_days, 1),
            "worst_month_pnl": round(self.worst_month_pnl, 2),
            "worst_month_label": self.worst_month_label,
            "current_month_pnl": round(self.current_month_pnl, 2),
            "recovery_needed_pct": round(self.recovery_needed_pct, 4),
        }


@dataclass
class EquityPoint:
    """Single point on the equity curve."""

    timestamp: datetime
    equity: float
    pnl_delta: float  # P&L change that produced this point


class DrawdownTracker:
    """
    Real-time drawdown tracker.

    Maintains an append-only equity curve and provides O(1) drawdown queries.

    Usage:
        tracker = DrawdownTracker(bankroll=5000.0)
        tracker.record_pnl(+25.0)   # After a winning trade
        tracker.record_pnl(-15.0)   # After a losing trade
        state = tracker.get_state()
        if state.is_breaching(-0.15):
            trigger_circuit_breaker()
    """

    def __init__(self, bankroll: float = 5000.0):
        self.bankroll = bankroll
        self.equity_curve: List[EquityPoint] = []

        # Running state (O(1) queries)
        self._current_equity = bankroll
        self._peak_equity = bankroll
        self._peak_timestamp: Optional[datetime] = None
        self._max_dd_usd = 0.0
        self._max_dd_pct = 0.0

        # Monthly tracking
        self._monthly_pnl: Dict[str, float] = defaultdict(float)

        # Initialize with starting equity
        now = datetime.now(timezone.utc)
        self.equity_curve.append(
            EquityPoint(
                timestamp=now,
                equity=bankroll,
                pnl_delta=0.0,
            )
        )
        self._peak_timestamp = now

    def record_pnl(self, pnl_delta: float, timestamp: Optional[datetime] = None):
        """
        Record a P&L change (from a trade resolution or fill).

        Args:
            pnl_delta: P&L change in USD (positive = profit, negative = loss)
            timestamp: When this happened (default: now)
        """
        ts = timestamp or datetime.now(timezone.utc)

        # Update equity
        self._current_equity += pnl_delta
        self.equity_curve.append(
            EquityPoint(
                timestamp=ts,
                equity=self._current_equity,
                pnl_delta=pnl_delta,
            )
        )

        # Update peak
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity
            self._peak_timestamp = ts

        # Update max drawdown
        dd_usd = self._current_equity - self._peak_equity
        dd_pct = dd_usd / self._peak_equity if self._peak_equity > 0 else 0.0

        if dd_usd < self._max_dd_usd:
            self._max_dd_usd = dd_usd
        if dd_pct < self._max_dd_pct:
            self._max_dd_pct = dd_pct

        # Monthly tracking
        month_key = ts.strftime("%Y-%m")
        self._monthly_pnl[month_key] += pnl_delta

        logger.debug("Equity: %.2f (peak=%.2f, dd=%.2f%%)", self._current_equity, self._peak_equity, dd_pct * 100)

    def get_state(self) -> DrawdownState:
        """Get current drawdown state (O(1))."""
        now = datetime.now(timezone.utc)

        dd_usd = self._current_equity - self._peak_equity
        dd_pct = dd_usd / self._peak_equity if self._peak_equity > 0 else 0.0

        # Time since peak
        time_since_peak_hours = 0.0
        if self._peak_timestamp:
            time_since_peak_hours = (now - self._peak_timestamp).total_seconds() / 3600.0

        # Time under water (0 if at peak)
        time_under_water_days = 0.0
        if dd_usd < -1e-9:
            time_under_water_days = time_since_peak_hours / 24.0

        # Worst month
        worst_month_pnl = 0.0
        worst_month_label = ""
        if self._monthly_pnl:
            worst_key = min(self._monthly_pnl, key=self._monthly_pnl.get)  # type: ignore[arg-type]
            worst_month_pnl = self._monthly_pnl[worst_key]
            worst_month_label = worst_key

        # Current month
        current_month_key = now.strftime("%Y-%m")
        current_month_pnl = self._monthly_pnl.get(current_month_key, 0.0)

        # Recovery needed
        recovery_needed_pct = 0.0
        if self._current_equity > 0 and dd_usd < -1e-9:
            recovery_needed_pct = (self._peak_equity - self._current_equity) / self._current_equity

        return DrawdownState(
            current_equity=self._current_equity,
            peak_equity=self._peak_equity,
            current_drawdown_usd=dd_usd,
            current_drawdown_pct=dd_pct,
            max_drawdown_usd=self._max_dd_usd,
            max_drawdown_pct=self._max_dd_pct,
            peak_timestamp=self._peak_timestamp,
            time_since_peak_hours=time_since_peak_hours,
            time_under_water_days=time_under_water_days,
            worst_month_pnl=worst_month_pnl,
            worst_month_label=worst_month_label,
            current_month_pnl=current_month_pnl,
            recovery_needed_pct=recovery_needed_pct,
        )

    def get_daily_pnl(self) -> Dict[str, float]:
        """
        Get P&L grouped by calendar day.

        Returns:
            {"2026-02-13": 25.50, "2026-02-12": -10.30, ...}
        """
        daily: Dict[str, float] = defaultdict(float)
        for point in self.equity_curve:
            if point.pnl_delta != 0:
                day_key = point.timestamp.strftime("%Y-%m-%d")
                daily[day_key] += point.pnl_delta
        return dict(daily)

    def get_monthly_pnl(self) -> Dict[str, float]:
        """Get P&L grouped by calendar month."""
        return dict(self._monthly_pnl)

    def reset(self, new_bankroll: Optional[float] = None):
        """
        Reset tracker (e.g., after capital injection).

        Args:
            new_bankroll: New starting equity (default: keep current)
        """
        equity = new_bankroll if new_bankroll is not None else self._current_equity
        self.__init__(bankroll=equity)  # type: ignore[misc]
        logger.info("DrawdownTracker reset. New equity: %.2f", equity)
