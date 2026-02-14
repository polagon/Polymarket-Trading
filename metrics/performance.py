"""
Performance Engine — Centralized metric computation for Astra.

DESIGN PRINCIPLES:
1. Single source of truth: ALL Sharpe/Sortino/Calmar/PF/etc. computed here
2. Annualized correctly: per-trade metrics scaled by sqrt(N_trades/year)
3. Rolling windows: 30d, 90d, all-time computed from trade ledger
4. Sample-size gated: metrics report NaN when N < minimum threshold
5. Deterministic: same inputs always produce same outputs (no random state)

CRITICAL DISTINCTION:
- Per-trade Sharpe: mean(R) / std(R) — measures signal quality
- Annualized Sharpe: per_trade * sqrt(trades_per_year) — measures capital efficiency
  Both are reported. Gates use annualized. Dashboard shows both.

WHAT THIS REPLACES:
- paper_trader.py get_stats() Sharpe/Sortino (ad-hoc, per-trade only)
- strategies/base.py StrategyMetrics.sharpe_ratio (population variance, no annualization)
- truth_report.py portfolio_snapshot (empty placeholders for Calmar/drawdown)
- trade_logger.py get_current_drawdown() (cumulative only, no rolling)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("astra.metrics")


# ============================================================================
# TRADE RECORD — Canonical input to the metrics engine
# ============================================================================


@dataclass
class TradeRecord:
    """
    One resolved trade. The metrics engine consumes a list of these.

    Designed to be populated from:
    - PaperPosition (paper_trader.py) — via from_paper_position()
    - Fill stream (user_ws.py) — via from_fill()
    - SQLite trade_logger — via from_db_row()
    """

    trade_id: str
    condition_id: str

    # Timing
    entry_time: datetime  # UTC
    exit_time: datetime  # UTC

    # Economics
    entry_price: float  # Price paid per share
    exit_price: float  # Resolution price (0 or 1 for binary)
    size_usd: float  # Position size in USD
    pnl_usd: float  # Realized P&L in USD
    fee_usd: float = 0.0  # Fees paid

    # Attribution
    direction: str = ""  # "BUY YES" / "BUY NO"
    category: str = "other"
    strategy: str = "unknown"  # "maker" / "taker" / "satellite" / "astra_v2"
    cluster_id: str = ""

    # Calibration (optional — for Brier score)
    our_probability: Optional[float] = None
    market_price_at_entry: Optional[float] = None
    outcome: Optional[bool] = None  # True=YES resolved, False=NO resolved

    @property
    def return_pct(self) -> float:
        """Return as fraction of capital deployed."""
        if self.size_usd <= 0:
            return 0.0
        return self.pnl_usd / self.size_usd

    @property
    def holding_hours(self) -> float:
        """Hours between entry and exit."""
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 3600.0

    @property
    def net_pnl_usd(self) -> float:
        """P&L after fees."""
        return self.pnl_usd - self.fee_usd

    @property
    def brier_score(self) -> Optional[float]:
        """Brier score: (p_hat - outcome)^2. Lower = better."""
        if self.our_probability is None or self.outcome is None:
            return None
        actual = 1.0 if self.outcome else 0.0
        return (self.our_probability - actual) ** 2


# ============================================================================
# METRICS SNAPSHOT — Output of the engine
# ============================================================================


@dataclass
class MetricsSnapshot:
    """
    Complete metrics snapshot for a given window.

    NaN values mean "insufficient data" — consumers should check
    sample_gate_passed before using any metric for decisions.
    """

    # Window metadata
    window_label: str  # "30d", "90d", "all_time"
    n_trades: int
    n_wins: int
    n_losses: int
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None

    # Sample size gate
    sample_gate_passed: bool = False  # True if n_trades >= min for this window
    min_trades_required: int = 0

    # Core return metrics
    total_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0
    net_pnl_usd: float = 0.0
    win_rate: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    profit_factor: float = float("nan")  # gross_profit / gross_loss

    # Expectancy
    expectancy_usd: float = 0.0  # avg P&L per trade
    expectancy_pct: float = 0.0  # avg return per trade

    # Risk-adjusted (per-trade, NOT annualized)
    sharpe_per_trade: float = float("nan")
    sortino_per_trade: float = float("nan")

    # Risk-adjusted (ANNUALIZED — these are what gates use)
    sharpe_annualized: float = float("nan")
    sortino_annualized: float = float("nan")
    calmar_ratio: float = float("nan")  # CAGR / |MDD|

    # Drawdown
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    worst_month_pct: float = 0.0
    time_under_water_days: float = 0.0

    # Tail risk
    var_95_pct: float = float("nan")  # 5th percentile return
    cvar_95_pct: float = float("nan")  # Mean of returns below VaR
    var_99_pct: float = float("nan")
    cvar_99_pct: float = float("nan")

    # Throughput
    trades_per_day: float = 0.0
    trades_per_year_est: float = 0.0  # For annualization
    avg_holding_hours: float = 0.0
    total_volume_usd: float = 0.0

    # Volatility
    return_std: float = float("nan")  # Per-trade return std
    return_std_annualized: float = float("nan")

    # Calibration (Brier)
    brier_score_mean: float = float("nan")
    brier_n: int = 0

    # Strategy attribution
    strategy_pnl: Dict[str, float] = field(default_factory=dict)
    category_pnl: Dict[str, float] = field(default_factory=dict)

    # Confidence intervals (95%)
    sharpe_ci_low: float = float("nan")
    sharpe_ci_high: float = float("nan")
    win_rate_ci_low: float = float("nan")
    win_rate_ci_high: float = float("nan")

    def to_dict(self) -> dict:
        """Serialize to dict, replacing NaN/inf with None for JSON safety.

        Standard JSON (RFC 8259) does not support NaN, Infinity, or -Infinity.
        All such values are mapped to None so that json.dumps() never fails
        and downstream consumers get deterministic behavior.
        """
        d = {}  # type: ignore[var-annotated]
        for k, v in self.__dict__.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None  # type: ignore[assignment]
            elif isinstance(v, datetime):
                d[k] = v.isoformat()
            else:
                d[k] = v
        return d


# ============================================================================
# PERFORMANCE ENGINE — The single source of truth
# ============================================================================

# Minimum trades for each window before metrics are considered meaningful
MIN_TRADES_GATE = {
    "7d": 10,
    "30d": 30,
    "90d": 50,
    "all_time": 20,
}

# Trading days per year for annualization
# Polymarket is 24/7/365 — use 365 calendar days
CALENDAR_DAYS_PER_YEAR = 365.0


class PerformanceEngine:
    """
    Centralized performance computation engine.

    Usage:
        engine = PerformanceEngine()
        engine.ingest_trades(trade_records)  # or ingest incrementally
        snap_30d = engine.compute("30d")
        snap_all = engine.compute("all_time")

    The engine is STATELESS per computation — it operates on the
    trade list it holds. No hidden accumulators to drift.
    """

    def __init__(self, bankroll: float = 5000.0):
        """
        Args:
            bankroll: Starting capital (for return/drawdown % calculations)
        """
        self.bankroll = bankroll
        self.trades: List[TradeRecord] = []

    def ingest_trades(self, trades: List[TradeRecord]):
        """
        Replace internal trade list. Sorts by exit_time.

        CRITICAL: This is a full replacement, not append.
        For incremental, use add_trade().
        """
        self.trades = sorted(trades, key=lambda t: t.exit_time)
        logger.info("PerformanceEngine ingested %d trades", len(self.trades))

    def add_trade(self, trade: TradeRecord):
        """Add a single trade (maintains sorted order)."""
        self.trades.append(trade)
        self.trades.sort(key=lambda t: t.exit_time)

    def compute(self, window: str = "all_time") -> MetricsSnapshot:
        """
        Compute full MetricsSnapshot for given window.

        Args:
            window: "7d", "30d", "90d", or "all_time"

        Returns:
            MetricsSnapshot with all metrics computed
        """
        trades = self._filter_window(window)
        n = len(trades)

        min_trades = MIN_TRADES_GATE.get(window, 20)
        gate_passed = n >= min_trades

        snap = MetricsSnapshot(
            window_label=window,
            n_trades=n,
            n_wins=0,
            n_losses=0,
            sample_gate_passed=gate_passed,
            min_trades_required=min_trades,
        )

        if n == 0:
            return snap

        # Window bounds
        snap.window_start = trades[0].exit_time
        snap.window_end = trades[-1].exit_time

        # Basic counts
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        snap.n_wins = len(wins)
        snap.n_losses = len(losses)

        # P&L
        snap.total_pnl_usd = sum(t.pnl_usd for t in trades)
        snap.total_fees_usd = sum(t.fee_usd for t in trades)
        snap.net_pnl_usd = snap.total_pnl_usd - snap.total_fees_usd
        snap.win_rate = snap.n_wins / n if n > 0 else 0.0

        # Win/loss averages
        if wins:
            snap.avg_win_usd = sum(t.pnl_usd for t in wins) / len(wins)
        if losses:
            snap.avg_loss_usd = sum(t.pnl_usd for t in losses) / len(losses)

        # Profit factor = gross_profit / |gross_loss|
        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0.0
        if gross_loss > 1e-9:
            snap.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            snap.profit_factor = float("inf")
        # else stays NaN

        # Expectancy
        snap.expectancy_usd = snap.total_pnl_usd / n

        # Returns array (per-trade, as fraction of capital)
        returns = np.array([t.return_pct for t in trades])
        snap.expectancy_pct = float(np.mean(returns))

        # Volume
        snap.total_volume_usd = sum(t.size_usd for t in trades)

        # Holding time
        holding_hours = [t.holding_hours for t in trades]
        snap.avg_holding_hours = float(np.mean(holding_hours)) if holding_hours else 0.0

        # Throughput
        if snap.window_start and snap.window_end:
            window_days = max((snap.window_end - snap.window_start).total_seconds() / 86400.0, 1.0)
            snap.trades_per_day = n / window_days
            snap.trades_per_year_est = snap.trades_per_day * CALENDAR_DAYS_PER_YEAR
        else:
            snap.trades_per_day = 0.0
            snap.trades_per_year_est = 0.0

        # === RISK METRICS (only if gate passed) ===
        if gate_passed and n >= 2:
            snap = self._compute_risk_metrics(snap, trades, returns)

        # === DRAWDOWN ===
        snap = self._compute_drawdowns(snap, trades)

        # === TAIL RISK ===
        if gate_passed and n >= 10:
            snap = self._compute_tail_risk(snap, returns)

        # === CALIBRATION ===
        snap = self._compute_calibration(snap, trades)

        # === ATTRIBUTION ===
        snap = self._compute_attribution(snap, trades)

        # === CONFIDENCE INTERVALS ===
        if gate_passed and n >= 10:
            snap = self._compute_confidence_intervals(snap, returns)

        return snap

    # ── Window filtering ────────────────────────────────────────────────

    def _filter_window(self, window: str) -> List[TradeRecord]:
        """Filter trades to the requested window."""
        if window == "all_time" or not self.trades:
            return list(self.trades)

        now = datetime.now(timezone.utc)
        window_days = {
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "180d": 180,
            "365d": 365,
        }

        days = window_days.get(window)
        if days is None:
            logger.warning("Unknown window '%s', returning all trades", window)
            return list(self.trades)

        cutoff = now - timedelta(days=days)
        return [t for t in self.trades if t.exit_time >= cutoff]

    # ── Risk-adjusted metrics ───────────────────────────────────────────

    def _compute_risk_metrics(
        self, snap: MetricsSnapshot, trades: List[TradeRecord], returns: np.ndarray
    ) -> MetricsSnapshot:
        """Compute Sharpe, Sortino, Calmar (per-trade + annualized)."""
        n = len(returns)

        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))  # Unbiased estimator
        snap.return_std = std_r

        # ── Per-trade Sharpe ────────────────────────────────────────────
        # Sharpe = E[R] / sigma(R), risk-free = 0
        if std_r > 1e-9:
            snap.sharpe_per_trade = mean_r / std_r
        else:
            snap.sharpe_per_trade = 0.0

        # ── Per-trade Sortino ───────────────────────────────────────────
        # Only penalize downside volatility
        downside = returns[returns < 0]
        if len(downside) >= 3:
            downside_std = float(np.std(downside, ddof=1))
            if downside_std > 1e-9:
                snap.sortino_per_trade = mean_r / downside_std
            else:
                snap.sortino_per_trade = 0.0 if mean_r <= 0 else float("inf")

        # ── Annualization ───────────────────────────────────────────────
        # The correct scaling for IID returns:
        #   Sharpe_ann = Sharpe_per_trade * sqrt(N_trades_per_year)
        #
        # CRITICAL: N_trades_per_year is estimated from the actual trade
        # frequency, NOT assumed. This prevents inflating Sharpe when
        # only 20 trades happened over 6 months.
        trades_per_year = snap.trades_per_year_est
        if trades_per_year > 0:
            sqrt_n = math.sqrt(trades_per_year)

            if not math.isnan(snap.sharpe_per_trade):
                snap.sharpe_annualized = snap.sharpe_per_trade * sqrt_n

            if not math.isnan(snap.sortino_per_trade):
                snap.sortino_annualized = snap.sortino_per_trade * sqrt_n

            # Annualized volatility
            if not math.isnan(snap.return_std):
                snap.return_std_annualized = snap.return_std * sqrt_n

        # ── Calmar = CAGR / |MDD| ──────────────────────────────────────
        # Computed after drawdown calculation (called separately)
        # Will be filled in _compute_drawdowns

        return snap

    # ── Drawdown tracking ───────────────────────────────────────────────

    def _compute_drawdowns(self, snap: MetricsSnapshot, trades: List[TradeRecord]) -> MetricsSnapshot:
        """Compute max drawdown, worst month, time-under-water."""
        if not trades:
            return snap

        # Equity curve (cumulative P&L)
        equity = np.zeros(len(trades) + 1)
        equity[0] = self.bankroll
        for i, t in enumerate(trades):
            equity[i + 1] = equity[i] + t.pnl_usd

        # Peak equity
        peak = np.maximum.accumulate(equity)

        # Drawdown series (negative values)
        dd_usd = equity - peak
        dd_pct = np.where(peak > 0, dd_usd / peak, 0.0)

        # Max drawdown
        snap.max_drawdown_usd = float(np.min(dd_usd))
        snap.max_drawdown_pct = float(np.min(dd_pct))
        snap.current_drawdown_pct = float(dd_pct[-1])

        # Time under water (days from peak to recovery)
        snap.time_under_water_days = self._compute_time_under_water(trades, equity, peak)

        # Worst calendar month return
        snap.worst_month_pct = self._compute_worst_month(trades)

        # ── Calmar ratio (now that we have MDD) ──────────────────────────
        # Calmar = CAGR / |MDD|
        if abs(snap.max_drawdown_pct) > 1e-9 and snap.window_start and snap.window_end:
            window_days = (snap.window_end - snap.window_start).total_seconds() / 86400.0
            if window_days > 0:
                total_return = (equity[-1] - self.bankroll) / self.bankroll
                # Annualized return (compounded)
                years = window_days / CALENDAR_DAYS_PER_YEAR
                if years > 0 and total_return > -1.0:
                    cagr = (1 + total_return) ** (1.0 / years) - 1.0
                    snap.calmar_ratio = cagr / abs(snap.max_drawdown_pct)

        return snap

    def _compute_time_under_water(
        self,
        trades: List[TradeRecord],
        equity: np.ndarray,
        peak: np.ndarray,
    ) -> float:
        """
        Compute current time-under-water in days.

        Time from when we last hit a new equity high to now.
        If currently at peak, returns 0.
        """
        if len(equity) < 2:
            return 0.0

        # Find most recent peak index
        last_peak_idx = 0
        for i in range(len(equity)):
            if equity[i] >= peak[i] - 1e-9:  # At or near peak
                last_peak_idx = i

        if last_peak_idx >= len(trades):
            return 0.0  # Currently at peak

        # Time from peak trade to last trade
        if last_peak_idx == 0:
            # Never recovered from initial
            peak_time = trades[0].entry_time
        else:
            peak_time = trades[last_peak_idx - 1].exit_time

        last_trade_time = trades[-1].exit_time
        return (last_trade_time - peak_time).total_seconds() / 86400.0

    def _compute_worst_month(self, trades: List[TradeRecord]) -> float:
        """
        Compute worst calendar month return as percentage.

        Groups trades by (year, month) and finds the worst total return.
        """
        if not trades:
            return 0.0

        monthly_pnl: Dict[Tuple[int, int], float] = {}
        for t in trades:
            key = (t.exit_time.year, t.exit_time.month)
            monthly_pnl[key] = monthly_pnl.get(key, 0.0) + t.pnl_usd

        if not monthly_pnl:
            return 0.0

        # Return as % of bankroll
        worst = min(monthly_pnl.values())
        return worst / self.bankroll if self.bankroll > 0 else 0.0

    # ── Tail risk (VaR / CVaR) ──────────────────────────────────────────

    def _compute_tail_risk(self, snap: MetricsSnapshot, returns: np.ndarray) -> MetricsSnapshot:
        """Compute Value-at-Risk and Conditional VaR."""
        # VaR at 95% (5th percentile of returns)
        snap.var_95_pct = float(np.percentile(returns, 5))
        snap.var_99_pct = float(np.percentile(returns, 1))

        # CVaR = E[R | R < VaR] (expected loss in the tail)
        tail_95 = returns[returns <= snap.var_95_pct]
        if len(tail_95) > 0:
            snap.cvar_95_pct = float(np.mean(tail_95))

        tail_99 = returns[returns <= snap.var_99_pct]
        if len(tail_99) > 0:
            snap.cvar_99_pct = float(np.mean(tail_99))

        return snap

    # ── Calibration (Brier score) ───────────────────────────────────────

    def _compute_calibration(self, snap: MetricsSnapshot, trades: List[TradeRecord]) -> MetricsSnapshot:
        """Compute mean Brier score across trades with probability estimates."""
        brier_scores = [t.brier_score for t in trades if t.brier_score is not None]
        snap.brier_n = len(brier_scores)
        if brier_scores:
            snap.brier_score_mean = float(np.mean(brier_scores))
        return snap

    # ── Strategy/category attribution ───────────────────────────────────

    def _compute_attribution(self, snap: MetricsSnapshot, trades: List[TradeRecord]) -> MetricsSnapshot:
        """Compute P&L breakdown by strategy and category."""
        strategy_pnl: Dict[str, float] = {}
        category_pnl: Dict[str, float] = {}

        for t in trades:
            strategy_pnl[t.strategy] = strategy_pnl.get(t.strategy, 0.0) + t.pnl_usd
            category_pnl[t.category] = category_pnl.get(t.category, 0.0) + t.pnl_usd

        snap.strategy_pnl = strategy_pnl
        snap.category_pnl = category_pnl
        return snap

    # ── Confidence intervals ────────────────────────────────────────────

    def _compute_confidence_intervals(self, snap: MetricsSnapshot, returns: np.ndarray) -> MetricsSnapshot:
        """
        Compute 95% confidence intervals for Sharpe and win rate.

        Sharpe CI: Uses the Mertens (2002) formula:
            SE(Sharpe) ~ sqrt((1 + 0.5 * Sharpe^2) / n)
            CI = Sharpe +/- 1.96 * SE

        Win rate CI: Wilson score interval (better than normal approx for small n).
        """
        n = len(returns)
        z = 1.96  # 95% CI

        # Sharpe CI (Mertens approximation)
        if not math.isnan(snap.sharpe_per_trade) and n >= 10:
            se_sharpe = math.sqrt((1 + 0.5 * snap.sharpe_per_trade**2) / n)

            # If annualized, scale the CI too
            if snap.trades_per_year_est > 0:
                sqrt_n_year = math.sqrt(snap.trades_per_year_est)
                se_ann = se_sharpe * sqrt_n_year
                snap.sharpe_ci_low = snap.sharpe_annualized - z * se_ann
                snap.sharpe_ci_high = snap.sharpe_annualized + z * se_ann
            else:
                snap.sharpe_ci_low = snap.sharpe_per_trade - z * se_sharpe
                snap.sharpe_ci_high = snap.sharpe_per_trade + z * se_sharpe

        # Win rate CI (Wilson score)
        if n >= 5:
            p = snap.win_rate
            denom = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
            snap.win_rate_ci_low = max(0.0, center - spread)
            snap.win_rate_ci_high = min(1.0, center + spread)

        return snap

    # ── Convenience methods ─────────────────────────────────────────────

    def compute_all_windows(self) -> Dict[str, MetricsSnapshot]:
        """Compute snapshots for all standard windows."""
        windows = ["7d", "30d", "90d", "all_time"]
        return {w: self.compute(w) for w in windows}

    def get_gate_metrics(self) -> dict:
        """
        Extract metrics specifically needed for gate evaluation.

        Returns dict with keys matching config.py gate thresholds.
        """
        snap_90d = self.compute("90d")
        snap_all = self.compute("all_time")

        return {
            # Gate B metrics
            "sharpe_90d": snap_90d.sharpe_annualized,
            "n_trades_90d": snap_90d.n_trades,
            "win_rate_90d": snap_90d.win_rate,
            "gate_passed_90d": snap_90d.sample_gate_passed,
            # Gate C metrics
            "sharpe_ann_90d": snap_90d.sharpe_annualized,
            "calmar_90d": snap_90d.calmar_ratio,
            "max_drawdown_90d": snap_90d.max_drawdown_pct,
            # Cross-window
            "sharpe_ann_all": snap_all.sharpe_annualized,
            "total_trades": snap_all.n_trades,
            "total_pnl": snap_all.total_pnl_usd,
            "brier_score": snap_all.brier_score_mean,
            # Confidence
            "sharpe_ci_low_90d": snap_90d.sharpe_ci_low,
            "sharpe_ci_high_90d": snap_90d.sharpe_ci_high,
        }

    def format_summary(self, window: str = "all_time") -> str:
        """Human-readable summary for console/logs."""
        snap = self.compute(window)
        gate = "PASS" if snap.sample_gate_passed else f"NEED {snap.min_trades_required - snap.n_trades} more"

        def fmt(v: float, suffix: str = "") -> str:
            if math.isnan(v):
                return "N/A"
            if math.isinf(v):
                return "inf" if v > 0 else "-inf"
            return f"{v:.2f}{suffix}"

        lines = [
            f"=== Astra Metrics [{snap.window_label}] ===",
            f"Trades: {snap.n_trades} (gate: {gate})",
            f"Win rate: {snap.win_rate:.0%}  |  PnL: ${snap.total_pnl_usd:+.2f}",
            f"Profit factor: {fmt(snap.profit_factor)}  |  Expectancy: ${snap.expectancy_usd:+.2f}/trade",
            "",
            f"Sharpe (per-trade): {fmt(snap.sharpe_per_trade)}",
            f"Sharpe (annualized): {fmt(snap.sharpe_annualized)} [{fmt(snap.sharpe_ci_low)}, {fmt(snap.sharpe_ci_high)}]",
            f"Sortino (annualized): {fmt(snap.sortino_annualized)}",
            f"Calmar: {fmt(snap.calmar_ratio)}",
            "",
            f"Max DD: {fmt(snap.max_drawdown_pct * 100, '%')}  |  Worst month: {fmt(snap.worst_month_pct * 100, '%')}",
            f"Time under water: {snap.time_under_water_days:.0f} days",
            f"VaR(95%): {fmt(snap.var_95_pct * 100, '%')}  |  CVaR(95%): {fmt(snap.cvar_95_pct * 100, '%')}",
            "",
            f"Brier score: {fmt(snap.brier_score_mean)} (n={snap.brier_n})",
            f"Trades/day: {snap.trades_per_day:.1f}  |  Avg hold: {snap.avg_holding_hours:.1f}h",
        ]

        return "\n".join(lines)


# ============================================================================
# FACTORY: Create TradeRecord from PaperPosition
# ============================================================================


def trade_record_from_paper_position(pos, trade_id: Optional[str] = None) -> Optional[TradeRecord]:
    """
    Convert a resolved PaperPosition to a TradeRecord.

    Args:
        pos: PaperPosition instance (from paper_trader.py)
        trade_id: Optional ID (defaults to condition_id)

    Returns:
        TradeRecord or None if position is not resolved or has no P&L
    """
    if not pos.resolved or pos.pnl is None:
        return None

    try:
        entry_time = datetime.fromisoformat(pos.timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        entry_time = datetime.now(timezone.utc)

    try:
        exit_time = datetime.fromisoformat(pos.resolution_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        exit_time = datetime.now(timezone.utc)

    return TradeRecord(
        trade_id=trade_id or pos.condition_id,
        condition_id=pos.condition_id,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=pos.entry_price,
        exit_price=pos.exit_price or 0.0,
        size_usd=pos.position_size,
        pnl_usd=pos.pnl,
        fee_usd=0.0,  # Paper trading has no fees
        direction=pos.direction,
        category=pos.category,
        strategy="astra_v2",  # Default for paper trades
        our_probability=pos.our_probability,
        market_price_at_entry=pos.market_price_at_entry,
        outcome=pos.outcome,
    )
