"""
Tests for the centralized metrics engine.

Validates:
1. Sharpe/Sortino/Calmar computation correctness
2. Annualization formula
3. Drawdown tracking
4. Sample-size gates
5. Confidence intervals
6. VaR/CVaR
7. Profit factor / expectancy
8. Edge cases (0 trades, 1 trade, all wins, all losses)
"""

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.confidence import (
    SampleGate,
    check_all_gates,
    confidence_interval,
    min_trades_for_sharpe_precision,
    sample_size_gate,
    sharpe_confidence_interval,
)
from metrics.drawdown import DrawdownState, DrawdownTracker
from metrics.performance import MetricsSnapshot, PerformanceEngine, TradeRecord

# ============================================================================
# HELPERS
# ============================================================================


def make_trade(
    pnl: float,
    size: float = 100.0,
    entry_price: float = 0.50,
    days_ago: float = 0,
    holding_hours: float = 24.0,
    category: str = "crypto",
    strategy: str = "astra_v2",
    our_prob: float = None,  # type: ignore[assignment]
    outcome: bool = None,  # type: ignore[assignment]
) -> TradeRecord:
    """Create a TradeRecord for testing."""
    now = datetime.now(timezone.utc)
    exit_time = now - timedelta(days=days_ago)
    entry_time = exit_time - timedelta(hours=holding_hours)

    return TradeRecord(
        trade_id=f"test_{hash((pnl, size, days_ago))}",
        condition_id=f"cid_{hash((pnl, days_ago))}",
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=1.0 if pnl > 0 else 0.0,
        size_usd=size,
        pnl_usd=pnl,
        fee_usd=0.0,
        direction="BUY YES",
        category=category,
        strategy=strategy,
        our_probability=our_prob,
        outcome=outcome,
    )


def make_trades_sequence(pnls: list, spread_days: float = 1.0) -> list:
    """Create a sequence of trades spread evenly over time."""
    trades = []
    for i, pnl in enumerate(pnls):
        trades.append(
            make_trade(
                pnl=pnl,
                days_ago=(len(pnls) - i) * spread_days,
            )
        )
    return trades


# ============================================================================
# TEST: PerformanceEngine
# ============================================================================


class TestPerformanceEngine:
    def test_empty_trades(self):
        engine = PerformanceEngine(bankroll=5000)
        snap = engine.compute("all_time")
        assert snap.n_trades == 0
        assert snap.total_pnl_usd == 0.0
        assert snap.sample_gate_passed is False
        assert math.isnan(snap.sharpe_per_trade)

    def test_single_trade(self):
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades([make_trade(pnl=50.0)])
        snap = engine.compute("all_time")
        assert snap.n_trades == 1
        assert snap.total_pnl_usd == 50.0
        assert snap.n_wins == 1
        assert snap.n_losses == 0
        assert snap.win_rate == 1.0
        assert snap.sample_gate_passed is False  # Need ≥20

    def test_basic_pnl(self):
        trades = make_trades_sequence([50, -30, 20, -10, 40, -20, 30, -15, 25, -5])
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert snap.n_trades == 10
        assert snap.total_pnl_usd == 85.0  # sum of all PnLs
        assert snap.n_wins == 5
        assert snap.n_losses == 5
        assert snap.win_rate == 0.5

    def test_win_rate(self):
        trades = make_trades_sequence([10, 20, 30, -5, -10])
        engine = PerformanceEngine(bankroll=1000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert snap.n_wins == 3
        assert snap.n_losses == 2
        assert abs(snap.win_rate - 0.6) < 0.01

    def test_profit_factor(self):
        """Profit factor = gross_profit / |gross_loss|"""
        trades = make_trades_sequence([100, -50, 80, -30])
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        gross_profit = 100 + 80  # 180
        gross_loss = 50 + 30  # 80
        expected_pf = gross_profit / gross_loss  # 2.25
        assert abs(snap.profit_factor - expected_pf) < 0.01

    def test_profit_factor_no_losses(self):
        trades = make_trades_sequence([10, 20, 30])
        engine = PerformanceEngine(bankroll=1000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")
        assert snap.profit_factor == float("inf")

    def test_expectancy(self):
        pnls = [50, -30, 20, -10]
        trades = make_trades_sequence(pnls)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        expected_exp = sum(pnls) / len(pnls)  # 30/4 = 7.5
        assert abs(snap.expectancy_usd - expected_exp) < 0.01

    def test_sharpe_positive_returns(self):
        """All positive returns should give positive Sharpe."""
        pnls = [
            10,
            20,
            15,
            12,
            8,
            18,
            22,
            14,
            11,
            16,
            13,
            19,
            17,
            9,
            21,
            15,
            12,
            20,
            11,
            14,
            16,
            18,
            13,
            15,
            20,
            12,
            17,
            19,
            11,
            14,
        ]
        trades = make_trades_sequence(pnls, spread_days=1.0)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert snap.sharpe_per_trade > 0
        assert snap.sharpe_annualized > snap.sharpe_per_trade  # Annualized should be larger
        assert snap.sample_gate_passed  # 30 trades ≥ 20 gate

    def test_sortino_only_penalizes_downside(self):
        """Sortino should be >= Sharpe when there's upside variance."""
        # Mix of returns with more upside variance than downside
        pnls = [
            50,
            -10,
            80,
            -5,
            60,
            -8,
            70,
            -3,
            40,
            -15,
            55,
            -7,
            45,
            -12,
            65,
            -6,
            35,
            -9,
            75,
            -4,
            50,
            -10,
            80,
            -5,
            60,
            -8,
            70,
            -3,
            40,
            -15,
        ]
        trades = make_trades_sequence(pnls, spread_days=1.0)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert snap.sortino_per_trade > snap.sharpe_per_trade

    def test_annualization_formula(self):
        """Sharpe_ann = Sharpe_per_trade * sqrt(trades_per_year)."""
        pnls = [
            10,
            -5,
            8,
            -3,
            12,
            -4,
            7,
            -6,
            15,
            -2,
            9,
            -7,
            11,
            -3,
            14,
            -5,
            8,
            -4,
            13,
            -6,
            10,
            -5,
            8,
            -3,
            12,
            -4,
            7,
            -6,
            15,
            -2,
        ]
        trades = make_trades_sequence(pnls, spread_days=1.0)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        # Verify annualization: sharpe_ann ≈ sharpe_per_trade * sqrt(tpy)
        if snap.trades_per_year_est > 0:
            expected_ann = snap.sharpe_per_trade * math.sqrt(snap.trades_per_year_est)
            assert abs(snap.sharpe_annualized - expected_ann) < 0.01

    def test_drawdown_computation(self):
        """Drawdown should correctly track peak-to-trough."""
        # Equity curve: 5000 → 5050 → 5020 → 5070 → 5010
        pnls = [50, -30, 50, -60]
        trades = make_trades_sequence(pnls, spread_days=1.0)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        # Max drawdown should be -60 from peak of 5070 to 5010
        assert snap.max_drawdown_usd < -50  # At least -60
        assert snap.max_drawdown_pct < 0

    def test_var_cvar(self):
        """VaR and CVaR should be computed for sufficient trades."""
        pnls = [
            10,
            -20,
            15,
            -30,
            5,
            -10,
            20,
            -25,
            8,
            -15,
            12,
            -18,
            22,
            -35,
            7,
            -12,
            18,
            -22,
            9,
            -28,
            14,
            -16,
            25,
            -32,
            6,
            -8,
            19,
            -20,
            11,
            -14,
            13,
            -17,
            21,
            -33,
            8,
            -11,
            16,
            -23,
            10,
            -26,
        ]
        trades = make_trades_sequence(pnls, spread_days=0.5)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert not math.isnan(snap.var_95_pct)
        assert not math.isnan(snap.cvar_95_pct)
        # CVaR should be more extreme than VaR
        assert snap.cvar_95_pct <= snap.var_95_pct

    def test_brier_score(self):
        """Brier score should be computed from trades with probability data."""
        trades = [
            make_trade(pnl=50, our_prob=0.8, outcome=True),
            make_trade(pnl=-50, our_prob=0.7, outcome=False, days_ago=1),
            make_trade(pnl=30, our_prob=0.6, outcome=True, days_ago=2),
        ]
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert snap.brier_n == 3
        # Brier for trade 1: (0.8 - 1.0)^2 = 0.04
        # Brier for trade 2: (0.7 - 0.0)^2 = 0.49
        # Brier for trade 3: (0.6 - 1.0)^2 = 0.16
        expected_mean = (0.04 + 0.49 + 0.16) / 3
        assert abs(snap.brier_score_mean - expected_mean) < 0.001

    def test_strategy_attribution(self):
        """P&L should be correctly attributed to strategies."""
        trades = [
            make_trade(pnl=50, strategy="maker"),
            make_trade(pnl=-20, strategy="taker", days_ago=1),
            make_trade(pnl=30, strategy="maker", days_ago=2),
            make_trade(pnl=-10, strategy="satellite", days_ago=3),
        ]
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert snap.strategy_pnl["maker"] == 80.0
        assert snap.strategy_pnl["taker"] == -20.0
        assert snap.strategy_pnl["satellite"] == -10.0

    def test_window_filtering(self):
        """7d window should only include recent trades."""
        trades = [
            make_trade(pnl=100, days_ago=1),  # Within 7d
            make_trade(pnl=50, days_ago=3),  # Within 7d
            make_trade(pnl=-200, days_ago=30),  # Outside 7d
        ]
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)

        snap_7d = engine.compute("7d")
        snap_all = engine.compute("all_time")

        assert snap_7d.n_trades == 2
        assert snap_7d.total_pnl_usd == 150.0
        assert snap_all.n_trades == 3
        assert snap_all.total_pnl_usd == -50.0

    def test_all_losses(self):
        """Should handle all-loss portfolio correctly."""
        pnls = [-10, -20, -15, -30, -5]
        trades = make_trades_sequence(pnls)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert snap.n_wins == 0
        assert snap.n_losses == 5
        assert snap.win_rate == 0.0
        assert snap.total_pnl_usd == -80.0
        # No wins → profit factor is 0/|gross_loss| = 0
        assert snap.profit_factor == 0.0

    def test_worst_month(self):
        """Worst month should reflect the worst calendar month."""
        # All trades in the same month
        pnls = [50, -100, 20, -80, 30]  # Net: -80
        trades = make_trades_sequence(pnls, spread_days=1.0)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        # Worst month should be negative
        assert snap.worst_month_pct < 0

    def test_confidence_intervals_present(self):
        """CI should be computed when we have enough trades."""
        pnls = [10, -5] * 20  # 40 trades
        trades = make_trades_sequence(pnls, spread_days=0.5)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")

        assert not math.isnan(snap.sharpe_ci_low)
        assert not math.isnan(snap.sharpe_ci_high)
        assert snap.sharpe_ci_low < snap.sharpe_ci_high
        assert not math.isnan(snap.win_rate_ci_low)
        assert snap.win_rate_ci_low < snap.win_rate_ci_high

    def test_to_dict_replaces_nan(self):
        """NaN values should become None in dict representation."""
        engine = PerformanceEngine(bankroll=5000)
        snap = engine.compute("all_time")
        d = snap.to_dict()
        # Check that NaN values are None (not NaN strings)
        assert d["sharpe_per_trade"] is None
        assert d["calmar_ratio"] is None

    def test_to_dict_replaces_inf(self):
        """Inf/-inf values should become None in serialized output (JSON safety)."""
        import json

        # All wins → profit_factor = +inf, sortino can be inf
        pnls = [
            10,
            20,
            15,
            12,
            8,
            18,
            22,
            14,
            11,
            16,
            13,
            19,
            17,
            9,
            21,
            15,
            12,
            20,
            11,
            14,
            16,
            18,
            13,
            15,
            20,
            12,
            17,
            19,
            11,
            14,
        ]
        trades = make_trades_sequence(pnls, spread_days=1.0)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)
        snap = engine.compute("all_time")
        d = snap.to_dict()

        # Must be valid JSON — no inf, -inf, nan
        serialized = json.dumps(d)  # Should not raise
        assert "Infinity" not in serialized
        assert "NaN" not in serialized
        assert "inf" not in serialized.lower().replace('"window_info"', "")

        # profit_factor should be None (was +inf from all-wins)
        assert d["profit_factor"] is None

    def test_gate_metrics(self):
        """Gate metrics should return correct values for gate evaluation."""
        pnls = [10, -5] * 30  # 60 trades
        trades = make_trades_sequence(pnls, spread_days=0.5)
        engine = PerformanceEngine(bankroll=5000)
        engine.ingest_trades(trades)

        gates = engine.get_gate_metrics()
        assert "sharpe_90d" in gates
        assert "calmar_90d" in gates
        assert "total_trades" in gates
        assert gates["total_trades"] == 60


# ============================================================================
# TEST: DrawdownTracker
# ============================================================================


class TestDrawdownTracker:
    def test_initial_state(self):
        tracker = DrawdownTracker(bankroll=5000)
        state = tracker.get_state()
        assert state.current_equity == 5000
        assert state.peak_equity == 5000
        assert state.current_drawdown_usd == 0
        assert state.current_drawdown_pct == 0

    def test_profit_updates_peak(self):
        tracker = DrawdownTracker(bankroll=5000)
        tracker.record_pnl(100)
        state = tracker.get_state()
        assert state.current_equity == 5100
        assert state.peak_equity == 5100
        assert state.current_drawdown_pct == 0

    def test_loss_creates_drawdown(self):
        tracker = DrawdownTracker(bankroll=5000)
        tracker.record_pnl(100)  # Peak at 5100
        tracker.record_pnl(-200)  # Down to 4900
        state = tracker.get_state()

        assert state.current_equity == 4900
        assert state.peak_equity == 5100
        assert abs(state.current_drawdown_usd - (-200)) < 0.01
        assert abs(state.current_drawdown_pct - (-200 / 5100)) < 0.001

    def test_max_drawdown_tracking(self):
        tracker = DrawdownTracker(bankroll=5000)
        tracker.record_pnl(500)  # 5500
        tracker.record_pnl(-1000)  # 4500 (DD = -1000)
        tracker.record_pnl(2000)  # 6500 (new peak)
        tracker.record_pnl(-500)  # 6000 (DD = -500, but max was -1000)

        state = tracker.get_state()
        assert state.max_drawdown_usd == -1000
        assert state.current_drawdown_usd == -500

    def test_recovery_needed(self):
        tracker = DrawdownTracker(bankroll=5000)
        tracker.record_pnl(1000)  # Peak: 6000
        tracker.record_pnl(-1500)  # Current: 4500

        state = tracker.get_state()
        # Recovery needed: (6000 - 4500) / 4500 = 33.3%
        assert abs(state.recovery_needed_pct - 1500 / 4500) < 0.01

    def test_monthly_pnl_tracking(self):
        tracker = DrawdownTracker(bankroll=5000)
        now = datetime.now(timezone.utc)

        tracker.record_pnl(100, timestamp=now)
        tracker.record_pnl(-50, timestamp=now)

        monthly = tracker.get_monthly_pnl()
        month_key = now.strftime("%Y-%m")
        assert month_key in monthly
        assert monthly[month_key] == 50.0

    def test_is_breaching(self):
        tracker = DrawdownTracker(bankroll=5000)
        tracker.record_pnl(-1000)  # 20% drawdown

        state = tracker.get_state()
        # -0.15 threshold = 15% max drawdown
        assert state.is_breaching(-0.15)  # 20% > 15%
        assert not state.is_breaching(-0.25)  # 20% < 25%


# ============================================================================
# TEST: Confidence Intervals
# ============================================================================


class TestConfidence:
    def test_normal_ci(self):
        values = [10, 12, 11, 13, 9, 14, 10, 12, 11, 13]
        low, high = confidence_interval(values, confidence=0.95, method="normal")  # type: ignore[arg-type]

        assert low < high
        mean = sum(values) / len(values)
        assert low < mean < high

    def test_bootstrap_ci(self):
        values = [10, 12, 11, 13, 9, 14, 10, 12, 11, 13]
        low, high = confidence_interval(values, confidence=0.95, method="bootstrap")  # type: ignore[arg-type]

        assert low < high
        mean = sum(values) / len(values)
        assert low < mean < high

    def test_wilson_ci(self):
        values = [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]  # 70% win rate
        low, high = confidence_interval(values, confidence=0.95, method="wilson")  # type: ignore[arg-type]

        assert 0 <= low < high <= 1
        assert low < 0.7 < high

    def test_empty_values(self):
        low, high = confidence_interval([], confidence=0.95)
        assert math.isnan(low)
        assert math.isnan(high)

    def test_sample_gate(self):
        gate = sample_size_gate("sharpe_ratio", n_actual=15)
        assert not gate.passed
        assert gate.shortfall == 15  # 30 - 15

        gate = sample_size_gate("sharpe_ratio", n_actual=30)
        assert gate.passed
        assert gate.shortfall == 0

    def test_check_all_gates(self):
        gates = check_all_gates(n_trades=25)
        # Should pass some, fail others
        assert gates["expectancy"].passed  # Needs 10
        assert not gates["sharpe_ratio"].passed  # Needs 30

    def test_sharpe_ci(self):
        low, high = sharpe_confidence_interval(
            sharpe=1.5,
            n_trades=100,
            confidence=0.95,
            annualization_factor=1.0,
        )
        assert low < 1.5 < high
        assert low > 0  # Sharpe 1.5 with 100 trades should have positive lower bound

    def test_sharpe_ci_small_n(self):
        low, high = sharpe_confidence_interval(
            sharpe=2.0,
            n_trades=3,
        )
        assert math.isnan(low)
        assert math.isnan(high)

    def test_min_trades_calculation(self):
        n = min_trades_for_sharpe_precision(
            target_precision=0.5,
            expected_sharpe=1.0,
        )
        assert n > 0
        assert n < 200  # Should be reasonable for 0.5 precision


# ============================================================================
# RUN TESTS
# ============================================================================


def run_all_tests():
    """Run all test classes and report results."""
    test_classes = [
        TestPerformanceEngine,
        TestDrawdownTracker,
        TestConfidence,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            test_name = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {test_name}")
            except AssertionError as e:
                failed += 1
                errors.append((test_name, str(e)))
                print(f"  FAIL  {test_name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((test_name, str(e)))
                print(f"  ERROR {test_name}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
