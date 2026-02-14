"""
Tests for RiskEngine — hard risk halts with cooldown timer.

All reason assertions use exact enums from models.reasons.

Loop 4: Risk halts first. Prefer missing opportunity over bleeding.
"""

from __future__ import annotations

import pytest

from models.reasons import (
    REASON_RISK_BROKEN_REGIME,
    REASON_RISK_CATEGORY_CAP,
    REASON_RISK_COOLDOWN,
    REASON_RISK_DAILY_LOSS,
    REASON_RISK_DRAWDOWN,
    REASON_RISK_OK,
)
from risk.risk_engine import RiskEngine


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine(
        portfolio_engine=None,
        daily_loss_halt_pct=0.05,
        drawdown_halt_pct=0.15,
        category_caps={"crypto_threshold": 0.30},
        cooldown_seconds=300,
        initial_equity=5000.0,
    )


class TestCanTrade:
    """Tests for can_trade checks."""

    def test_normal_conditions_allowed(self, engine: RiskEngine) -> None:
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=100.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="NORMAL",
            now=1000.0,
        )
        assert allowed is True
        assert reason == REASON_RISK_OK

    def test_broken_regime_vetoed(self, engine: RiskEngine) -> None:
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=100.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="BROKEN",
            now=1000.0,
        )
        assert allowed is False
        assert reason == REASON_RISK_BROKEN_REGIME

    def test_category_cap_vetoed(self, engine: RiskEngine) -> None:
        """Exceeding category cap → veto."""
        # Cap is 30% of 5000 = 1500
        engine.update_category_exposure("crypto_threshold", 1400.0)
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=200.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="NORMAL",
            now=1000.0,
        )
        assert allowed is False
        assert reason == REASON_RISK_CATEGORY_CAP

    def test_category_cap_allowed_when_under(self, engine: RiskEngine) -> None:
        engine.update_category_exposure("crypto_threshold", 1000.0)
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=100.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="NORMAL",
            now=1000.0,
        )
        assert allowed is True


class TestDailyLossHalt:
    """Tests for daily loss halt."""

    def test_daily_loss_triggers_halt(self, engine: RiskEngine) -> None:
        """Daily loss exceeding threshold → halt."""
        # Threshold = 5% of 5000 = 250
        engine.record_pnl(-260.0, current_equity=4740.0, now=1000.0)
        assert engine.is_halted is True
        assert engine.halt_reason == REASON_RISK_DAILY_LOSS

    def test_daily_loss_halt_blocks_trades(self, engine: RiskEngine) -> None:
        engine.record_pnl(-260.0, current_equity=4740.0, now=1000.0)
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=100.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="NORMAL",
            now=1001.0,
        )
        assert allowed is False
        # Cooldown check fires first in can_trade(), so reason is cooldown_active
        assert reason == REASON_RISK_COOLDOWN

    def test_daily_loss_below_threshold_no_halt(self, engine: RiskEngine) -> None:
        engine.record_pnl(-200.0, current_equity=4800.0, now=1000.0)
        assert engine.is_halted is False


class TestDrawdownHalt:
    """Tests for drawdown halt."""

    def test_drawdown_triggers_halt(self, engine: RiskEngine) -> None:
        """Drawdown exceeding 15% → halt."""
        # Peak = 5000, drawdown threshold = 15% → halt at 4250
        engine.record_pnl(-800.0, current_equity=4200.0, now=1000.0)
        assert engine.is_halted is True
        assert engine.halt_reason == REASON_RISK_DRAWDOWN

    def test_drawdown_math_correct(self, engine: RiskEngine) -> None:
        """Verify: (peak - current) / peak."""
        # Peak starts at 5000, daily loss threshold = 5% of 5000 = 250
        # Use smaller losses that stay under daily loss threshold
        # but test drawdown accumulation over multiple days

        # Fresh engine with higher daily loss tolerance
        eng = RiskEngine(
            daily_loss_halt_pct=1.0,  # disable daily loss for this test
            drawdown_halt_pct=0.15,
            initial_equity=5000.0,
            cooldown_seconds=300,
        )

        # 4500 → dd = (5000-4500)/5000 = 0.10 → below 0.15 → no halt
        eng.record_pnl(-500.0, current_equity=4500.0, now=1000.0)
        assert eng.is_halted is False

        # 4200 → dd = (5000-4200)/5000 = 0.16 → above 0.15 → halt
        eng.record_pnl(-300.0, current_equity=4200.0, now=1001.0)
        assert eng.is_halted is True
        assert eng.halt_reason == REASON_RISK_DRAWDOWN

    def test_peak_equity_updates(self, engine: RiskEngine) -> None:
        """Peak equity should track high-water mark."""
        engine.record_pnl(500.0, current_equity=5500.0, now=1000.0)
        assert engine.peak_equity == 5500.0

        # Equity drops but peak stays
        engine.record_pnl(-200.0, current_equity=5300.0, now=1001.0)
        assert engine.peak_equity == 5500.0


class TestCooldown:
    """Tests for cooldown timer after halt."""

    def test_cooldown_blocks_after_halt(self, engine: RiskEngine) -> None:
        """After halt, cooldown must expire before trading resumes."""
        engine.record_pnl(-260.0, current_equity=4740.0, now=1000.0)
        assert engine.is_halted is True

        # During cooldown
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=100.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="NORMAL",
            now=1100.0,  # 100s into 300s cooldown
        )
        assert allowed is False
        assert reason == REASON_RISK_COOLDOWN

    def test_cooldown_expires(self, engine: RiskEngine) -> None:
        """After cooldown expires, trading resumes."""
        engine.record_pnl(-260.0, current_equity=4740.0, now=1000.0)

        # After cooldown (300s)
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=100.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="NORMAL",
            now=1301.0,
        )
        assert allowed is True
        assert reason == REASON_RISK_OK

    def test_reset_daily_does_not_clear_halt(self, engine: RiskEngine) -> None:
        """reset_daily clears P&L counter but NOT halt state."""
        engine.record_pnl(-260.0, current_equity=4740.0, now=1000.0)
        assert engine.is_halted is True

        engine.reset_daily()
        assert engine.daily_pnl == 0.0

        # Still halted despite reset
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=100.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="NORMAL",
            now=1100.0,
        )
        assert allowed is False

    def test_cooldown_prevents_restart_bleed(self, engine: RiskEngine) -> None:
        """Halt + immediate reset_daily → still halted until cooldown."""
        engine.record_pnl(-260.0, current_equity=4740.0, now=1000.0)
        engine.reset_daily()  # P&L reset

        # P&L is 0 now, but cooldown still active
        allowed, reason = engine.can_trade(
            market=None,
            size_usd=100.0,
            category="crypto_threshold",
            toxicity_flag=False,
            spread_regime="NORMAL",
            now=1100.0,
        )
        assert allowed is False
        assert reason == REASON_RISK_COOLDOWN
