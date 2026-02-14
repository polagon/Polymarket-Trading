"""
Risk Engine — hard risk halts with cooldown timer.

Wraps existing PortfolioRiskEngine. Does NOT replace it — adds:
  - Daily loss halt
  - Drawdown halt
  - Cooldown timer (prevents halt-restart-bleed loops)
  - Per-category exposure caps
  - BROKEN regime veto

All reason strings are stable enums from models.reasons.

Loop 4: Risk halts first. Prefer missing opportunity over bleeding.
"""

from __future__ import annotations

import logging
from typing import Optional

from models.reasons import (
    REASON_RISK_BROKEN_REGIME,
    REASON_RISK_CATEGORY_CAP,
    REASON_RISK_COOLDOWN,
    REASON_RISK_DAILY_LOSS,
    REASON_RISK_DRAWDOWN,
    REASON_RISK_OK,
    REASON_RISK_PORTFOLIO_CAP,
)

logger = logging.getLogger(__name__)

_EPS = 1e-9


class RiskEngine:
    """Hard risk halts with cooldown timer.

    Wraps existing PortfolioRiskEngine for position-level checks
    while adding system-level halt/cooldown/regime vetoes.

    Args:
        portfolio_engine: Existing PortfolioRiskEngine (or None for standalone use).
        daily_loss_halt_pct: Daily loss threshold to trigger halt (0.05 = 5%).
        drawdown_halt_pct: Peak-to-trough drawdown threshold (0.15 = 15%).
        category_caps: Per-category exposure caps as fraction of equity.
        cooldown_seconds: Seconds to remain halted after trigger.
        initial_equity: Starting equity for peak tracking.
    """

    def __init__(
        self,
        portfolio_engine: object = None,
        daily_loss_halt_pct: float = 0.05,
        drawdown_halt_pct: float = 0.15,
        category_caps: Optional[dict[str, float]] = None,
        cooldown_seconds: int = 300,
        initial_equity: float = 5000.0,
    ) -> None:
        self._portfolio = portfolio_engine
        self._daily_loss_halt_pct = daily_loss_halt_pct
        self._drawdown_halt_pct = drawdown_halt_pct
        self._category_caps = category_caps or {}
        self._cooldown_seconds = cooldown_seconds

        self._halted = False
        self._halt_reason = ""
        self._halt_until: float = 0.0  # epoch when cooldown expires

        self._daily_pnl = 0.0
        self._peak_equity = max(initial_equity, _EPS)  # floor at initial
        self._current_equity = initial_equity

        # Per-category exposure tracking
        self._category_exposure: dict[str, float] = {}

    def can_trade(
        self,
        market: object,
        size_usd: float,
        category: str,
        toxicity_flag: bool,
        spread_regime: str,
        now: float = 0.0,
    ) -> tuple[bool, str]:
        """Check if a trade is allowed.

        Checks in order:
          1. Cooldown active
          2. Halted (daily loss / drawdown)
          3. BROKEN regime
          4. Portfolio engine (if available)
          5. Category cap

        Args:
            market: Market object (passed to portfolio engine).
            size_usd: Proposed trade size in USD.
            category: Market category.
            toxicity_flag: Whether flow is toxic.
            spread_regime: Current spread regime string.
            now: Current epoch time.

        Returns:
            (allowed, reason) tuple with stable reason enum.
        """
        # 1. Cooldown check
        if self._halt_until > 0 and now < self._halt_until:
            return (False, REASON_RISK_COOLDOWN)

        # If cooldown has expired, clear halt state
        if self._halted and self._halt_until > 0 and now >= self._halt_until:
            self._halted = False
            self._halt_reason = ""
            self._halt_until = 0.0
            logger.info("Risk halt cooldown expired, resuming")

        # 2. Halted check (may still be halted if cooldown hasn't expired)
        if self._halted:
            return (False, self._halt_reason)

        # 3. BROKEN regime → unconditional veto
        if spread_regime == "BROKEN":
            return (False, REASON_RISK_BROKEN_REGIME)

        # 4. Portfolio engine check (if available)
        if self._portfolio is not None and hasattr(self._portfolio, "can_enter_position"):
            try:
                allowed = self._portfolio.can_enter_position(market, size_usd)
                if not allowed:
                    return (False, REASON_RISK_PORTFOLIO_CAP)
            except Exception:
                pass  # Don't crash on portfolio engine errors

        # 5. Category cap
        cap = self._category_caps.get(category, 1.0)
        current_exposure = self._category_exposure.get(category, 0.0)
        max_exposure = cap * self._peak_equity
        if current_exposure + size_usd > max_exposure:
            return (False, REASON_RISK_CATEGORY_CAP)

        return (True, REASON_RISK_OK)

    def record_pnl(self, pnl: float, current_equity: float, now: float = 0.0) -> None:
        """Record P&L and check halt conditions.

        Args:
            pnl: P&L from this trade/event.
            current_equity: Current total equity.
            now: Current epoch time.
        """
        self._daily_pnl += pnl
        self._current_equity = current_equity

        # Check daily loss
        daily_loss_threshold = self._daily_loss_halt_pct * self._peak_equity
        if self._daily_pnl < -daily_loss_threshold:
            self._halt(REASON_RISK_DAILY_LOSS, now)

        # Update peak equity (floor at initial to avoid div weirdness)
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Check drawdown: (peak - current) / max(peak, eps)
        drawdown = (self._peak_equity - current_equity) / max(self._peak_equity, _EPS)
        if drawdown > self._drawdown_halt_pct:
            self._halt(REASON_RISK_DRAWDOWN, now)

    def _halt(self, reason: str, now: float) -> None:
        """Trigger a halt with cooldown."""
        self._halted = True
        self._halt_reason = reason
        self._halt_until = now + self._cooldown_seconds
        logger.warning(f"Risk halt triggered: {reason} (cooldown until {self._halt_until})")

    def reset_daily(self) -> None:
        """Reset daily P&L counter. Does NOT clear halt state or cooldown."""
        self._daily_pnl = 0.0

    def update_category_exposure(self, category: str, exposure_usd: float) -> None:
        """Update tracked exposure for a category."""
        self._category_exposure[category] = exposure_usd

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_equity(self) -> float:
        return self._current_equity
