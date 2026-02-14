"""
EV Gate — conservative lower-bound EV with size-aware friction model.

CRITICAL: All inputs and outputs are in FRACTIONAL units.
Never mix USD and fractions inside EV math.

Loop 4: No taker escalation. MAKER_ONLY or NO_TRADE.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.reasons import (
    REASON_EV_INVALID_MARKET_PRICE,
    REASON_EV_INVALID_PROB_BOUNDS,
    REASON_EV_MAKER_ONLY,
    REASON_EV_NET_LB_BELOW_THRESHOLD,
)

# Imported at module level; actual values come from config at call site.
_DEFAULT_EPS = 1e-9
_DEFAULT_DEPTH_FLOOR = 100.0


@dataclass
class EVGateResult:
    """Result of EV gate evaluation. All cost fields are fractions.

    Attributes:
        approved: True if trade passes EV gate.
        ev_gross_lb: Lower-bound gross EV (fraction / probability points).
        fee_est_frac: Fee estimate as fraction (e.g. 0.02 = 2%).
        slippage_est_frac: Slippage estimate as fraction.
        adverse_buffer_frac: Adverse selection buffer as fraction.
        ev_net_lb: Net EV after all frictions (fraction).
        execution_mode: "MAKER_ONLY" or "NO_TRADE".
        reason: Stable reason enum from models.reasons.
    """

    approved: bool
    ev_gross_lb: float
    fee_est_frac: float
    slippage_est_frac: float
    adverse_buffer_frac: float
    ev_net_lb: float
    execution_mode: str
    reason: str


def _make_veto(reason: str) -> EVGateResult:
    """Create a NO_TRADE result with zeroed fields."""
    return EVGateResult(
        approved=False,
        ev_gross_lb=0.0,
        fee_est_frac=0.0,
        slippage_est_frac=0.0,
        adverse_buffer_frac=0.0,
        ev_net_lb=0.0,
        execution_mode="NO_TRADE",
        reason=reason,
    )


def evaluate(
    p_hat: float,
    p_low: float,
    p_high: float,
    market_price: float,
    side: str,
    size_usd: float,
    fees_pct: float,
    spread_frac: float,
    depth_proxy_usd: float,
    toxicity_multiplier: float,
    maker_threshold: float,
    *,
    base_slip: float = 0.003,
    k_spread: float = 0.5,
    k_depth: float = 0.01,
    base_buffer: float = 0.002,
    eps: float = _DEFAULT_EPS,
    depth_floor_usd: float = _DEFAULT_DEPTH_FLOOR,
) -> EVGateResult:
    """Evaluate EV gate with conservative lower-bound and fractional frictions.

    Args:
        p_hat: Point estimate probability.
        p_low: Calibrated lower bound.
        p_high: Calibrated upper bound.
        market_price: Current market price (0-1).
        side: "BUY_YES" or "BUY_NO".
        size_usd: Order size in USD.
        fees_pct: Fee rate as fraction (0.02 = 2%).
        spread_frac: (best_ask - best_bid) / max(mid, eps) — fraction.
        depth_proxy_usd: USD liquidity on consumed side.
        toxicity_multiplier: 0.0 (clean) to 1.0 (toxic).
        maker_threshold: Minimum ev_net_lb for MAKER_ONLY.
        base_slip: Base slippage fraction (default 0.003).
        k_spread: Spread sensitivity coefficient (default 0.5).
        k_depth: Depth sensitivity coefficient (default 0.01).
        base_buffer: Base adverse selection buffer (default 0.002).
        eps: Epsilon for numerical stability (default 1e-9).
        depth_floor_usd: Minimum depth proxy USD (default 100.0).

    Returns:
        EVGateResult with all fractional cost fields.
    """
    # ── Sanity checks (must be first) ──
    if not (0 <= p_low <= p_hat + eps and p_hat <= p_high + eps and p_high <= 1 + eps):
        return _make_veto(REASON_EV_INVALID_PROB_BOUNDS)
    if p_low < -eps or p_high > 1 + eps:
        return _make_veto(REASON_EV_INVALID_PROB_BOUNDS)

    if not (0 <= market_price <= 1):
        return _make_veto(REASON_EV_INVALID_MARKET_PRICE)

    # ── Clamp depth proxy ──
    depth_proxy_usd = max(depth_proxy_usd, depth_floor_usd)

    # ── Conservative EV ──
    if side == "BUY_YES":
        ev_gross_lb = p_low - market_price
    elif side == "BUY_NO":
        ev_gross_lb = market_price - p_high
    else:
        return _make_veto(f"ev_veto: invalid_side_{side}")

    # ── Deterministic friction model (all fractions) ──
    fee_est_frac = fees_pct
    slippage_est_frac = base_slip + k_spread * spread_frac + k_depth * (size_usd / depth_proxy_usd)
    adverse_buffer_frac = base_buffer * (1.0 + toxicity_multiplier)

    ev_net_lb = ev_gross_lb - fee_est_frac - slippage_est_frac - adverse_buffer_frac

    # ── Approval rule ──
    if ev_net_lb <= maker_threshold:
        return EVGateResult(
            approved=False,
            ev_gross_lb=ev_gross_lb,
            fee_est_frac=fee_est_frac,
            slippage_est_frac=slippage_est_frac,
            adverse_buffer_frac=adverse_buffer_frac,
            ev_net_lb=ev_net_lb,
            execution_mode="NO_TRADE",
            reason=REASON_EV_NET_LB_BELOW_THRESHOLD,
        )

    return EVGateResult(
        approved=True,
        ev_gross_lb=ev_gross_lb,
        fee_est_frac=fee_est_frac,
        slippage_est_frac=slippage_est_frac,
        adverse_buffer_frac=adverse_buffer_frac,
        ev_net_lb=ev_net_lb,
        execution_mode="MAKER_ONLY",
        reason=REASON_EV_MAKER_ONLY,
    )
