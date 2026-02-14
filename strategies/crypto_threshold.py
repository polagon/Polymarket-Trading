"""
CryptoThresholdStrategy — primary category strategy for Loop 4.

Only fires on markets mapped to "crypto_threshold" with a valid DefinitionContract.
Emits a decision artifact for EVERY evaluation (SKIPs included).
MAKER_ONLY execution mode only.

Gate chain: Definition → EV → Risk → PLACE_ORDER.
"""

from __future__ import annotations

import logging
from typing import Optional

from definitions.registry import DefinitionRegistry
from gates import definition_gate
from gates.ev_gate import evaluate as ev_evaluate
from models.reasons import (
    REASON_DEFINITION_OK,
    REASON_EV_MAKER_ONLY,
    REASON_RISK_OK,
)
from risk.risk_engine import RiskEngine
from scanner.strategies.base import BaseStrategy, StrategyContext, TradeSignal
from signals.flow_toxicity import FlowToxicityAnalyzer
from telemetry.trade_telemetry import TradeTelemetry

logger = logging.getLogger(__name__)

# Default EV gate config
_DEFAULT_EV_CONFIG = {
    "fees_pct": 0.02,
    "base_slip": 0.003,
    "k_spread": 0.5,
    "k_depth": 0.01,
    "base_buffer": 0.002,
    "maker_threshold": 0.005,
    "eps": 1e-9,
    "depth_floor_usd": 100.0,
}

_EPS = 1e-9


class CryptoThresholdStrategy(BaseStrategy):
    """Crypto threshold strategy — lognormal price model on binary strike markets.

    Only activates on markets explicitly mapped to crypto_threshold AND present
    in the DefinitionRegistry.

    Args:
        registry: DefinitionRegistry for contract lookups.
        ev_gate_config: EV gate parameters (fees_pct, thresholds, etc).
        risk_engine: RiskEngine for halt/cap checks.
        toxicity_analyzer: FlowToxicityAnalyzer for microstructure signals.
        telemetry: TradeTelemetry for decision artifacts.
        cycle_id: Current cycle ID (updated externally).
    """

    def __init__(
        self,
        registry: DefinitionRegistry,
        ev_gate_config: Optional[dict] = None,
        risk_engine: Optional[RiskEngine] = None,
        toxicity_analyzer: Optional[FlowToxicityAnalyzer] = None,
        telemetry: Optional[TradeTelemetry] = None,
    ) -> None:
        super().__init__(name="crypto_threshold")
        self._registry = registry
        self._ev_config = {**_DEFAULT_EV_CONFIG, **(ev_gate_config or {})}
        self._risk_engine = risk_engine
        self._toxicity = toxicity_analyzer
        self._telemetry = telemetry
        self.cycle_id = 0

    def evaluate(
        self,
        market: object,
        context: StrategyContext,
    ) -> Optional[TradeSignal]:
        """Evaluate a market through the full gate chain.

        Gate chain: Definition → EV → Risk → PLACE_ORDER.
        Emits decision artifacts for every evaluation (SKIPs included).

        Returns TradeSignal if all gates pass, None otherwise.
        """
        from scanner.market_fetcher import Market

        if not isinstance(market, Market):
            return None

        market_id = market.condition_id
        gates_result: dict = {}
        action = "SKIP"
        order_params = None

        # Default inputs/frictions for artifact
        best_bid = getattr(market, "yes_bid", 0.0) or 0.0
        best_ask = getattr(market, "yes_ask", 0.0) or 0.0
        mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0.0
        spread_frac = (best_ask - best_bid) / max(mid, _EPS) if mid > 0 else 0.0
        depth_proxy_usd = getattr(market, "liquidity", 0.0) or 5000.0

        inputs = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread_frac": spread_frac,
            "depth_proxy_usd": depth_proxy_usd,
        }

        # Default estimator values
        p_hat = 0.0
        p_low = 0.0
        p_high = 0.0
        estimator_info = {
            "p_hat": 0.0,
            "p_low": 0.0,
            "p_high": 0.0,
            "calibration_bucket": "crypto_threshold",
            "estimator_version": "lognormal_v1",
        }
        frictions = {
            "fee_est_frac": 0.0,
            "slippage_est_frac": 0.0,
            "adverse_buffer_frac": 0.0,
            "toxicity_score": 0.0,
            "spread_regime": "NORMAL",
        }
        ev_net_lb = 0.0

        # ── Gate 1: Definition ──
        def_ok, def_reason = definition_gate.check(market_id, self._registry)
        gates_result["definition"] = {"ok": def_ok, "reason": def_reason}

        definition_present = def_ok
        definition_hash = None
        if def_ok:
            contract = self._registry.get(market_id)
            if contract:
                definition_hash = contract.definition_hash

        if not def_ok:
            self._emit_decision(
                market_id,
                definition_present,
                definition_hash,
                inputs,
                estimator_info,
                frictions,
                ev_net_lb,
                gates_result,
                action,
                order_params,
            )
            return None

        # ── Extract estimator values ──
        # Use price data from context if available
        price_data = context.price_data if context.price_data else {}

        # For crypto threshold: use lognormal model estimate from context
        # or fallback to market price as p_hat
        market_price = best_ask  # Use ask for BUY_YES evaluation
        p_hat = price_data.get("p_hat", market_price)
        p_low = price_data.get("p_low", p_hat * 0.9)
        p_high = price_data.get("p_high", min(p_hat * 1.1, 1.0))

        estimator_info = {
            "p_hat": p_hat,
            "p_low": p_low,
            "p_high": p_high,
            "calibration_bucket": "crypto_threshold",
            "estimator_version": "lognormal_v1",
        }

        # ── Get toxicity state ──
        toxicity_multiplier = 0.0
        spread_regime = "NORMAL"
        toxicity_score = 0.0

        if self._toxicity:
            state = self._toxicity.get_state(market_id)
            toxicity_multiplier = self._toxicity.get_toxicity_multiplier(market_id)
            spread_regime = state.spread_regime
            toxicity_score = state.toxicity_score

        # Determine side
        side = "BUY_YES"

        # ── Gate 2: EV ──
        ev_result = ev_evaluate(
            p_hat=p_hat,
            p_low=p_low,
            p_high=p_high,
            market_price=market_price,
            side=side,
            size_usd=50.0,  # default evaluation size
            fees_pct=self._ev_config["fees_pct"],
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            toxicity_multiplier=toxicity_multiplier,
            maker_threshold=self._ev_config["maker_threshold"],
            base_slip=self._ev_config["base_slip"],
            k_spread=self._ev_config["k_spread"],
            k_depth=self._ev_config["k_depth"],
            base_buffer=self._ev_config["base_buffer"],
            eps=self._ev_config["eps"],
            depth_floor_usd=self._ev_config["depth_floor_usd"],
        )

        gates_result["ev"] = {"ok": ev_result.approved, "reason": ev_result.reason}
        ev_net_lb = ev_result.ev_net_lb

        frictions = {
            "fee_est_frac": ev_result.fee_est_frac,
            "slippage_est_frac": ev_result.slippage_est_frac,
            "adverse_buffer_frac": ev_result.adverse_buffer_frac,
            "toxicity_score": toxicity_score,
            "spread_regime": spread_regime,
        }

        if not ev_result.approved:
            self._emit_decision(
                market_id,
                definition_present,
                definition_hash,
                inputs,
                estimator_info,
                frictions,
                ev_net_lb,
                gates_result,
                action,
                order_params,
            )
            return None

        # ── Gate 3: Risk ──
        risk_ok = True
        risk_reason = REASON_RISK_OK
        if self._risk_engine:
            risk_ok, risk_reason = self._risk_engine.can_trade(
                market=market,
                size_usd=50.0,
                category="crypto_threshold",
                toxicity_flag=(toxicity_multiplier > 0.7),
                spread_regime=spread_regime,
            )

        gates_result["risk"] = {"ok": risk_ok, "reason": risk_reason}

        if not risk_ok:
            self._emit_decision(
                market_id,
                definition_present,
                definition_hash,
                inputs,
                estimator_info,
                frictions,
                ev_net_lb,
                gates_result,
                action,
                order_params,
            )
            return None

        # ── All gates pass → PLACE_ORDER ──
        action = "PLACE_ORDER"
        order_params = {
            "side": side,
            "price": market_price,
            "size_usd": 50.0,
            "mode": "MAKER_ONLY",
            "ttl_seconds": 120,
        }

        self._emit_decision(
            market_id,
            definition_present,
            definition_hash,
            inputs,
            estimator_info,
            frictions,
            ev_net_lb,
            gates_result,
            action,
            order_params,
        )

        # Compute Kelly sizing
        edge = ev_result.ev_gross_lb
        kelly_pct = max(edge / max(1.0 - market_price, _EPS), 0.0) * 0.25  # quarter Kelly

        return TradeSignal(
            condition_id=market_id,
            question=getattr(market, "question", ""),
            direction=side.replace("_", " "),
            confidence=min(p_hat, 0.95),
            target_price=market_price,
            kelly_pct=kelly_pct,
            edge=edge,
            ev_after_costs=ev_net_lb,
            source="crypto_threshold",
            reasoning=f"EV_net_lb={ev_net_lb:.4f} maker_only spread_regime={spread_regime}",
        )

    def _emit_decision(
        self,
        market_id: str,
        definition_present: bool,
        definition_hash: Optional[str],
        inputs: dict,
        estimator: dict,
        frictions: dict,
        ev_net_lb: float,
        gates: dict,
        action: str,
        order: Optional[dict],
    ) -> None:
        """Emit decision artifact if telemetry is available."""
        if self._telemetry:
            try:
                self._telemetry.emit_decision(
                    cycle_id=self.cycle_id,
                    market_id=market_id,
                    definition_present=definition_present,
                    definition_hash=definition_hash,
                    inputs=inputs,
                    estimator=estimator,
                    frictions=frictions,
                    ev_net_lb=ev_net_lb,
                    gates=gates,
                    action=action,
                    order=order,
                )
            except Exception as e:
                logger.error(f"Failed to emit decision artifact: {e}")
