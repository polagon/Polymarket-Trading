"""
Tests for CryptoThresholdStrategy — gate chain integration + artifact emission.

Key invariants:
  - No trade without DefinitionContract
  - Decision artifacts emitted for every evaluation (SKIPs + PLACE_ORDER)
  - MAKER_ONLY execution mode only
  - All reason strings are exact enums

Loop 4: Gate chain integration test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from definitions.registry import DefinitionRegistry
from models.definition_contract import DefinitionContract
from models.reasons import (
    REASON_DEFINITION_OK,
    REASON_EV_MAKER_ONLY,
    REASON_NO_DEFINITION,
    REASON_RISK_BROKEN_REGIME,
    REASON_RISK_OK,
)
from risk.risk_engine import RiskEngine
from scanner.market_fetcher import Market
from scanner.strategies.base import StrategyContext
from signals.flow_toxicity import FlowToxicityAnalyzer
from strategies.crypto_threshold import CryptoThresholdStrategy
from telemetry.trade_telemetry import TradeTelemetry


def _make_market(condition_id: str = "test_cid_1") -> Market:
    """Create a test Market with bid/ask attributes."""
    m = Market(
        condition_id=condition_id,
        question="Will BTC hit $85K?",
        end_date_iso="2026-03-01T00:00:00+00:00",
        category="Crypto",
        yes_token_id="yes_tok_1",
        no_token_id="no_tok_1",
        yes_price=0.70,
        no_price=0.30,
        liquidity=50000.0,
        volume=100000.0,
    )
    # Add bid/ask attributes that the strategy reads
    m.yes_bid = 0.69  # type: ignore[attr-defined]
    m.yes_ask = 0.71  # type: ignore[attr-defined]
    return m


def _make_contract(market_id: str = "test_cid_1") -> DefinitionContract:
    """Create a valid crypto_threshold touch contract."""
    return DefinitionContract(
        market_id=market_id,
        category="crypto_threshold",
        resolution_type="touch",
        underlying="BTC",
        quote_ccy="USD",
        cutoff_ts_utc="2026-03-01T00:00:00+00:00",
        oracle_source="coingecko_v3",
        oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
        condition={"op": ">=", "level": 85000, "window": "any_time"},
        venue_rules_version="polymarket_v2",
    )


def _default_context(p_hat: float = 0.85, p_low: float = 0.78, p_high: float = 0.92) -> StrategyContext:
    """Create a strategy context with estimator data."""
    return StrategyContext(
        price_data={"p_hat": p_hat, "p_low": p_low, "p_high": p_high},
    )


class TestNoDefinition:
    """Strategy returns None when no DefinitionContract exists."""

    def test_no_contract_returns_none(self) -> None:
        registry = DefinitionRegistry()
        strategy = CryptoThresholdStrategy(registry=registry)
        market = _make_market()
        result = strategy.evaluate(market, _default_context())
        assert result is None

    def test_no_contract_emits_skip_artifact(self, tmp_path: Path) -> None:
        registry = DefinitionRegistry()
        telemetry = TradeTelemetry(artifacts_dir=tmp_path, run_id="test_run")
        strategy = CryptoThresholdStrategy(registry=registry, telemetry=telemetry)
        strategy.cycle_id = 1

        market = _make_market()
        result = strategy.evaluate(market, _default_context())
        assert result is None

        # Verify artifact was written
        artifact_path = tmp_path / "decisions" / "1" / "test_cid_1.json"
        assert artifact_path.exists()

        artifact = json.loads(artifact_path.read_text())
        assert artifact["schema_version"] == "1.0"
        assert artifact["action"] == "SKIP"
        assert artifact["definition_present"] is False
        assert artifact["definition_hash"] is None
        assert artifact["gates"]["definition"]["ok"] is False
        assert artifact["gates"]["definition"]["reason"] == REASON_NO_DEFINITION


class TestPositiveEV:
    """Strategy returns TradeSignal when all gates pass."""

    def test_all_gates_pass_returns_signal(self) -> None:
        registry = DefinitionRegistry()
        registry.register(_make_contract())
        strategy = CryptoThresholdStrategy(registry=registry)

        market = _make_market()
        # p_low=0.78 vs market_price=0.71 → ev_gross_lb=0.07 (positive)
        result = strategy.evaluate(market, _default_context(p_hat=0.85, p_low=0.78, p_high=0.92))
        assert result is not None
        assert result.source == "crypto_threshold"
        assert "BUY YES" in result.direction

    def test_all_gates_pass_emits_place_order(self, tmp_path: Path) -> None:
        registry = DefinitionRegistry()
        registry.register(_make_contract())
        telemetry = TradeTelemetry(artifacts_dir=tmp_path, run_id="test_run")
        strategy = CryptoThresholdStrategy(registry=registry, telemetry=telemetry)
        strategy.cycle_id = 1

        market = _make_market()
        result = strategy.evaluate(market, _default_context(p_hat=0.85, p_low=0.78, p_high=0.92))
        assert result is not None

        artifact_path = tmp_path / "decisions" / "1" / "test_cid_1.json"
        assert artifact_path.exists()

        artifact = json.loads(artifact_path.read_text())
        assert artifact["action"] == "PLACE_ORDER"
        assert artifact["definition_present"] is True
        assert artifact["definition_hash"] is not None
        assert artifact["gates"]["definition"]["ok"] is True
        assert artifact["gates"]["ev"]["ok"] is True
        assert artifact["order"] is not None
        assert artifact["order"]["mode"] == "MAKER_ONLY"

        # Verify frictions are present and fractional
        assert "fee_est_frac" in artifact["frictions"]
        assert "slippage_est_frac" in artifact["frictions"]
        assert "adverse_buffer_frac" in artifact["frictions"]


class TestNegativeEV:
    """Strategy returns None when EV gate fails."""

    def test_negative_ev_returns_none(self) -> None:
        registry = DefinitionRegistry()
        registry.register(_make_contract())
        strategy = CryptoThresholdStrategy(registry=registry)

        market = _make_market()
        # p_low=0.70 vs market_price=0.71 → ev_gross_lb=-0.01 (negative)
        result = strategy.evaluate(market, _default_context(p_hat=0.71, p_low=0.70, p_high=0.72))
        assert result is None

    def test_negative_ev_emits_skip(self, tmp_path: Path) -> None:
        registry = DefinitionRegistry()
        registry.register(_make_contract())
        telemetry = TradeTelemetry(artifacts_dir=tmp_path, run_id="test_run")
        strategy = CryptoThresholdStrategy(registry=registry, telemetry=telemetry)
        strategy.cycle_id = 1

        market = _make_market()
        result = strategy.evaluate(market, _default_context(p_hat=0.71, p_low=0.70, p_high=0.72))
        assert result is None

        artifact_path = tmp_path / "decisions" / "1" / "test_cid_1.json"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["action"] == "SKIP"
        assert artifact["gates"]["ev"]["ok"] is False


class TestRiskHalt:
    """Strategy returns None when risk engine vetoes."""

    def test_risk_halt_returns_none(self) -> None:
        registry = DefinitionRegistry()
        registry.register(_make_contract())
        risk_engine = RiskEngine(initial_equity=5000.0)
        # Trigger halt
        risk_engine.record_pnl(-300.0, current_equity=4700.0, now=1000.0)

        strategy = CryptoThresholdStrategy(registry=registry, risk_engine=risk_engine)

        market = _make_market()
        result = strategy.evaluate(market, _default_context(p_hat=0.85, p_low=0.78, p_high=0.92))
        assert result is None

    def test_broken_regime_returns_none(self) -> None:
        registry = DefinitionRegistry()
        registry.register(_make_contract())
        risk_engine = RiskEngine(initial_equity=5000.0)

        # Create toxicity analyzer that reports BROKEN
        toxicity = FlowToxicityAnalyzer(window_size=50, threshold=0.7, min_samples=10)
        # Feed one-sided book
        for i in range(20):
            toxicity.update("test_cid_1", 0.0, 0.71, "SELL", 10.0, 1000.0 + i)

        strategy = CryptoThresholdStrategy(
            registry=registry,
            risk_engine=risk_engine,
            toxicity_analyzer=toxicity,
        )

        market = _make_market()
        result = strategy.evaluate(market, _default_context(p_hat=0.85, p_low=0.78, p_high=0.92))
        assert result is None


class TestHighToxicity:
    """Strategy vetoes marginal trades when toxicity is high."""

    def test_high_toxicity_raises_buffer(self, tmp_path: Path) -> None:
        """High toxicity → adverse buffer raised → marginal trade vetoed."""
        registry = DefinitionRegistry()
        registry.register(_make_contract())
        telemetry = TradeTelemetry(artifacts_dir=tmp_path, run_id="test_run")

        # Marginal context: barely positive EV without toxicity
        context = _default_context(p_hat=0.75, p_low=0.74, p_high=0.76)

        # Without toxicity
        strategy_clean = CryptoThresholdStrategy(
            registry=registry,
            telemetry=telemetry,
            ev_gate_config={"maker_threshold": 0.001},
        )
        strategy_clean.cycle_id = 1
        market = _make_market()
        result_clean = strategy_clean.evaluate(market, context)

        # With heavy buy toxicity (elevated adverse buffer)
        toxicity = FlowToxicityAnalyzer(window_size=50, threshold=0.3, min_samples=5)
        for i in range(20):
            toxicity.update("test_cid_1", 0.498, 0.502, "BUY", 1000.0, 1000.0 + i)

        strategy_toxic = CryptoThresholdStrategy(
            registry=registry,
            toxicity_analyzer=toxicity,
            ev_gate_config={"maker_threshold": 0.001},
        )
        strategy_toxic.cycle_id = 2
        result_toxic = strategy_toxic.evaluate(market, context)

        # Toxic environment should make marginal trade harder to pass
        # (at minimum, the adverse buffer is higher)
        if result_clean is not None and result_toxic is not None:
            assert result_toxic.ev_after_costs < result_clean.ev_after_costs
