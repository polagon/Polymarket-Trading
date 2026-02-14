"""
Tests for DefinitionContract, lint, registry, and definition gate.

Loop 4: Fail-closed definition validation.
All reason assertions use exact enum values from models.reasons.
"""

from __future__ import annotations

import pytest

from definitions.lint import LintResult, lint
from definitions.registry import DefinitionRegistry, build_from_market
from gates.definition_gate import check
from models.definition_contract import DefinitionContract, compute_definition_hash
from models.reasons import (
    LINT_INVALID_CONDITION_LEVEL_TYPE,
    LINT_INVALID_CONDITION_WINDOW,
    LINT_INVALID_CUTOFF,
    LINT_INVALID_ORACLE_ROUNDING,
    LINT_MISSING_CONDITION_LEVEL,
    LINT_MISSING_CONDITION_OP,
    LINT_MISSING_ORACLE_FEED,
    LINT_MISSING_ORACLE_FINALITY,
    LINT_MISSING_ORACLE_ROUNDING,
    LINT_UNKNOWN_CONDITION_KEY,
    LINT_UNKNOWN_ORACLE_KEY,
    REASON_DEFINITION_INCOMPLETE,
    REASON_DEFINITION_OK,
    REASON_NO_DEFINITION,
)

# ── Helpers ──


def _valid_touch_contract(market_id: str = "test_market_1") -> DefinitionContract:
    """Create a valid crypto_threshold touch contract."""
    return DefinitionContract(
        market_id=market_id,
        category="crypto_threshold",
        resolution_type="touch",
        underlying="BTC",
        quote_ccy="USD",
        cutoff_ts_utc="2026-03-01T00:00:00+00:00",
        oracle_source="coingecko_v3",
        oracle_details={
            "feed": "bitcoin",
            "rounding": "floor_int",
            "finality": "1h_vwap",
        },
        condition={
            "op": ">=",
            "level": 85000,
            "window": "any_time",
        },
        venue_rules_version="polymarket_v2",
    )


def _valid_close_contract(market_id: str = "test_market_2") -> DefinitionContract:
    """Create a valid crypto_threshold close contract."""
    return DefinitionContract(
        market_id=market_id,
        category="crypto_threshold",
        resolution_type="close",
        underlying="ETH",
        quote_ccy="USD",
        cutoff_ts_utc="2026-03-01T00:00:00+00:00",
        oracle_source="coingecko_v3",
        oracle_details={
            "feed": "ethereum",
            "rounding": "floor_int",
        },
        condition={
            "op": ">=",
            "level": 3000,
            "window": "at_close",
            "measurement_time": "2026-03-01T00:00:00+00:00",
        },
        venue_rules_version="polymarket_v2",
    )


# ── DefinitionContract Tests ──


class TestDefinitionContract:
    """Tests for the frozen DefinitionContract dataclass."""

    def test_hash_computed_automatically(self) -> None:
        """Hash is computed in __post_init__, not user-supplied."""
        contract = _valid_touch_contract()
        assert contract.definition_hash != ""
        assert len(contract.definition_hash) == 64  # SHA-256 hex

    def test_hash_deterministic(self) -> None:
        """Same inputs always produce same hash."""
        c1 = _valid_touch_contract()
        c2 = _valid_touch_contract()
        assert c1.definition_hash == c2.definition_hash

    def test_hash_changes_with_different_fields(self) -> None:
        """Different inputs produce different hash."""
        c1 = _valid_touch_contract(market_id="market_a")
        c2 = _valid_touch_contract(market_id="market_b")
        assert c1.definition_hash != c2.definition_hash

    def test_reject_user_supplied_hash(self) -> None:
        """Passing non-empty definition_hash raises ValueError."""
        with pytest.raises(ValueError, match="definition_hash_must_be_computed"):
            DefinitionContract(
                market_id="test",
                category="crypto_threshold",
                resolution_type="touch",
                underlying="BTC",
                quote_ccy="USD",
                cutoff_ts_utc="2026-03-01T00:00:00+00:00",
                oracle_source="coingecko_v3",
                oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
                condition={"op": ">=", "level": 85000, "window": "any_time"},
                venue_rules_version="polymarket_v2",
                definition_hash="user_supplied_hash",
            )

    def test_frozen_immutable(self) -> None:
        """Contract fields cannot be changed after creation."""
        contract = _valid_touch_contract()
        with pytest.raises(AttributeError):
            contract.market_id = "changed"  # type: ignore[misc]

    def test_compute_definition_hash_function(self) -> None:
        """compute_definition_hash produces consistent results."""
        fields = {
            "market_id": "test",
            "category": "crypto_threshold",
            "resolution_type": "touch",
            "underlying": "BTC",
            "quote_ccy": "USD",
            "cutoff_ts_utc": "2026-03-01T00:00:00+00:00",
            "oracle_source": "coingecko_v3",
            "oracle_details": {"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            "condition": {"op": ">=", "level": 85000, "window": "any_time"},
            "venue_rules_version": "polymarket_v2",
        }
        h1 = compute_definition_hash(fields)
        h2 = compute_definition_hash(fields)
        assert h1 == h2
        assert len(h1) == 64


# ── Lint Tests ──


class TestLint:
    """Tests for category-aware semantic completeness validation."""

    def test_valid_touch_contract_passes(self) -> None:
        result = lint(_valid_touch_contract())
        assert result.ok is True
        assert result.missing == []

    def test_valid_close_contract_passes(self) -> None:
        result = lint(_valid_close_contract())
        assert result.ok is True
        assert result.missing == []

    def test_missing_condition_op(self) -> None:
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"level": 85000, "window": "any_time"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert LINT_MISSING_CONDITION_OP in result.missing

    def test_missing_condition_level(self) -> None:
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"op": ">=", "window": "any_time"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert LINT_MISSING_CONDITION_LEVEL in result.missing

    def test_float_level_fails(self) -> None:
        """condition.level as float for crypto_threshold → FAIL."""
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"op": ">=", "level": 85000.0, "window": "any_time"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert LINT_INVALID_CONDITION_LEVEL_TYPE in result.missing

    def test_unknown_condition_key_fails(self) -> None:
        """Unknown keys in condition dict → FAIL (not warn)."""
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"op": ">=", "level": 85000, "window": "any_time", "extra_key": True},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        found = [m for m in result.missing if m.startswith(LINT_UNKNOWN_CONDITION_KEY)]
        assert len(found) == 1
        assert "extra_key" in found[0]

    def test_unknown_oracle_key_fails(self) -> None:
        """Unknown keys in oracle_details → FAIL."""
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={
                "feed": "bitcoin",
                "rounding": "floor_int",
                "finality": "1h_vwap",
                "bogus": "field",
            },
            condition={"op": ">=", "level": 85000, "window": "any_time"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        found = [m for m in result.missing if m.startswith(LINT_UNKNOWN_ORACLE_KEY)]
        assert len(found) == 1

    def test_invalid_rounding_fails(self) -> None:
        """Invalid rounding rule → FAIL."""
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={
                "feed": "bitcoin",
                "rounding": "bankers_round",
                "finality": "1h_vwap",
            },
            condition={"op": ">=", "level": 85000, "window": "any_time"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert LINT_INVALID_ORACLE_ROUNDING in result.missing

    def test_missing_oracle_finality_for_touch(self) -> None:
        """Touch contracts require finality in oracle_details."""
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int"},
            condition={"op": ">=", "level": 85000, "window": "any_time"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert LINT_MISSING_ORACLE_FINALITY in result.missing

    def test_close_does_not_require_finality(self) -> None:
        """Close contracts do not require finality."""
        result = lint(_valid_close_contract())
        assert result.ok is True

    def test_invalid_condition_window_touch(self) -> None:
        """Touch with window != any_time → FAIL."""
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"op": ">=", "level": 85000, "window": "at_close"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert LINT_INVALID_CONDITION_WINDOW in result.missing

    def test_invalid_cutoff_fails(self) -> None:
        """Unparseable cutoff_ts_utc → FAIL."""
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="not-a-date",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"op": ">=", "level": 85000, "window": "any_time"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert LINT_INVALID_CUTOFF in result.missing

    def test_close_missing_measurement_time(self) -> None:
        """Close contract without measurement_time → FAIL."""
        contract = DefinitionContract(
            market_id="test",
            category="crypto_threshold",
            resolution_type="close",
            underlying="ETH",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "ethereum", "rounding": "floor_int"},
            condition={"op": ">=", "level": 3000, "window": "at_close"},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert "missing_measurement_time" in result.missing

    def test_unsupported_category(self) -> None:
        contract = DefinitionContract(
            market_id="test",
            category="weather_exceedance",
            resolution_type="touch",
            underlying="TEMP",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="noaa",
            oracle_details={"feed": "temp"},
            condition={"op": ">=", "level": 100},
            venue_rules_version="polymarket_v2",
        )
        result = lint(contract)
        assert result.ok is False
        assert any("unsupported_category" in m for m in result.missing)


# ── Registry Tests ──


class TestRegistry:
    """Tests for DefinitionRegistry."""

    def test_register_valid_contract(self) -> None:
        registry = DefinitionRegistry()
        contract = _valid_touch_contract()
        registry.register(contract)
        assert registry.get("test_market_1") is contract
        assert len(registry) == 1

    def test_register_incomplete_contract_raises(self) -> None:
        """Registering a lint-failing contract raises ValueError."""
        registry = DefinitionRegistry()
        contract = DefinitionContract(
            market_id="bad",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"level": 85000, "window": "any_time"},  # missing op
            venue_rules_version="polymarket_v2",
        )
        with pytest.raises(ValueError, match="lint failed"):
            registry.register(contract)

    def test_get_nonexistent_returns_none(self) -> None:
        registry = DefinitionRegistry()
        assert registry.get("nonexistent") is None

    def test_has_and_contains(self) -> None:
        registry = DefinitionRegistry()
        registry.register(_valid_touch_contract())
        assert registry.has("test_market_1") is True
        assert "test_market_1" in registry
        assert "nonexistent" not in registry


# ── build_from_market Tests ──


class TestBuildFromMarket:
    """Tests for build_from_market helper."""

    def test_valid_build(self) -> None:
        contract = build_from_market(
            market_id="m1",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"op": ">=", "level": 85000, "window": "any_time"},
        )
        assert contract is not None
        assert contract.definition_hash != ""

    def test_incomplete_returns_none(self) -> None:
        """Missing required field returns None (no partial objects)."""
        contract = build_from_market(
            market_id="m1",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin"},  # missing rounding, finality
            condition={"op": ">=", "level": 85000, "window": "any_time"},
        )
        assert contract is None


# ── Definition Gate Tests ──


class TestDefinitionGate:
    """Tests for gates/definition_gate.py."""

    def test_no_contract_returns_false(self) -> None:
        registry = DefinitionRegistry()
        ok, reason = check("nonexistent", registry)
        assert ok is False
        assert reason == REASON_NO_DEFINITION

    def test_valid_contract_returns_true(self) -> None:
        registry = DefinitionRegistry()
        registry.register(_valid_touch_contract())
        ok, reason = check("test_market_1", registry)
        assert ok is True
        assert reason == REASON_DEFINITION_OK

    def test_incomplete_contract_returns_reason(self) -> None:
        """Contract with lint failures returns stable reason string."""
        registry = DefinitionRegistry()
        # Manually insert a contract that would fail lint (bypass register validation)
        bad_contract = DefinitionContract(
            market_id="bad",
            category="crypto_threshold",
            resolution_type="touch",
            underlying="BTC",
            quote_ccy="USD",
            cutoff_ts_utc="2026-03-01T00:00:00+00:00",
            oracle_source="coingecko_v3",
            oracle_details={"feed": "bitcoin", "rounding": "floor_int", "finality": "1h_vwap"},
            condition={"level": 85000, "window": "any_time"},  # missing op
            venue_rules_version="polymarket_v2",
        )
        # Bypass register to test gate directly
        registry._contracts["bad"] = bad_contract
        ok, reason = check("bad", registry)
        assert ok is False
        assert reason.startswith(REASON_DEFINITION_INCOMPLETE)
        assert LINT_MISSING_CONDITION_OP in reason
