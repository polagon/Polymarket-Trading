"""Tests for definitions/loader.py — file-based contract loading."""

import json
from pathlib import Path

import pytest

from definitions.loader import dump_definitions, load_definitions
from definitions.registry import DefinitionRegistry


def _valid_contract(market_id: str = "0xabc123") -> dict:
    """Minimal valid crypto_threshold touch contract."""
    return {
        "market_id": market_id,
        "category": "crypto_threshold",
        "resolution_type": "touch",
        "underlying": "BTC",
        "quote_ccy": "USD",
        "cutoff_ts_utc": "2030-12-31T23:59:59+00:00",
        "oracle_source": "coingecko_v3",
        "oracle_details": {"feed": "bitcoin/usd", "rounding": "floor_int", "finality": "1h_close"},
        "condition": {"op": ">=", "level": 100000, "window": "any_time"},
        "venue_rules_version": "polymarket_v2",
    }


class TestLoadDefinitions:
    """Core loader behavior."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps([_valid_contract()]))
        reg = DefinitionRegistry()
        n = load_definitions(f, reg)
        assert n == 1
        assert len(reg) == 1
        assert reg.has("0xabc123")

    def test_load_multiple_contracts(self, tmp_path: Path) -> None:
        contracts = [_valid_contract("0xaaa"), _valid_contract("0xbbb")]
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps(contracts))
        reg = DefinitionRegistry()
        n = load_definitions(f, reg)
        assert n == 2
        assert reg.has("0xaaa")
        assert reg.has("0xbbb")

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        f = tmp_path / "does_not_exist.json"
        reg = DefinitionRegistry()
        n = load_definitions(f, reg)
        assert n == 0
        assert len(reg) == 0

    def test_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "contracts.json"
        f.write_text("[]")
        reg = DefinitionRegistry()
        n = load_definitions(f, reg)
        assert n == 0

    def test_hash_is_computed_not_user_supplied(self, tmp_path: Path) -> None:
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps([_valid_contract()]))
        reg = DefinitionRegistry()
        load_definitions(f, reg)
        c = reg.get("0xabc123")
        assert c is not None
        assert len(c.definition_hash) == 64  # SHA-256 hex


class TestLoadDefinitionsErrors:
    """Error paths — invalid files hard-fail."""

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "contracts.json"
        f.write_text("{not valid json")
        reg = DefinitionRegistry()
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_definitions(f, reg)

    def test_not_a_list_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "contracts.json"
        f.write_text('{"market_id": "x"}')
        reg = DefinitionRegistry()
        with pytest.raises(ValueError, match="Expected JSON list"):
            load_definitions(f, reg)

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        bad = _valid_contract()
        bad["extra_field"] = "oops"
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps([bad]))
        reg = DefinitionRegistry()
        with pytest.raises(ValueError, match="unknown keys"):
            load_definitions(f, reg)

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        bad = _valid_contract()
        del bad["underlying"]
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps([bad]))
        reg = DefinitionRegistry()
        with pytest.raises(ValueError, match="missing keys"):
            load_definitions(f, reg)

    def test_lint_failure_raises(self, tmp_path: Path) -> None:
        bad = _valid_contract()
        bad["condition"]["level"] = 1.5  # float level → lint FAIL
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps([bad]))
        reg = DefinitionRegistry()
        with pytest.raises(ValueError, match="invalid"):
            load_definitions(f, reg)

    def test_entry_not_dict_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps(["not a dict"]))
        reg = DefinitionRegistry()
        with pytest.raises(ValueError, match="expected dict"):
            load_definitions(f, reg)


class TestDumpDefinitions:
    """dump_definitions formatting."""

    def test_empty_registry(self) -> None:
        reg = DefinitionRegistry()
        result = dump_definitions(reg)
        assert "No definitions" in result

    def test_with_contracts(self, tmp_path: Path) -> None:
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps([_valid_contract()]))
        reg = DefinitionRegistry()
        load_definitions(f, reg)
        result = dump_definitions(reg)
        assert "1 definition" in result
        assert "BTC/USD" in result
        assert "touch" in result


class TestRealContractsFile:
    """Validate the actual contracts file ships valid."""

    def test_real_contracts_file_loads(self) -> None:
        path = Path(__file__).resolve().parent.parent / "definitions" / "contracts.crypto_threshold.json"
        if not path.exists():
            pytest.skip("No contracts file shipped")
        reg = DefinitionRegistry()
        n = load_definitions(path, reg)
        # 0 contracts is valid (generator may output 0 passers if all fail filters)
        assert n >= 0, "Should load without error"
        for c in reg.all_contracts():
            assert len(c.definition_hash) == 64
            assert c.category == "crypto_threshold"
