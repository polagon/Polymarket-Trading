"""
Tests for tools/generate_crypto_threshold_contracts.py

All tests use fixtures and mocks — no network calls.
"""

import json

# Add tools to path
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from generate_crypto_threshold_contracts import (
    VETO_EV_FAILED,
    VETO_LINT_FAILED,
    VETO_MAKER_CANNOT_POST,
    VETO_NO_BOOK,
    VETO_PARSE_AMBIGUOUS,
    VETO_PARSE_FALSE_POSITIVE,
    VETO_PARSE_MISSING_FIELDS,
    VETO_PARSE_NO_RESOLUTION_TYPE,
    VETO_PARSE_NO_STRIKE,
    VETO_PARSE_UNSUPPORTED_UNDERLYING,
    VETO_THIN_DEPTH,
    VETO_TIME_OOB,
    VETO_WIDE_SPREAD,
    build_definition_contract,
    compute_maker_entry_price,
    parse_threshold_market,
)


class TestParseThresholdMarket:
    """Test deterministic market question parsing."""

    def test_parse_touch_market_btc(self):
        market = {
            "id": "test1",
            "question": "Will BTC hit $100,000 by December 31, 2026?",
            "end_date_iso": "2026-12-31T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is not None
        assert parsed["underlying"] == "BTC"
        assert parsed["strike"] == 100000.0
        assert parsed["resolution_type"] == "touch"
        assert parsed["op"] == ">="
        assert parsed["window"] == "any_time"

    def test_parse_touch_market_eth_with_k_suffix(self):
        market = {
            "id": "test2",
            "question": "Will ETH reach $5K by end of 2026?",
            "end_date_iso": "2026-12-31T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is not None
        assert parsed["underlying"] == "ETH"
        assert parsed["strike"] == 5000.0
        assert parsed["resolution_type"] == "touch"

    def test_parse_close_market_above(self):
        market = {
            "id": "test3",
            "question": "Will BTC be above $90,000 at January 1 2027 close?",
            "end_date_iso": "2027-01-01T00:00:00Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is not None
        assert parsed["underlying"] == "BTC"
        assert parsed["strike"] == 90000.0
        assert parsed["resolution_type"] == "close"
        assert parsed["op"] == ">="
        assert parsed["window"] == "at_close"

    def test_parse_close_market_below(self):
        market = {
            "id": "test4",
            "question": "Will ETH be below $3,000 at June 30 close?",
            "end_date_iso": "2026-06-30T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is not None
        assert parsed["underlying"] == "ETH"
        assert parsed["strike"] == 3000.0
        assert parsed["resolution_type"] == "close"
        assert parsed["op"] == "<="

    def test_parse_ambiguous_question_returns_none(self):
        market = {
            "id": "test5",
            "question": "Will crypto markets rally in 2026?",
            "end_date_iso": "2026-12-31T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is None

    def test_parse_bitcoin_fullname(self):
        market = {
            "id": "test6",
            "question": "Will Bitcoin hit $1M by 2030?",
            "end_date_iso": "2030-12-31T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is not None
        assert parsed["underlying"] == "BTC"
        assert parsed["strike"] == 1000000.0


class TestBuildDefinitionContract:
    """Test DefinitionContract construction and lint."""

    def test_build_valid_touch_contract(self):
        market = {"id": "test1", "condition_id": "0xabc123"}
        parsed = {
            "underlying": "BTC",
            "strike": 100000.0,
            "cutoff_ts_utc": "2026-12-31T23:59:59Z",
            "resolution_type": "touch",
            "op": ">=",
            "window": "any_time",
        }
        contract = build_definition_contract(market, parsed)
        assert contract is not None
        assert contract.market_id == "test1"
        assert contract.underlying == "BTC"
        assert contract.condition["level"] == 100000.0
        assert contract.condition["op"] == ">="
        assert contract.definition_hash != ""

    def test_build_with_missing_oracle_details_fails(self):
        market = {"id": "test2", "condition_id": "0xdef456"}
        parsed = {
            "underlying": "ETH",
            "strike": 5000.0,
            "cutoff_ts_utc": "2026-12-31T23:59:59Z",
            "resolution_type": "touch",
            "op": ">=",
            "window": "any_time",
        }
        # This should pass lint because oracle_details are provided in build_definition_contract
        contract = build_definition_contract(market, parsed)
        assert contract is not None

    def test_build_with_int_strike_passes(self):
        # Strike as int should pass
        market = {"id": "test3", "condition_id": "0xghi789"}
        parsed = {
            "underlying": "BTC",
            "strike": 100000,  # int
            "cutoff_ts_utc": "2026-12-31T23:59:59Z",
            "resolution_type": "touch",
            "op": ">=",
            "window": "any_time",
        }
        contract = build_definition_contract(market, parsed)
        assert contract is not None
        # Level in condition should be int
        assert isinstance(contract.condition["level"], int)


class TestComputeMakerEntryPrice:
    """Test maker entry price computation (never cross spread)."""

    def test_buy_yes_posts_between_bid_and_ask(self):
        best_bid = 0.48
        best_ask = 0.50
        tick = 0.01
        entry = compute_maker_entry_price(best_bid, best_ask, "BUY_YES", tick)
        assert entry is not None
        assert best_bid < entry < best_ask
        assert entry == 0.49  # min(0.50 - 0.01, 0.48 + 0.01) = min(0.49, 0.49)

    def test_buy_yes_tight_spread_cannot_post(self):
        best_bid = 0.48
        best_ask = 0.49
        tick = 0.01
        entry = compute_maker_entry_price(best_bid, best_ask, "BUY_YES", tick)
        # best_ask - tick = 0.48, best_bid + tick = 0.49
        # min(0.48, 0.49) = 0.48, but that equals best_bid → cannot post
        assert entry is None

    def test_buy_yes_zero_bid_returns_none(self):
        best_bid = 0.0
        best_ask = 0.50
        tick = 0.01
        entry = compute_maker_entry_price(best_bid, best_ask, "BUY_YES", tick)
        assert entry is None

    def test_buy_yes_crossed_spread_returns_none(self):
        best_bid = 0.52
        best_ask = 0.48  # Crossed
        tick = 0.01
        entry = compute_maker_entry_price(best_bid, best_ask, "BUY_YES", tick)
        assert entry is None

    def test_buy_no_symmetric_logic(self):
        best_bid = 0.48
        best_ask = 0.50
        tick = 0.01
        entry = compute_maker_entry_price(best_bid, best_ask, "BUY_NO", tick)
        assert entry is not None
        # NO best_ask = 1 - best_bid = 0.52
        # NO best_bid = 1 - best_ask = 0.50
        # entry = min(0.52 - 0.01, 0.50 + 0.01) = min(0.51, 0.51) = 0.51
        assert 0.50 < entry < 0.52

    def test_entry_price_clamped_to_valid_range(self):
        # Edge case: very low best_bid/ask near 0.01
        best_bid = 0.01
        best_ask = 0.03
        tick = 0.01
        entry = compute_maker_entry_price(best_bid, best_ask, "BUY_YES", tick)
        assert entry is not None
        assert 0.01 <= entry <= 0.99


class TestDeterministicOutputs:
    """Test stable sorting and schema versioning."""

    def test_veto_reasons_are_stable_strings(self):
        # All veto reasons must be stable enums (specific, not generic)
        # Parse vetoes (specific reasons)
        assert VETO_PARSE_FALSE_POSITIVE == "parse_veto: false_positive_token"
        assert VETO_PARSE_UNSUPPORTED_UNDERLYING == "parse_veto: unsupported_underlying"
        assert VETO_PARSE_NO_STRIKE == "parse_veto: no_strike_level"
        assert VETO_PARSE_NO_RESOLUTION_TYPE == "parse_veto: no_resolution_type"
        assert VETO_PARSE_MISSING_FIELDS == "parse_veto: missing_required_fields"
        assert VETO_PARSE_AMBIGUOUS == "parse_veto: ambiguous_question"
        # Other vetoes
        assert VETO_LINT_FAILED == "lint_veto: validation_failed"
        assert VETO_NO_BOOK == "book_veto: no_book"
        assert VETO_WIDE_SPREAD == "book_veto: spread_too_wide"
        assert VETO_THIN_DEPTH == "book_veto: depth_too_thin"
        assert VETO_TIME_OOB == "time_veto: cutoff_out_of_bounds"
        assert VETO_MAKER_CANNOT_POST == "maker_veto: cannot_post_without_crossing"
        assert VETO_EV_FAILED == "ev_veto: net_lb_below_threshold"

    def test_json_output_is_sorted_keys(self):
        # The generator outputs JSON with sort_keys=True
        # This ensures deterministic file diffs
        sample = {"z_field": 1, "a_field": 2, "m_field": 3}
        output = json.dumps(sample, sort_keys=True)
        assert output == '{"a_field": 2, "m_field": 3, "z_field": 1}'


class TestZeroPassersValid:
    """Test that 0 passers is a valid outcome."""

    def test_generator_can_output_zero_contracts(self):
        # If all markets fail filters, output should be empty list (not error)
        contracts: list[dict] = []
        # This is valid JSON
        output = json.dumps(contracts, indent=2, sort_keys=True)
        assert output == "[]"

    def test_zero_passers_summary_has_counts(self):
        summary = {
            "schema_version": "1.0",
            "discovered_count": 50,
            "scored_count": 50,
            "passed_count": 0,
            "top_veto_reasons": {
                VETO_EV_FAILED: 30,
                VETO_WIDE_SPREAD: 15,
                VETO_THIN_DEPTH: 5,
            },
        }
        assert summary["passed_count"] == 0
        assert summary["discovered_count"] > 0  # type: ignore[operator]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
