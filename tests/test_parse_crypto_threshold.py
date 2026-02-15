"""
Tests for tools/parse_crypto_threshold.py

Loop 5.1: Strict word-boundary parsing with Ethena/WBTC/stETH rejection.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from parse_crypto_threshold import (
    detect_resolution_type,
    detect_underlying,
    extract_strike,
    parse_threshold_market,
)


class TestDetectUnderlying:
    """Test strict word-boundary detection for BTC/ETH/SOL."""

    def test_btc_word_boundary_match(self):
        assert detect_underlying("Will BTC hit $100K?") == "BTC"

    def test_bitcoin_full_word_match(self):
        assert detect_underlying("Will Bitcoin reach $1M?") == "BTC"

    def test_eth_word_boundary_match(self):
        assert detect_underlying("Will ETH hit $10K?") == "ETH"

    def test_ethereum_full_word_match(self):
        assert detect_underlying("Will Ethereum reach $5K?") == "ETH"

    def test_sol_rejected_out_of_scope(self):
        """SOL is out of scope for Loop 5.x (BTC/ETH only)."""
        assert detect_underlying("Will SOL reach $500?") is None

    def test_solana_rejected_out_of_scope(self):
        """Solana is out of scope for Loop 5.x (BTC/ETH only)."""
        assert detect_underlying("Will Solana hit $1000?") is None

    def test_wbtc_rejected_as_false_positive(self):
        """WBTC should be rejected (not BTC)."""
        assert detect_underlying("Will WBTC depeg from BTC?") is None

    def test_ethena_rejected_as_false_positive(self):
        """Ethena should be rejected (not ETH)."""
        assert detect_underlying("Will Ethena TVL exceed $10B?") is None

    def test_steth_rejected_as_false_positive(self):
        """stETH should be rejected (not ETH)."""
        assert detect_underlying("Will stETH maintain peg to ETH?") is None

    def test_reth_rejected_as_false_positive(self):
        """rETH should be rejected (not ETH)."""
        assert detect_underlying("Will rETH yield exceed 5%?") is None

    def test_cbeth_rejected_as_false_positive(self):
        """cbETH should be rejected (not ETH)."""
        assert detect_underlying("Will cbETH trading volume grow?") is None

    def test_renbtc_rejected_as_false_positive(self):
        """renBTC should be rejected (not BTC)."""
        assert detect_underlying("Will renBTC supply increase?") is None

    def test_susde_rejected_as_false_positive(self):
        """sUSDe should be rejected (Ethena product)."""
        assert detect_underlying("Will sUSDe yield stay above 15%?") is None

    def test_case_insensitive_btc(self):
        assert detect_underlying("Will btc hit $100k?") == "BTC"

    def test_case_insensitive_ethereum(self):
        assert detect_underlying("Will ethereum reach $5k?") == "ETH"

    def test_no_underlying_detected_returns_none(self):
        assert detect_underlying("Will crypto markets rally?") is None


class TestExtractStrike:
    """Test strike price extraction (unchanged from Loop 5)."""

    def test_extract_dollars_with_commas(self):
        assert extract_strike("Will BTC hit $100,000?") == 100000.0

    def test_extract_k_suffix(self):
        assert extract_strike("Will ETH reach $5K?") == 5000.0

    def test_extract_m_suffix(self):
        assert extract_strike("Will BTC hit $1M?") == 1000000.0

    def test_extract_decimal_k(self):
        assert extract_strike("Will ETH hit $3.5K?") == 3500.0

    def test_no_strike_returns_none(self):
        assert extract_strike("Will crypto rally?") is None


class TestDetectResolutionType:
    """Test resolution type detection (unchanged from Loop 5)."""

    def test_hit_keyword_returns_touch(self):
        res_type, op, window = detect_resolution_type("Will BTC hit $100K?")
        assert res_type == "touch"
        assert op == ">="
        assert window == "any_time"

    def test_reach_keyword_returns_touch(self):
        res_type, op, window = detect_resolution_type("Will ETH reach $10K by year end?")
        assert res_type == "touch"
        assert op == ">="
        assert window == "any_time"

    def test_close_above_returns_close(self):
        res_type, op, window = detect_resolution_type("Will BTC be above $90K at January 1 close?")
        assert res_type == "close"
        assert op == ">="
        assert window == "at_close"

    def test_close_below_returns_close(self):
        res_type, op, window = detect_resolution_type("Will ETH be below $3K at June 30 close?")
        assert res_type == "close"
        assert op == "<="
        assert window == "at_close"

    def test_ambiguous_returns_unknown(self):
        res_type, _op, _window = detect_resolution_type("Will BTC rally?")
        assert res_type == "unknown"


class TestParseThresholdMarket:
    """Test full parsing pipeline with strict filtering."""

    def test_parse_valid_btc_touch_market(self):
        market = {
            "id": "test1",
            "question": "Will BTC hit $100,000 by December 31?",
            "end_date_iso": "2026-12-31T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is not None
        assert parsed["underlying"] == "BTC"
        assert parsed["strike"] == 100000.0
        assert parsed["resolution_type"] == "touch"
        assert parsed["op"] == ">="
        assert parsed["window"] == "any_time"

    def test_parse_valid_eth_close_market(self):
        market = {
            "id": "test2",
            "question": "Will ETH be below $3,000 at June 30 close?",
            "end_date_iso": "2026-06-30T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is not None
        assert parsed["underlying"] == "ETH"
        assert parsed["strike"] == 3000.0
        assert parsed["resolution_type"] == "close"
        assert parsed["op"] == "<="

    def test_parse_ethena_market_returns_none(self):
        """Ethena market should be rejected."""
        market = {
            "id": "ethena_fp",
            "question": "Will Ethena protocol TVL exceed $10B?",
            "end_date_iso": "2026-09-30T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is None, "Ethena market should be rejected as false positive"

    def test_parse_wbtc_market_returns_none(self):
        """WBTC market should be rejected."""
        market = {
            "id": "wbtc_fp",
            "question": "Will WBTC depeg by >2% from BTC?",
            "end_date_iso": "2026-06-30T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is None, "WBTC market should be rejected as false positive"

    def test_parse_steth_market_returns_none(self):
        """stETH market should be rejected."""
        market = {
            "id": "steth_fp",
            "question": "Will stETH maintain peg to ETH?",
            "end_date_iso": "2026-06-30T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is None, "stETH market should be rejected as false positive"

    def test_parse_susde_market_returns_none(self):
        """sUSDe market should be rejected."""
        market = {
            "id": "susde_fp",
            "question": "Will sUSDe yield stay above 15%?",
            "end_date_iso": "2026-06-30T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is None, "sUSDe market should be rejected as false positive"

    def test_parse_ambiguous_question_returns_none(self):
        market = {
            "id": "ambiguous",
            "question": "Will crypto markets rally in 2026?",
            "end_date_iso": "2026-12-31T23:59:59Z",
        }
        parsed = parse_threshold_market(market)
        assert parsed is None

    def test_parse_missing_question_returns_none(self):
        market = {"id": "missing_q", "end_date_iso": "2026-12-31T23:59:59Z"}
        parsed = parse_threshold_market(market)
        assert parsed is None

    def test_parse_missing_end_date_returns_none(self):
        market = {
            "id": "missing_date",
            "question": "Will BTC hit $100K?",
        }
        parsed = parse_threshold_market(market)
        assert parsed is None


class TestFixtureIntegration:
    """Test parser against fixture markets from discover_crypto_gamma."""

    def test_fixture_btc_markets_parse_successfully(self):
        """BTC fixture markets should parse with underlying='BTC'."""
        from discover_crypto_gamma import discover_crypto_markets_fixture

        result = discover_crypto_markets_fixture()
        btc_markets = [m for m in result.markets if "BTC" in m["question"] or "Bitcoin" in m["question"]]

        parsed_count = 0
        for market in btc_markets:
            parsed = parse_threshold_market(market)
            if parsed is not None:
                assert parsed["underlying"] == "BTC"
                parsed_count += 1

        assert parsed_count > 0, "At least one BTC fixture market should parse successfully"

    def test_fixture_ethena_markets_rejected(self):
        """Ethena fixture markets should ALL be rejected."""
        from discover_crypto_gamma import discover_crypto_markets_fixture

        result = discover_crypto_markets_fixture()
        ethena_markets = [m for m in result.markets if "Ethena" in m.get("question", "")]

        for market in ethena_markets:
            parsed = parse_threshold_market(market)
            assert parsed is None, f"Ethena market should be rejected: {market['question']}"

    def test_fixture_wbtc_market_rejected(self):
        """WBTC fixture market should be rejected."""
        from discover_crypto_gamma import discover_crypto_markets_fixture

        result = discover_crypto_markets_fixture()
        wbtc_markets = [m for m in result.markets if "WBTC" in m.get("question", "")]

        for market in wbtc_markets:
            parsed = parse_threshold_market(market)
            assert parsed is None, f"WBTC market should be rejected: {market['question']}"

    def test_fixture_steth_market_rejected(self):
        """stETH fixture market should be rejected."""
        from discover_crypto_gamma import discover_crypto_markets_fixture

        result = discover_crypto_markets_fixture()
        steth_markets = [m for m in result.markets if "stETH" in m.get("question", "")]

        for market in steth_markets:
            parsed = parse_threshold_market(market)
            assert parsed is None, f"stETH market should be rejected: {market['question']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
