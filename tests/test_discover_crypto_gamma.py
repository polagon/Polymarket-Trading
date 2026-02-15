"""
Tests for tools/discover_crypto_gamma.py

Loop 5.1: Fixture-based tests proving btc_count > 0 (fixing ZERO BTC bug).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from discover_crypto_gamma import discover_crypto_markets_fixture


class TestFixtureDiscovery:
    """Test fixture-based discovery returns valid markets with correct counts."""

    def test_fixture_returns_markets_and_metadata(self):
        markets, metadata = discover_crypto_markets_fixture()
        assert isinstance(markets, list)
        assert isinstance(metadata, dict)
        assert len(markets) > 0
        assert "total_count" in metadata
        assert "btc_count" in metadata
        assert "eth_count" in metadata
        assert "sol_count" in metadata

    def test_fixture_btc_count_greater_than_zero(self):
        """CRITICAL: Proves btc_count > 0, fixing ZERO BTC discovery bug."""
        _markets, metadata = discover_crypto_markets_fixture()
        btc_count = metadata["btc_count"]
        assert btc_count > 0, f"Expected btc_count > 0, got {btc_count} (ZERO BTC BUG)"

    def test_fixture_eth_count_greater_than_zero(self):
        _markets, metadata = discover_crypto_markets_fixture()
        eth_count = metadata["eth_count"]
        assert eth_count > 0, f"Expected eth_count > 0, got {eth_count}"

    def test_fixture_total_count_matches_market_list_length(self):
        markets, metadata = discover_crypto_markets_fixture()
        assert metadata["total_count"] == len(markets)

    def test_fixture_includes_ethena_false_positives(self):
        """Fixture includes Ethena markets to test parser rejection."""
        markets, _metadata = discover_crypto_markets_fixture()
        ethena_markets = [m for m in markets if "Ethena" in m.get("question", "")]
        assert len(ethena_markets) >= 1, "Fixture should include at least 1 Ethena false positive"

    def test_fixture_includes_wbtc_false_positive(self):
        """Fixture includes WBTC market to test parser rejection."""
        markets, _metadata = discover_crypto_markets_fixture()
        wbtc_markets = [m for m in markets if "WBTC" in m.get("question", "")]
        assert len(wbtc_markets) >= 1, "Fixture should include at least 1 WBTC false positive"

    def test_fixture_includes_steth_false_positive(self):
        """Fixture includes stETH market to test parser rejection."""
        markets, _metadata = discover_crypto_markets_fixture()
        steth_markets = [m for m in markets if "stETH" in m.get("question", "")]
        assert len(steth_markets) >= 1, "Fixture should include at least 1 stETH false positive"

    def test_fixture_metadata_counts_exclude_false_positives(self):
        """Metadata counts should NOT include Ethena/WBTC/stETH in ETH count."""
        markets, metadata = discover_crypto_markets_fixture()

        # Manual count: ETH markets without Ethena
        actual_eth = sum(
            1
            for m in markets
            if ("ETH" in m["question"] or "Ethereum" in m["question"])
            and "Ethena" not in m["question"]
            and "stETH" not in m["question"]
        )

        assert metadata["eth_count"] == actual_eth, (
            f"eth_count should exclude Ethena/stETH, expected {actual_eth}, got {metadata['eth_count']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
