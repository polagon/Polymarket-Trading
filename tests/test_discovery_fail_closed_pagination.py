"""
Tests for fail-closed pagination exhaustion detection (Loop 6.1).

Validates:
- Generator hard-stops when pagination_exhausted == True
- discovery.json artifact includes pagination_exhausted field
- SystemExit with descriptive message
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

pytestmark = pytest.mark.anyio


class TestPaginationExhaustionFailClosed:
    """Test generator hard-stops when pagination is exhausted."""

    async def test_generator_exits_on_pagination_exhausted(self, tmp_path):
        """Generator raises SystemExit when pagination_exhausted == True."""
        from discover_crypto_gamma import DiscoveryResult
        from generate_crypto_threshold_contracts import discover_markets

        from models.reasons import REASON_DISCOVERY_PAGINATION_EXHAUSTED

        # Mock discover_crypto_markets to return pagination_exhausted=True
        exhausted_result = DiscoveryResult(
            markets=[
                {
                    "id": "test_market",
                    "question": "Will BTC hit $100K?",
                    "end_date_iso": "2026-12-31T23:59:59Z",
                    "condition_id": "0xtest",
                    "clobTokenIds": ["0x1", "0x2"],
                }
            ],
            metadata={"total_count": 1, "btc_count": 1, "eth_count": 0},
            reason=REASON_DISCOVERY_PAGINATION_EXHAUSTED,  # Pagination exhausted
            tag_ids_used=["crypto"],
            pages_fetched=10,
            discovered_at="2026-02-15T10:00:00Z",
            pagination_exhausted=True,  # Partial universe
        )

        with patch("generate_crypto_threshold_contracts.discover_crypto_markets", return_value=exhausted_result):
            # Should raise SystemExit on discovery_reason check (not pagination_exhausted check)
            with pytest.raises(SystemExit, match="Discovery veto"):
                await discover_markets(mode="fixture", underlyings=["BTC", "ETH"], write_artifact=False)

    async def test_generator_proceeds_on_pagination_ok(self):
        """Generator proceeds when pagination_exhausted == False."""
        from discover_crypto_gamma import DiscoveryResult
        from generate_crypto_threshold_contracts import discover_markets

        from models.reasons import REASON_DISCOVERY_OK

        # Mock discover_crypto_markets to return pagination_exhausted=False
        ok_result = DiscoveryResult(
            markets=[
                {
                    "id": "test_market",
                    "question": "Will BTC hit $100K?",
                    "end_date_iso": "2026-12-31T23:59:59Z",
                    "condition_id": "0xtest",
                    "clobTokenIds": ["0x1", "0x2"],
                }
            ],
            metadata={"total_count": 1, "btc_count": 1, "eth_count": 0},
            reason=REASON_DISCOVERY_OK,
            tag_ids_used=["crypto"],
            pages_fetched=3,
            discovered_at="2026-02-15T10:00:00Z",
            pagination_exhausted=False,  # Complete universe
        )

        with patch("generate_crypto_threshold_contracts.discover_crypto_markets", return_value=ok_result):
            # Should NOT raise SystemExit
            markets, metadata, result = await discover_markets(
                mode="fixture", underlyings=["BTC", "ETH"], write_artifact=False
            )

            assert len(markets) == 1
            assert metadata["btc_count"] == 1
            assert result.reason == REASON_DISCOVERY_OK
            assert not result.pagination_exhausted

    async def test_discovery_artifact_includes_pagination_exhausted(self, tmp_path):
        """discovery.json artifact includes pagination_exhausted field."""
        import json

        from discover_crypto_gamma import DiscoveryResult
        from generate_crypto_threshold_contracts import discover_markets

        from models.reasons import REASON_DISCOVERY_PAGINATION_EXHAUSTED

        exhausted_result = DiscoveryResult(
            markets=[],
            metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
            reason=REASON_DISCOVERY_PAGINATION_EXHAUSTED,
            tag_ids_used=["crypto"],
            pages_fetched=10,
            discovered_at="2026-02-15T10:00:00Z",
            pagination_exhausted=True,
        )

        # Patch the artifact path to write to tmp_path
        artifact_dir = tmp_path / "artifacts" / "universe"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        import generate_crypto_threshold_contracts

        original_path = generate_crypto_threshold_contracts.Path

        def mock_path(p):
            if str(p) == "artifacts/universe/discovery.json":
                return artifact_dir / "discovery.json"
            return original_path(p)

        with patch("generate_crypto_threshold_contracts.discover_crypto_markets", return_value=exhausted_result):
            with patch("generate_crypto_threshold_contracts.Path", side_effect=mock_path):
                # Should raise SystemExit but write artifact first
                with pytest.raises(SystemExit, match="Discovery veto"):
                    await discover_markets(mode="fixture", underlyings=["BTC", "ETH"], write_artifact=True)

                # Artifact should exist
                artifact_path = artifact_dir / "discovery.json"
                assert artifact_path.exists()

                # Verify contents
                data = json.loads(artifact_path.read_text())
                assert data["pagination_exhausted"] is True
                assert data["discovery_reason"] == REASON_DISCOVERY_PAGINATION_EXHAUSTED
                assert data["pages_fetched"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
