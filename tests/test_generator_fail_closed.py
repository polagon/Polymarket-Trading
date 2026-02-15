"""
Test fail-closed discovery guard in generator.

Verifies that contract generation refuses to proceed when discovery fails.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add tools/ to path
tools_path = Path(__file__).resolve().parent.parent / "tools"
if str(tools_path) not in sys.path:
    sys.path.insert(0, str(tools_path))

# Use anyio backend (already in project deps)
pytestmark = pytest.mark.anyio


class TestFailClosedDiscovery:
    """Test generator hard-stops on degraded discovery."""

    async def test_generator_exits_nonzero_on_discovery_failure(self, tmp_path):
        """Generator raises SystemExit when discovery_reason != discovery_ok."""
        from discover_crypto_gamma import DiscoveryResult
        from generate_crypto_threshold_contracts import discover_markets

        from models.reasons import REASON_DISCOVERY_TAG_NOT_FOUND

        # Mock discover_crypto_markets to return a failed discovery
        failed_result = DiscoveryResult(
            markets=[],
            metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
            reason=REASON_DISCOVERY_TAG_NOT_FOUND,
            tag_ids_used=[],
            pages_fetched=0,
            discovered_at="2026-02-15T10:00:00Z",
        )

        with patch("generate_crypto_threshold_contracts.discover_crypto_markets", return_value=failed_result):
            # Change to tmp_path for artifact writing
            with patch("generate_crypto_threshold_contracts.Path") as mock_path_cls:
                mock_path_cls.return_value = tmp_path / "artifacts" / "universe" / "discovery.json"
                mock_path_cls.return_value.parent.mkdir(parents=True, exist_ok=True)

                # Should raise SystemExit with discovery veto message
                with pytest.raises(SystemExit, match="Discovery veto.*tag_not_found"):
                    await discover_markets(mode="fixture", underlyings=["BTC", "ETH"], write_artifact=False)

    async def test_generator_writes_discovery_artifact_even_on_failure(self, tmp_path):
        """Discovery artifact is written even when discovery fails."""
        from discover_crypto_gamma import DiscoveryResult
        from generate_crypto_threshold_contracts import discover_markets

        from models.reasons import REASON_DISCOVERY_GAMMA_HTTP_ERROR

        # Mock discover_crypto_markets to return a failed discovery
        failed_result = DiscoveryResult(
            markets=[],
            metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
            reason=REASON_DISCOVERY_GAMMA_HTTP_ERROR,
            tag_ids_used=["crypto"],
            pages_fetched=0,
            discovered_at="2026-02-15T10:00:00Z",
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

        with patch("generate_crypto_threshold_contracts.discover_crypto_markets", return_value=failed_result):
            with patch("generate_crypto_threshold_contracts.Path", side_effect=mock_path):
                # Should raise SystemExit but write artifact first
                with pytest.raises(SystemExit, match="Discovery veto.*gamma_http_error"):
                    await discover_markets(mode="fixture", underlyings=["BTC", "ETH"], write_artifact=True)

                # Artifact should exist
                artifact_path = artifact_dir / "discovery.json"
                assert artifact_path.exists()

                # Verify contents
                data = json.loads(artifact_path.read_text())
                assert data["discovery_reason"] == REASON_DISCOVERY_GAMMA_HTTP_ERROR
                assert data["discovery_mode"] == "fixture"
                assert data["total_count"] == 0

    async def test_generator_proceeds_on_discovery_ok(self):
        """Generator proceeds normally when discovery_reason == discovery_ok."""
        from discover_crypto_gamma import DiscoveryResult
        from generate_crypto_threshold_contracts import discover_markets

        from models.reasons import REASON_DISCOVERY_OK

        # Mock discover_crypto_markets to return successful discovery
        success_result = DiscoveryResult(
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
            pages_fetched=1,
            discovered_at="2026-02-15T10:00:00Z",
        )

        with patch("generate_crypto_threshold_contracts.discover_crypto_markets", return_value=success_result):
            # Should NOT raise SystemExit
            markets, metadata, result = await discover_markets(
                mode="fixture", underlyings=["BTC", "ETH"], write_artifact=False
            )

            assert len(markets) == 1
            assert metadata["btc_count"] == 1
            assert result.reason == REASON_DISCOVERY_OK
