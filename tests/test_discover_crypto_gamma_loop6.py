"""
Tests for Loop 6.0 live discovery with fail-closed behavior.

Validates:
- Fixture mode returns expected BTC/ETH counts (no SOL)
- Live mode blocked under PYTEST_CURRENT_TEST
- Invalid responses map to correct veto reasons
- DiscoveryResult includes all required provenance metadata
"""

import os

# Ensure tools/ is in path
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

tools_path = Path(__file__).resolve().parent.parent / "tools"
if str(tools_path) not in sys.path:
    sys.path.insert(0, str(tools_path))

from discover_crypto_gamma import (
    DiscoveryResult,
    GammaClient,
    discover_crypto_markets,
    discover_crypto_markets_fixture,
    discover_crypto_markets_live,
)

from models.reasons import (
    REASON_DISCOVERY_GAMMA_HTTP_ERROR,
    REASON_DISCOVERY_GAMMA_TIMEOUT,
    REASON_DISCOVERY_INVALID_RESPONSE,
    REASON_DISCOVERY_OK,
    REASON_DISCOVERY_TAG_NOT_FOUND,
)


class TestFixtureMode:
    """Test fixture-based discovery (no network)."""

    def test_fixture_returns_expected_counts_btc_eth_only(self):
        """Fixture mode returns 3 BTC, 2 ETH markets (no SOL)."""
        result = discover_crypto_markets_fixture()

        assert isinstance(result, DiscoveryResult)
        assert result.reason == REASON_DISCOVERY_OK
        assert result.metadata["total_count"] == 10
        assert result.metadata["btc_count"] == 3
        assert result.metadata["eth_count"] == 2
        assert "sol_count" not in result.metadata  # SOL out of scope

        # Provenance metadata
        assert result.tag_ids_used == ["crypto_fixture"]
        assert result.pages_fetched == 1
        assert result.discovered_at != ""

    def test_fixture_mode_via_public_api(self):
        """discover_crypto_markets(mode='fixture') works."""
        result = discover_crypto_markets(mode="fixture")

        assert result.reason == REASON_DISCOVERY_OK
        assert result.metadata["btc_count"] == 3
        assert result.metadata["eth_count"] == 2

    def test_fixture_markets_have_required_fields(self):
        """All fixture markets have id, question, end_date_iso."""
        result = discover_crypto_markets_fixture()

        for m in result.markets:
            assert "id" in m
            assert "question" in m
            assert "end_date_iso" in m
            assert "condition_id" in m
            assert "clobTokenIds" in m


class TestLiveModeBlockedUnderPytest:
    """Test that live mode is blocked when PYTEST_CURRENT_TEST is set."""

    def test_live_mode_blocked_under_pytest_env(self):
        """Live mode raises ValueError when PYTEST_CURRENT_TEST is set."""
        # PYTEST_CURRENT_TEST is auto-set by pytest, so this should always block
        assert os.getenv("PYTEST_CURRENT_TEST") is not None

        with pytest.raises(ValueError, match="Live discovery blocked.*PYTEST_CURRENT_TEST"):
            discover_crypto_markets(mode="live")


class TestGammaClientErrorHandling:
    """Test GammaClient fail-closed behavior with mocked requests."""

    @patch("discover_crypto_gamma.GammaClient._get_session")
    def test_tag_not_found_maps_to_reason_tag_not_found(self, mock_get_session):
        """GET /tags returns 200 but no 'crypto' tag → REASON_DISCOVERY_TAG_NOT_FOUND."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": "sports", "label": "Sports"}]  # No crypto tag
        mock_session.get.return_value = mock_resp
        mock_get_session.return_value = mock_session

        client = GammaClient()
        tag_id = client.find_crypto_tag()
        assert tag_id is None

        # Now test full discovery (should return TAG_NOT_FOUND)
        # We need to patch _should_use_live_discovery to allow this test to run
        with patch("discover_crypto_gamma._should_use_live_discovery", return_value=True):
            result = discover_crypto_markets_live()
            assert result.reason == REASON_DISCOVERY_TAG_NOT_FOUND
            assert result.metadata["total_count"] == 0

    @patch("discover_crypto_gamma.GammaClient._get_session")
    def test_http_500_maps_to_gamma_http_error(self, mock_get_session):
        """GET /tags returns 500 after retries → REASON_DISCOVERY_GAMMA_HTTP_ERROR."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_session.get.return_value = mock_resp
        mock_get_session.return_value = mock_session

        with patch("discover_crypto_gamma._should_use_live_discovery", return_value=True):
            result = discover_crypto_markets_live()
            assert result.reason == REASON_DISCOVERY_GAMMA_HTTP_ERROR
            assert result.metadata["total_count"] == 0

    @patch("discover_crypto_gamma.GammaClient._get_session")
    def test_timeout_maps_to_gamma_timeout(self, mock_get_session):
        """Timeout after retries → REASON_DISCOVERY_GAMMA_TIMEOUT."""
        import requests

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.Timeout("Timeout")
        mock_get_session.return_value = mock_session

        with patch("discover_crypto_gamma._should_use_live_discovery", return_value=True):
            result = discover_crypto_markets_live()
            assert result.reason == REASON_DISCOVERY_GAMMA_TIMEOUT
            assert result.metadata["total_count"] == 0

    @patch("discover_crypto_gamma.GammaClient._get_session")
    def test_invalid_json_response_handled_gracefully(self, mock_get_session):
        """GET /tags returns invalid JSON → REASON_DISCOVERY_INVALID_RESPONSE or HTTP_ERROR."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_session.get.return_value = mock_resp
        mock_get_session.return_value = mock_session

        with patch("discover_crypto_gamma._should_use_live_discovery", return_value=True):
            result = discover_crypto_markets_live()
            # Should return HTTP_ERROR (exception caught in find_crypto_tag)
            assert result.reason in [REASON_DISCOVERY_GAMMA_HTTP_ERROR, REASON_DISCOVERY_INVALID_RESPONSE]


class TestGammaClientMarketNormalization:
    """Test market shape normalization from Gamma API."""

    def test_normalize_market_with_endDate_key(self):
        """Normalize market with 'endDate' instead of 'end_date_iso'."""
        client = GammaClient()
        raw = {
            "id": "test_market",
            "question": "Will BTC hit $100K?",
            "endDate": "2026-12-31T23:59:59Z",
            "conditionId": "0xabc",
            "clobTokenIds": ["0x1", "0x2"],
        }
        normalized = client._normalize_market(raw)

        assert normalized is not None
        assert normalized["id"] == "test_market"
        assert normalized["end_date_iso"] == "2026-12-31T23:59:59Z"
        assert normalized["condition_id"] == "0xabc"

    def test_normalize_market_with_tokens_array(self):
        """Normalize market with 'tokens' array instead of 'clobTokenIds'."""
        client = GammaClient()
        raw = {
            "id": "test_market",
            "question": "Will ETH hit $10K?",
            "end_date_iso": "2026-12-31T23:59:59Z",
            "condition_id": "0xdef",
            "tokens": [{"token_id": "0xa"}, {"token_id": "0xb"}],
        }
        normalized = client._normalize_market(raw)

        assert normalized is not None
        assert normalized["clobTokenIds"] == ["0xa", "0xb"]

    def test_normalize_market_missing_required_fields_returns_none(self):
        """Normalize returns None if required fields missing."""
        client = GammaClient()
        raw = {"id": "test_market"}  # Missing question, end_date_iso
        normalized = client._normalize_market(raw)

        assert normalized is None


class TestCountsByUnderlying:
    """Test that counts_by_underlying only includes BTC/ETH (no SOL)."""

    def test_fixture_counts_by_underlying_only_btc_eth(self):
        """Fixture metadata counts only BTC and ETH."""
        result = discover_crypto_markets_fixture()

        # Check metadata keys
        assert "btc_count" in result.metadata
        assert "eth_count" in result.metadata
        assert "sol_count" not in result.metadata  # SOL out of scope

        # Check actual counts match fixture
        assert result.metadata["btc_count"] == 3
        assert result.metadata["eth_count"] == 2
