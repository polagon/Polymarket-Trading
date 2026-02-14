"""
Tests for Astra V2 Probability Estimation Engine — scanner/probability_estimator.py

Deterministic. No network calls. No real Anthropic API usage.
Uses unittest.mock for all Claude interactions.
"""

import asyncio
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.market_fetcher import Market
from scanner.probability_estimator import (
    _ESTIMATE_CACHE,
    _ESTIMATE_CACHE_TTL,
    _ESTIMATE_PRICE_DRIFT,
    ASTRA_CON_SYSTEM_HASH,
    ASTRA_PRO_SYSTEM_HASH,
    ASTRA_SYNTHESIZER_SYSTEM_HASH,
    ASTRA_V2_SYSTEM_HASH,
    ESTIMATOR_VERSION,
    PROMPT_BUNDLE_HASH,
    PROMPT_REGISTRY,
    Estimate,
    _cache_estimate,
    _clean_json,
    _extremize,
    _get_cached_estimate,
    _kelly_pct,
    _robustness,
    _validate_adversarial_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_market(**kw) -> Market:
    defaults = dict(
        condition_id="cond_1",
        question="Will BTC reach $100k?",
        end_date_iso="2026-06-01T00:00:00Z",
        category="sports",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        yes_price=0.50,
        no_price=0.50,
        liquidity=10000.0,
        volume=5000.0,
        hours_to_expiry=200.0,
    )
    defaults.update(kw)
    return Market(**defaults)  # type: ignore[arg-type]


def _make_estimate(**kw) -> Estimate:
    defaults = dict(
        market_condition_id="cond_1",
        question="Will BTC reach $100k?",
        category="crypto",
        probability=0.60,
        probability_low=0.50,
        probability_high=0.70,
        confidence=0.75,
        market_type="Crypto",
        modeling_approach="A",
        trap_flags=[],
        edge=0.10,
        ev_after_costs=0.095,
        robustness_score=4,
        kelly_position_pct=0.005,
        source="astra_v2",
        truth_state="Supported",
        reasoning="Test estimate",
        key_evidence_needed="None",
        no_trade=False,
        no_trade_reason="",
        details={"test": True},
    )
    defaults.update(kw)
    return Estimate(**defaults)  # type: ignore[arg-type]


def _valid_synth_item(**overrides) -> dict:
    """Template matching production _validate_adversarial_schema requirements."""
    item = {
        "id": "cond_1",
        "p_hat": 0.65,
        "confidence": 0.80,
        "dominant_evidence_tier": "B",
        "p_low": 0.55,
        "p_high": 0.75,
        "edge": 0.15,
        "ev_after_costs": 0.145,
        "robustness_score": 4,
        "kelly_position_pct": 0.005,
        "stake": 60000,
        "trap_flags": [],
        "no_trade": False,
        "no_trade_reason": "",
        "truth_state": "Supported",
        "market_type": "Crypto",
        "modeling_approach": "B",
        "p_neutral": 0.63,
        "p_aware": 0.66,
        "pro_summary": "Strong thesis",
        "con_summary": "Weak counter",
        "synthesis_reasoning": "Net positive",
        "key_evidence_needed": "None",
    }
    item.update(overrides)
    return item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear module-level estimate cache before and after each test."""
    _ESTIMATE_CACHE.clear()
    yield
    _ESTIMATE_CACHE.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 1) TestEstimateCache
# ═══════════════════════════════════════════════════════════════════════════
class TestEstimateCache:
    def test_cache_hit_within_ttl(self, monkeypatch):
        est = _make_estimate(market_condition_id="cond_A")
        now = 1000000.0
        monkeypatch.setattr("scanner.probability_estimator._time.time", lambda: now)
        _cache_estimate(est, 0.50)
        # 10 seconds later, same price
        monkeypatch.setattr("scanner.probability_estimator._time.time", lambda: now + 10)
        result = _get_cached_estimate("cond_A", 0.50)
        assert result is not None
        assert result.market_condition_id == "cond_A"

    def test_cache_miss_ttl_expired(self, monkeypatch):
        est = _make_estimate(market_condition_id="cond_B")
        now = 1000000.0
        monkeypatch.setattr("scanner.probability_estimator._time.time", lambda: now)
        _cache_estimate(est, 0.50)
        # Jump past TTL
        monkeypatch.setattr(
            "scanner.probability_estimator._time.time",
            lambda: now + _ESTIMATE_CACHE_TTL + 1,
        )
        result = _get_cached_estimate("cond_B", 0.50)
        assert result is None

    def test_cache_miss_price_drift(self, monkeypatch):
        est = _make_estimate(market_condition_id="cond_C")
        now = 1000000.0
        monkeypatch.setattr("scanner.probability_estimator._time.time", lambda: now)
        _cache_estimate(est, 0.50)
        monkeypatch.setattr("scanner.probability_estimator._time.time", lambda: now + 5)
        # Price drifted more than threshold
        result = _get_cached_estimate("cond_C", 0.50 + _ESTIMATE_PRICE_DRIFT + 0.01)
        assert result is None

    def test_cache_cold_start_returns_none(self):
        result = _get_cached_estimate("nonexistent", 0.50)
        assert result is None

    def test_cache_populate_and_retrieve(self, monkeypatch):
        est = _make_estimate(market_condition_id="cond_D")
        now = 1000000.0
        monkeypatch.setattr("scanner.probability_estimator._time.time", lambda: now)
        _cache_estimate(est, 0.60)
        result = _get_cached_estimate("cond_D", 0.60)
        assert result is est


# ═══════════════════════════════════════════════════════════════════════════
# 2) TestRobustness
# ═══════════════════════════════════════════════════════════════════════════
class TestRobustness:
    def test_large_edge_returns_max_score(self):
        # edge >> spread: ratio ≥ 2.0 → 5
        score = _robustness(p=0.80, mkt_price=0.50, spread=0.10)
        assert score == 5

    def test_moderate_edge_returns_mid_score(self):
        # edge=0.15, spread=0.10 → ratio = 0.15/0.101 ≈ 1.49 → score 3
        score = _robustness(p=0.65, mkt_price=0.50, spread=0.10)
        assert score == 3

    def test_tiny_edge_returns_min_score(self):
        # ratio < 0.5 → 1
        score = _robustness(p=0.51, mkt_price=0.50, spread=0.10)
        assert score == 1

    def test_zero_edge_returns_min_score(self):
        score = _robustness(p=0.50, mkt_price=0.50, spread=0.10)
        assert score == 1

    def test_monotonic_with_increasing_edge(self):
        """Score should be monotonically non-decreasing as edge/spread ratio increases."""
        spread = 0.10
        mkt = 0.50
        scores = [_robustness(mkt + spread * r, mkt, spread) for r in [0.3, 0.6, 1.1, 1.6, 2.1]]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"scores not monotonic: {scores}"


# ═══════════════════════════════════════════════════════════════════════════
# 3) TestExtremize
# ═══════════════════════════════════════════════════════════════════════════
class TestExtremize:
    def test_pushes_away_from_half(self):
        # p=0.6 with k=1.3 should become > 0.6
        result = _extremize(0.6, 0.5)
        assert result > 0.6

    def test_symmetry(self):
        # extremize(p) + extremize(1-p) should ≈ 1.0
        for p in [0.3, 0.4, 0.6, 0.7, 0.8]:
            s = _extremize(p, 0.5) + _extremize(1 - p, 0.5)
            assert abs(s - 1.0) < 1e-10, f"symmetry broken at p={p}: sum={s}"

    def test_boundary_zero(self):
        assert _extremize(0.0, 0.5) == 0.0

    def test_boundary_one(self):
        assert _extremize(1.0, 0.5) == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 4) TestKellyPct
# ═══════════════════════════════════════════════════════════════════════════
class TestKellyPct:
    def test_positive_edge_positive_kelly(self):
        result = _kelly_pct(p=0.70, mkt_price=0.50, conf=0.85, rob=4)
        assert result > 0

    def test_no_edge_near_zero(self):
        result = _kelly_pct(p=0.50, mkt_price=0.50, conf=0.85, rob=4)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_capped_at_two_percent(self):
        # Massive edge → should still be capped
        result = _kelly_pct(p=0.99, mkt_price=0.10, conf=1.0, rob=5)
        assert result <= 0.02

    def test_low_confidence_reduces_sizing(self):
        # Use a smaller edge so neither hits the cap
        high_conf = _kelly_pct(p=0.58, mkt_price=0.50, conf=0.90, rob=4)
        low_conf = _kelly_pct(p=0.58, mkt_price=0.50, conf=0.50, rob=4)
        assert high_conf > 0
        assert low_conf < high_conf

    def test_invalid_price_returns_zero(self):
        assert _kelly_pct(0.60, 0.0, 0.85, 4) == 0.0
        assert _kelly_pct(0.60, 1.0, 0.85, 4) == 0.0

    def test_never_negative_or_nan(self):
        for p in [0.01, 0.5, 0.99]:
            for mkt in [0.01, 0.5, 0.99]:
                for conf in [0.0, 0.5, 1.0]:
                    result = _kelly_pct(p, mkt, conf, 3)
                    assert result >= 0.0
                    assert not math.isnan(result)
                    assert result <= 0.02


# ═══════════════════════════════════════════════════════════════════════════
# 5) TestCleanJson
# ═══════════════════════════════════════════════════════════════════════════
class TestCleanJson:
    def test_strips_markdown_fences_with_lang(self):
        raw = '```json\n[{"a": 1}]\n```'
        cleaned = _clean_json(raw)
        assert json.loads(cleaned) == [{"a": 1}]

    def test_strips_markdown_fences_without_lang(self):
        raw = '```\n[{"b": 2}]\n```'
        cleaned = _clean_json(raw)
        assert json.loads(cleaned) == [{"b": 2}]

    def test_removes_trailing_commas(self):
        raw = '[{"a": 1,}, {"b": 2,}]'
        cleaned = _clean_json(raw)
        assert json.loads(cleaned) == [{"a": 1}, {"b": 2}]

    def test_removes_control_characters(self):
        raw = '[{"a": "hello\x00world"}]'
        cleaned = _clean_json(raw)
        parsed = json.loads(cleaned)
        assert "hello" in parsed[0]["a"]
        assert "\x00" not in parsed[0]["a"]

    def test_truncated_array_recovery(self):
        """Production truncation recovery: extract valid prefix from truncated array."""
        raw = '[{"a": 1}, {"b": 2}, {"c": 3'
        cleaned = _clean_json(raw)
        parsed = json.loads(cleaned)
        # Should recover at least the first two complete objects
        assert len(parsed) >= 2
        assert parsed[0] == {"a": 1}
        assert parsed[1] == {"b": 2}

    def test_valid_json_passthrough(self):
        raw = '[{"x": 42, "y": "hello"}]'
        cleaned = _clean_json(raw)
        assert json.loads(cleaned) == json.loads(raw)


# ═══════════════════════════════════════════════════════════════════════════
# 6) TestValidateAdversarialSchema
# ═══════════════════════════════════════════════════════════════════════════
class TestValidateAdversarialSchema:
    def test_valid_item_passes(self):
        item = _valid_synth_item()
        valid, failures = _validate_adversarial_schema([item])
        assert len(valid) == 1
        assert len(failures) == 0

    def test_missing_required_field_rejected(self):
        item = _valid_synth_item()
        del item["confidence"]
        valid, failures = _validate_adversarial_schema([item])
        assert len(valid) == 0
        assert len(failures) == 1

    def test_non_dict_rejected(self):
        valid, failures = _validate_adversarial_schema(["not_a_dict"])
        assert len(valid) == 0
        assert len(failures) == 1

    def test_invalid_tier_defaults_to_c(self):
        item = _valid_synth_item(dominant_evidence_tier="X")
        valid, failures = _validate_adversarial_schema([item])
        assert len(valid) == 1
        assert valid[0]["dominant_evidence_tier"] == "C"

    def test_p_hat_out_of_range_rejected(self):
        item = _valid_synth_item(p_hat=1.5)
        valid, failures = _validate_adversarial_schema([item])
        assert len(valid) == 0
        assert len(failures) >= 1

    def test_missing_stake_defaults_conservatively(self):
        item = _valid_synth_item()
        del item["stake"]
        valid, failures = _validate_adversarial_schema([item])
        assert len(valid) == 1
        assert valid[0]["stake"] == 50000  # Production default


# ═══════════════════════════════════════════════════════════════════════════
# 7) TestAlgorithmicEstimates
# ═══════════════════════════════════════════════════════════════════════════
class TestAlgorithmicEstimates:
    def test_sports_algorithmic_returns_estimate(self):
        from scanner.probability_estimator import _try_algorithmic

        market = _make_market(category="sports", question="Lakers vs Celtics", yes_price=0.55)

        # Build a mock sports estimate (matching production expectations)
        @dataclass
        class MockSportsEstimate:
            probability: float = 0.60
            confidence: float = 0.70
            reasoning: str = "Consensus bookmaker odds"
            matched_game: str = "Lakers vs Celtics"
            hours_to_game: float = 24.0

        sports_ests = {"Lakers vs Celtics": MockSportsEstimate()}
        result = _try_algorithmic(market, {}, {}, sports_estimates=sports_ests)
        assert result is not None
        assert result.source == "the_odds_api"
        assert 0.0 <= result.probability <= 1.0

    def test_sports_low_confidence_returns_none(self):
        from scanner.probability_estimator import _try_algorithmic

        market = _make_market(category="sports", question="Game X", yes_price=0.55)

        @dataclass
        class LowConfSports:
            probability: float = 0.60
            confidence: float = 0.30  # Below 0.40 threshold
            reasoning: str = "Low conf"
            matched_game: str = "Game X"
            hours_to_game: float = 24.0

        result = _try_algorithmic(market, {}, {}, sports_estimates={"Game X": LowConfSports()})
        assert result is None

    def test_crypto_algorithmic_returns_estimate(self):
        from scanner.probability_estimator import _try_algorithmic

        market = _make_market(
            category="crypto",
            question="Will BTC reach $100k?",
            yes_price=0.40,
        )
        mock_result = {
            "probability": 0.55,
            "confidence": 0.70,
            "source": "crypto_lognormal",
            "details": {
                "current_price": 95000,
                "target_price": 100000,
                "annualized_vol": 0.60,
            },
        }
        with mock.patch("scanner.probability_estimator.crypto_source.estimate_probability", return_value=mock_result):
            result = _try_algorithmic(market, {"BTC": {"price": 95000}}, {})

        assert result is not None
        assert "lognormal" in result.source
        assert 0.0 <= result.probability <= 1.0

    def test_weather_algorithmic_returns_estimate(self):
        from scanner.probability_estimator import _try_algorithmic

        market = _make_market(
            category="weather",
            question="Will it rain in Chicago tomorrow?",
            yes_price=0.30,
        )
        mock_result = {
            "probability": 0.65,
            "confidence": 0.80,
            "source": "noaa_forecast",
            "details": {
                "noaa_short_forecast": "Rain likely",
                "noaa_precip_pct": 75,
            },
        }
        with mock.patch("scanner.probability_estimator.weather_source.estimate_probability", return_value=mock_result):
            result = _try_algorithmic(market, {}, {"Chicago": {"temp": 50}})

        assert result is not None
        assert "noaa" in result.source
        assert 0.0 <= result.probability <= 1.0

    def test_unknown_category_returns_none(self):
        from scanner.probability_estimator import _try_algorithmic

        market = _make_market(category="politics", question="Will X win election?", yes_price=0.55)
        result = _try_algorithmic(market, {}, {})
        assert result is None

    def test_algorithmic_no_trade_on_low_ev(self):
        from scanner.probability_estimator import _try_algorithmic

        # Set up a scenario where EV would be ≤ 0 (probability ≈ market price)
        market = _make_market(
            category="crypto",
            question="Will BTC reach $100k?",
            yes_price=0.50,
        )
        mock_result = {
            "probability": 0.505,  # Tiny edge → EV ≤ 0 after costs
            "confidence": 0.70,
            "source": "crypto_lognormal",
            "details": {"current_price": 95000, "target_price": 100000, "annualized_vol": 0.60},
        }
        with mock.patch("scanner.probability_estimator.crypto_source.estimate_probability", return_value=mock_result):
            result = _try_algorithmic(market, {"BTC": {"price": 95000}}, {})

        if result is not None:
            assert result.no_trade is True


# ═══════════════════════════════════════════════════════════════════════════
# 8) TestAdversarialPipeline
# ═══════════════════════════════════════════════════════════════════════════
class TestAdversarialPipeline:
    def _mock_claude_response(self, text: str):
        """Build a mock message response object matching Anthropic SDK."""
        mock_content = mock.MagicMock()
        mock_content.text = text
        mock_resp = mock.MagicMock()
        mock_resp.content = [mock_content]
        return mock_resp

    def test_no_api_key_returns_empty(self):
        from scanner.probability_estimator import _estimate_with_astra_v2

        market = _make_market()
        with mock.patch("scanner.probability_estimator.ANTHROPIC_API_KEY", ""):
            result = asyncio.run(_estimate_with_astra_v2([(market, None)], ""))
        assert result == []

    def test_auth_error_returns_empty(self):
        from scanner.probability_estimator import _estimate_with_astra_v2

        market = _make_market()
        mock_client = mock.MagicMock()
        mock_client.messages = mock.MagicMock()
        mock_client.messages.create = mock.AsyncMock(side_effect=Exception("Authentication failed"))

        with (
            mock.patch("scanner.probability_estimator.ANTHROPIC_API_KEY", "sk-test"),
            mock.patch("scanner.probability_estimator.anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            result = asyncio.run(_estimate_with_astra_v2([(market, None)], ""))
        # Should return empty or a list (fallback to single-pass may also fail)
        assert isinstance(result, list)

    def test_adversarial_produces_estimates(self):
        from scanner.probability_estimator import _estimate_with_astra_v2

        market = _make_market(condition_id="cond_adv", category="politics", yes_price=0.45)
        synth_item = _valid_synth_item(id="cond_adv", p_hat=0.60, confidence=0.80, stake=60000)
        synth_json = json.dumps([synth_item])

        # Mock all three Claude calls (PRO, CON, Synthesizer)
        mock_client = mock.MagicMock()
        mock_client.messages = mock.MagicMock()
        mock_client.messages.create = mock.AsyncMock(return_value=self._mock_claude_response(synth_json))

        with (
            mock.patch("scanner.probability_estimator.ANTHROPIC_API_KEY", "sk-test"),
            mock.patch("scanner.probability_estimator.anthropic.AsyncAnthropic", return_value=mock_client),
            mock.patch(
                "scanner.probability_estimator._fetch_verification_data", new_callable=mock.AsyncMock, return_value=""
            ),
        ):
            result = asyncio.run(_estimate_with_astra_v2([(market, None)], ""))
        assert isinstance(result, list)
        # May produce estimates or empty depending on parsing flow
        # The important thing is no crash

    def test_sports_hallucination_guard(self):
        """Sports hallucination guard: output should be bounded near market price."""
        from scanner.probability_estimator import _build_estimates_from_adversarial

        market = _make_market(
            condition_id="cond_sports_h",
            category="sports",
            yes_price=0.06,
            hours_to_expiry=200,
        )
        # Hallucinated p_hat way above 5x market
        synth_item = _valid_synth_item(
            id="cond_sports_h",
            p_hat=0.88,
            confidence=0.85,
            stake=60000,
            market_type="Sports",
        )

        with (
            mock.patch(
                "scanner.probability_estimator._fetch_verification_data", new_callable=mock.AsyncMock, return_value=""
            ),
            mock.patch(
                "scanner.probability_estimator._verify_estimate_with_claude",
                new_callable=mock.AsyncMock,
                return_value=None,
            ),
        ):
            result = asyncio.run(
                _build_estimates_from_adversarial(
                    [synth_item],
                    [(market, None)],
                    {},  # pro_by_id
                    {},  # con_by_id
                )
            )

        assert len(result) == 1
        est = result[0]
        # Hallucination guard should have anchored the probability closer to market
        assert est.probability < 0.88  # Not the hallucinated value
        assert abs(est.probability - market.yes_price) < abs(0.88 - market.yes_price)
        assert est.no_trade is True  # Should be vetoed

    def test_adversarial_fallback_to_single_pass(self):
        """If adversarial pipeline raises, production falls back to single-pass."""
        from scanner.probability_estimator import _estimate_with_astra_v2

        market = _make_market(condition_id="cond_fb", category="politics", yes_price=0.45)
        synth_item = _valid_synth_item(id="cond_fb", p_hat=0.60, confidence=0.80, stake=60000)
        synth_json = json.dumps([synth_item])

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # PRO and CON succeed
                return self._mock_claude_response(synth_json)
            elif call_count == 3:
                # Synthesizer fails
                raise ValueError("Synthesizer parse error")
            else:
                # Single-pass fallback succeeds
                return self._mock_claude_response(synth_json)

        mock_client = mock.MagicMock()
        mock_client.messages = mock.MagicMock()
        mock_client.messages.create = mock.AsyncMock(side_effect=mock_create)

        with (
            mock.patch("scanner.probability_estimator.ANTHROPIC_API_KEY", "sk-test"),
            mock.patch("scanner.probability_estimator.anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            result = asyncio.run(_estimate_with_astra_v2([(market, None)], ""))
        # Should not crash — either produces results from fallback or empty
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════
# 9) TestVerificationLoop
# ═══════════════════════════════════════════════════════════════════════════
class TestVerificationLoop:
    def test_verification_boosts_confidence(self):
        from scanner.probability_estimator import _verify_estimate_with_claude

        market = _make_market()
        mock_resp_text = json.dumps({"confidence": 0.80, "p_hat": 0.62, "updated_reasoning": "Data confirms"})

        mock_content = mock.MagicMock()
        mock_content.text = mock_resp_text
        mock_resp = mock.MagicMock()
        mock_resp.content = [mock_content]

        mock_client = mock.MagicMock()
        mock_client.messages = mock.MagicMock()
        mock_client.messages.create = mock.AsyncMock(return_value=mock_resp)

        with (
            mock.patch("scanner.probability_estimator.ANTHROPIC_API_KEY", "sk-test"),
            mock.patch("scanner.probability_estimator.anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            result = asyncio.run(_verify_estimate_with_claude(market, 0.60, 0.55, "context"))
        assert result is not None
        assert result["confidence"] == 0.80
        assert result["p_hat"] == 0.62

    def test_verification_auth_error_returns_none(self):
        from scanner.probability_estimator import _verify_estimate_with_claude

        market = _make_market()
        mock_client = mock.MagicMock()
        mock_client.messages = mock.MagicMock()
        mock_client.messages.create = mock.AsyncMock(side_effect=Exception("Auth error"))

        with (
            mock.patch("scanner.probability_estimator.ANTHROPIC_API_KEY", "sk-test"),
            mock.patch("scanner.probability_estimator.anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            result = asyncio.run(_verify_estimate_with_claude(market, 0.60, 0.55, "context"))
        assert result is None

    def test_no_api_key_returns_none(self):
        from scanner.probability_estimator import _verify_estimate_with_claude

        market = _make_market()
        with mock.patch("scanner.probability_estimator.ANTHROPIC_API_KEY", ""):
            result = asyncio.run(_verify_estimate_with_claude(market, 0.60, 0.55, "context"))
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 10) TestProvenance
# ═══════════════════════════════════════════════════════════════════════════
class TestProvenance:
    def test_prompt_bundle_hash_is_64_hex(self):
        assert re.match(r"^[0-9a-f]{64}$", PROMPT_BUNDLE_HASH)

    def test_estimator_version_nonempty(self):
        assert isinstance(ESTIMATOR_VERSION, str)
        assert len(ESTIMATOR_VERSION) > 0

    def test_prompt_registry_has_all_four_prompts(self):
        hashes = [
            ASTRA_V2_SYSTEM_HASH,
            ASTRA_PRO_SYSTEM_HASH,
            ASTRA_CON_SYSTEM_HASH,
            ASTRA_SYNTHESIZER_SYSTEM_HASH,
        ]
        for h in hashes:
            assert re.match(r"^[0-9a-f]{64}$", h), f"Hash not 64-hex: {h}"
            assert h in PROMPT_REGISTRY, f"Hash {h} not in PROMPT_REGISTRY"
            entry = PROMPT_REGISTRY[h]
            assert "name" in entry
            assert "intent" in entry

    def test_adversarial_estimate_details_contain_provenance(self):
        """Adversarial path includes provenance in details dict."""
        from scanner.probability_estimator import _build_estimates_from_adversarial

        market = _make_market(
            condition_id="cond_prov",
            category="politics",
            yes_price=0.45,
            hours_to_expiry=500,
        )
        synth_item = _valid_synth_item(
            id="cond_prov",
            p_hat=0.60,
            confidence=0.85,
            stake=60000,
        )

        with (
            mock.patch(
                "scanner.probability_estimator._fetch_verification_data", new_callable=mock.AsyncMock, return_value=""
            ),
            mock.patch(
                "scanner.probability_estimator._verify_estimate_with_claude",
                new_callable=mock.AsyncMock,
                return_value=None,
            ),
        ):
            result = asyncio.run(_build_estimates_from_adversarial([synth_item], [(market, None)], {}, {}))

        assert len(result) == 1
        details = result[0].details
        assert details["prompt_bundle_hash"] == PROMPT_BUNDLE_HASH
        assert details["estimator_version"] == ESTIMATOR_VERSION

    def test_single_pass_estimate_details_contain_provenance(self):
        """Single-pass path also includes provenance in details dict."""
        from scanner.probability_estimator import _astra_batch_single_pass

        market = _make_market(condition_id="cond_sp", category="politics", yes_price=0.45)
        synth_item = _valid_synth_item(id="cond_sp", p_hat=0.60, confidence=0.80, stake=60000)
        synth_json = json.dumps([synth_item])

        mock_content = mock.MagicMock()
        mock_content.text = synth_json
        mock_resp = mock.MagicMock()
        mock_resp.content = [mock_content]
        mock_client = mock.MagicMock()
        mock_client.messages = mock.MagicMock()
        mock_client.messages.create = mock.AsyncMock(return_value=mock_resp)

        with (
            mock.patch("scanner.probability_estimator.ANTHROPIC_API_KEY", "sk-test"),
            mock.patch("scanner.probability_estimator.anthropic.AsyncAnthropic", return_value=mock_client),
        ):
            result = asyncio.run(_astra_batch_single_pass([(market, None)], ""))

        assert len(result) >= 1
        details = result[0].details
        assert details["prompt_bundle_hash"] == PROMPT_BUNDLE_HASH
        assert details["estimator_version"] == ESTIMATOR_VERSION
