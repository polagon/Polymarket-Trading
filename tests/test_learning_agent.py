"""
Tests for Astra V2 Learning Agent — scanner/learning_agent.py

Deterministic. No network calls. No real Anthropic API usage.
All file I/O uses tmp_path to avoid touching real memory/ directory.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.learning_agent import (
    CalibrationBucket,
    LearningAgent,
    Prediction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_prediction(**kw) -> Prediction:
    defaults = dict(
        market_condition_id="cond_1",
        question="Will X happen?",
        category="politics",
        our_probability=0.70,
        probability_low=0.60,
        probability_high=0.80,
        market_price=0.50,
        direction="BUY YES",
        source="astra_v2",
        truth_state="Supported",
        reasoning="Test prediction",
        key_unknowns="None",
        no_trade=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    defaults.update(kw)
    return Prediction(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """Create a LearningAgent with file paths pointing to tmp_path."""
    memory_file = tmp_path / "predictions.json"
    strategy_file = tmp_path / "strategy.md"
    journal_file = tmp_path / "journal.json"
    overrides_file = tmp_path / "strategy_overrides.json"

    monkeypatch.setattr("scanner.learning_agent.MEMORY_FILE", memory_file)
    monkeypatch.setattr("scanner.learning_agent.STRATEGY_FILE", strategy_file)
    monkeypatch.setattr("scanner.learning_agent.JOURNAL_FILE", journal_file)
    monkeypatch.setattr("scanner.learning_agent.OVERRIDES_FILE", overrides_file)

    return LearningAgent()


# ═══════════════════════════════════════════════════════════════════════════
# 1) TestPredictionLifecycle
# ═══════════════════════════════════════════════════════════════════════════
class TestPredictionLifecycle:
    def test_record_prediction_adds_entry(self, agent):
        pred = _make_prediction()
        agent.record_prediction(pred)
        assert len(agent._predictions) == 1
        assert agent._predictions[0].market_condition_id == "cond_1"

    def test_duplicate_condition_id_ignored(self, agent):
        pred1 = _make_prediction(market_condition_id="cond_dup")
        pred2 = _make_prediction(market_condition_id="cond_dup", our_probability=0.80)
        agent.record_prediction(pred1)
        agent.record_prediction(pred2)
        assert len(agent._predictions) == 1
        # First one kept
        assert agent._predictions[0].our_probability == 0.70

    def test_update_outcome_true_computes_brier(self, agent):
        pred = _make_prediction(our_probability=0.70, market_condition_id="cond_true")
        agent.record_prediction(pred)
        agent.update_outcome("cond_true", outcome=True, profit_loss=10.0)
        p = agent._predictions[0]
        assert p.resolved is True
        assert p.outcome is True
        # Brier = (0.70 - 1.0)^2 = 0.09
        assert p.brier_score == pytest.approx(0.09, abs=1e-6)

    def test_update_outcome_false_computes_brier(self, agent):
        pred = _make_prediction(our_probability=0.70, market_condition_id="cond_false")
        agent.record_prediction(pred)
        agent.update_outcome("cond_false", outcome=False, profit_loss=-10.0)
        p = agent._predictions[0]
        assert p.resolved is True
        assert p.outcome is False
        # Brier = (0.70 - 0.0)^2 = 0.49
        assert p.brier_score == pytest.approx(0.49, abs=1e-6)

    def test_update_already_resolved_ignored(self, agent):
        pred = _make_prediction(market_condition_id="cond_resolved")
        agent.record_prediction(pred)
        agent.update_outcome("cond_resolved", outcome=True, profit_loss=10.0)
        original_brier = agent._predictions[0].brier_score
        # Second update should be ignored
        agent.update_outcome("cond_resolved", outcome=False, profit_loss=-10.0)
        assert agent._predictions[0].brier_score == original_brier
        assert agent._predictions[0].outcome is True


# ═══════════════════════════════════════════════════════════════════════════
# 2) TestCalibrationBuckets
# ═══════════════════════════════════════════════════════════════════════════
class TestCalibrationBuckets:
    def test_insufficient_data_returns_empty(self, agent):
        # Less than 10 resolved predictions
        for i in range(5):
            pred = _make_prediction(market_condition_id=f"cond_{i}", our_probability=0.5 + i * 0.05)
            agent.record_prediction(pred)
            agent.update_outcome(f"cond_{i}", outcome=(i % 2 == 0))
        buckets = agent.calibration_buckets()
        assert buckets == []

    def test_sufficient_data_returns_buckets(self, agent):
        # 15 resolved predictions spanning multiple buckets
        for i in range(15):
            p = 0.10 + (i / 15) * 0.80  # 0.10 to 0.90
            pred = _make_prediction(
                market_condition_id=f"cond_{i}",
                our_probability=round(p, 3),
            )
            agent.record_prediction(pred)
            agent.update_outcome(f"cond_{i}", outcome=(i % 3 != 0))
        buckets = agent.calibration_buckets()
        assert len(buckets) > 0
        for b in buckets:
            assert isinstance(b, CalibrationBucket)
            assert b.n > 0

    def test_bucket_bias_computation(self, agent):
        """With known data, verify bias = actual_rate - predicted_avg."""
        # All predictions at 0.70 → bucket "0.7-0.8"
        # If 6 out of 10 resolve True, actual_rate = 0.60
        # bias = 0.60 - 0.70 = -0.10 (overconfident)
        for i in range(12):
            pred = _make_prediction(
                market_condition_id=f"cal_{i}",
                our_probability=0.70,
            )
            agent.record_prediction(pred)
            agent.update_outcome(f"cal_{i}", outcome=(i < 6))

        buckets = agent.calibration_buckets()
        b70 = [b for b in buckets if "0.7" in b.bucket]
        assert len(b70) == 1
        assert b70[0].actual_rate == pytest.approx(6 / 12, abs=0.01)
        assert b70[0].bias == pytest.approx(6 / 12 - 0.70, abs=0.01)

    def test_brier_avg_computation(self, agent):
        """Brier average should match manual calculation."""
        # prediction=0.80 resolved True → brier = (0.8-1.0)^2 = 0.04
        # prediction=0.80 resolved False → brier = (0.8-0.0)^2 = 0.64
        for i in range(10):
            pred = _make_prediction(
                market_condition_id=f"brier_{i}",
                our_probability=0.80,
            )
            agent.record_prediction(pred)
            outcome = i < 5  # 5 True, 5 False
            agent.update_outcome(f"brier_{i}", outcome=outcome)

        buckets = agent.calibration_buckets()
        b80 = [b for b in buckets if "0.8" in b.bucket]
        assert len(b80) == 1
        # Expected: avg of 5 * 0.04 + 5 * 0.64 = 3.40 / 10 = 0.34
        assert b80[0].brier_avg == pytest.approx(0.34, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# 3) TestStats
# ═══════════════════════════════════════════════════════════════════════════
class TestStats:
    def test_no_resolved_returns_none_fields(self, agent):
        pred = _make_prediction()
        agent.record_prediction(pred)
        stats = agent.get_stats()
        assert stats["resolved"] == 0
        assert stats["accuracy"] is None
        assert stats["brier_score_avg"] is None

    def test_resolved_stats_correct(self, agent):
        # 3 resolved: 2 correct (BUY YES + True, BUY NO + False), 1 wrong
        pred1 = _make_prediction(market_condition_id="s1", direction="BUY YES", our_probability=0.70)
        pred2 = _make_prediction(market_condition_id="s2", direction="BUY NO", our_probability=0.30)
        pred3 = _make_prediction(market_condition_id="s3", direction="BUY YES", our_probability=0.80)
        agent.record_prediction(pred1)
        agent.record_prediction(pred2)
        agent.record_prediction(pred3)
        agent.update_outcome("s1", outcome=True, profit_loss=10.0)
        agent.update_outcome("s2", outcome=False, profit_loss=5.0)
        agent.update_outcome("s3", outcome=False, profit_loss=-15.0)  # Wrong

        stats = agent.get_stats()
        assert stats["resolved"] == 3
        assert stats["accuracy"] == pytest.approx(2 / 3, abs=0.01)
        assert stats["brier_score_avg"] is not None
        assert stats["total_pnl"] == pytest.approx(0.0, abs=0.01)

    def test_stats_by_category(self, agent):
        pred1 = _make_prediction(market_condition_id="cat1", category="sports", our_probability=0.60)
        pred2 = _make_prediction(market_condition_id="cat2", category="politics", our_probability=0.70)
        agent.record_prediction(pred1)
        agent.record_prediction(pred2)
        agent.update_outcome("cat1", outcome=True)
        agent.update_outcome("cat2", outcome=True)

        stats = agent.get_stats()
        by_cat = stats["by_category"]
        assert "sports" in by_cat
        assert "politics" in by_cat
        assert by_cat["sports"]["n"] == 1
        assert by_cat["politics"]["n"] == 1

    def test_stats_by_source(self, agent):
        pred1 = _make_prediction(market_condition_id="src1", source="astra_v2")
        pred2 = _make_prediction(market_condition_id="src2", source="crypto_lognormal")
        agent.record_prediction(pred1)
        agent.record_prediction(pred2)
        agent.update_outcome("src1", outcome=True)
        agent.update_outcome("src2", outcome=False)

        stats = agent.get_stats()
        by_src = stats["by_source"]
        assert "astra_v2" in by_src
        assert "crypto_lognormal" in by_src
        assert by_src["astra_v2"]["n"] == 1
        assert by_src["crypto_lognormal"]["n"] == 1
