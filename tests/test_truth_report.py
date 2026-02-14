"""
Tests for Truth Report writer (daily JSON output).

Validates:
- Maker/taker separation
- Atomic write behavior
- Schema correctness
- Metric computation
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from models.types import Fill
from reporting.truth_report import TruthReportBuilder, write_daily_report


@pytest.fixture
def truth_builder():
    """Create TruthReportBuilder for testing."""
    return TruthReportBuilder(date="2026-02-12")


@pytest.fixture
def sample_fill():
    """Create sample Fill for testing."""
    return Fill(
        fill_id="fill_123",
        order_id="order_456",
        condition_id="condition_789",
        token_id="token_yes",
        side="BUY",
        price=0.55,
        size_tokens=100.0,
        timestamp=datetime(2026, 2, 12, 10, 30, 0),  # type: ignore[arg-type]
        maker=True,
        fee_rate_bps=200,
    )


def test_record_maker_fill(truth_builder, sample_fill):
    """Test recording maker fill."""
    truth_builder.record_fill(
        fill=sample_fill,
        cluster_id="cluster_1",
        pnl=5.0,
        is_maker=True,
        realized_spread=0.01,
        markout_2m=0.005,
    )

    assert len(truth_builder.maker_fills) == 1
    assert len(truth_builder.taker_fills) == 0
    assert truth_builder.maker_fills[0]["pnl"] == 5.0
    assert truth_builder.maker_fills[0]["realized_spread"] == 0.01
    assert truth_builder.cluster_pnl["cluster_1"] == 5.0


def test_record_taker_fill(truth_builder, sample_fill):
    """Test recording taker fill."""
    sample_fill.maker = False

    truth_builder.record_fill(
        fill=sample_fill,
        cluster_id="cluster_1",
        pnl=-2.0,
        is_maker=False,
    )

    assert len(truth_builder.maker_fills) == 0
    assert len(truth_builder.taker_fills) == 1
    assert truth_builder.taker_fills[0]["pnl"] == -2.0


def test_maker_taker_separation(truth_builder, sample_fill):
    """Test maker vs taker separation (CRITICAL)."""
    # Maker fill
    truth_builder.record_fill(
        fill=sample_fill,
        cluster_id="cluster_1",
        pnl=5.0,
        is_maker=True,
        realized_spread=0.01,
    )

    # Taker fill
    sample_fill.fill_id = "fill_124"
    sample_fill.maker = False
    truth_builder.record_fill(
        fill=sample_fill,
        cluster_id="cluster_1",
        pnl=-2.0,
        is_maker=False,
    )

    # Verify separation
    assert len(truth_builder.maker_fills) == 1
    assert len(truth_builder.taker_fills) == 1

    # Finalize and check attribution
    report = truth_builder.finalize()
    assert report["component_attribution"]["maker_pnl"] == 5.0
    assert report["component_attribution"]["taker_pnl"] == -2.0
    assert report["component_attribution"]["total_pnl"] == 3.0


def test_quote_events(truth_builder):
    """Test quote event tracking."""
    truth_builder.record_quote_event("quote", count=10)
    truth_builder.record_quote_event("cancel", count=3)
    truth_builder.record_quote_event("replace", count=2)

    assert truth_builder.quote_count == 10
    assert truth_builder.cancel_count == 3
    assert truth_builder.replace_count == 2


def test_health_metrics(truth_builder):
    """Test health metrics tracking."""
    truth_builder.record_health(
        ws_market_uptime_seconds=86000.0,
        ws_user_uptime_seconds=85000.0,
        cancel_all_trigger=True,
        reconciliation_error=False,
        toxic_market_pause=True,
    )

    assert truth_builder.ws_market_uptime_seconds == 86000.0
    assert truth_builder.cancel_all_triggers == 1
    assert truth_builder.toxic_market_pauses == 1

    # Check uptime ratios in finalized report
    report = truth_builder.finalize()
    assert report["health_metrics"]["ws_market_uptime"] > 0.99
    assert report["health_metrics"]["cancel_all_triggers"] == 1


def test_portfolio_snapshot(truth_builder):
    """Test portfolio snapshot."""
    truth_builder.set_portfolio_snapshot(
        daily_return=0.012,
        weekly_return=0.045,
        monthly_return=0.18,
        sharpe_90d=2.1,
        calmar_90d=2.4,
        max_drawdown=-0.08,
        cluster_exposures={"cluster_1": 0.12, "cluster_2": 0.08},
        aggregate_exposure=0.35,
        max_market_inventory=45.0,
    )

    report = truth_builder.finalize()
    assert report["portfolio_metrics"]["sharpe_90d"] == 2.1
    assert report["portfolio_metrics"]["cluster_exposures"]["cluster_1"] == 0.12


def test_maker_truth_metrics_computation(truth_builder, sample_fill):
    """Test maker truth metrics computation."""
    # Record multiple maker fills with varying metrics
    for i in range(5):
        truth_builder.record_fill(
            fill=sample_fill,
            cluster_id="cluster_1",
            pnl=5.0 + i,
            is_maker=True,
            realized_spread=0.01,
            markout_2m=0.005,
        )
        sample_fill.fill_id = f"fill_{i}"

    # Record quote events
    truth_builder.record_quote_event("quote", count=100)
    truth_builder.record_quote_event("cancel", count=20)

    report = truth_builder.finalize()
    maker_metrics = report["maker_truth_metrics"]

    # Verify metrics
    assert maker_metrics["quote_count"] == 100
    assert maker_metrics["cancel_count"] == 20
    assert maker_metrics["fill_count"] == 5
    assert maker_metrics["realized_spread_bps"] == 100.0  # 0.01 * 10000
    assert maker_metrics["markout_2m_bps"] == 50.0  # 0.005 * 10000
    assert maker_metrics["fill_rate_per_quote"] == 0.05  # 5 fills / 100 quotes


def test_empty_report(truth_builder):
    """Test finalize with no fills."""
    report = truth_builder.finalize()

    assert report["date"] == "2026-02-12"
    assert report["component_attribution"]["total_pnl"] == 0.0
    assert report["maker_truth_metrics"]["fill_count"] == 0
    assert report["maker_truth_metrics"]["realized_spread_bps"] == 0.0


def test_atomic_write():
    """Test atomic write behavior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir) / "reports"

        # Create sample report
        builder = TruthReportBuilder(date="2026-02-12")
        builder.set_portfolio_snapshot(
            daily_return=0.012,
            weekly_return=0.045,
            monthly_return=0.18,
            sharpe_90d=2.1,
            calmar_90d=2.4,
            max_drawdown=-0.08,
            cluster_exposures={},
            aggregate_exposure=0.35,
            max_market_inventory=45.0,
        )
        report = builder.finalize()

        # Write report
        output_path = write_daily_report(report, reports_dir=reports_dir)

        # Verify file exists
        assert output_path.exists()
        assert output_path.name == "2026-02-12.json"

        # Verify contents
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["date"] == "2026-02-12"
        assert loaded["portfolio_metrics"]["sharpe_90d"] == 2.1


def test_schema_correctness():
    """Test that finalized report matches schema."""
    builder = TruthReportBuilder(date="2026-02-12")

    # Add some data
    sample_fill = Fill(
        fill_id="fill_123",
        order_id="order_456",
        condition_id="condition_789",
        token_id="token_yes",
        side="BUY",
        price=0.55,
        size_tokens=100.0,
        timestamp=datetime(2026, 2, 12, 10, 30, 0),  # type: ignore[arg-type]
        maker=True,
        fee_rate_bps=200,
    )

    builder.record_fill(
        fill=sample_fill,
        cluster_id="cluster_1",
        pnl=5.0,
        is_maker=True,
        realized_spread=0.01,
    )

    builder.record_quote_event("quote", count=10)
    builder.record_health(cancel_all_trigger=True)

    builder.set_portfolio_snapshot(
        daily_return=0.012,
        weekly_return=0.045,
        monthly_return=0.18,
        sharpe_90d=2.1,
        calmar_90d=2.4,
        max_drawdown=-0.08,
        cluster_exposures={"cluster_1": 0.12},
        aggregate_exposure=0.35,
        max_market_inventory=45.0,
    )

    report = builder.finalize()

    # Verify schema structure
    required_keys = [
        "date",
        "portfolio_metrics",
        "component_attribution",
        "risk_metrics",
        "gate_outcomes",
        "health_metrics",
        "maker_truth_metrics",
    ]

    for key in required_keys:
        assert key in report, f"Missing required key: {key}"

    # Verify maker_truth_metrics structure
    maker_metrics = report["maker_truth_metrics"]
    required_maker_keys = [
        "quote_count",
        "cancel_count",
        "fill_count",
        "realized_spread_bps",
        "markout_2m_bps",
        "fill_rate_per_quote",
    ]

    for key in required_maker_keys:
        assert key in maker_metrics, f"Missing maker metric: {key}"


def test_multiple_clusters(truth_builder, sample_fill):
    """Test tracking multiple clusters."""
    # Cluster 1 fills
    truth_builder.record_fill(
        fill=sample_fill,
        cluster_id="cluster_1",
        pnl=5.0,
        is_maker=True,
    )

    # Cluster 2 fills
    sample_fill.fill_id = "fill_124"
    truth_builder.record_fill(
        fill=sample_fill,
        cluster_id="cluster_2",
        pnl=3.0,
        is_maker=True,
    )

    report = truth_builder.finalize()

    # Verify cluster P&L
    assert report["risk_metrics"]["cluster_pnl"]["cluster_1"] == 5.0
    assert report["risk_metrics"]["cluster_pnl"]["cluster_2"] == 3.0
