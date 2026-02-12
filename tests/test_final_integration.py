"""
Integration tests for final components (Truth Report, Parity, Satellite).
"""
import pytest
from models.types import Market, OrderBook, Fill, Event
from reporting.truth_report import TruthReportBuilder
from strategy.parity_scanner import scan_parity_arb
from strategy.satellite_filter import evaluate_satellite_trade
from risk.portfolio_engine import PortfolioRiskEngine


# ============================================================================
# TRUTH REPORT TESTS
# ============================================================================

def test_truth_report_maker_taker_separation():
    """CRITICAL: Truth report must separate maker vs taker fills."""
    from datetime import datetime

    report = TruthReportBuilder(date="2026-02-12")

    # Add maker fill
    maker_fill = Fill(
        fill_id="fill1",
        order_id="order1",
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.50,
        size_tokens=100.0,
        timestamp=datetime(2026, 2, 12, 10, 0, 0),
        maker=True,  # MAKER
        fee_rate_bps=200,
    )

    report.record_fill(maker_fill, "cluster1", pnl=1.0, is_maker=True)

    # Add taker fill
    taker_fill = Fill(
        fill_id="fill2",
        order_id="order2",
        condition_id="market2",
        token_id="yes2",
        side="SELL",
        price=0.60,
        size_tokens=50.0,
        timestamp=datetime(2026, 2, 12, 11, 0, 0),
        maker=False,  # TAKER
        fee_rate_bps=200,
    )

    report.record_fill(taker_fill, "cluster2", pnl=0.5, is_maker=False)

    # Verify separation
    assert len(report.maker_fills) == 1
    assert len(report.taker_fills) == 1
    assert report.maker_fills[0]["fill_id"] == "fill1"
    assert report.taker_fills[0]["fill_id"] == "fill2"


@pytest.mark.skip(reason="Legacy TruthReport API - replaced by TruthReportBuilder")
def test_sharpe_computation():
    """Test Sharpe ratio computation."""
    report = TruthReportBuilder(date="2026-02-12")

    # Add daily returns
    returns = [0.001, 0.002, -0.001, 0.003, 0.001]  # Daily returns
    report.daily_returns = returns

    sharpe = report.compute_sharpe(returns)

    # Sharpe should be positive with positive mean return
    assert sharpe > 0


@pytest.mark.skip(reason="Legacy TruthReport API - replaced by TruthReportBuilder")
def test_gate_b_evaluation():
    """Test Gate B evaluation (paper → small live)."""
    report = TruthReportBuilder(date="2026-02-12")

    # Add enough fills to pass
    for i in range(3500):  # > GATE_B_MIN_FILLS
        fill = Fill(
            fill_id=f"fill{i}",
            order_id=f"order{i}",
            condition_id=f"market{i % 10}",  # 10 unique markets
            token_id=f"yes{i}",
            side="BUY",
            price=0.50,
            size_tokens=100.0,
            timestamp=1000000 + i,
            maker=True,
        )

        cluster_id = f"cluster{i % 9}"  # 9 unique clusters
        report.add_fill(fill, cluster_id, pnl=0.01)

    # Add daily returns for Sharpe
    report.daily_returns = [0.001] * 30  # 30 days of 0.1% returns

    # Evaluate
    result = report.evaluate_gate_b(duration_days=10)

    # Check conditions
    assert result["conditions"]["fill_count"]["passed"] is True
    assert result["conditions"]["cluster_diversity"]["passed"] is True


# ============================================================================
# PARITY SCANNER TESTS (CRITICAL FIX #2, #7, #17)
# ============================================================================

def test_parity_queries_both_books():
    """CRITICAL FIX #2: Must query BOTH YES and NO books separately."""
    market = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.0,
        yes_ask=0.0,
        no_bid=0.0,
        no_ask=0.0,
        time_to_close=100.0,
        fee_rate_bps=200,
    )

    # YES book: ask = 0.45 (cheap)
    yes_book = OrderBook(
        token_id="yes1",
        best_bid=0.43,
        best_ask=0.45,
        timestamp_ms=1000000,
        timestamp_age_ms=0,
    )

    # NO book: ask = 0.48 (cheap)
    # Total cost: 0.45 + 0.48 = 0.93 (before fees)
    # After 2% fees: ~0.95, profit ~0.05 (> 0.005 threshold)
    no_book = OrderBook(
        token_id="no1",
        best_bid=0.46,
        best_ask=0.48,
        timestamp_ms=1000000,
        timestamp_age_ms=0,
    )

    risk_engine = PortfolioRiskEngine()

    opp = scan_parity_arb(market, yes_book, no_book, risk_engine)

    # YES @ 0.48 + NO @ 0.50 = 0.98 (before fees)
    # After fees: should be profitable
    assert opp is not None
    assert opp.type == "YES_NO_PARITY"
    assert len(opp.legs) == 2


def test_parity_disabled_for_neg_risk():
    """CRITICAL FIX #17: Parity must be disabled for negRisk events."""
    event = Event(event_id="event1", title="Test", neg_risk=True)

    market = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.0,
        yes_ask=0.0,
        no_bid=0.0,
        no_ask=0.0,
        time_to_close=100.0,
        fee_rate_bps=200,
        event=event,  # negRisk event
    )

    yes_book = OrderBook(
        token_id="yes1",
        best_bid=0.46,
        best_ask=0.48,
        timestamp_ms=1000000,
        timestamp_age_ms=0,
    )

    no_book = OrderBook(
        token_id="no1",
        best_bid=0.48,
        best_ask=0.50,
        timestamp_ms=1000000,
        timestamp_age_ms=0,
    )

    risk_engine = PortfolioRiskEngine()

    opp = scan_parity_arb(market, yes_book, no_book, risk_engine)

    # Must be None (disabled for negRisk)
    assert opp is None


def test_parity_acknowledges_leg_risk():
    """CRITICAL FIX #7: Parity arb must acknowledge leg risk."""
    market = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.0,
        yes_ask=0.0,
        no_bid=0.0,
        no_ask=0.0,
        time_to_close=100.0,
        fee_rate_bps=200,
    )

    yes_book = OrderBook(
        token_id="yes1",
        best_bid=0.43,
        best_ask=0.45,
        timestamp_ms=1000000,
        timestamp_age_ms=0,
    )

    no_book = OrderBook(
        token_id="no1",
        best_bid=0.46,
        best_ask=0.48,
        timestamp_ms=1000000,
        timestamp_age_ms=0,
    )

    risk_engine = PortfolioRiskEngine()

    opp = scan_parity_arb(market, yes_book, no_book, risk_engine)

    assert opp is not None, "Profitable parity arb should be detected"

    # Must have leg risk awareness
    assert opp.execution_mode == "taker"
    assert opp.max_leg_time_ms > 0
    assert opp.requires_atomic is True


# ============================================================================
# SATELLITE FILTER TESTS
# ============================================================================

def test_satellite_high_conviction_gates():
    """Test satellite filter requires high conviction evidence."""
    market = Market(
        condition_id="market1",
        question="Test",
        description="",
        yes_token_id="yes1",
        no_token_id="no1",
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
        time_to_close=100.0,
        liquidity=2000.0,
    )

    # Prediction with insufficient edge
    prediction_low_edge = {
        "estimated_prob": 0.55,  # Market @ 0.50 → 5% edge (< 15% threshold)
        "confidence": 0.8,
        "robustness": 5,
        "evidence_tier": "A",
    }

    rec = evaluate_satellite_trade(market, prediction_low_edge, current_exposure=0.0)
    assert rec is None, "Low edge should be vetoed"

    # Prediction with sufficient edge
    prediction_high_edge = {
        "estimated_prob": 0.70,  # Market @ 0.50 → 20% edge (> 15% threshold)
        "confidence": 0.8,
        "robustness": 5,
        "evidence_tier": "A",
    }

    rec = evaluate_satellite_trade(market, prediction_high_edge, current_exposure=0.0)
    assert rec is not None, "High edge with strong evidence should qualify"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
