#!/usr/bin/env python3
"""
Quick startup test for main_maker.py runtime.

Tests that all components initialize without errors.
"""
import asyncio
import logging
from main_maker import MarketMakerRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


async def test_startup():
    """Test that runtime initializes successfully."""
    logger.info("=" * 80)
    logger.info("STARTUP TEST")
    logger.info("=" * 80)

    # Initialize runtime
    logger.info("Initializing runtime...")
    runtime = MarketMakerRuntime(clob_client=None)

    # Check components
    logger.info("Checking components...")
    assert runtime.order_store is not None, "OrderStateStore not initialized"
    assert runtime.executor is not None, "CLOBExecutor not initialized"
    assert runtime.risk_engine is not None, "PortfolioRiskEngine not initialized"
    assert runtime.markout_tracker is not None, "MarkoutTracker not initialized"
    assert runtime.event_refresher is not None, "EventRefresher not initialized"
    assert runtime.paper_simulator is not None, "PaperTradingSimulator not initialized (PAPER_MODE=true)"

    logger.info("✅ All components initialized")

    # Check order store loaded
    stats = runtime.order_store.get_stats()
    logger.info(f"Order store stats: {stats}")

    # Check risk engine
    portfolio_stats = runtime.risk_engine.get_stats()
    logger.info(f"Portfolio stats: {portfolio_stats}")

    logger.info("=" * 80)
    logger.info("✅ STARTUP TEST PASSED")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_startup())
