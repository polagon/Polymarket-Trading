"""
Astra V2 — Strategy Registry

All trading strategies are BaseStrategy subclasses registered here.
Import the registry to run all strategies in a single pass.

Usage:
    from scanner.strategies import build_registry
    registry = build_registry()
    signals = registry.run_all(markets, context)
"""

from scanner.strategies.arb_strategy import ArbitrageStrategy
from scanner.strategies.base import (
    BaseStrategy,
    StrategyContext,
    StrategyMetrics,
    StrategyRegistry,
    TradeSignal,
)
from scanner.strategies.longshot_strategy import LongshotStrategy


def build_registry() -> StrategyRegistry:
    """
    Build and return the default strategy registry with all active strategies.

    To add a new strategy:
      1. Subclass BaseStrategy in scanner/strategies/
      2. Import it here
      3. Add registry.register(MyNewStrategy()) below
    """
    registry = StrategyRegistry()
    registry.register(LongshotStrategy())
    registry.register(ArbitrageStrategy())
    return registry


__all__ = [
    "BaseStrategy",
    "TradeSignal",
    "StrategyContext",
    "StrategyMetrics",
    "StrategyRegistry",
    "LongshotStrategy",
    "ArbitrageStrategy",
    "build_registry",
]
