"""
LongshotStrategy — BaseStrategy subclass wrapping the longshot bias screener.

Uses the structural edge from Polymarket microstructure research:
  - Markets priced 1–8¢ YES systematically overpriced by 22–57%
  - BUY NO signals with consistent positive EV across all categories

Adapts LongshotSignal → TradeSignal for the unified strategy registry.
"""

from typing import Optional

from scanner.longshot_screener import screen_longshot_markets
from scanner.market_fetcher import Market
from scanner.strategies.base import BaseStrategy, StrategyContext, TradeSignal


class LongshotStrategy(BaseStrategy):
    """
    Systematic BUY NO strategy for markets priced 1–8¢ YES.

    Edge source: structural longshot bias (market overprices YES at low prices).
    Not dependent on knowing the actual outcome — pure microstructure edge.
    """

    def __init__(self):
        super().__init__(name="longshot")

    def evaluate(
        self,
        market: Market,
        context: StrategyContext,
    ) -> Optional[TradeSignal]:
        """
        Screen a single market for longshot bias opportunity.
        Returns a BUY NO TradeSignal if criteria met, else None.
        """
        # Re-use existing screener logic by passing a single-item list
        signals = screen_longshot_markets([market])
        if not signals:
            return None

        sig = signals[0]

        # Apply VIX Kelly multiplier from context
        kelly = sig.kelly_pct * context.vix_kelly_mult

        return TradeSignal(
            condition_id=market.condition_id,
            question=market.question,
            direction="BUY NO",
            confidence=sig.confidence,
            target_price=sig.no_price,
            kelly_pct=round(kelly, 4),
            edge=sig.structural_edge,
            ev_after_costs=sig.ev_after_costs,
            source=self.name,
            reasoning=sig.reasoning,
        )

    def evaluate_batch(
        self,
        markets: list[Market],
        context: StrategyContext,
    ) -> list[TradeSignal]:
        """
        Vectorised batch evaluation — more efficient than calling evaluate() per market.
        Uses the screener's native batch path then converts to TradeSignals.
        """
        if not self._enabled:
            return []

        signals = screen_longshot_markets(markets)
        result = []

        for sig in signals:
            self.metrics.total_signals += 1
            kelly = sig.kelly_pct * context.vix_kelly_mult
            ts = TradeSignal(
                condition_id=sig.market.condition_id,
                question=sig.market.question,
                direction="BUY NO",
                confidence=sig.confidence,
                target_price=sig.no_price,
                kelly_pct=round(kelly, 4),
                edge=sig.structural_edge,
                ev_after_costs=sig.ev_after_costs,
                source=self.name,
                reasoning=sig.reasoning,
            )
            if ts.is_actionable:
                result.append(ts)

        return result
