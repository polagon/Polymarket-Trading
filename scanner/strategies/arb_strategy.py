"""
ArbitrageStrategy — BaseStrategy subclass wrapping the YES/NO arbitrage scanner.

Detects markets where YES + NO combined price < $0.97 (after fees).
Profit is guaranteed at resolution regardless of outcome.

Note: Requires live CLOB execution to capture both legs atomically.
In paper trading mode, signals are recorded but only one leg can be tracked.
"""
from typing import Optional

from scanner.market_fetcher import Market
from scanner.longshot_screener import scan_for_arbitrage
from scanner.strategies.base import BaseStrategy, TradeSignal, StrategyContext


class ArbitrageStrategy(BaseStrategy):
    """
    Guaranteed-profit strategy when YES + NO < $0.97.

    Filters:
      - Combined price < 0.97 (3¢ margin for fees/slippage)
      - Liquidity ≥ $2,000
      - Resolution within 7 days (metaggdev finding: long-dated arb loses guaranteed edge)

    Edge type: structural (maths, not prediction) — profit guaranteed at resolution.
    """

    def __init__(self):
        super().__init__(name="arbitrage")

    def evaluate(
        self,
        market: Market,
        context: StrategyContext,
    ) -> Optional[TradeSignal]:
        """
        Check a single market for arbitrage opportunity.
        Returns a BUY YES TradeSignal (YES leg) if arb found, else None.

        Note: True arb requires executing both legs. This signal represents
        the YES leg; the NO leg would need to be submitted simultaneously via CLOB.
        """
        signals = scan_for_arbitrage([market])
        if not signals:
            return None

        sig = signals[0]

        # For paper trading, signal the YES leg (the cheaper of the two)
        if sig.yes_price <= sig.no_price:
            direction = "BUY YES"
            entry_price = sig.yes_price
        else:
            direction = "BUY NO"
            entry_price = sig.no_price

        # Kelly for arb: edge = guaranteed profit %, risk = 0 (guaranteed), so
        # use a conservative fixed fractional approach capped at 2%
        kelly = min(0.02, sig.guaranteed_profit_pct * 0.5)

        reasoning = (
            f"Arbitrage: YES={sig.yes_price:.3f} + NO={sig.no_price:.3f} "
            f"= {sig.combined_price:.3f} (< 0.97). "
            f"Guaranteed profit ≈ {sig.guaranteed_profit_pct:.1%} after fees. "
            f"Min position: ${sig.min_position_usd:.0f}. "
            f"⚠ Needs both legs via CLOB for guaranteed profit."
        )

        return TradeSignal(
            condition_id=market.condition_id,
            question=market.question,
            direction=direction,
            confidence=0.90,   # High confidence — maths, not prediction
            target_price=entry_price,
            kelly_pct=round(kelly, 4),
            edge=1.0 - sig.combined_price,   # The spread is the edge
            ev_after_costs=sig.guaranteed_profit_pct,
            source=self.name,
            reasoning=reasoning,
        )

    def evaluate_batch(
        self,
        markets: list[Market],
        context: StrategyContext,
    ) -> list[TradeSignal]:
        """
        Vectorised batch evaluation — uses arb scanner's native batch path.
        """
        if not self._enabled:
            return []

        arb_signals = scan_for_arbitrage(markets)
        result = []

        for sig in arb_signals:
            self.metrics.total_signals += 1

            if sig.yes_price <= sig.no_price:
                direction = "BUY YES"
                entry_price = sig.yes_price
            else:
                direction = "BUY NO"
                entry_price = sig.no_price

            kelly = min(0.02, sig.guaranteed_profit_pct * 0.5)

            reasoning = (
                f"Arbitrage: YES={sig.yes_price:.3f} + NO={sig.no_price:.3f} "
                f"= {sig.combined_price:.3f}. "
                f"Profit ≈ {sig.guaranteed_profit_pct:.1%} after fees. "
                f"Min: ${sig.min_position_usd:.0f}. ⚠ Needs CLOB for both legs."
            )

            ts = TradeSignal(
                condition_id=sig.market.condition_id,
                question=sig.market.question,
                direction=direction,
                confidence=0.90,
                target_price=entry_price,
                kelly_pct=round(kelly, 4),
                edge=1.0 - sig.combined_price,
                ev_after_costs=sig.guaranteed_profit_pct,
                source=self.name,
                reasoning=reasoning,
            )
            if ts.is_actionable:
                result.append(ts)

        return result
