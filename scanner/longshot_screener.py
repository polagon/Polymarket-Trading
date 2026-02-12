"""
Astra V2 — Longshot Bias Screener + YES/NO Arbitrage Scanner

Based on empirical microstructure research (jbecker.dev):
  - 1¢ YES contracts win only 0.43% of the time (implied: 1.0%)
  - 1¢ YES EV: -41% | 1¢ NO EV: +23% — massive asymmetry
  - Makers earn +1.12% per trade vs takers -1.12% systematically
  - Optimism Tax: takers disproportionately buy YES longshots

Category-specific maker-taker gaps:
  Entertainment: 4.79pp | World Events: 7.32pp | Sports: 2.23pp | Finance: 0.17pp

Strategy:
  1. LONGSHOT SCREENER: Scan markets priced 1-8¢. Flag BUY NO signals.
     Edge is systematic (market overprices YES longshots by 57%).

  2. ARBITRAGE SCANNER: If YES_price + NO_price < 0.97 (after fees),
     guaranteed profit regardless of outcome.

Also based on: CyberK "AI Agents Making Millions on Polymarket"
  "If an agent can acquire both sides at a combined cost below $1,
   profit becomes guaranteed regardless of the outcome."
"""
from dataclasses import dataclass
from typing import Optional

from scanner.market_fetcher import Market
from config import MIN_MARKET_LIQUIDITY


# Longshot thresholds (from microstructure research)
LONGSHOT_MAX_YES_PRICE = 0.08    # Markets priced below 8¢ YES
LONGSHOT_MIN_YES_PRICE = 0.01    # Avoid dust markets
LONGSHOT_BASE_EDGE = 0.025       # Conservative: even 2.5% structural edge at 5¢
LONGSHOT_MIN_LIQUIDITY = 1000    # Need at least $1k to have meaningful order

# Arbitrage threshold
ARB_MAX_COMBINED = 0.97          # YES + NO combined price (0.97 leaves 3¢ for fees/slippage)
ARB_MIN_LIQUIDITY = 2000         # Need both sides liquid

# Category-specific confidence multipliers (from research)
CATEGORY_CONFIDENCE = {
    "entertainment": 0.80,
    "other": 0.72,       # "world events" proxy
    "sports": 0.68,
    "weather": 0.65,
    "crypto": 0.55,
    "politics": 0.60,    # explicitly lower — institutional entrants closing gaps
}


@dataclass
class LongshotSignal:
    """A BUY NO signal from the longshot bias screener."""
    market: Market
    yes_price: float          # e.g. 0.04 (4¢)
    no_price: float           # e.g. 0.96
    structural_edge: float    # Expected edge from longshot bias (positive = BUY NO)
    ev_after_costs: float
    robustness: int           # 1-5
    confidence: float
    reasoning: str
    source: str = "longshot_screener"

    @property
    def direction(self) -> str:
        return "BUY NO"

    @property
    def kelly_pct(self) -> float:
        """
        Kelly sizing for BUY NO position.

        We buy NO at price market_no_price = 1 - yes_price.
        If NO wins: receive $1.00 per share (net gain = 1 - market_no_price = yes_price)
        If YES wins: lose market_no_price.

        Kelly formula: f* = p - (1-p) / b
          where b = net odds = yes_price / (1 - yes_price) = yes / no
                p = our estimated probability that NO wins (1 - true_yes)
        """
        market_no_price = 1.0 - self.yes_price
        if market_no_price <= 0 or market_no_price >= 1:
            return 0.0
        # True YES probability after accounting for longshot overpricing
        true_yes = self.yes_price - self.structural_edge
        our_p_no = max(0.0, min(0.999, 1.0 - true_yes))
        # Net odds for BUY NO: win yes_price per dollar risked (market_no_price)
        # b = net gain per unit bet = yes_price / market_no_price
        b = self.yes_price / market_no_price
        if b <= 0:
            return 0.0
        kelly_raw = our_p_no - (1.0 - our_p_no) / b
        kelly_adj = kelly_raw * self.confidence * (self.robustness / 5)
        return round(min(0.015, max(0.0, kelly_adj)), 4)  # cap at 1.5% — conservative for NO bets


@dataclass
class ArbitrageSignal:
    """A guaranteed-profit signal from YES+NO combined price < $1."""
    market: Market
    yes_price: float
    no_price: float
    combined_price: float     # yes + no
    guaranteed_profit_pct: float  # (1 - combined) / combined
    min_position_usd: float   # Minimum to make meaningful profit after fees
    source: str = "arbitrage_scanner"


def screen_longshot_markets(markets: list[Market]) -> list[LongshotSignal]:
    """
    Screen for markets exhibiting longshot bias opportunity.

    Returns BUY NO signals for markets where YES is priced 1-8¢.
    Edge is structural (market systematically overprices YES at these levels)
    NOT dependent on knowing the actual outcome.
    """
    signals = []

    for market in markets:
        yes = market.yes_price
        no = market.no_price

        # Filter: correct price range for longshot bias
        if yes < LONGSHOT_MIN_YES_PRICE or yes > LONGSHOT_MAX_YES_PRICE:
            continue

        # Filter: need real liquidity
        if market.liquidity < LONGSHOT_MIN_LIQUIDITY:
            continue

        # Filter: needs both token IDs (can actually trade NO)
        if not market.no_token_id:
            continue

        # Category confidence
        cat = market.category.lower() if market.category else "other"
        conf_base = CATEGORY_CONFIDENCE.get(cat, 0.65)

        # Structural edge calculation (from microstructure data):
        # At 1¢: actual win rate ≈ 0.43% vs implied 1.0% → 57% overpriced
        # At 5¢: actual win rate ≈ 3.5% vs implied 5% → 30% overpriced
        # At 8¢: actual win rate ≈ 6.2% vs implied 8% → 22% overpriced
        # Linear interpolation:
        overpricing_factor = 0.22 + (LONGSHOT_MAX_YES_PRICE - yes) / LONGSHOT_MAX_YES_PRICE * 0.35
        true_yes_probability = yes * (1 - overpricing_factor)
        structural_edge = yes - true_yes_probability  # How much YES is overpriced

        ev = structural_edge - 0.005   # Subtract fees

        if ev <= 0:
            continue

        # Robustness: how robust is this edge to estimation error?
        # The longshot bias is consistent — even if we're off, direction is clear
        if yes <= 0.03:
            rob = 4   # Very clear longshot — overpricing strong
        elif yes <= 0.05:
            rob = 4
        else:
            rob = 3   # 6-8¢ — still significant but less extreme

        # Extra confidence boost for high-liquidity longshots (more toxic flow = more overpricing)
        if market.liquidity > 50000:
            conf_base = min(conf_base + 0.08, 0.88)

        reasoning = (
            f"Longshot bias: YES@{yes:.2%} historically overpriced. "
            f"True p(YES) ≈ {true_yes_probability:.2%} (market implies {yes:.2%}). "
            f"Structural edge = {structural_edge:.2%} on BUY NO. "
            f"Category: {cat} ({conf_base:.0%} conf tier)."
        )

        signals.append(LongshotSignal(
            market=market,
            yes_price=yes,
            no_price=no or (1 - yes),
            structural_edge=structural_edge,
            ev_after_costs=round(ev, 4),
            robustness=rob,
            confidence=conf_base,
            reasoning=reasoning,
        ))

    # Sort by EV * confidence (best opportunities first)
    return sorted(signals, key=lambda s: s.ev_after_costs * s.confidence, reverse=True)


def scan_for_arbitrage(markets: list[Market]) -> list[ArbitrageSignal]:
    """
    Scan for markets where YES + NO prices sum to less than 1.0 (minus fees).

    This should theoretically never happen in an efficient market.
    In practice it occurs on:
    - Newly created thin markets (wide spreads, no arbitrageurs yet)
    - Markets with ambiguous resolution criteria (liquidity flight)
    - API lag / price update delays

    When both sides are acquired at combined cost < $1, profit is GUARANTEED
    at resolution regardless of outcome.

    Note: This only works with actual order execution via CLOB API.
    In paper trading mode, this signals the opportunity but cannot execute both legs.
    """
    signals = []

    for market in markets:
        yes = market.yes_price
        no = market.no_price

        if yes <= 0 or no <= 0:
            continue
        if market.liquidity < ARB_MIN_LIQUIDITY:
            continue
        # Skip long-dated markets — arb edge erodes with time (metaggdev finding)
        # Long-dated combined price can stay <1 for days with no resolution in sight
        if market.hours_to_expiry > 168:   # 7 days
            continue

        combined = yes + no

        if combined >= ARB_MAX_COMBINED:
            continue

        # Guaranteed profit = (1 - combined) / combined
        gross_profit_pct = (1.0 - combined) / combined
        # After fees (approximately 0.5-1% per leg)
        net_profit_pct = gross_profit_pct - 0.01

        if net_profit_pct <= 0:
            continue

        # Min position to make it worth executing (>$5 profit after gas/fees)
        min_position = max(50.0, 5.0 / net_profit_pct)

        signals.append(ArbitrageSignal(
            market=market,
            yes_price=yes,
            no_price=no,
            combined_price=combined,
            guaranteed_profit_pct=round(net_profit_pct, 4),
            min_position_usd=round(min_position, 2),
        ))

    # Best arbitrage first (highest guaranteed profit %)
    return sorted(signals, key=lambda s: s.guaranteed_profit_pct, reverse=True)


def summarize_longshot_stats(signals: list[LongshotSignal]) -> str:
    """Quick summary of longshot scan results for the terminal report."""
    if not signals:
        return ""
    top = signals[0]
    return (
        f"[dim]Longshot screener: {len(signals)} BUY NO signals | "
        f"Top: {top.market.question[:40]}... "
        f"YES@{top.yes_price:.1%} EV={top.ev_after_costs:+.3f}[/dim]"
    )
