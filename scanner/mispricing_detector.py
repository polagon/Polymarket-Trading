"""
Astra V2 — Opportunity Detector

Applies Astra's Module 3 minimum bar to flag tradeable mispricings:
  - EV after costs > 0
  - Robustness score ≥ 3
  - Confidence ≥ 0.60
  - No trap flags that triggered no_trade
  - Not marked no_trade by Astra
"""

from dataclasses import dataclass

from config import MIN_CONFIDENCE, MIN_MARKET_LIQUIDITY, MISPRICING_THRESHOLD
from scanner.market_fetcher import Market
from scanner.probability_estimator import Estimate


def _apply_longshot_calibration(our_estimate: float, market_price: float) -> float:
    """
    Favorite-longshot bias calibration correction (arXiv:1811.12516, arXiv:2010.12508).

    Prediction markets structurally overprice longshots (<10% YES) and may
    underprice strong favorites (>85% YES). This bias persists indefinitely
    because it is theoretically grounded in rational equilibrium with noisy beliefs.

    For markets priced below 5%:   shade down 28%   (extreme longshot region)
    For markets priced below 10%:  shade down 20%   (strong longshot region)
    For markets above 92%:         shade up   6%    (strong favorite region)
    For markets above 85%:         shade up   4%    (favorite region)
    For everything else:           no adjustment
    """
    if market_price < 0.05:
        return our_estimate * 0.72
    elif market_price < 0.10:
        return our_estimate * 0.80
    elif market_price > 0.92:
        return min(0.99, our_estimate * 1.06)
    elif market_price > 0.85:
        return min(0.99, our_estimate * 1.04)
    return our_estimate


@dataclass
class Opportunity:
    market: Market
    estimate: Estimate
    market_price: float  # Current YES price (0-1)
    our_estimate: float  # Astra's p_hat (0-1)
    edge: float  # p_hat - market_implied (signed)
    direction: str  # "BUY YES" or "BUY NO"
    ev_after_costs: float  # Expected value after fees/slippage
    robustness_score: int  # 1-5 (Astra V2 Module 3)
    score: float  # abs(edge) * confidence * (robustness/5) — sorting key
    kelly_pct: float  # Suggested position as % of bankroll


def find_opportunities(
    markets: list[Market],
    estimates: list[Estimate],
    whale_signals: list | None = None,
) -> list[Opportunity]:
    """
    Astra V2 opportunity filter.

    Only surfaces markets that pass ALL of:
    1. Astra didn't flag no_trade
    2. EV after costs > 0
    3. Robustness ≥ 3
    4. Confidence ≥ MIN_CONFIDENCE
    5. Liquidity ≥ MIN_MARKET_LIQUIDITY
    6. |edge| ≥ MISPRICING_THRESHOLD

    whale_signals: list of WhaleSignal objects from whale_tracker.
    Markets with active whale volume spikes get a score boost (Tier B signal).
    """
    # Build whale signal lookup by market condition_id
    # WhaleSignal has .condition_id and .sigma (standard deviations above baseline)
    whale_boost: dict[str, float] = {}
    if whale_signals:
        for ws in whale_signals:
            cid = getattr(ws, "condition_id", None)
            sigma = getattr(ws, "sigma", 0.0)
            if cid and sigma > 2.0:
                # >2σ volume spike: boost score by up to 20% (capped)
                whale_boost[cid] = min(0.20, (sigma - 2.0) * 0.05)
    estimate_map = {e.market_condition_id: e for e in estimates}
    opportunities = []

    for market in markets:
        est = estimate_map.get(market.condition_id)
        if not est:
            continue

        # Astra veto
        if est.no_trade:
            continue

        # Basic quality gates
        if est.confidence < MIN_CONFIDENCE:
            continue
        if market.liquidity < MIN_MARKET_LIQUIDITY:
            continue
        if market.yes_price <= 0 or market.yes_price >= 1:
            continue

        # Dead-market guard: skip markets where the price is already near resolution.
        # If YES < 1% or YES > 99%, the market has almost certainly already resolved
        # or has overwhelming consensus. Buying at these prices gives near-zero EV
        # and any tiny error in our estimate causes a total loss.
        # The $0.001 US revenue trade was a perfect example of this failure mode.
        if market.yes_price < 0.02 or market.yes_price > 0.98:
            continue

        # YES+NO Arbitrage Scanner (arXiv:2010.12508, recovered ResearchGate papers)
        # If yes_price + no_price < 1.00 after fees, risk-free profit exists
        # Polymarket uses AMM pricing, so YES + NO should equal exactly $1.00
        # Any deviation > 2¢ (after 2% fees) is arbitrage opportunity
        no_price = 1.0 - market.yes_price
        arbitrage_edge = 1.0 - (market.yes_price + no_price)

        # Account for 2% trading fees on both sides (1% maker + 1% taker)
        net_arbitrage = arbitrage_edge - 0.02

        if net_arbitrage > 0.005:  # Profitable after fees (0.5¢ minimum)
            # Risk-free arbitrage: buy both YES and NO, guaranteed profit on resolution
            opportunities.append(
                Opportunity(
                    market=market,
                    estimate=est,
                    market_price=market.yes_price,
                    our_estimate=0.50,  # Neutral probability (we profit either way)
                    edge=net_arbitrage,
                    direction="BUY BOTH",  # Special flag for arbitrage
                    ev_after_costs=net_arbitrage,  # Pure arbitrage EV
                    robustness_score=5,  # Maximum robustness (risk-free)
                    score=999.0 + (net_arbitrage * 100),  # Always top priority, scaled by edge
                    kelly_pct=0.25,  # Max Kelly for risk-free arb (limited by liquidity)
                )
            )
            continue  # Skip normal mispricing logic for arbitrage opportunities

        # Astra V2 Module 3 minimum bar
        if est.ev_after_costs <= 0:
            continue
        if est.robustness_score < 3:
            continue

        # Apply favorite-longshot calibration correction (arXiv:1811.12516)
        # Shade our estimate toward the market price in extreme longshot/favorite regions
        calibrated_p = _apply_longshot_calibration(est.probability, market.yes_price)

        edge = calibrated_p - market.yes_price
        if abs(edge) < MISPRICING_THRESHOLD:
            continue

        # Hallucination sanity check for BUY YES on longshot markets.
        # If our probability is > 5× the market price on a sub-20¢ market,
        # it's almost certainly model hallucination (e.g. 88% vs 6% market).
        # Prediction markets are efficient — 6¢ markets don't become 88% winners
        # without a massive, observable catalyst. Veto these signals.
        if edge > 0 and market.yes_price < 0.20 and calibrated_p > market.yes_price * 5:
            continue

        direction = "BUY YES" if edge > 0 else "BUY NO"

        # Score = edge quality × confidence × robustness
        score = abs(edge) * est.confidence * (est.robustness_score / 5)

        # Whale signal boost: active volume spike = Tier B confirmation signal
        wb = whale_boost.get(market.condition_id, 0.0)
        if wb > 0:
            score = score * (1.0 + wb)  # e.g. 3σ spike → +5% score boost

        opportunities.append(
            Opportunity(
                market=market,
                estimate=est,
                market_price=market.yes_price,
                our_estimate=calibrated_p,  # use calibrated estimate
                edge=edge,
                direction=direction,
                ev_after_costs=est.ev_after_costs,
                robustness_score=est.robustness_score,
                score=score,
                kelly_pct=est.kelly_position_pct,
            )
        )

    return sorted(opportunities, key=lambda x: x.score, reverse=True)
