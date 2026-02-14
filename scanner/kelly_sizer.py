"""
Kelly Criterion position sizing for binary prediction markets.
Uses fractional Kelly (25%) with hard position size caps.
"""

from config import MAX_KELLY_FRACTION, MAX_POSITION_PCT


def kelly_fraction(probability: float, market_price: float) -> float:
    """
    Compute the full Kelly fraction for a binary market bet.

    For a YES bet:
      - Win probability = probability
      - Win payoff per $1 bet = (1 - market_price) / market_price
      - Loss amount = 1 (lose stake)

    Kelly formula: f = W - (L / R)
      where W = win prob, L = lose prob, R = win-to-loss ratio
    """
    win_prob = probability
    lose_prob = 1.0 - probability

    if market_price <= 0 or market_price >= 1:
        return 0.0

    # For YES bet
    win_payoff = (1.0 - market_price) / market_price

    if win_payoff <= 0:
        return 0.0

    full_kelly = win_prob - (lose_prob / win_payoff)
    return max(0.0, full_kelly)


def size_position(
    probability: float,
    market_price: float,
    direction: str,
    bankroll: float,
    confidence: float,
) -> dict:
    """
    Calculate recommended position size.

    Returns dict with:
    - position_dollars: amount to bet in USD
    - position_pct: as fraction of bankroll
    - full_kelly_pct: what full Kelly would be
    - expected_value: expected profit per dollar bet
    - rationale: human-readable explanation
    """
    # For NO bets, flip probabilities
    if direction == "BUY NO":
        # We're betting on NO outcome
        # NO price = 1 - YES price
        no_price = 1.0 - market_price
        effective_probability = 1.0 - probability
        effective_price = no_price
    else:
        effective_probability = probability
        effective_price = market_price

    full_k = kelly_fraction(effective_probability, effective_price)

    # Scale Kelly by confidence (less confident = smaller position)
    confidence_adjusted = full_k * confidence

    # Apply fractional Kelly
    fractional_k = confidence_adjusted * MAX_KELLY_FRACTION

    # Cap at max position
    capped = min(fractional_k, MAX_POSITION_PCT)

    position_dollars = bankroll * capped

    # Minimum position size $5 (below this, fees eat too much)
    if position_dollars < 5.0:
        return {
            "position_dollars": 0.0,
            "position_pct": 0.0,
            "full_kelly_pct": round(full_k, 4),
            "expected_value": _expected_value(effective_probability, effective_price),
            "rationale": f"Position too small (${position_dollars:.2f} < $5 minimum)",
        }

    return {
        "position_dollars": round(position_dollars, 2),
        "position_pct": round(capped, 4),
        "full_kelly_pct": round(full_k, 4),
        "expected_value": _expected_value(effective_probability, effective_price),
        "rationale": (
            f"Full Kelly={full_k:.1%}, "
            f"×{confidence:.0%} confidence, "
            f"×{MAX_KELLY_FRACTION:.0%} fractional = {capped:.1%} of bankroll"
        ),
    }


def _expected_value(probability: float, market_price: float) -> float:
    """Expected profit per $1 bet."""
    win_payoff = (1.0 - market_price) / market_price
    return round(probability * win_payoff - (1.0 - probability), 4)
