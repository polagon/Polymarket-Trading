"""
Satellite Filter - High-conviction info-edge trades (Astra V2 integration).

10-20% of capital allocated to satellite trades with exceptional evidence.
"""
import logging
from typing import Optional

from models.types import Market
from config import (
    SATELLITE_MIN_EDGE,
    SATELLITE_MIN_ROBUSTNESS,
    SATELLITE_MIN_LIQUIDITY,
    SATELLITE_REQUIRE_TIER_A_OR_B,
    SATELLITE_RISK_BUDGET_PCT,
    BANKROLL,
)

logger = logging.getLogger(__name__)


def evaluate_satellite_trade(
    market: Market,
    prediction: dict,
    current_exposure: float,
) -> Optional[dict]:
    """
    Evaluate if prediction qualifies as satellite trade.

    Satellite trades are high-conviction bets on information edge,
    NOT market structure edge.

    CRITICAL: Requires exceptional evidence to overcome risk budget.

    Args:
        market: Market instance
        prediction: {
            "estimated_prob": float,
            "confidence": float,
            "robustness": int (1-5),
            "evidence_tier": str ("A", "B", "C"),
            "reasoning": str,
        }
        current_exposure: Current satellite risk budget used

    Returns:
        Trade recommendation dict or None if not qualified
    """
    # Extract prediction fields
    estimated_prob = prediction.get("estimated_prob", 0.5)
    confidence = prediction.get("confidence", 0.0)
    robustness = prediction.get("robustness", 0)
    evidence_tier = prediction.get("evidence_tier", "C")

    # Get market price
    market_price = (market.yes_bid + market.yes_ask) / 2.0

    # Compute edge
    edge = abs(estimated_prob - market_price)

    # Gate 1: Minimum edge
    if edge < SATELLITE_MIN_EDGE:
        logger.debug(
            f"Satellite veto: {market.condition_id} edge={edge:.2%} < {SATELLITE_MIN_EDGE:.2%}"
        )
        return None

    # Gate 2: Minimum robustness
    if robustness < SATELLITE_MIN_ROBUSTNESS:
        logger.debug(
            f"Satellite veto: {market.condition_id} robustness={robustness} < {SATELLITE_MIN_ROBUSTNESS}"
        )
        return None

    # Gate 3: Minimum liquidity
    if market.liquidity < SATELLITE_MIN_LIQUIDITY:
        logger.debug(
            f"Satellite veto: {market.condition_id} liquidity=${market.liquidity:.0f} < ${SATELLITE_MIN_LIQUIDITY:.0f}"
        )
        return None

    # Gate 4: Tier A or B evidence required
    if SATELLITE_REQUIRE_TIER_A_OR_B and evidence_tier not in ["A", "B"]:
        logger.debug(
            f"Satellite veto: {market.condition_id} evidence_tier={evidence_tier} (need A or B)"
        )
        return None

    # Gate 5: Risk budget available
    max_satellite_budget = BANKROLL * SATELLITE_RISK_BUDGET_PCT
    if current_exposure >= max_satellite_budget:
        logger.warning(
            f"Satellite veto: {market.condition_id} risk budget exhausted "
            f"({current_exposure:.2f} / {max_satellite_budget:.2f})"
        )
        return None

    # Qualified! Compute position size
    side = "BUY" if estimated_prob > market_price else "SELL"

    # Size using Kelly fraction (conservative)
    kelly_fraction = edge / (1.0 - confidence) if confidence < 1.0 else 0.0
    kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%

    target_size_usd = kelly_fraction * (max_satellite_budget - current_exposure)

    recommendation = {
        "market_id": market.condition_id,
        "side": side,
        "target_size_usd": target_size_usd,
        "edge": edge,
        "robustness": robustness,
        "evidence_tier": evidence_tier,
        "reasoning": prediction.get("reasoning", ""),
    }

    logger.info(
        f"Satellite trade qualified: {market.condition_id} {side} "
        f"edge={edge:.2%} robustness={robustness}/5 tier={evidence_tier} "
        f"size=${target_size_usd:.2f}"
    )

    return recommendation


def load_astra_predictions(predictions_file: str) -> dict:
    """
    Load Astra V2 predictions from file.

    Args:
        predictions_file: Path to predictions.json

    Returns:
        {condition_id: prediction_dict}
    """
    import json
    from pathlib import Path

    path = Path(predictions_file)
    if not path.exists():
        logger.warning(f"Astra predictions file not found: {predictions_file}")
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)

        # Convert list to dict keyed by market_condition_id
        if isinstance(data, list):
            predictions_dict = {
                pred.get("market_condition_id"): pred
                for pred in data
                if pred.get("market_condition_id")
            }
            logger.info(f"Loaded {len(predictions_dict)} Astra predictions from {predictions_file}")
            return predictions_dict
        elif isinstance(data, dict):
            logger.info(f"Loaded {len(data)} Astra predictions from {predictions_file}")
            return data
        else:
            logger.error(f"Unexpected predictions format: {type(data)}")
            return {}

    except Exception as e:
        logger.error(f"Failed to load Astra predictions: {e}")
        return {}


def scan_satellite_opportunities(
    markets: list,
    predictions: dict,
    current_satellite_exposure: float,
) -> list:
    """
    Scan for satellite trade opportunities.

    Args:
        markets: List of Market objects
        predictions: {condition_id: prediction_dict}
        current_satellite_exposure: Current satellite risk used

    Returns:
        List of trade recommendations
    """
    recommendations = []

    for market in markets:
        prediction = predictions.get(market.condition_id)
        if not prediction:
            continue

        recommendation = evaluate_satellite_trade(
            market, prediction, current_satellite_exposure
        )

        if recommendation:
            recommendations.append(recommendation)

    return recommendations
