"""
Resolution Risk Score (RRS) - Ambiguity + dispute susceptibility detection.

CRITICAL: Prevents "right but loses" events from ambiguous resolution criteria.

Hard veto gates:
- RRS > 0.35 → refuse maker quoting
- RRS > 0.25 → refuse satellite trades
"""
import logging
from typing import Dict

from models.types import Market

logger = logging.getLogger(__name__)

# Category-specific dispute priors
DISPUTE_PRIORS = {
    "politics": 0.15,
    "sports": 0.10,
    "crypto": 0.20,
    "weather": 0.05,
    "entertainment": 0.25,
    "other": 0.20,
}

# Ambiguity markers in market questions/descriptions
AMBIGUITY_KEYWORDS = [
    "materially",
    "significantly",
    "substantially",
    "end of year",
    "by end of",
    "official",
    "major",
    "publicly",
    "widely",
]


def compute_rrs(market: Market, metadata: Dict) -> float:
    """
    Compute Resolution Risk Score (0-1, higher = riskier).

    Components:
    1. Rule ambiguity (40%): NLP analysis of question + description
    2. Dispute susceptibility (30%): Category priors + market history
    3. Time-to-resolution pressure (20%): Shorter = riskier
    4. Clarification flag (10%): Has market been clarified?

    Args:
        market: Market instance
        metadata: Market raw metadata (for clarification flags)

    Returns:
        RRS score (0-1). Hard veto gates:
        - RRS > 0.35 → no maker quoting
        - RRS > 0.25 → no satellite trades
    """
    # 1. Ambiguity score
    ambiguity_score = _detect_ambiguity(market.question, market.description)

    # 2. Dispute susceptibility
    category_prior = DISPUTE_PRIORS.get(market.category, 0.2)

    # CRITICAL: If clarification/dispute history unavailable, bias conservative
    if metadata.get("dispute_history") is None:
        # Unknown must bias high during close windows
        if market.time_to_close and market.time_to_close < 72:
            dispute_score = 0.6
        else:
            dispute_score = category_prior
    else:
        dispute_score = category_prior

    # 3. Time-to-close risk
    if market.time_to_close is None:
        time_risk = 0.5  # Unknown
    elif market.time_to_close < 48:
        time_risk = 1.0  # Very high risk
    elif market.time_to_close < 168:
        time_risk = 0.5
    else:
        time_risk = 0.3

    # 4. Clarification flag
    clarification_risk = 0.8 if metadata.get("clarification_posted") else 0.2

    # Weighted sum
    rrs = (
        0.40 * ambiguity_score +
        0.30 * dispute_score +
        0.20 * time_risk +
        0.10 * clarification_risk
    )

    if rrs > 0.35:
        logger.warning(
            f"High RRS={rrs:.3f} for {market.condition_id}: {market.question[:80]}"
        )

    return rrs


def _detect_ambiguity(question: str, description: str) -> float:
    """
    Detect ambiguity in market question/description.

    Returns:
        Ambiguity score (0-1)
    """
    text = (question + " " + description).lower()

    # Count ambiguity markers
    marker_count = sum(1 for keyword in AMBIGUITY_KEYWORDS if keyword in text)

    # Normalize to 0-1 (cap at 5 markers = max ambiguity)
    ambiguity_score = min(marker_count / 5.0, 1.0)

    # Additional heuristics
    # Very short questions are often ambiguous
    if len(question) < 30:
        ambiguity_score = max(ambiguity_score, 0.4)

    # Questions without clear resolution criteria
    if "resolve" not in text and "determined" not in text:
        ambiguity_score = max(ambiguity_score, 0.3)

    return ambiguity_score
