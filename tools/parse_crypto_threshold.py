"""
Strict word-boundary parser for crypto threshold markets.

Loop 5.1: Eliminate false positives (Ethena, WBTC, stETH) using regex word boundaries.

Key rules:
- \bBTC\b matches "BTC" but NOT "WBTC", "OBTC", "renBTC"
- \bETH\b matches "ETH" but NOT "Ethena", "stETH", "sETH", "rETH"
- Explicit rejection list for known false positives
- "Bitcoin" and "Ethereum" full-word matches also accepted
"""

import re
from typing import Any, Optional

# ── Rejection lists ──────────────────────────────────────────────────

# Known false-positive tokens (case-insensitive substring check)
FALSE_POSITIVE_TOKENS = {
    "ethena",
    "sUSDe",
    "USDe",
    "wbtc",  # Wrapped BTC is not BTC
    "renbtc",  # Ren BTC is not BTC
    "steth",  # Staked ETH is not ETH
    "seth",  # Synthetix sETH
    "reth",  # Rocket Pool rETH
    "cbeth",  # Coinbase wrapped ETH
}


def _is_false_positive(question: str) -> bool:
    """Check if question contains known false-positive tokens.

    Case-insensitive substring match against FALSE_POSITIVE_TOKENS.
    Returns True if ANY token is found (reject market).
    """
    q_lower = question.lower()
    return any(token in q_lower for token in FALSE_POSITIVE_TOKENS)


# ── Underlying detection with word boundaries ────────────────────────


def detect_underlying(question: str) -> Optional[str]:
    """Detect underlying asset using strict word-boundary regex.

    Returns "BTC", "ETH", "SOL", or None.

    Rules:
    - \bBTC\b or \bBitcoin\b → "BTC"
    - \bETH\b or \bEthereum\b → "ETH"
    - \bSOL\b or \bSolana\b → "SOL"
    - FALSE_POSITIVE_TOKENS present → None (reject)

    Examples:
        "Will BTC hit $100K?" → "BTC"
        "Will Bitcoin reach $1M?" → "BTC"
        "Will WBTC depeg?" → None (false positive)
        "Will Ethena TVL exceed $10B?" → None (false positive)
        "Will stETH maintain peg?" → None (false positive)
    """
    # Check false positives first
    if _is_false_positive(question):
        return None

    # Word-boundary patterns (case-insensitive)
    if re.search(r"\bBTC\b", question, re.IGNORECASE) or re.search(r"\bBitcoin\b", question, re.IGNORECASE):
        return "BTC"

    if re.search(r"\bETH\b", question, re.IGNORECASE) or re.search(r"\bEthereum\b", question, re.IGNORECASE):
        return "ETH"

    if re.search(r"\bSOL\b", question, re.IGNORECASE) or re.search(r"\bSolana\b", question, re.IGNORECASE):
        return "SOL"

    return None


# ── Strike extraction (unchanged from Loop 5) ────────────────────────


def extract_strike(question: str) -> Optional[float]:
    """Extract numeric strike price from question.

    Handles formats:
    - $100,000 → 100000.0
    - $5K → 5000.0
    - $1M → 1000000.0
    - $3.5K → 3500.0
    """
    # Remove commas
    q = question.replace(",", "")

    # Match $X.XK, $XK, $X.XM, $XM, or $XXXXX
    # Pattern: $ followed by digits, optional decimal, optional K/M suffix
    pattern = r"\$(\d+(?:\.\d+)?)\s*([KMk m]?)"
    match = re.search(pattern, q)
    if not match:
        return None

    value_str, suffix = match.groups()
    value = float(value_str)

    suffix_lower = suffix.lower().strip()
    if suffix_lower == "k":
        value *= 1000
    elif suffix_lower == "m":
        value *= 1000000

    return value


# ── Resolution type detection (unchanged from Loop 5) ────────────────


def detect_resolution_type(question: str) -> tuple[str, str, str]:
    """Detect resolution_type, op, and window from question.

    Returns (resolution_type, op, window):
    - resolution_type: "touch" or "close"
    - op: ">=" or "<="
    - window: "any_time" or "at_close"

    Touch patterns: "hit", "reach", "touch"
    Close patterns: "be above/below at X close", "close above/below"
    """
    q_lower = question.lower()

    # Touch patterns (any_time window)
    if any(kw in q_lower for kw in ["hit", "reach", "touch"]):
        return "touch", ">=", "any_time"

    # Close patterns (at_close window)
    if "close" in q_lower or "at " in q_lower:
        if "below" in q_lower:
            return "close", "<=", "at_close"
        if "above" in q_lower:
            return "close", ">=", "at_close"

    # Default: cannot determine
    return "unknown", "unknown", "unknown"


# ── Main parser ──────────────────────────────────────────────────────


def parse_threshold_market(market: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Parse crypto threshold market with strict word-boundary filtering.

    Returns parsed dict with:
        - underlying: "BTC" | "ETH" | "SOL"
        - strike: float
        - cutoff_ts_utc: ISO8601 string
        - resolution_type: "touch" | "close"
        - op: ">=" | "<="
        - window: "any_time" | "at_close"

    Returns None if:
    - Question contains false-positive tokens (Ethena, WBTC, stETH, etc.)
    - No valid underlying detected (word-boundary match failed)
    - Strike price not found
    - Resolution type ambiguous
    - Missing required fields (question, end_date_iso)
    """
    question = market.get("question", "")
    end_date_iso = market.get("end_date_iso", "")

    if not question or not end_date_iso:
        return None

    # Strict underlying detection (word-boundary + false-positive filter)
    underlying = detect_underlying(question)
    if underlying is None:
        return None

    # Extract strike
    strike = extract_strike(question)
    if strike is None:
        return None

    # Detect resolution type
    resolution_type, op, window = detect_resolution_type(question)
    if resolution_type == "unknown":
        return None

    return {
        "underlying": underlying,
        "strike": strike,
        "cutoff_ts_utc": end_date_iso,
        "resolution_type": resolution_type,
        "op": op,
        "window": window,
    }
