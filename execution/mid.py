"""
Canonical mid price calculation with fallback rules.

CRITICAL FIX (ChatGPT Final): Must handle one-sided books, stale books, and log fallbacks.
"""

import logging
from typing import Optional

from config import MID_FALLBACK_ONE_SIDED_OFFSET, MID_FALLBACK_STALE_AGE_MS

logger = logging.getLogger(__name__)


def compute_mid(
    bid: Optional[float],
    ask: Optional[float],
    last_mid: Optional[float] = None,
    book_age_ms: int = 0,
) -> Optional[float]:
    """
    Compute mid price with deterministic fallback rules.

    CRITICAL: This is the single source of truth for "mid" across the system.

    Fallback hierarchy:
    1. Both bid and ask exist → (bid + ask) / 2
    2. One-sided book → offset from single side by MID_FALLBACK_ONE_SIDED_OFFSET
    3. Stale book (age > threshold) → last_mid if available
    4. No data → None (refuse to trade)

    Args:
        bid: Best bid price (or None)
        ask: Best ask price (or None)
        last_mid: Last known mid price for this market (fallback)
        book_age_ms: Book staleness in milliseconds

    Returns:
        Mid price (0-1) or None if unable to compute

    Logging:
        Always logs when using fallback rules (critical for post-trade debugging)
    """
    # Check staleness first
    if book_age_ms > MID_FALLBACK_STALE_AGE_MS:
        if last_mid is not None:
            logger.warning(
                f"Book stale ({book_age_ms}ms > {MID_FALLBACK_STALE_AGE_MS}ms), using last_mid={last_mid:.4f}"
            )
            return last_mid
        else:
            logger.error(f"Book stale ({book_age_ms}ms) and no last_mid available. Refusing to compute mid.")
            return None

    # Standard case: both bid and ask
    if bid is not None and ask is not None:
        # Sanity check
        if bid > ask:
            logger.error(f"Crossed book: bid={bid:.4f} > ask={ask:.4f}. Refusing to compute mid.")
            return None
        return (bid + ask) / 2.0

    # One-sided book fallbacks
    if bid is not None and ask is None:
        # Only bid available
        mid = bid + MID_FALLBACK_ONE_SIDED_OFFSET
        mid = min(0.99, mid)  # Clamp to valid range
        logger.warning(
            f"One-sided book (bid only): bid={bid:.4f}, using mid={mid:.4f} (offset +{MID_FALLBACK_ONE_SIDED_OFFSET})"
        )
        return mid

    if ask is not None and bid is None:
        # Only ask available
        mid = ask - MID_FALLBACK_ONE_SIDED_OFFSET
        mid = max(0.01, mid)  # Clamp to valid range
        logger.warning(
            f"One-sided book (ask only): ask={ask:.4f}, using mid={mid:.4f} (offset -{MID_FALLBACK_ONE_SIDED_OFFSET})"
        )
        return mid

    # No bid, no ask
    if last_mid is not None:
        logger.warning("No bid/ask available, using last_mid={last_mid:.4f}")
        return last_mid

    logger.error("No bid, no ask, no last_mid. Refusing to compute mid.")
    return None


def is_mid_reliable(
    bid: Optional[float],
    ask: Optional[float],
    book_age_ms: int,
) -> bool:
    """
    Check if mid is reliable (both sides present, not stale).

    Use this to decide whether to quote or wait for better data.

    Args:
        bid: Best bid
        ask: Best ask
        book_age_ms: Book staleness

    Returns:
        True if mid is reliable (standard case), False otherwise
    """
    if book_age_ms > MID_FALLBACK_STALE_AGE_MS:
        return False
    if bid is None or ask is None:
        return False
    if bid > ask:
        return False
    return True
