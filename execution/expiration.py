"""
GTD (Good-Til-Date) expiration timestamp computation.

CRITICAL FIX #1, #9, #20: Polymarket GTD requires unix timestamp with safety buffer.
Buffer is configurable and logs rejections for empirical tuning.
"""

import logging
import time

from config import GTD_DESIRED_SECONDS, GTD_SAFETY_BUFFER_SECONDS

logger = logging.getLogger(__name__)


def compute_gtd_expiration(desired_seconds: int = GTD_DESIRED_SECONDS) -> int:
    """
    Compute GTD expiration timestamp with configurable safety buffer.

    CRITICAL FIX #20: Buffer is configurable to allow empirical tuning.
    Platform rejections should be logged to tighten/relax buffer.

    Args:
        desired_seconds: How long you want the order to live (default from config)

    Returns:
        Unix timestamp for expiration

    Formula:
        expiration = now + GTD_SAFETY_BUFFER_SECONDS + desired_seconds

    Example:
        >>> # If GTD_SAFETY_BUFFER_SECONDS=60 and desired_seconds=120
        >>> expiration = compute_gtd_expiration(120)
        >>> # expiration = now + 60 + 120 = now + 180 seconds

    Logging:
        Logs at DEBUG level for audit trail. If CLOB rejects with "expiration too soon",
        increase GTD_SAFETY_BUFFER_SECONDS in config.py.
    """
    now = int(time.time())
    buffer = GTD_SAFETY_BUFFER_SECONDS
    expiration = now + buffer + desired_seconds

    logger.debug(
        f"GTD expiration: now={now}, buffer={buffer}s, desired={desired_seconds}s, "
        f"result={expiration} ({buffer + desired_seconds}s total)"
    )

    return expiration


def log_gtd_rejection(error_message: str, expiration: int):
    """
    Log GTD rejection from CLOB for empirical buffer tuning.

    CRITICAL FIX #20: Makes buffer empirically tunable.

    Args:
        error_message: Error from CLOB API
        expiration: The expiration timestamp that was rejected

    Example:
        >>> log_gtd_rejection("expiration must be at least 60 seconds in future", 1234567890)
    """
    now = int(time.time())
    delta = expiration - now

    logger.error(
        f"GTD REJECTION: {error_message}\n"
        f"  Expiration: {expiration} (unix)\n"
        f"  Now: {now} (unix)\n"
        f"  Delta: {delta}s\n"
        f"  Current buffer: {GTD_SAFETY_BUFFER_SECONDS}s\n"
        f"  ACTION: Consider increasing GTD_SAFETY_BUFFER_SECONDS in config.py"
    )


def is_gtd_near_expiry(expiration: int, threshold_seconds: int = 30) -> bool:
    """
    Check if GTD order is approaching expiration.

    Use this to decide whether to cancel/replace quotes.

    Args:
        expiration: Unix timestamp when order expires
        threshold_seconds: How many seconds before expiry to consider "near" (default 30s)

    Returns:
        True if order will expire within threshold_seconds

    Example:
        >>> expiration = int(time.time()) + 25  # Expires in 25 seconds
        >>> is_gtd_near_expiry(expiration, threshold_seconds=30)
        True  # Within 30s threshold
    """
    now = int(time.time())
    time_until_expiry = expiration - now
    return time_until_expiry <= threshold_seconds
