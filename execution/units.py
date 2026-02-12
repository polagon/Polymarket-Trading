"""
Unit conversion helpers for consistent internal sizing.

CRITICAL FIX #10: Internal sizing MUST be consistent.
- Order sizing: tokens
- Inventory aggregation: mark-to-mid USD
- Risk caps: USD
"""


def tokens_to_usd(size_tokens: float, mid_price: float, side: str) -> float:
    """
    Convert token quantity to USD exposure (mark-to-mid).

    CRITICAL FIX #10: Ensures consistent inventory aggregation.

    Args:
        size_tokens: Quantity in tokens
        mid_price: Current mid price (0-1)
        side: "BUY" or "SELL"

    Returns:
        USD exposure (positive)

    Rationale:
        - BUY: Buying YES tokens at mid_price costs mid_price per token
        - SELL: Selling YES tokens (shorting) exposes us to (1 - mid_price) per token

    Examples:
        >>> tokens_to_usd(100.0, 0.50, "BUY")
        50.0  # Buying 100 YES @ 50¢ = $50 exposure

        >>> tokens_to_usd(100.0, 0.50, "SELL")
        50.0  # Selling 100 YES (short) @ 50¢ = $50 exposure to downside
    """
    if side == "BUY":
        # Buying YES tokens at mid_price
        return size_tokens * mid_price
    elif side == "SELL":
        # Selling YES tokens (shorting) exposes to (1 - mid_price)
        return size_tokens * (1.0 - mid_price)
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")


def usd_to_tokens(size_usd: float, mid_price: float, side: str) -> float:
    """
    Convert USD exposure to token quantity.

    Args:
        size_usd: Target USD exposure
        mid_price: Current mid price (0-1)
        side: "BUY" or "SELL"

    Returns:
        Token quantity needed

    Examples:
        >>> usd_to_tokens(50.0, 0.50, "BUY")
        100.0  # Need 100 tokens @ 50¢ to get $50 exposure

        >>> usd_to_tokens(50.0, 0.50, "SELL")
        100.0  # Need to sell 100 tokens @ 50¢ for $50 exposure
    """
    if side == "BUY":
        if mid_price <= 0:
            return 0.0
        return size_usd / mid_price
    elif side == "SELL":
        if mid_price >= 1.0:
            return 0.0
        return size_usd / (1.0 - mid_price)
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")


def aggregate_exposure_usd(token_positions: dict[str, float], mid_prices: dict[str, float]) -> float:
    """
    Aggregate token-level inventory to total USD exposure.

    CRITICAL FIX #5: Portfolio risk aggregation uses mark-to-mid.
    CRITICAL: mid_prices must come from execution.mid.compute_mid(), not ad-hoc calculations.

    Args:
        token_positions: {token_id: signed_quantity} (positive=long, negative=short)
        mid_prices: {token_id: mid_price} (from execution.mid.compute_mid)

    Returns:
        Total USD exposure (sum of abs values)

    Example:
        >>> positions = {"YES_token_1": 100.0, "NO_token_2": -50.0}
        >>> mids = {"YES_token_1": 0.50, "NO_token_2": 0.40}
        >>> aggregate_exposure_usd(positions, mids)
        80.0  # |100*0.50| + |-50*(1-0.40)| = 50 + 30 = 80
    """
    total_exposure = 0.0

    for token_id, signed_qty in token_positions.items():
        if signed_qty == 0:
            continue

        mid = mid_prices.get(token_id, 0.5)  # Default to 0.5 if unknown

        if signed_qty > 0:
            # Long position
            exposure = signed_qty * mid
        else:
            # Short position (negative quantity)
            exposure = abs(signed_qty) * (1.0 - mid)

        total_exposure += exposure

    return total_exposure
