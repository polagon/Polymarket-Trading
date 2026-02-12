"""
Canonical fee calculation helpers.

CRITICAL FIX #2, #11: Fee rate MUST come from market metadata (feeRateBps).
NEVER hardcode fees.
"""


def effective_cost_buy(ask_price: float, size_tokens: float, fee_rate_bps: int) -> float:
    """
    Compute total cost to BUY tokens including fees.

    Args:
        ask_price: Best ask price (what we pay per token, 0-1)
        size_tokens: Quantity to buy
        fee_rate_bps: Fee rate in basis points (e.g., 200 = 2%)

    Returns:
        Total USD cost including fees

    Example:
        >>> effective_cost_buy(0.50, 100.0, 200)  # Buy 100 tokens @ 50¢ with 2% fee
        51.0  # $50 base + $1 fee = $51
    """
    fee_rate = fee_rate_bps / 10000.0
    base_cost = ask_price * size_tokens
    total_cost = base_cost * (1.0 + fee_rate)
    return total_cost


def effective_proceeds_sell(bid_price: float, size_tokens: float, fee_rate_bps: int) -> float:
    """
    Compute net proceeds from SELLING tokens after fees.

    Args:
        bid_price: Best bid price (what we receive per token, 0-1)
        size_tokens: Quantity to sell
        fee_rate_bps: Fee rate in basis points (e.g., 200 = 2%)

    Returns:
        Net USD proceeds after fees

    Example:
        >>> effective_proceeds_sell(0.50, 100.0, 200)  # Sell 100 tokens @ 50¢ with 2% fee
        49.0  # $50 gross - $1 fee = $49
    """
    fee_rate = fee_rate_bps / 10000.0
    base_proceeds = bid_price * size_tokens
    net_proceeds = base_proceeds * (1.0 - fee_rate)
    return net_proceeds


def compute_fee_paid(base_amount: float, fee_rate_bps: int, is_buyer: bool) -> float:
    """
    Compute fee paid for a trade.

    Args:
        base_amount: Base trade amount (price * size)
        fee_rate_bps: Fee rate in basis points
        is_buyer: True if buyer (fee on cost), False if seller (fee on proceeds)

    Returns:
        Fee amount in USD
    """
    fee_rate = fee_rate_bps / 10000.0
    if is_buyer:
        # Buyer pays fee on cost
        return base_amount * fee_rate
    else:
        # Seller pays fee on proceeds
        return base_amount * fee_rate


def net_parity_profit(yes_ask: float, no_ask: float, fee_rate_bps: int, size: float = 1.0) -> float:
    """
    Compute net profit from YES/NO parity arbitrage.

    CRITICAL FIX #7: This is NOT risk-free due to leg risk.

    Args:
        yes_ask: Best ask for YES token
        no_ask: Best ask for NO token
        fee_rate_bps: Fee rate (same for both legs typically)
        size: Size to trade (default 1.0 token)

    Returns:
        Net profit after fees. Positive = profitable arb.

    Example:
        >>> net_parity_profit(0.48, 0.50, 200, 1.0)
        0.02  # (1.00 - 0.48 - 0.50) - fees
    """
    yes_cost = effective_cost_buy(yes_ask, size, fee_rate_bps)
    no_cost = effective_cost_buy(no_ask, size, fee_rate_bps)
    total_cost = yes_cost + no_cost
    payout = 1.0 * size  # YES+NO always pays $1.00 per pair
    profit = payout - total_cost
    return profit
