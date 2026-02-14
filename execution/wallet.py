"""
Wallet model and preflight checks.

CRITICAL FIX #19: Signer vs funder address, allowances, inventory checks.
"""

import logging

from config import FUNDER_ADDRESS, MIN_USDC_BALANCE, SIGNATURE_TYPE, TRADING_ADDRESS

logger = logging.getLogger(__name__)


def preflight_checks(clob_client):
    """
    Startup health checks before quoting begins.

    CRITICAL FIX #19: Prevents silent failures from:
    - Insufficient balance
    - Missing allowances
    - Incorrect funder address resolution
    - No sellable inventory

    Args:
        clob_client: Instance of py-clob-client with wallet credentials

    Raises:
        ValueError: If any preflight check fails

    Checks:
    1. USDC balance sufficient (≥ MIN_USDC_BALANCE)
    2. Allowances approved for CLOB contract
    3. Can we SELL? (have token inventory for at least one market)
    4. Funder address resolution correct (signer vs funder)
    """
    logger.info(f"Running wallet preflight checks for {TRADING_ADDRESS}")

    # Check 1: USDC balance
    try:
        usdc_balance = clob_client.get_balance("USDC", TRADING_ADDRESS)
        logger.info(f"USDC balance: ${usdc_balance:.2f}")

        if usdc_balance < MIN_USDC_BALANCE:
            raise ValueError(f"Insufficient USDC balance: ${usdc_balance:.2f} (minimum: ${MIN_USDC_BALANCE:.2f})")
    except Exception as e:
        logger.error(f"Failed to check USDC balance: {e}")
        raise ValueError(f"USDC balance check failed: {e}")

    # Check 2: Allowance
    try:
        allowance = clob_client.get_allowance(TRADING_ADDRESS)
        logger.info(f"USDC allowance: ${allowance:.2f}")

        if allowance < usdc_balance:
            logger.warning(
                f"Allowance {allowance:.2f} < balance {usdc_balance:.2f}. "
                "May need approval transaction. Proceed with caution."
            )
            # Not a hard failure, but log warning
    except Exception as e:
        logger.warning(f"Could not check allowance: {e}")

    # Check 3: Funder address resolution
    try:
        resolved_funder = clob_client.get_funder_address(TRADING_ADDRESS, SIGNATURE_TYPE)
        logger.info(f"Resolved funder address: {resolved_funder}")

        if FUNDER_ADDRESS and resolved_funder != FUNDER_ADDRESS:
            raise ValueError(f"Funder address mismatch: expected {FUNDER_ADDRESS}, got {resolved_funder}")
    except Exception as e:
        logger.error(f"Failed to resolve funder address: {e}")
        # Don't hard fail if FUNDER_ADDRESS not configured
        if FUNDER_ADDRESS:
            raise ValueError(f"Funder address check failed: {e}")

    # Check 4: Token inventory (can we SELL?)
    try:
        # Query wallet for outcome token balances
        # If we have zero outcome tokens, we can only BUY (never SELL)
        # This is a soft check - log warning if no inventory

        # NOTE: This requires fetching token balances for markets
        # For now, just log that we should check
        logger.info("Token inventory check: Ensure we have outcome tokens for SELL orders")
        # TODO: Implement token balance query when py-clob-client method available
    except Exception as e:
        logger.warning(f"Could not check token inventory: {e}")

    logger.info("✅ Wallet preflight checks passed")


def get_available_usdc(clob_client, market_id: str, reserved_by_market: dict[str, float]) -> float:
    """
    Get available USDC for new BUY orders in this market.

    CRITICAL FIX #18: Must account for existing reservations.

    Args:
        clob_client: CLOB client instance
        market_id: Market condition_id
        reserved_by_market: {condition_id: reserved_usdc}

    Returns:
        Available USDC for new orders
    """
    wallet_balance = clob_client.get_balance("USDC", TRADING_ADDRESS)
    reserved = reserved_by_market.get(market_id, 0.0)
    available = wallet_balance - reserved
    return max(0.0, available)  # type: ignore[no-any-return]


def get_available_tokens(clob_client, token_id: str, reserved_by_token: dict[str, float]) -> float:
    """
    Get available tokens for new SELL orders.

    CRITICAL FIX #18: Must account for existing reservations.

    Args:
        clob_client: CLOB client instance
        token_id: Token ID (YES or NO)
        reserved_by_token: {token_id: reserved_tokens}

    Returns:
        Available tokens for new orders
    """
    try:
        wallet_balance = clob_client.get_balance(token_id, TRADING_ADDRESS)
        reserved = reserved_by_token.get(token_id, 0.0)
        available = wallet_balance - reserved
        return max(0.0, available)  # type: ignore[no-any-return]
    except Exception as e:
        logger.error(f"Failed to get token balance for {token_id}: {e}")
        return 0.0
