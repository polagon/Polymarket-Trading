"""
Parity Scanner - YES/NO consistency arbitrage.

CRITICAL FIX #2, #7: Query BOTH YES and NO books separately (not identity formula).
"""

import logging
from typing import List, Optional

from config import DISABLE_PARITY_FOR_NEG_RISK, PARITY_MIN_PROFIT
from execution import fees
from models.types import ArbitrageOpp, Market, OrderBook
from risk.portfolio_engine import PortfolioRiskEngine

logger = logging.getLogger(__name__)


def scan_parity_arb(
    market: Market,
    yes_book: OrderBook,
    no_book: OrderBook,
    risk_engine: PortfolioRiskEngine,
) -> Optional[ArbitrageOpp]:
    """
    Scan for YES/NO parity arbitrage.

    CRITICAL FIX #2: Query BOTH order books separately.
    NEVER use no_price = 1 - yes_price identity formula.

    CRITICAL FIX #7: This is NOT risk-free (leg risk exists).

    CRITICAL FIX #17: Disabled for negRisk markets.

    Args:
        market: Market instance
        yes_book: Order book for YES token
        no_book: Order book for NO token
        risk_engine: Portfolio risk engine

    Returns:
        ArbitrageOpp if profitable, None otherwise
    """
    # Check if parity trading allowed (CRITICAL FIX #17)
    if not risk_engine.can_trade_parity_arb(market):
        return None

    # Get best asks from BOTH books (CRITICAL FIX #2)
    yes_ask = yes_book.best_ask
    no_ask = no_book.best_ask

    if yes_ask is None or no_ask is None:
        return None

    # Check if books are fresh
    if yes_book.timestamp_age_ms > 5000 or no_book.timestamp_age_ms > 5000:
        logger.debug(f"Parity scan: {market.condition_id} books stale")
        return None

    # Compute net profit after fees
    net_profit = fees.net_parity_profit(yes_ask, no_ask, market.fee_rate_bps, size=1.0)

    if net_profit < PARITY_MIN_PROFIT:
        return None

    # Found profitable arb!
    opp = ArbitrageOpp(
        type="YES_NO_PARITY",
        markets=[market],
        legs=[
            {"token": market.yes_token_id, "side": "BUY", "price": yes_ask},
            {"token": market.no_token_id, "side": "BUY", "price": no_ask},
        ],
        expected_profit=net_profit,
        execution_mode="taker",  # Parity is taker mode
        max_leg_time_ms=5000,  # CRITICAL FIX #7: Leg risk awareness
        requires_atomic=True,
    )

    logger.info(
        f"Parity arb found: {market.condition_id} "
        f"YES@{yes_ask:.4f} + NO@{no_ask:.4f} = {yes_ask + no_ask:.4f} "
        f"(profit={net_profit:.4f})"
    )

    return opp


def scan_all_parity(
    markets: List[Market],
    yes_books: dict,
    no_books: dict,
    risk_engine: PortfolioRiskEngine,
) -> List[ArbitrageOpp]:
    """
    Scan all markets for parity arb.

    Args:
        markets: List of markets
        yes_books: {condition_id: OrderBook} for YES tokens
        no_books: {condition_id: OrderBook} for NO tokens
        risk_engine: Portfolio risk engine

    Returns:
        List of arbitrage opportunities
    """
    opps = []

    for market in markets:
        yes_book = yes_books.get(market.condition_id)
        no_book = no_books.get(market.condition_id)

        if not yes_book or not no_book:
            continue

        opp = scan_parity_arb(market, yes_book, no_book, risk_engine)
        if opp:
            opps.append(opp)

    if opps:
        logger.info(f"Parity scan: found {len(opps)} opportunities")

    return opps
