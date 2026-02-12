"""
Portfolio Risk Engine - Single source of truth for all risk caps.

CRITICAL FIXES:
- #5: Token-level inventory, mark-to-mid USD aggregation
- #17: negRisk events → single cluster
- #18: Balance reservations from open orders
- #21: Near-close ratchet uses time_to_close
"""
import logging
import hashlib
from typing import Optional, Tuple
from collections import defaultdict

from models.types import Market, Event, PortfolioExposure, Cluster
from execution import units
from risk import market_state
from config import (
    BANKROLL,
    MAX_CLUSTER_EXPOSURE_PCT,
    MAX_AGG_EXPOSURE_PCT,
    MAX_MARKET_INVENTORY_PCT,
    SATELLITE_RISK_BUDGET_PCT,
    TAKER_RISK_BUDGET_PCT,
    NEAR_RESOLUTION_HOURS,
    NEAR_RESOLUTION_CAP_MULTIPLIER,
    NEG_RISK_SINGLE_CLUSTER,
)

logger = logging.getLogger(__name__)


class PortfolioRiskEngine:
    """
    Central risk management engine.

    Strategy modules CANNOT bypass this engine.
    """

    def __init__(self):
        self.exposure = PortfolioExposure()
        self.cluster_cache: dict[str, str] = {}  # condition_id → cluster_id

    def assign_cluster(self, market: Market) -> str:
        """
        Assign market to cluster (deterministic).

        CRITICAL FIX #17: If negRisk event, all markets in event → one cluster.

        Args:
            market: Market with optional event association

        Returns:
            cluster_id (deterministic)
        """
        # Check if already cached
        if market.condition_id in self.cluster_cache:
            return self.cluster_cache[market.condition_id]

        # negRisk event handling
        if market.event and (market.event.neg_risk or market.event.augmented_neg_risk):
            if NEG_RISK_SINGLE_CLUSTER:
                cluster_id = f"negRisk_event_{market.event.event_id}"
                logger.info(
                    f"Market {market.condition_id}: negRisk event → cluster {cluster_id}"
                )
                self.cluster_cache[market.condition_id] = cluster_id
                return cluster_id

        # Normal clustering: deterministic hash based on market metadata
        cluster_id = self._deterministic_cluster_assignment(market)
        self.cluster_cache[market.condition_id] = cluster_id
        return cluster_id

    def _deterministic_cluster_assignment(self, market: Market) -> str:
        """
        Deterministic cluster assignment based on market metadata.

        CRITICAL: Same market text/rules → same cluster_id across restarts.

        Strategy:
        - Extract entities (NER), dates, resolution source
        - Hash to create deterministic cluster_id
        - Default fallback: category-based clustering

        Args:
            market: Market instance

        Returns:
            cluster_id (deterministic string)
        """
        # Simple implementation: use category + resolution_source hash
        # TODO: Implement proper NER-based clustering

        key_str = f"{market.category}|{market.resolution_source}"
        cluster_hash = hashlib.md5(key_str.encode()).hexdigest()[:8]

        cluster_id = f"cluster_{market.category}_{cluster_hash}"
        return cluster_id

    def can_enter_position(
        self, market: Market, size_usd: float, cluster_id: Optional[str] = None, rrs: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Check if new position violates any risk cap.

        CRITICAL FIXES:
        - #21: Near-close ratchet uses time_to_close
        - Market state multipliers applied

        Args:
            market: Market instance
            size_usd: Position size in USD
            cluster_id: Cluster ID (computed if not provided)
            rrs: Resolution Risk Score

        Returns:
            (allowed: bool, reason: str)
        """
        if cluster_id is None:
            cluster_id = self.assign_cluster(market)

        # Get state-based position cap multiplier
        multiplier = market_state.compute_position_cap_multiplier(market, rrs)

        # Per-market cap
        current_market_exposure = self.exposure.market_exposures.get(market.condition_id, 0.0)
        max_market = BANKROLL * MAX_MARKET_INVENTORY_PCT * multiplier

        if current_market_exposure + size_usd > max_market:
            return (
                False,
                f"Market cap exceeded: ${current_market_exposure + size_usd:.2f} > ${max_market:.2f}",
            )

        # Cluster cap
        current_cluster_exposure = self.exposure.cluster_exposures.get(cluster_id, 0.0)
        max_cluster = BANKROLL * MAX_CLUSTER_EXPOSURE_PCT

        if current_cluster_exposure + size_usd > max_cluster:
            return (
                False,
                f"Cluster cap exceeded: {cluster_id} ${current_cluster_exposure + size_usd:.2f} > ${max_cluster:.2f}",
            )

        # Aggregate cap
        total_exposure = self.exposure.total_exposure_usd
        max_agg = BANKROLL * MAX_AGG_EXPOSURE_PCT

        if total_exposure + size_usd > max_agg:
            return (
                False,
                f"Aggregate cap exceeded: ${total_exposure + size_usd:.2f} > ${max_agg:.2f}",
            )

        # Near-close ratchet (CRITICAL FIX #21)
        if market.time_to_close and market.time_to_close < NEAR_RESOLUTION_HOURS:
            # Tighten caps by NEAR_RESOLUTION_CAP_MULTIPLIER
            adjusted_max = max_market * NEAR_RESOLUTION_CAP_MULTIPLIER
            if size_usd > adjusted_max:
                return (
                    False,
                    f"Near-close cap: max position reduced to ${adjusted_max:.2f}",
                )

        return (True, "OK")

    def update_exposure(
        self, cluster_id: str, market_id: str, token_id: str, delta_usd: float, delta_tokens: float
    ):
        """
        Update exposure after position change (entry or exit).

        CRITICAL FIX #5: Token-level inventory tracking.

        Args:
            cluster_id: Cluster ID
            market_id: Market condition_id
            token_id: Token ID (YES or NO)
            delta_usd: Change in USD exposure (mark-to-mid)
            delta_tokens: Change in token count (signed)
        """
        # Update cluster exposure
        self.exposure.cluster_exposures[cluster_id] = (
            self.exposure.cluster_exposures.get(cluster_id, 0.0) + delta_usd
        )

        # Update market exposure
        self.exposure.market_exposures[market_id] = (
            self.exposure.market_exposures.get(market_id, 0.0) + delta_usd
        )

        # Update token inventory (CRITICAL FIX #5)
        self.exposure.token_inventory[token_id] = (
            self.exposure.token_inventory.get(token_id, 0.0) + delta_tokens
        )

        # Update aggregate
        self.exposure.total_exposure_usd += delta_usd

        logger.debug(
            f"Exposure updated: {market_id} {cluster_id} "
            f"delta_usd={delta_usd:.2f} delta_tokens={delta_tokens:.2f}"
        )

    def reserve_for_order(self, order: dict):
        """
        Reserve balance when order is submitted.

        CRITICAL FIX #18: Track reserved balances from open orders.

        Args:
            order: Order dict with condition_id, token_id, side, price, size_in_tokens
        """
        if order["side"] == "BUY":
            # Reserve USDC needed for this order
            cost = order["price"] * order["size_in_tokens"]
            market_id = order["condition_id"]
            self.exposure.reserved_usdc_by_market[market_id] = (
                self.exposure.reserved_usdc_by_market.get(market_id, 0.0) + cost
            )
            logger.debug(f"Reserved ${cost:.2f} USDC for BUY order in {market_id}")

        else:  # SELL
            # Reserve tokens being sold
            token_id = order["token_id"]
            size = order["size_in_tokens"]
            self.exposure.reserved_tokens_by_token_id[token_id] = (
                self.exposure.reserved_tokens_by_token_id.get(token_id, 0.0) + size
            )
            logger.debug(f"Reserved {size:.2f} tokens for SELL order of {token_id}")

    def release_reservation(self, order: dict):
        """
        Release reservation when order filled/canceled.

        CRITICAL FIX #18: Must release reservations to free balance.

        Args:
            order: Order dict
        """
        if order["side"] == "BUY":
            cost = order["price"] * order["size_in_tokens"]
            market_id = order["condition_id"]
            current = self.exposure.reserved_usdc_by_market.get(market_id, 0.0)
            self.exposure.reserved_usdc_by_market[market_id] = max(0.0, current - cost)
            logger.debug(f"Released ${cost:.2f} USDC reservation for {market_id}")

        else:  # SELL
            token_id = order["token_id"]
            size = order["size_in_tokens"]
            current = self.exposure.reserved_tokens_by_token_id.get(token_id, 0.0)
            self.exposure.reserved_tokens_by_token_id[token_id] = max(0.0, current - size)
            logger.debug(f"Released {size:.2f} tokens reservation for {token_id}")

    def update_reservation_partial_fill(self, order_id: str, fill_size: float, order: dict):
        """
        Reduce reservation on partial fill (GAP #4 FIX).

        CRITICAL: Must be called whenever OrderStateStore.update_partial_fill() is called.

        Args:
            order_id: Order ID
            fill_size: Size filled (in tokens)
            order: Order dict with condition_id, token_id, side, price
        """
        if order["side"] == "BUY":
            # Release USDC proportional to fill
            released_usdc = fill_size * order["price"]
            market_id = order["condition_id"]

            current = self.exposure.reserved_usdc_by_market.get(market_id, 0.0)
            self.exposure.reserved_usdc_by_market[market_id] = max(0.0, current - released_usdc)

            logger.info(
                f"Partial fill reservation release: {order_id} "
                f"released ${released_usdc:.2f} USDC (filled {fill_size} tokens)"
            )

        else:  # SELL
            # Release tokens proportional to fill
            token_id = order["token_id"]

            current = self.exposure.reserved_tokens_by_token_id.get(token_id, 0.0)
            self.exposure.reserved_tokens_by_token_id[token_id] = max(0.0, current - fill_size)

            logger.info(
                f"Partial fill reservation release: {order_id} "
                f"released {fill_size:.2f} tokens"
            )

    def transfer_reservation_on_replace(self, old_order: dict, new_order: dict):
        """
        Transfer reservation from old → new order (GAP #4 FIX).

        CRITICAL: Prevents double reservation on cancel/replace.

        Args:
            old_order: Order being replaced
            new_order: New order
        """
        # Release old reservation
        self.release_reservation(old_order)

        # Reserve for new order
        self.reserve_for_order(new_order)

        logger.info(
            f"Reservation transferred: old={old_order.get('order_id', 'unknown')} → "
            f"new={new_order.get('order_id', 'unknown')}"
        )

    def can_trade_parity_arb(self, market: Market) -> bool:
        """
        Check if parity arbitrage is allowed for this market.

        CRITICAL FIX #17: Disable parity scans for negRisk events.

        Args:
            market: Market instance

        Returns:
            True if parity arb allowed, False otherwise
        """
        if market.event and (market.event.neg_risk or market.event.augmented_neg_risk):
            logger.warning(
                f"Market {market.condition_id}: negRisk event → parity arb disabled"
            )
            return False

        return True

    def get_cluster_exposure(self, cluster_id: str) -> float:
        """Get current exposure for a cluster."""
        return self.exposure.cluster_exposures.get(cluster_id, 0.0)

    def get_market_exposure(self, market_id: str) -> float:
        """Get current exposure for a market."""
        return self.exposure.market_exposures.get(market_id, 0.0)

    def get_aggregate_exposure(self) -> float:
        """Get total portfolio exposure."""
        return self.exposure.total_exposure_usd

    def get_available_usdc(self, market_id: str, wallet_balance: float) -> float:
        """
        Get available USDC for new BUY orders in this market (GAP #4 FIX).

        CRITICAL: Must account for existing reservations.

        Args:
            market_id: Market condition_id
            wallet_balance: Total USDC balance from wallet

        Returns:
            Available USDC for new orders
        """
        reserved = self.exposure.reserved_usdc_by_market.get(market_id, 0.0)
        # Note: This is per-market. In reality, reservations are global across all markets.
        # For production, should sum ALL reservations across all markets.
        total_reserved = sum(self.exposure.reserved_usdc_by_market.values())
        available = wallet_balance - total_reserved
        return max(0.0, available)

    def get_available_tokens(self, token_id: str, wallet_balance: float) -> float:
        """
        Get available tokens for new SELL orders (GAP #4 FIX).

        Args:
            token_id: Token ID
            wallet_balance: Total token balance from wallet

        Returns:
            Available tokens for new orders
        """
        reserved = self.exposure.reserved_tokens_by_token_id.get(token_id, 0.0)
        available = wallet_balance - reserved
        return max(0.0, available)

    def get_stats(self) -> dict:
        """Get portfolio risk statistics."""
        return {
            "total_exposure_usd": self.exposure.total_exposure_usd,
            "cluster_count": len(self.exposure.cluster_exposures),
            "market_count": len(self.exposure.market_exposures),
            "satellite_risk_used": self.exposure.satellite_risk_used_usd,
            "taker_risk_used": self.exposure.taker_risk_used_usd,
            "reserved_usdc_total": sum(self.exposure.reserved_usdc_by_market.values()),
            "reserved_tokens_total": sum(self.exposure.reserved_tokens_by_token_id.values()),
        }
