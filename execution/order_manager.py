"""
Order Manager — maker-only bounded execution with TTL, stale-cancel, and never-cross.

Hard rules:
  - post_only always True
  - Never cross at submit or replace
  - TTL always enforced
  - Stale → cancel (not flag)
  - Bounded replace by max_chase_ticks
  - No improving replace when toxic or WIDE/BROKEN (cancel-only)
  - Prices are tick-rounded AWAY from crossing

Loop 4: No taker escalation.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models.reasons import (
    REASON_ORDER_CHASE_EXCEEDED,
    REASON_ORDER_CROSS_REJECTED,
    REASON_ORDER_STALE,
    REASON_ORDER_TOXIC_NO_IMPROVE,
    REASON_ORDER_TTL_EXPIRED,
)


class ExecutionMode(str, Enum):
    MAKER_ONLY = "MAKER_ONLY"
    NO_TRADE = "NO_TRADE"


@dataclass
class ManagedOrder:
    """A managed order with lifecycle tracking.

    Attributes:
        order_id: Unique order identifier.
        market_id: Market condition_id.
        side: "BUY_YES" or "BUY_NO".
        price: Order price (tick-aligned).
        size: Order size in USD.
        post_only: Always True in Loop 4.
        ttl_seconds: Time-to-live in seconds.
        created_at: Epoch timestamp of creation.
        last_refreshed_at: Epoch timestamp of last refresh.
        canceled: Whether the order has been canceled.
        cancel_reason: Reason for cancellation.
    """

    order_id: str
    market_id: str
    side: str
    price: float
    size: float
    post_only: bool = True
    ttl_seconds: int = 120
    created_at: float = 0.0
    last_refreshed_at: float = 0.0
    canceled: bool = False
    cancel_reason: str = ""


def _round_price_away_from_cross(price: float, side: str, tick_size: float) -> float:
    """Round price to nearest tick AWAY from crossing.

    BUY: round DOWN to avoid crossing.
    SELL: round UP to avoid crossing.
    """
    if side.startswith("BUY"):
        return math.floor(price / tick_size) * tick_size
    else:
        return math.ceil(price / tick_size) * tick_size


def _would_cross(price: float, side: str, best_bid: float, best_ask: float) -> bool:
    """Check if a price would cross the book.

    BUY at or above best_ask → crossing.
    SELL at or below best_bid → crossing.
    """
    if side.startswith("BUY"):
        return price >= best_ask
    else:
        return price <= best_bid


class OrderManager:
    """Maker-only order manager with TTL, stale-cancel, and never-cross enforcement.

    Args:
        default_ttl: Default time-to-live in seconds (120).
        stale_threshold: Seconds before order is considered stale (60).
        max_chase_ticks: Maximum price improvement distance in ticks (3).
        tick_size: Price tick size (0.01 for Polymarket cent ticks).
    """

    def __init__(
        self,
        default_ttl: int = 120,
        stale_threshold: int = 60,
        max_chase_ticks: int = 3,
        tick_size: float = 0.01,
    ) -> None:
        self._orders: dict[str, ManagedOrder] = {}
        self._default_ttl = default_ttl
        self._stale_threshold = stale_threshold
        self._max_chase_ticks = max_chase_ticks
        self._tick_size = tick_size

    def submit(
        self,
        market_id: str,
        side: str,
        price: float,
        size: float,
        best_bid: float,
        best_ask: float,
        ttl: Optional[int] = None,
        now: float = 0.0,
    ) -> ManagedOrder:
        """Submit a new maker-only order.

        Price is tick-rounded away from crossing before validation.

        Args:
            market_id: Market condition_id.
            side: "BUY_YES" or "BUY_NO".
            price: Desired price (will be tick-rounded).
            size: Order size in USD.
            best_bid: Current best bid.
            best_ask: Current best ask.
            ttl: Time-to-live override (default: default_ttl).
            now: Current epoch time.

        Returns:
            The created ManagedOrder.

        Raises:
            ValueError: If the tick-rounded price would cross.
        """
        # Round price away from crossing
        rounded_price = _round_price_away_from_cross(price, side, self._tick_size)

        # Cross check after rounding
        if _would_cross(rounded_price, side, best_bid, best_ask):
            raise ValueError(
                f"{REASON_ORDER_CROSS_REJECTED}: {side} @ {rounded_price:.4f} vs bid={best_bid:.4f}/ask={best_ask:.4f}"
            )

        order = ManagedOrder(
            order_id=uuid.uuid4().hex[:16],
            market_id=market_id,
            side=side,
            price=rounded_price,
            size=size,
            post_only=True,
            ttl_seconds=ttl or self._default_ttl,
            created_at=now,
            last_refreshed_at=now,
        )
        self._orders[order.order_id] = order
        return order

    def enforce_ttl(self, now: float) -> list[str]:
        """Cancel all orders past TTL. Returns list of canceled order_ids."""
        canceled_ids: list[str] = []
        for order in list(self._orders.values()):
            if order.canceled:
                continue
            elapsed = now - order.created_at
            if elapsed >= order.ttl_seconds:
                order.canceled = True
                order.cancel_reason = REASON_ORDER_TTL_EXPIRED
                canceled_ids.append(order.order_id)
        return canceled_ids

    def check_stale(self, now: float) -> list[str]:
        """Cancel (not flag) orders not refreshed within stale_threshold.

        Returns list of canceled order_ids.
        """
        canceled_ids: list[str] = []
        for order in list(self._orders.values()):
            if order.canceled:
                continue
            since_refresh = now - order.last_refreshed_at
            if since_refresh >= self._stale_threshold:
                order.canceled = True
                order.cancel_reason = REASON_ORDER_STALE
                canceled_ids.append(order.order_id)
        return canceled_ids

    def can_improve_price(self, toxicity_flag: bool, spread_regime: str) -> bool:
        """Check if price improvement (cancel/replace) is allowed.

        Returns False if toxic or WIDE/BROKEN regime — prevents converting
        informed flow into adverse selection.
        """
        if toxicity_flag:
            return False
        if spread_regime in ("WIDE", "BROKEN"):
            return False
        return True

    def cancel_replace(
        self,
        order_id: str,
        new_price: float,
        new_size: float,
        best_bid: float,
        best_ask: float,
        toxicity_flag: bool,
        spread_regime: str,
    ) -> Optional[ManagedOrder]:
        """Cancel old order and create new one within chase distance.

        Returns None and only cancels if:
          - toxic or WIDE/BROKEN (no improvement allowed)
          - new price would cross
          - chase distance exceeded

        Args:
            order_id: Order to replace.
            new_price: Desired new price.
            new_size: Desired new size.
            best_bid: Current best bid.
            best_ask: Current best ask.
            toxicity_flag: Whether flow is toxic.
            spread_regime: Current spread regime.

        Returns:
            New ManagedOrder if replacement succeeded, None if only canceled.
        """
        old_order = self._orders.get(order_id)
        if old_order is None or old_order.canceled:
            return None

        # Check if improvement is allowed
        if not self.can_improve_price(toxicity_flag, spread_regime):
            # Cancel only, no replacement
            old_order.canceled = True
            old_order.cancel_reason = REASON_ORDER_TOXIC_NO_IMPROVE
            return None

        # Round new price
        rounded_new = _round_price_away_from_cross(new_price, old_order.side, self._tick_size)

        # Cross check
        if _would_cross(rounded_new, old_order.side, best_bid, best_ask):
            old_order.canceled = True
            old_order.cancel_reason = REASON_ORDER_CROSS_REJECTED
            return None

        # Chase distance check
        chase_distance = abs(rounded_new - old_order.price)
        max_chase = self._max_chase_ticks * self._tick_size
        if chase_distance > max_chase + 1e-9:  # eps tolerance for float comparison
            old_order.canceled = True
            old_order.cancel_reason = REASON_ORDER_CHASE_EXCEEDED
            return None

        # Cancel old order
        old_order.canceled = True
        old_order.cancel_reason = "replaced"

        # Create new order
        now = old_order.created_at  # preserve original time context
        new_order = ManagedOrder(
            order_id=uuid.uuid4().hex[:16],
            market_id=old_order.market_id,
            side=old_order.side,
            price=rounded_new,
            size=new_size,
            post_only=True,
            ttl_seconds=old_order.ttl_seconds,
            created_at=now,
            last_refreshed_at=now,
        )
        self._orders[new_order.order_id] = new_order
        return new_order

    def cancel(self, order_id: str, reason: str) -> bool:
        """Cancel a specific order.

        Returns True if order was found and canceled.
        """
        order = self._orders.get(order_id)
        if order is None or order.canceled:
            return False
        order.canceled = True
        order.cancel_reason = reason
        return True

    def cancel_all(self, market_id: str = "") -> int:
        """Cancel all active orders, optionally filtered by market_id.

        Returns count of newly canceled orders.
        """
        count = 0
        for order in self._orders.values():
            if order.canceled:
                continue
            if market_id and order.market_id != market_id:
                continue
            order.canceled = True
            order.cancel_reason = "cancel_all"
            count += 1
        return count

    def refresh(self, order_id: str, now: float = 0.0) -> bool:
        """Refresh an order's stale timer.

        Returns True if order was found and refreshed.
        """
        order = self._orders.get(order_id)
        if order is None or order.canceled:
            return False
        order.last_refreshed_at = now
        return True

    def get_active_orders(self, market_id: str = "") -> list[ManagedOrder]:
        """Get all active (non-canceled) orders.

        Args:
            market_id: Optional filter by market_id.

        Returns:
            List of active ManagedOrders.
        """
        result = []
        for order in self._orders.values():
            if order.canceled:
                continue
            if market_id and order.market_id != market_id:
                continue
            result.append(order)
        return result
