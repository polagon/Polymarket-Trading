"""
Order State Store - Persistent tracking of every live order.

CRITICAL FIXES:
- #5: Stores BOTH human-friendly AND execution-truth fields for reconciliation
- GAP #3: Partial fill tracking + atomic storage
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from models.types import StoredOrder, OrderStatus
from config import MEMORY_DIR

logger = logging.getLogger(__name__)

ORDER_STORE_FILE = MEMORY_DIR / "order_state_store.json"


class OrderStateStore:
    """
    Persistent store for all orders.

    Responsibilities:
    - Track order lifecycle (PENDING → LIVE → FILLED/CANCELED/EXPIRED)
    - Store execution-truth fields for reconciliation
    - Reconcile with CLOB on restart
    """

    def __init__(self):
        self.orders: dict[str, StoredOrder] = {}
        self._load()

    def _load(self):
        """Load orders from disk."""
        if not ORDER_STORE_FILE.exists():
            logger.info("Order State Store: No existing store, starting fresh")
            return

        try:
            with open(ORDER_STORE_FILE, "r") as f:
                data = json.load(f)

            self.orders = {}
            for order_id, order_dict in data.items():
                # Convert dict to StoredOrder
                self.orders[order_id] = StoredOrder(
                    order_id=order_dict["order_id"],
                    condition_id=order_dict["condition_id"],
                    token_id=order_dict["token_id"],
                    side=order_dict["side"],
                    price=order_dict["price"],
                    size_in_tokens=order_dict["size_in_tokens"],
                    order_type=order_dict["order_type"],
                    post_only=order_dict["post_only"],
                    expiration=order_dict.get("expiration"),
                    fee_rate_bps=order_dict["fee_rate_bps"],
                    maker_amount=order_dict.get("maker_amount", ""),
                    taker_amount=order_dict.get("taker_amount", ""),
                    nonce=order_dict.get("nonce", 0),
                    salt=order_dict.get("salt", 0),
                    signature=order_dict.get("signature", ""),
                    order_hash=order_dict.get("order_hash", ""),
                    status=OrderStatus(order_dict["status"]),
                    placed_at=order_dict["placed_at"],
                    last_seen_ws=order_dict["last_seen_ws"],
                    cancel_reason=order_dict.get("cancel_reason"),
                    clob_error=order_dict.get("clob_error"),
                    # GAP #2 & #3 FIX: Load new fields
                    origin=order_dict.get("origin", "OTHER"),
                    original_size=order_dict.get("original_size", order_dict["size_in_tokens"]),
                    remaining_size=order_dict.get("remaining_size", order_dict["size_in_tokens"]),
                    filled_size=order_dict.get("filled_size", 0.0),
                )

            logger.info(f"Order State Store: Loaded {len(self.orders)} orders from disk")

        except Exception as e:
            logger.error(f"Failed to load Order State Store: {e}", exc_info=True)
            self.orders = {}

    def _save(self):
        """
        Save orders to disk with atomic write (GAP #3 FIX).

        CRITICAL: Uses temp file + atomic rename to prevent corruption on crash.
        """
        try:
            # Convert StoredOrder objects to dicts
            data = {}
            for order_id, order in self.orders.items():
                data[order_id] = {
                    "order_id": order.order_id,
                    "condition_id": order.condition_id,
                    "token_id": order.token_id,
                    "side": order.side,
                    "price": order.price,
                    "size_in_tokens": order.size_in_tokens,
                    "order_type": order.order_type,
                    "post_only": order.post_only,
                    "expiration": order.expiration,
                    "fee_rate_bps": order.fee_rate_bps,
                    "maker_amount": order.maker_amount,
                    "taker_amount": order.taker_amount,
                    "nonce": order.nonce,
                    "salt": order.salt,
                    "signature": order.signature,
                    "order_hash": order.order_hash,
                    "status": order.status.value,
                    "placed_at": order.placed_at,
                    "last_seen_ws": order.last_seen_ws,
                    "cancel_reason": order.cancel_reason,
                    "clob_error": order.clob_error,
                    # GAP #2 & #3 FIX: Additional fields
                    "origin": order.origin,
                    "original_size": order.original_size,
                    "remaining_size": order.remaining_size,
                    "filled_size": order.filled_size,
                }

            ORDER_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)

            # GAP #3 FIX: Atomic write via temp file + rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=ORDER_STORE_FILE.parent, prefix=".order_store_", suffix=".tmp"
            )
            try:
                with os.fdopen(temp_fd, "w") as f:
                    json.dump(data, f, indent=2)

                # Atomic rename
                os.replace(temp_path, ORDER_STORE_FILE)

            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e

        except Exception as e:
            logger.error(f"Failed to save Order State Store: {e}", exc_info=True)

    def add_order(self, order: StoredOrder):
        """
        Add new order to store.

        Args:
            order: StoredOrder instance
        """
        self.orders[order.order_id] = order
        self._save()
        logger.info(
            f"Order added: {order.order_id} {order.side} {order.size_in_tokens} "
            f"{order.token_id} @ {order.price:.4f} ({order.order_type})"
        )

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        ws_timestamp: Optional[str] = None,
        cancel_reason: Optional[str] = None,
        clob_error: Optional[str] = None,
    ):
        """
        Update order status from WebSocket or CLOB response.

        Args:
            order_id: Order ID
            status: New status
            ws_timestamp: ISO timestamp from WS update
            cancel_reason: Reason for cancellation (if applicable)
            clob_error: Error from CLOB (if applicable)
        """
        if order_id not in self.orders:
            logger.warning(f"Cannot update unknown order: {order_id}")
            return

        order = self.orders[order_id]
        old_status = order.status
        order.status = status

        if ws_timestamp:
            order.last_seen_ws = ws_timestamp

        if cancel_reason:
            order.cancel_reason = cancel_reason

        if clob_error:
            order.clob_error = clob_error

        self._save()

        logger.info(f"Order status updated: {order_id} {old_status.value} → {status.value}")

    def get_live_orders(self) -> list[StoredOrder]:
        """
        Get all orders with status=LIVE.

        Returns:
            List of live orders
        """
        return [o for o in self.orders.values() if o.status == OrderStatus.LIVE]

    def get_orders_by_market(self, condition_id: str) -> list[StoredOrder]:
        """Get all orders for a specific market."""
        return [o for o in self.orders.values() if o.condition_id == condition_id]

    def get_live_orders_by_market(self, condition_id: str) -> list[StoredOrder]:
        """Get live orders for a specific market."""
        return [
            o
            for o in self.orders.values()
            if o.condition_id == condition_id and o.status == OrderStatus.LIVE
        ]

    def reconcile_with_clob(self, clob_open_orders: list[dict]):
        """
        Reconcile Order State Store with CLOB on startup.

        CRITICAL: This prevents divergence between local state and CLOB truth.

        Args:
            clob_open_orders: List of open orders from CLOB API
                [{"orderId": "...", "orderHash": "...", ...}, ...]

        Returns:
            dict with reconciliation stats
        """
        logger.info(f"Reconciling Order State Store with CLOB...")

        clob_order_ids = {o["orderId"] for o in clob_open_orders}
        store_live_order_ids = {
            order_id for order_id, order in self.orders.items() if order.status == OrderStatus.LIVE
        }

        # Orders in store but not in CLOB → mark CANCELED
        missing_in_clob = store_live_order_ids - clob_order_ids
        for order_id in missing_in_clob:
            self.update_order_status(
                order_id,
                OrderStatus.CANCELED,
                cancel_reason="Reconciliation: order not found in CLOB",
            )

        # Orders in CLOB but not in store → add as UNEXPECTED
        unexpected_in_clob = clob_order_ids - store_live_order_ids
        for clob_order in clob_open_orders:
            if clob_order["orderId"] in unexpected_in_clob:
                logger.warning(
                    f"UNEXPECTED ORDER in CLOB: {clob_order['orderId']} "
                    f"(not in local store, possibly from previous session)"
                )
                # Optionally: add to store with LIVE status
                # For now, just log

        stats = {
            "store_live_count": len(store_live_order_ids),
            "clob_open_count": len(clob_order_ids),
            "missing_in_clob": len(missing_in_clob),
            "unexpected_in_clob": len(unexpected_in_clob),
        }

        logger.info(f"Reconciliation complete: {stats}")
        return stats

    def cancel_stale_orders(self, staleness_threshold_ms: int):
        """
        Mark orders as EXPIRED if not seen in WS for too long.

        CRITICAL: Prevents quoting with stale orders that may have filled/canceled.

        Args:
            staleness_threshold_ms: Max age before considering stale
        """
        now = datetime.now(timezone.utc)
        stale_count = 0

        for order_id, order in self.orders.items():
            if order.status != OrderStatus.LIVE:
                continue

            if not order.last_seen_ws:
                # No WS update yet, check placed_at
                placed_dt = datetime.fromisoformat(order.placed_at)
            else:
                placed_dt = datetime.fromisoformat(order.last_seen_ws)

            age_ms = (now - placed_dt.replace(tzinfo=timezone.utc)).total_seconds() * 1000

            if age_ms > staleness_threshold_ms:
                self.update_order_status(
                    order_id,
                    OrderStatus.EXPIRED,
                    cancel_reason=f"Stale: no WS update for {age_ms:.0f}ms",
                )
                stale_count += 1

        if stale_count > 0:
            logger.warning(f"Marked {stale_count} orders as EXPIRED (stale)")

    def update_partial_fill(self, order_id: str, fill_size: float, fill_timestamp: str):
        """
        Update order with partial fill (GAP #3 FIX).

        CRITICAL: Must release partial reservations after this is called.

        Args:
            order_id: Order ID
            fill_size: Size filled (in tokens)
            fill_timestamp: ISO timestamp of fill
        """
        if order_id not in self.orders:
            logger.warning(f"Cannot update partial fill for unknown order: {order_id}")
            return

        order = self.orders[order_id]

        # Update fill tracking
        order.filled_size += fill_size
        order.remaining_size -= fill_size
        order.last_seen_ws = fill_timestamp

        # If fully filled, mark as FILLED
        if order.remaining_size <= 0:
            order.status = OrderStatus.FILLED
            logger.info(
                f"Order {order_id} FULLY FILLED: "
                f"{order.filled_size}/{order.original_size} tokens"
            )
        else:
            # Partial fill, keep LIVE
            logger.info(
                f"Order {order_id} PARTIAL FILL: "
                f"{order.filled_size}/{order.original_size} tokens "
                f"(remaining: {order.remaining_size})"
            )

        self._save()

    def get_stats(self) -> dict:
        """Get summary statistics."""
        stats = {
            "total": len(self.orders),
            "pending": sum(1 for o in self.orders.values() if o.status == OrderStatus.PENDING),
            "live": sum(1 for o in self.orders.values() if o.status == OrderStatus.LIVE),
            "filled": sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED),
            "canceled": sum(1 for o in self.orders.values() if o.status == OrderStatus.CANCELED),
            "expired": sum(1 for o in self.orders.values() if o.status == OrderStatus.EXPIRED),
        }
        return stats
