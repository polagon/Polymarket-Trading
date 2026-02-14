"""
User WebSocket Feed - Order fills, updates, and balance changes.

CRITICAL: Emits Fill events for markout tracking.
"""

import asyncio
import json
import logging
import time
from typing import Callable, List, Optional

import websockets

from config import WS_RECONNECT_DELAY_SECONDS
from models.types import Fill, OrderStatus

logger = logging.getLogger(__name__)


class UserWebSocketFeed:
    """
    WebSocket feed for user-specific events.

    Listens for:
    - Order fills (for markout tracking)
    - Order status updates (for OrderStateStore reconciliation)
    - Balance changes
    """

    def __init__(
        self,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user",
        api_key: Optional[str] = None,
        on_fill_callback: Optional[Callable] = None,
        on_order_update_callback: Optional[Callable] = None,
        on_disconnect_callback: Optional[Callable] = None,
    ):
        """
        Initialize user WebSocket feed.

        Args:
            ws_url: WebSocket URL for user data
            api_key: API key for authentication
            on_fill_callback: Callback when fill received: callback(fill: Fill)
            on_order_update_callback: Callback when order status changes: callback(order_id, status)
            on_disconnect_callback: Callback when connection lost (triggers "unsafe" signal)
        """
        self.ws_url = ws_url
        self.api_key = api_key
        self.on_fill_callback = on_fill_callback
        self.on_order_update_callback = on_order_update_callback
        self.on_disconnect_callback = on_disconnect_callback

        # Connection state
        self.ws = None
        self.running = False
        self.last_message_time = 0

        # Fill history (for markout tracking)
        self.recent_fills: List[Fill] = []

    async def connect(self):
        """Connect to WebSocket with authentication."""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self.ws = await websockets.connect(self.ws_url, extra_headers=headers)  # type: ignore[assignment]
            self.running = True
            logger.info(f"Connected to user WebSocket: {self.ws_url}")
        except Exception as e:
            logger.error(f"Failed to connect to user WebSocket: {e}")
            raise

    async def subscribe(self):
        """
        Subscribe to user events.

        Subscribes to:
        - fills
        - order updates
        - balance changes
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")

        subscribe_msg = {
            "auth": {"api_key": self.api_key} if self.api_key else {},
            "type": "subscribe",
            "channel": "user",
        }

        await self.ws.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to user events")

    async def listen(self):
        """
        Listen for WebSocket messages.

        Runs continuously until stopped.
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")

        logger.info("Starting user WebSocket listener...")

        try:
            async for message in self.ws:
                self.last_message_time = int(time.time() * 1000)

                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("User WebSocket connection closed")
            self.running = False

            # Trigger unsafe callback (should trigger cancel-all)
            if self.on_disconnect_callback:
                self.on_disconnect_callback()

        except Exception as e:
            logger.error(f"User WebSocket listener error: {e}", exc_info=True)
            self.running = False

            if self.on_disconnect_callback:
                self.on_disconnect_callback()

    async def _handle_message(self, data: dict):
        """
        Handle incoming WebSocket message.

        Args:
            data: Parsed JSON message
        """
        msg_type = data.get("type")

        if msg_type == "fill":
            # Fill event (for markout tracking)
            await self._handle_fill(data)

        elif msg_type == "order":
            # Order status update
            await self._handle_order_update(data)

        elif msg_type == "balance":
            # Balance change (informational)
            logger.debug(f"Balance update: {data}")

        elif msg_type == "error":
            logger.error(f"User WebSocket error: {data}")

    async def _handle_fill(self, data: dict):
        """
        Handle fill event.

        CRITICAL: Creates Fill object for markout tracking.

        Args:
            data: Fill message
        """
        try:
            fill_id = data.get("id")
            order_id = data.get("order_id")
            market_id = data.get("market_id") or data.get("condition_id")
            token_id = data.get("token_id") or data.get("asset_id")

            side = data.get("side")  # "BUY" or "SELL"
            price = float(data.get("price", 0))
            size = float(data.get("size", 0))

            timestamp_ms = data.get("timestamp", int(time.time() * 1000))

            # Maker vs taker flag (CRITICAL for Truth Report separation)
            maker = data.get("maker", True)  # Default to maker if not specified

            # Fee paid
            fee_rate_bps = int(data.get("fee_rate_bps", 200))
            fee_paid = float(data.get("fee_paid", 0))

            # Create Fill object
            fill = Fill(
                fill_id=fill_id,  # type: ignore[arg-type]
                order_id=order_id,  # type: ignore[arg-type]
                condition_id=market_id,  # type: ignore[arg-type]
                token_id=token_id,  # type: ignore[arg-type]
                side=side,  # type: ignore[arg-type]
                price=price,
                size_tokens=size,
                timestamp=timestamp_ms,
                maker=maker,
                fee_rate_bps=fee_rate_bps,
                fee_paid_usd=fee_paid,
            )

            self.recent_fills.append(fill)

            # Trigger callback (for markout tracking)
            if self.on_fill_callback:
                self.on_fill_callback(fill)

            logger.info(
                f"Fill received: {fill_id} {side} {size:.2f} {token_id} @ {price:.4f} ({'maker' if maker else 'taker'})"
            )

        except Exception as e:
            logger.error(f"Failed to parse fill message: {e}", exc_info=True)

    async def _handle_order_update(self, data: dict):
        """
        Handle order status update.

        Updates OrderStateStore via callback.

        Args:
            data: Order update message
        """
        try:
            order_id = data.get("order_id") or data.get("id")
            status_str = data.get("status")

            # Map to OrderStatus enum
            status_map = {
                "open": OrderStatus.LIVE,
                "filled": OrderStatus.FILLED,
                "cancelled": OrderStatus.CANCELED,
                "expired": OrderStatus.EXPIRED,
            }

            status = status_map.get(status_str, OrderStatus.PENDING)  # type: ignore[arg-type]

            timestamp = data.get("timestamp")
            if timestamp:
                timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp / 1000))
            else:
                timestamp_iso = None

            # Trigger callback (updates OrderStateStore)
            if self.on_order_update_callback:
                self.on_order_update_callback(order_id, status, timestamp_iso)

            logger.info(f"Order update: {order_id} → {status.value}")

        except Exception as e:
            logger.error(f"Failed to parse order update: {e}", exc_info=True)

    def get_recent_fills(self, since_ms: Optional[int] = None) -> List[Fill]:
        """
        Get recent fills for markout tracking.

        Args:
            since_ms: Return fills since this timestamp (ms)

        Returns:
            List of Fill objects
        """
        if since_ms is None:
            return self.recent_fills

        return [f for f in self.recent_fills if f.timestamp >= since_ms]

    async def reconnect_loop(self):
        """
        Reconnect loop with exponential backoff.

        Automatically reconnects if connection drops.
        """
        while True:
            try:
                await self.connect()
                await self.subscribe()
                await self.listen()

            except Exception as e:
                logger.error(f"User WebSocket error: {e}. Reconnecting in {WS_RECONNECT_DELAY_SECONDS}s...")
                await asyncio.sleep(WS_RECONNECT_DELAY_SECONDS)

    def stop(self):
        """Stop WebSocket feed."""
        self.running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
            logger.info("User WebSocket stopped")

    def get_stats(self) -> dict:
        """Get feed statistics."""
        now_ms = int(time.time() * 1000)

        return {
            "running": self.running,
            "last_message_age_ms": now_ms - self.last_message_time if self.last_message_time else None,
            "recent_fills_count": len(self.recent_fills),
        }
