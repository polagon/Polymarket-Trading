"""
Market WebSocket Feed - L2 order book ingestion.

CRITICAL: Staleness detection triggers global "unsafe" signal for cancel-all.
"""
import asyncio
import json
import logging
import time
from typing import Optional, Callable, Dict
from collections import defaultdict, deque
import websockets

from models.types import OrderBook
from config import WS_STALENESS_THRESHOLD_MS, WS_RECONNECT_DELAY_SECONDS

logger = logging.getLogger(__name__)


class MarketWebSocketFeed:
    """
    WebSocket feed for market L2 order books.

    Maintains in-memory OrderBook per token_id.
    Tracks staleness and churn for QS computation.
    """

    def __init__(
        self,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        on_stale_callback: Optional[Callable] = None,
    ):
        """
        Initialize market WebSocket feed.

        Args:
            ws_url: WebSocket URL for market data
            on_stale_callback: Callback function triggered when book goes stale
        """
        self.ws_url = ws_url
        self.on_stale_callback = on_stale_callback

        # Order books indexed by token_id
        self.books: Dict[str, OrderBook] = {}

        # Churn tracking (updates per second, rolling window)
        self.update_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # 60s window
        self.last_update_times: Dict[str, int] = {}

        # Connection state
        self.ws = None
        self.running = False
        self.last_message_time = 0

        # Subscribed assets
        self.subscribed_assets: set[str] = set()

        # Feed health metrics (NEW)
        self.ws_messages_total = 0
        self.ws_json_parse_errors = 0
        self.book_messages_total = 0
        self.book_messages_by_asset: Dict[str, int] = defaultdict(int)
        self.subscriptions_outbound_total = 0

        # WS truth observability
        self.msg_type_counts: Dict[str, int] = defaultdict(int)
        self.unique_asset_ids_with_book: set[str] = set()
        self.first_book_batch_logged = False

    async def connect(self):
        """Connect to WebSocket."""
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.running = True
            logger.info(f"Connected to market WebSocket: {self.ws_url}")

            # Start PING keepalive loop
            asyncio.create_task(self._keepalive_loop())
        except Exception as e:
            logger.error(f"Failed to connect to market WebSocket: {e}")
            raise

    async def _keepalive_loop(self):
        """Send PING every 10 seconds to keep connection alive (per Polymarket docs)."""
        while self.running and self.ws:
            try:
                await asyncio.sleep(10)
                if self.ws and self.running:
                    await self.ws.send("PING")
                    logger.debug("Sent PING keepalive")
            except Exception as e:
                logger.error(f"Keepalive error: {e}")
                break

    async def subscribe(self, asset_ids: list[str]):
        """
        Subscribe to L2 order book updates for assets.

        Per Polymarket docs: send one subscription message per asset
        {"auth": {}, "type": "subscribe", "channel": "market", "asset_id": "..."}

        Args:
            asset_ids: List of asset IDs (token_ids) to subscribe to
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")

        if not asset_ids:
            return

        # Try batch subscription per Polymarket docs
        # Format: {"assets_ids": [...], "type": "market"}
        subscribe_msg = {
            "assets_ids": asset_ids,
            "type": "market"
        }

        payload_json = json.dumps(subscribe_msg)

        # Log first 2 batches in detail
        if self.subscriptions_outbound_total < 2:
            logger.info(
                f"OUTBOUND SUB #{self.subscriptions_outbound_total + 1}: "
                f"count={len(asset_ids)} "
                f"payload={payload_json[:300]}"
            )

        await self.ws.send(payload_json)
        self.subscriptions_outbound_total += 1

        for asset_id in asset_ids:
            self.subscribed_assets.add(asset_id)

        logger.info(
            f"Subscription batch {self.subscriptions_outbound_total} sent: "
            f"{len(asset_ids)} assets (total subscribed: {len(self.subscribed_assets)})"
        )

    async def unsubscribe(self, asset_ids: list[str]):
        """
        Unsubscribe from assets.

        Uses official Polymarket format:
        {"assets_ids": [...], "operation": "unsubscribe"}
        """
        if not self.ws or not asset_ids:
            return

        unsubscribe_msg = {
            "assets_ids": asset_ids,
            "operation": "unsubscribe"
        }

        await self.ws.send(json.dumps(unsubscribe_msg))
        self.subscriptions_outbound_total += 1

        for asset_id in asset_ids:
            if asset_id in self.subscribed_assets:
                self.subscribed_assets.remove(asset_id)

        logger.info(f"Unsubscribed from {len(asset_ids)} assets")

    async def listen(self):
        """
        Listen for WebSocket messages and update order books.

        Runs continuously until stopped.
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")

        logger.info("Starting market WebSocket listener...")

        try:
            async for message in self.ws:
                self.last_message_time = int(time.time() * 1000)  # milliseconds
                self.ws_messages_total += 1

                # Handle non-JSON messages (PONG, INVALID OPERATION, etc.)
                if isinstance(message, str) and not message.startswith(('{', '[')):
                    if message == "PONG":
                        logger.debug("Received PONG")
                        continue
                    else:
                        # Server error/warning message
                        logger.warning(f"Server message: {message}")
                        continue

                try:
                    data = json.loads(message)

                    # Track message types
                    if isinstance(data, list):
                        self.msg_type_counts["array_of_books"] += 1

                        # Log first batch of books
                        if not self.first_book_batch_logged:
                            logger.info(
                                f"FIRST BOOK BATCH: array length={len(data)} "
                                f"sample_keys={list(data[0].keys()) if data else []}"
                            )
                            self.first_book_batch_logged = True

                        for book_data in data:
                            await self._update_order_book(book_data)
                    else:
                        # Single message - track type
                        msg_type = data.get("type")
                        event_type = data.get("event_type")

                        # Track by type or event_type
                        type_key = msg_type or event_type or "unknown"
                        self.msg_type_counts[type_key] += 1

                        # Log first few unknown messages to debug
                        if type_key == "unknown" and self.msg_type_counts["unknown"] <= 3:
                            logger.warning(
                                f"Unknown message type: keys={list(data.keys())[:15]} "
                                f"sample={str(data)[:300]}"
                            )

                        await self._handle_message(data)
                except json.JSONDecodeError as e:
                    self.ws_json_parse_errors += 1
                    # Log first 200 chars of raw message
                    msg_preview = str(message)[:200]
                    logger.error(f"JSON parse error: {e}. Message preview: {msg_preview}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Market WebSocket connection closed: code={e.code} reason={e.reason}")
            self.running = False
        except Exception as e:
            logger.error(f"Market WebSocket listener error: {e}", exc_info=True)
            self.running = False

    async def _handle_message(self, data: dict):
        """
        Handle incoming WebSocket message.

        Updates OrderBook with new L2 data.

        Args:
            data: Parsed JSON message
        """
        msg_type = data.get("type")

        # Log all message types for debugging
        logger.debug(f"WS message type={msg_type} keys={list(data.keys())[:10]}")

        if msg_type == "book":
            # L2 book snapshot or update
            await self._update_order_book(data)

        elif msg_type == "trade":
            # Trade execution (optional handling)
            pass

        elif msg_type == "error":
            logger.error(f"WebSocket error message: {data}")

    async def _update_order_book(self, data: dict):
        """
        Update order book from WebSocket message.

        Args:
            data: Book update message
        """
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        # Track book messages
        self.book_messages_total += 1
        self.book_messages_by_asset[asset_id] += 1
        self.unique_asset_ids_with_book.add(asset_id)

        # Parse timestamp (comes as string from server)
        timestamp_raw = data.get("timestamp")
        if timestamp_raw:
            timestamp_ms = int(timestamp_raw)
        else:
            timestamp_ms = int(time.time() * 1000)

        # Parse bids and asks
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])

        # Convert to (price, size) tuples
        bids = [(float(b["price"]), float(b["size"])) for b in bids_raw]
        asks = [(float(a["price"]), float(a["size"])) for a in asks_raw]

        # Sort: bids descending, asks ascending
        bids.sort(reverse=True)
        asks.sort()

        # Extract best prices
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None

        # Update churn tracking
        now_ms = int(time.time() * 1000)
        self.update_counts[asset_id].append(now_ms)
        self.last_update_times[asset_id] = now_ms

        # Compute churn rate (updates per second)
        updates = self.update_counts[asset_id]
        if len(updates) >= 2:
            time_span_ms = updates[-1] - updates[0]
            time_span_s = time_span_ms / 1000.0
            churn_rate = len(updates) / time_span_s if time_span_s > 0 else 0.0
        else:
            churn_rate = 0.0

        # Get last_mid if book exists
        from execution.mid import compute_mid
        last_mid = None
        if asset_id in self.books:
            old_book = self.books[asset_id]
            last_mid = compute_mid(
                old_book.best_bid,
                old_book.best_ask,
                old_book.last_mid,
                old_book.timestamp_age_ms,
            )

        # Create/update OrderBook
        book = OrderBook(
            token_id=asset_id,
            best_bid=best_bid if best_bid else 0.0,
            best_ask=best_ask if best_ask else 1.0,
            bids=bids,
            asks=asks,
            timestamp_ms=timestamp_ms,
            timestamp_age_ms=0,  # Fresh update
            churn_rate=churn_rate,
            last_mid=last_mid,
        )

        self.books[asset_id] = book

        bid_str = f"{best_bid:.4f}" if best_bid else "0.0000"
        ask_str = f"{best_ask:.4f}" if best_ask else "1.0000"
        logger.debug(
            f"Book updated: {asset_id} "
            f"bid={bid_str} ask={ask_str} churn={churn_rate:.2f}/s"
        )

    def get_book(self, token_id: str) -> Optional[OrderBook]:
        """
        Get current order book for a token.

        Updates timestamp_age_ms before returning.

        Args:
            token_id: Token ID

        Returns:
            OrderBook or None if not subscribed
        """
        book = self.books.get(token_id)
        if not book:
            return None

        # Update staleness
        now_ms = int(time.time() * 1000)
        book.timestamp_age_ms = now_ms - book.timestamp_ms

        return book

    def check_staleness(self) -> bool:
        """
        Check if any book is stale.

        CRITICAL: If ANY subscribed book is stale, trigger unsafe signal.

        Returns:
            True if stale (unsafe), False if fresh
        """
        now_ms = int(time.time() * 1000)

        for asset_id in self.subscribed_assets:
            book = self.books.get(asset_id)
            if not book:
                # No data yet for subscribed asset
                continue

            age_ms = now_ms - book.timestamp_ms

            if age_ms > WS_STALENESS_THRESHOLD_MS:
                logger.warning(
                    f"STALE BOOK DETECTED: {asset_id} age={age_ms}ms "
                    f"(threshold={WS_STALENESS_THRESHOLD_MS}ms)"
                )

                # Trigger callback (should trigger cancel-all)
                if self.on_stale_callback:
                    self.on_stale_callback(asset_id, age_ms)

                return True

        return False

    async def run_staleness_monitor(self, interval_seconds: int = 5):
        """
        Periodic staleness monitor.

        Checks every N seconds for stale books.

        Args:
            interval_seconds: Check interval
        """
        logger.info(f"Starting staleness monitor (interval={interval_seconds}s)")

        while self.running:
            self.check_staleness()
            await asyncio.sleep(interval_seconds)

    async def reconnect_loop(self):
        """
        Reconnect loop with exponential backoff.

        Automatically reconnects if connection drops.
        """
        while True:
            try:
                await self.connect()

                # Re-subscribe to assets
                if self.subscribed_assets:
                    await self.subscribe(list(self.subscribed_assets))

                # Start listening
                await self.listen()

            except Exception as e:
                logger.error(f"Market WebSocket error: {e}. Reconnecting in {WS_RECONNECT_DELAY_SECONDS}s...")
                await asyncio.sleep(WS_RECONNECT_DELAY_SECONDS)

    def stop(self):
        """Stop WebSocket feed."""
        self.running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
            logger.info("Market WebSocket stopped")

    def get_stats(self) -> dict:
        """Get feed statistics."""
        now_ms = int(time.time() * 1000)

        return {
            "subscribed_count": len(self.subscribed_assets),
            "books_count": len(self.books),
            "running": self.running,
            "last_message_age_ms": now_ms - self.last_message_time if self.last_message_time else None,
            "avg_churn_rate": sum(b.churn_rate for b in self.books.values()) / len(self.books)
            if self.books
            else 0.0,
        }

    def get_feed_health(self) -> dict:
        """
        Get feed health metrics for observability.

        Returns:
            Dict with ws_msgs, book_msgs, parse_err, max_age_seconds
        """
        now_ms = int(time.time() * 1000)

        # Find max age across all subscribed books
        max_age_seconds = 0.0
        for asset_id in self.subscribed_assets:
            if asset_id in self.books:
                age_ms = now_ms - self.books[asset_id].timestamp_ms
                age_seconds = age_ms / 1000.0
                max_age_seconds = max(max_age_seconds, age_seconds)

        return {
            "ws_msgs": self.ws_messages_total,
            "book_msgs": self.book_messages_total,
            "parse_err": self.ws_json_parse_errors,
            "max_age_seconds": max_age_seconds,
            "subscribed_assets": len(self.subscribed_assets),
            "books_received": len(self.books),
            "unique_assets_with_book": len(self.unique_asset_ids_with_book),
            "msg_type_counts": dict(self.msg_type_counts),
        }
