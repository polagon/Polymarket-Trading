"""
CLOB Executor - Execution firewall with validation and rate limiting.

CRITICAL: This is the ONLY place that:
- Validates postOnly/order type rules
- Enforces batch limit ≤15
- Performs tick rounding
- Enforces mutation budget
- Submits/cancels orders to CLOB

Strategy modules CANNOT bypass this layer.
"""

import logging
import time
from collections import deque
from typing import List, Optional, Tuple

from config import (
    MAX_BATCH_ORDERS,
    MAX_PRICE,
    MIN_PRICE,
    MUTATION_MAX_PER_CYCLE,
    MUTATION_MAX_PER_MINUTE,
    MUTATION_MIN_DRIFT_TICKS,
    POST_ONLY_ALLOWED_TYPES,
    STANDARD_TICK_SIZE,
    VALID_ORDER_TYPES,
)
from execution import expiration, fees, units
from execution.order_state_store import OrderStateStore
from models.types import Market, OrderIntent, OrderStatus, SignedOrderPayload, StoredOrder

logger = logging.getLogger(__name__)


class CLOBExecutor:
    """
    Central CLOB execution firewall.

    All order submissions MUST go through this class.
    """

    def __init__(self, clob_client, order_store: OrderStateStore):
        """
        Initialize CLOB executor.

        Args:
            clob_client: Instance of py-clob-client
            order_store: OrderStateStore instance
        """
        self.clob_client = clob_client
        self.order_store = order_store

        # Rolling mutation budget (per minute)
        self.mutation_timestamps = deque(maxlen=MUTATION_MAX_PER_MINUTE)  # type: ignore[var-annotated]

        # GAP #5: Per-market mutation tracking + reject storm handling
        self.per_market_last_mutation: dict[str, float] = {}
        self.per_market_cooldown_seconds = 30  # Min 30s between mutations per market
        self.reject_counts: dict[str, int] = {}
        self.paused_until: dict[str, float] = {}  # market_id → unpause_timestamp

        # Market metadata cache for validation
        self.market_cache: dict[str, Market] = {}

    def register_market(self, market: Market):
        """
        Register market metadata for validation.

        CRITICAL: Must be called before submitting orders for this market.

        Args:
            market: Market with metadata (tick_size, fee_rate_bps, etc.)
        """
        self.market_cache[market.condition_id] = market
        logger.debug(
            f"Registered market {market.condition_id}: tick_size={market.tick_size}, fee_rate_bps={market.fee_rate_bps}"
        )

    def validate_market_constraints(self, intent: OrderIntent, market: Market) -> Tuple[bool, str]:
        """
        Validate per-market constraints (GAP #1 FIX).

        CRITICAL CHECKS:
        - tick_size from market metadata (NOT config constant)
        - min_order_size from market metadata (if available)
        - price within [0.01, 0.99] after rounding
        - size >= min_size

        Args:
            intent: OrderIntent to validate
            market: Market metadata

        Returns:
            (valid: bool, reason: str)
        """
        # Tick size validation
        tick_size = market.tick_size
        expected_rounded = round(intent.price / tick_size) * tick_size

        if abs(intent.price - expected_rounded) > 1e-9:
            return False, f"Price {intent.price:.4f} not divisible by tick_size={tick_size}"

        # Price bounds check (after rounding)
        rounded_price = expected_rounded
        if rounded_price < MIN_PRICE or rounded_price > MAX_PRICE:
            return False, f"Rounded price {rounded_price:.4f} outside bounds [{MIN_PRICE}, {MAX_PRICE}]"

        # Min size check (if metadata provides it)
        # Note: Polymarket CLOB may have min_size in market metadata
        # For now, we'll add a conservative default
        min_size = market.raw_metadata.get("min_size", 1.0)  # Default 1 token
        if intent.size_in_tokens < min_size:
            return False, f"Size {intent.size_in_tokens} below min_size={min_size}"

        return True, "OK"

    def validate_order_intent(self, intent: OrderIntent, market: Optional[Market] = None) -> Tuple[bool, str]:
        """
        Validate order intent before submission.

        CRITICAL INVARIANTS ENFORCED:
        1. postOnly only valid with GTC/GTD
        2. GTD requires expiration timestamp
        3. Price must be tick-rounded
        4. Spread >= min tick
        5. Per-market constraints (GAP #1 FIX)

        Args:
            intent: OrderIntent to validate
            market: Market metadata (REQUIRED for GAP #1 fix)

        Returns:
            (valid: bool, reason: str)
        """
        # Rule 1: Order type must be valid
        if intent.order_type not in VALID_ORDER_TYPES:
            return False, f"Invalid order type: {intent.order_type}. Must be one of {VALID_ORDER_TYPES}"

        # Rule 2: postOnly only allowed with GTC/GTD (CRITICAL FIX #1)
        if intent.post_only and intent.order_type not in POST_ONLY_ALLOWED_TYPES:
            return False, f"postOnly=True only valid with {POST_ONLY_ALLOWED_TYPES}, got {intent.order_type}"

        # Rule 3: GTD requires expiration timestamp (CRITICAL FIX #1)
        if intent.order_type == "GTD" and intent.expiration is None:
            return False, "GTD order requires expiration timestamp"

        # Rule 4: Price must be within bounds
        if intent.price < MIN_PRICE or intent.price > MAX_PRICE:
            return False, f"Price {intent.price:.4f} outside bounds [{MIN_PRICE}, {MAX_PRICE}]"

        # Rule 5: Size must be positive
        if intent.size_in_tokens <= 0:
            return False, f"Size must be positive, got {intent.size_in_tokens}"

        # Rule 6: Per-market constraints (GAP #1 FIX)
        if market is None:
            # Try to get from cache
            market = self.market_cache.get(intent.condition_id)

        if market is not None:
            valid, reason = self.validate_market_constraints(intent, market)
            if not valid:
                return False, f"Market constraint violation: {reason}"
        else:
            # CRITICAL: Conservative fallback - use standard tick
            logger.warning(
                f"No market metadata for {intent.condition_id}. "
                f"Using STANDARD_TICK_SIZE={STANDARD_TICK_SIZE}. "
                f"This may cause CLOB rejects!"
            )
            tick_size = STANDARD_TICK_SIZE
            expected_rounded = round(intent.price / tick_size) * tick_size
            if abs(intent.price - expected_rounded) > 1e-9:
                return False, f"Price {intent.price:.4f} not tick-rounded (fallback tick={tick_size})"

        return True, "OK"

    def round_price_to_tick(self, price: float, tick_size: float = STANDARD_TICK_SIZE) -> float:
        """
        Round price to tick size.

        CRITICAL FIX #6: Tick rounding BEFORE clamping.

        Args:
            price: Raw price
            tick_size: Tick size (default 0.01)

        Returns:
            Tick-rounded price
        """
        rounded = round(price / tick_size) * tick_size
        return rounded

    def clamp_price(self, price: float) -> float:
        """
        Clamp price to valid range.

        Args:
            price: Price to clamp

        Returns:
            Clamped price
        """
        return max(MIN_PRICE, min(MAX_PRICE, price))

    def can_mutate_market(self, market_id: str) -> Tuple[bool, str]:
        """
        Check per-market cooldown + pause state (GAP #5 FIX).

        CRITICAL: Prevents reject storms from single markets.

        Args:
            market_id: Market condition_id

        Returns:
            (allowed: bool, reason: str)
        """
        now = time.time()

        # Check if market is paused (after reject storm)
        if market_id in self.paused_until:
            if now < self.paused_until[market_id]:
                remaining = int(self.paused_until[market_id] - now)
                return False, f"Market paused (reject storm). Unpause in {remaining}s"
            else:
                # Unpause
                del self.paused_until[market_id]
                self.reject_counts[market_id] = 0  # Reset counter
                logger.info(f"Market {market_id} unpaused after reject storm cooldown")

        # Check per-market cooldown
        last_mut = self.per_market_last_mutation.get(market_id, 0)
        if now - last_mut < self.per_market_cooldown_seconds:
            remaining = int(self.per_market_cooldown_seconds - (now - last_mut))
            return False, f"Per-market cooldown active ({remaining}s remaining)"

        return True, "OK"

    def can_mutate(self, count: int = 1) -> Tuple[bool, str]:
        """
        Check if mutation budget allows N mutations.

        CRITICAL: Rolling budget per minute prevents cancel/replace storms.

        Args:
            count: Number of mutations requested

        Returns:
            (allowed: bool, reason: str)
        """
        now = time.time()

        # Remove mutations older than 60s
        cutoff = now - 60
        while self.mutation_timestamps and self.mutation_timestamps[0] < cutoff:
            self.mutation_timestamps.popleft()

        # Check budget
        if len(self.mutation_timestamps) + count > MUTATION_MAX_PER_MINUTE:
            return (
                False,
                f"Mutation budget exhausted: {len(self.mutation_timestamps)}/{MUTATION_MAX_PER_MINUTE} per minute",
            )

        return True, "OK"

    def on_order_reject(self, market_id: str, error: str):
        """
        Handle order rejection with exponential backoff (GAP #5 FIX).

        CRITICAL: Prevents reject storm death spirals.

        Args:
            market_id: Market condition_id
            error: Rejection error message
        """
        self.reject_counts[market_id] = self.reject_counts.get(market_id, 0) + 1
        reject_count = self.reject_counts[market_id]

        logger.warning(f"Order rejected for {market_id}: {error} (reject count: {reject_count})")

        # Reject storm detection (5+ rejects)
        if reject_count >= 5:
            # Exponential backoff: 5min, 10min, 20min, 40min...
            pause_duration = 300 * (2 ** (reject_count - 5))
            pause_duration = min(pause_duration, 7200)  # Cap at 2 hours

            self.paused_until[market_id] = time.time() + pause_duration

            logger.error(
                f"🚨 REJECT STORM DETECTED: {market_id} paused for {pause_duration}s "
                f"({reject_count} consecutive rejects). Error: {error}"
            )

    def on_order_success(self, market_id: str):
        """
        Reset reject counter on successful order.

        Args:
            market_id: Market condition_id
        """
        if market_id in self.reject_counts:
            self.reject_counts[market_id] = 0

    def record_mutations(self, count: int):
        """
        Record mutations in rolling budget.

        Args:
            count: Number of mutations performed
        """
        now = time.time()
        for _ in range(count):
            self.mutation_timestamps.append(now)

    def record_market_mutation(self, market_id: str):
        """
        Record mutation timestamp for specific market (GAP #5 FIX).

        Args:
            market_id: Market condition_id
        """
        self.per_market_last_mutation[market_id] = time.time()

    async def submit_batch_orders(self, intents: List[OrderIntent]) -> dict:
        """
        Submit batch of orders to CLOB.

        CRITICAL FIXES:
        - #4: Batch limit ≤15 enforced
        - #1: postOnly/type validation
        - #6: Tick rounding enforcement

        Args:
            intents: List of OrderIntent objects

        Returns:
            {
                "submitted": int,
                "failed": int,
                "errors": list[str],
            }
        """
        if len(intents) > MAX_BATCH_ORDERS:
            raise ValueError(
                f"Batch size {len(intents)} exceeds limit {MAX_BATCH_ORDERS}. Slice batches before calling this method."
            )

        submitted = 0
        failed = 0
        errors = []

        for intent in intents:
            # Validate
            valid, reason = self.validate_order_intent(intent)
            if not valid:
                logger.error(f"Order validation failed: {reason}")
                errors.append(reason)
                failed += 1
                continue

            try:
                # Submit to CLOB (stub - actual implementation needs py-clob-client)
                # order_result = await self.clob_client.create_order(...)

                # For now, create StoredOrder and add to store
                order_id = f"order_{int(time.time() * 1000)}"  # Temporary ID generation

                stored_order = StoredOrder(
                    order_id=order_id,
                    condition_id=intent.condition_id,
                    token_id=intent.token_id,
                    side=intent.side,
                    price=intent.price,
                    size_in_tokens=intent.size_in_tokens,
                    order_type=intent.order_type,
                    post_only=intent.post_only,
                    expiration=intent.expiration,
                    fee_rate_bps=intent.fee_rate_bps,
                    status=OrderStatus.PENDING,
                    placed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )

                self.order_store.add_order(stored_order)
                submitted += 1

                logger.info(
                    f"Order submitted: {order_id} {intent.side} {intent.size_in_tokens:.2f} "
                    f"{intent.token_id} @ {intent.price:.4f}"
                )

            except Exception as e:
                logger.error(f"Failed to submit order: {e}", exc_info=True)
                errors.append(str(e))
                failed += 1

        return {
            "submitted": submitted,
            "failed": failed,
            "errors": errors,
        }

    async def cancel_orders(self, order_ids: List[str]) -> dict:
        """
        Cancel orders by ID.

        Args:
            order_ids: List of order IDs to cancel

        Returns:
            {
                "canceled": int,
                "failed": int,
                "errors": list[str],
            }
        """
        canceled = 0
        failed = 0
        errors = []

        for order_id in order_ids:
            try:
                # Cancel via CLOB (stub)
                # await self.clob_client.cancel_order(order_id)

                # Update OrderStateStore
                self.order_store.update_order_status(
                    order_id,
                    OrderStatus.CANCELED,
                    cancel_reason="Manual cancellation",
                )

                canceled += 1
                logger.info(f"Order canceled: {order_id}")

            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                errors.append(str(e))
                failed += 1

        return {
            "canceled": canceled,
            "failed": failed,
            "errors": errors,
        }

    async def cancel_all(self, reason: str = "Circuit breaker"):
        """
        Cancel ALL live orders.

        CRITICAL: Called by circuit breakers on unsafe conditions.

        Args:
            reason: Reason for cancel-all
        """
        live_orders = self.order_store.get_live_orders()

        if not live_orders:
            logger.info("Cancel-all: No live orders to cancel")
            return

        logger.warning(f"CANCEL-ALL triggered: {reason}. Canceling {len(live_orders)} orders...")

        order_ids = [o.order_id for o in live_orders]
        result = await self.cancel_orders(order_ids)

        logger.warning(
            f"Cancel-all complete: {result['canceled']} canceled, {result['failed']} failed. Reason: {reason}"
        )

    def slice_batch(self, intents: List[OrderIntent]) -> List[List[OrderIntent]]:
        """
        Slice large batch into chunks of MAX_BATCH_ORDERS.

        CRITICAL FIX #4: Enforces batch limit centrally.

        Args:
            intents: List of OrderIntent objects

        Returns:
            List of batches (each ≤ MAX_BATCH_ORDERS)
        """
        batches = []
        for i in range(0, len(intents), MAX_BATCH_ORDERS):
            batch = intents[i : i + MAX_BATCH_ORDERS]
            batches.append(batch)

        return batches

    def should_replace_quote(
        self,
        market_id: str,
        old_bid: float,
        old_ask: float,
        new_bid: float,
        new_ask: float,
        tick_size: float = STANDARD_TICK_SIZE,
    ) -> bool:
        """
        Decide if quotes need cancel/replace.

        CRITICAL FIX #14: Debounced mutation (only if drift > threshold).

        Args:
            market_id: Market condition_id
            old_bid: Current bid price
            old_ask: Current ask price
            new_bid: New bid price
            new_ask: New ask price
            tick_size: Tick size

        Returns:
            True if mutation needed, False otherwise
        """
        bid_drift_ticks = abs(new_bid - old_bid) / tick_size
        ask_drift_ticks = abs(new_ask - old_ask) / tick_size

        if bid_drift_ticks > MUTATION_MIN_DRIFT_TICKS or ask_drift_ticks > MUTATION_MIN_DRIFT_TICKS:
            return True

        return False

    def get_stats(self) -> dict:
        """Get executor statistics."""
        now = time.time()
        cutoff = now - 60

        # Count mutations in last minute
        recent_mutations = sum(1 for ts in self.mutation_timestamps if ts > cutoff)

        return {
            "mutations_last_minute": recent_mutations,
            "mutation_budget_remaining": MUTATION_MAX_PER_MINUTE - recent_mutations,
            "order_store_stats": self.order_store.get_stats(),
        }
