"""
Unit tests for CLOB Executor.

CRITICAL: These tests lock execution invariants that prevent production failures.
"""

import time

import pytest

from config import MAX_BATCH_ORDERS, MUTATION_MAX_PER_MINUTE, POST_ONLY_ALLOWED_TYPES
from execution.clob_executor import CLOBExecutor
from execution.order_state_store import OrderStateStore
from models.types import OrderIntent

# ============================================================================
# ORDER VALIDATION TESTS (CRITICAL FIX #1)
# ============================================================================


def test_post_only_only_valid_with_gtc_gtd():
    """CRITICAL INVARIANT: postOnly only valid with GTC/GTD."""
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Valid: postOnly with GTD
    intent_valid = OrderIntent(
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.50,
        size_in_tokens=100.0,
        order_type="GTD",
        post_only=True,
        expiration=int(time.time()) + 180,
        fee_rate_bps=200,
    )

    valid, reason = executor.validate_order_intent(intent_valid)
    assert valid is True

    # Invalid: postOnly with FOK
    intent_invalid = OrderIntent(
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.50,
        size_in_tokens=100.0,
        order_type="FOK",
        post_only=True,
        fee_rate_bps=200,
    )

    valid, reason = executor.validate_order_intent(intent_invalid)
    assert valid is False
    assert "postOnly" in reason


def test_gtd_requires_expiration():
    """CRITICAL INVARIANT: GTD must have expiration timestamp."""
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Valid: GTD with expiration
    intent_valid = OrderIntent(
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.50,
        size_in_tokens=100.0,
        order_type="GTD",
        post_only=True,
        expiration=int(time.time()) + 180,
        fee_rate_bps=200,
    )

    valid, reason = executor.validate_order_intent(intent_valid)
    assert valid is True

    # Invalid: GTD without expiration
    intent_invalid = OrderIntent(
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.50,
        size_in_tokens=100.0,
        order_type="GTD",
        post_only=True,
        expiration=None,  # Missing!
        fee_rate_bps=200,
    )

    valid, reason = executor.validate_order_intent(intent_invalid)
    assert valid is False
    assert "expiration" in reason.lower()


def test_no_ttl_seconds_anywhere():
    """CRITICAL INVARIANT: Must use unix timestamp expiration, NOT ttl_seconds."""
    # This test verifies that OrderIntent does NOT have a ttl_seconds field

    intent = OrderIntent(
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.50,
        size_in_tokens=100.0,
        order_type="GTD",
        post_only=True,
        expiration=int(time.time()) + 180,
        fee_rate_bps=200,
    )

    # Verify no ttl_seconds attribute exists
    assert not hasattr(intent, "ttl_seconds"), "OrderIntent must NOT have ttl_seconds field"


# ============================================================================
# TICK ROUNDING TESTS (CRITICAL FIX #6)
# ============================================================================


def test_tick_rounding_before_clamping():
    """CRITICAL INVARIANT: Tick rounding must happen BEFORE clamping."""
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Raw price not tick-aligned
    raw_price = 0.5234

    # Round to tick
    rounded = executor.round_price_to_tick(raw_price, tick_size=0.01)
    assert rounded == 0.52, f"Expected 0.52, got {rounded}"

    # Clamp (should not affect already-rounded price)
    clamped = executor.clamp_price(rounded)
    assert clamped == 0.52


def test_tick_rounding_enforcement():
    """CRITICAL INVARIANT: Orders with non-tick-rounded prices must be rejected."""
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Non-tick-rounded price
    intent_bad = OrderIntent(
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.5234,  # NOT tick-rounded!
        size_in_tokens=100.0,
        order_type="GTD",
        post_only=True,
        expiration=int(time.time()) + 180,
        fee_rate_bps=200,
    )

    valid, reason = executor.validate_order_intent(intent_bad)
    assert valid is False
    assert "tick" in reason.lower()

    # Tick-rounded price
    intent_good = OrderIntent(
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.52,  # Tick-rounded
        size_in_tokens=100.0,
        order_type="GTD",
        post_only=True,
        expiration=int(time.time()) + 180,
        fee_rate_bps=200,
    )

    valid, reason = executor.validate_order_intent(intent_good)
    assert valid is True


# ============================================================================
# BATCH LIMIT TESTS (CRITICAL FIX #4)
# ============================================================================


def test_batch_limit_enforced():
    """CRITICAL INVARIANT: Batch size must be ≤ MAX_BATCH_ORDERS."""
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Create batch larger than limit
    intents = [
        OrderIntent(
            condition_id=f"market{i}",
            token_id=f"yes{i}",
            side="BUY",
            price=0.50,
            size_in_tokens=100.0,
            order_type="GTD",
            post_only=True,
            expiration=int(time.time()) + 180,
            fee_rate_bps=200,
        )
        for i in range(MAX_BATCH_ORDERS + 5)  # Over limit
    ]

    # Should raise error
    with pytest.raises(ValueError) as exc_info:
        import asyncio

        asyncio.run(executor.submit_batch_orders(intents))

    assert "batch size" in str(exc_info.value).lower()


def test_batch_slicing():
    """CRITICAL INVARIANT: Batch slicing must respect MAX_BATCH_ORDERS."""
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Create 50 intents
    intents = [
        OrderIntent(
            condition_id=f"market{i}",
            token_id=f"yes{i}",
            side="BUY",
            price=0.50,
            size_in_tokens=100.0,
            order_type="GTD",
            post_only=True,
            expiration=int(time.time()) + 180,
            fee_rate_bps=200,
        )
        for i in range(50)
    ]

    # Slice into batches
    batches = executor.slice_batch(intents)

    # Each batch must be ≤ MAX_BATCH_ORDERS
    for batch in batches:
        assert len(batch) <= MAX_BATCH_ORDERS

    # All intents must be included
    total_intents = sum(len(b) for b in batches)
    assert total_intents == 50


# ============================================================================
# MUTATION BUDGET TESTS (CRITICAL FIX #14)
# ============================================================================


def test_mutation_budget_rolling():
    """CRITICAL INVARIANT: Mutation budget enforced per minute."""
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Exhaust budget
    for i in range(MUTATION_MAX_PER_MINUTE):
        executor.record_mutations(1)

    # Next mutation should fail
    allowed, reason = executor.can_mutate(1)
    assert allowed is False
    assert "budget" in reason.lower()


def test_mutation_debouncing():
    """Test mutation debouncing (only if drift > threshold)."""
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Small drift (< 2 ticks) → no mutation
    should_replace = executor.should_replace_quote(
        market_id="market1",
        old_bid=0.50,
        old_ask=0.52,
        new_bid=0.501,  # Only 0.1 tick drift
        new_ask=0.52,
        tick_size=0.01,
    )

    assert should_replace is False, "Small drift should not trigger mutation"

    # Large drift (> 2 ticks) → mutation
    should_replace = executor.should_replace_quote(
        market_id="market1",
        old_bid=0.50,
        old_ask=0.52,
        new_bid=0.53,  # 3 tick drift
        new_ask=0.55,
        tick_size=0.01,
    )

    assert should_replace is True, "Large drift should trigger mutation"


# ============================================================================
# INTEGRATION SMOKE TEST
# ============================================================================


def test_full_order_submission_flow():
    """
    Smoke test: Full order submission with validation.

    Tests that all execution primitives work together.
    """
    store = OrderStateStore()
    executor = CLOBExecutor(clob_client=None, order_store=store)

    # Create valid order intent
    intent = OrderIntent(
        condition_id="market1",
        token_id="yes1",
        side="BUY",
        price=0.50,  # Tick-rounded
        size_in_tokens=100.0,
        order_type="GTD",
        post_only=True,
        expiration=int(time.time()) + 180,
        fee_rate_bps=200,
    )

    # Validate
    valid, reason = executor.validate_order_intent(intent)
    assert valid is True

    # Submit (stubbed, but validates)
    import asyncio

    result = asyncio.run(executor.submit_batch_orders([intent]))

    assert result["submitted"] == 1
    assert result["failed"] == 0

    # Verify order in store
    stats = store.get_stats()
    assert stats["pending"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
