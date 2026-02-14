"""
Tests for OrderManager — maker-only bounded execution.

Assertions use exact reason enums from models.reasons.
All orders are post_only=True, never cross, TTL/stale enforced.

Loop 4: No taker escalation.
"""

from __future__ import annotations

import pytest

from execution.order_manager import ManagedOrder, OrderManager
from models.reasons import (
    REASON_ORDER_CHASE_EXCEEDED,
    REASON_ORDER_CROSS_REJECTED,
    REASON_ORDER_STALE,
    REASON_ORDER_TOXIC_NO_IMPROVE,
    REASON_ORDER_TTL_EXPIRED,
)


@pytest.fixture
def mgr() -> OrderManager:
    return OrderManager(default_ttl=120, stale_threshold=60, max_chase_ticks=3, tick_size=0.01)


class TestSubmit:
    """Tests for order submission."""

    def test_submit_always_post_only(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        assert order.post_only is True

    def test_submit_sets_ttl(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        assert order.ttl_seconds == 120

    def test_submit_custom_ttl(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, ttl=60, now=1000.0)
        assert order.ttl_seconds == 60

    def test_submit_rejects_crossing_buy(self, mgr: OrderManager) -> None:
        """BUY at best_ask → crossing → reject."""
        with pytest.raises(ValueError, match=REASON_ORDER_CROSS_REJECTED):
            mgr.submit("m1", "BUY_YES", 0.52, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)

    def test_submit_rejects_crossing_buy_above(self, mgr: OrderManager) -> None:
        """BUY above best_ask → crossing → reject."""
        with pytest.raises(ValueError, match=REASON_ORDER_CROSS_REJECTED):
            mgr.submit("m1", "BUY_YES", 0.55, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)

    def test_submit_rounds_price_buy_down(self, mgr: OrderManager) -> None:
        """BUY price rounds DOWN to nearest tick."""
        order = mgr.submit("m1", "BUY_YES", 0.505, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        assert order.price == pytest.approx(0.50, abs=1e-9)

    def test_submit_in_active_orders(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        active = mgr.get_active_orders()
        assert len(active) == 1
        assert active[0].order_id == order.order_id


class TestTTL:
    """Tests for TTL enforcement."""

    def test_ttl_cancels_expired(self, mgr: OrderManager) -> None:
        mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        canceled = mgr.enforce_ttl(now=1121.0)  # 121s > 120s TTL
        assert len(canceled) == 1
        assert mgr.get_active_orders() == []

    def test_ttl_does_not_cancel_fresh(self, mgr: OrderManager) -> None:
        mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        canceled = mgr.enforce_ttl(now=1050.0)  # 50s < 120s TTL
        assert len(canceled) == 0
        assert len(mgr.get_active_orders()) == 1

    def test_ttl_cancel_reason(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        mgr.enforce_ttl(now=1121.0)
        assert order.cancel_reason == REASON_ORDER_TTL_EXPIRED


class TestStale:
    """Tests for stale detection and cancellation."""

    def test_stale_cancels_not_flags(self, mgr: OrderManager) -> None:
        """Stale detection CANCELS orders (not just flags)."""
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        canceled = mgr.check_stale(now=1061.0)  # 61s > 60s threshold
        assert len(canceled) == 1
        assert order.canceled is True
        assert order.cancel_reason == REASON_ORDER_STALE

    def test_refresh_resets_stale_timer(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        mgr.refresh(order.order_id, now=1050.0)
        canceled = mgr.check_stale(now=1061.0)  # 11s since refresh < 60s
        assert len(canceled) == 0
        assert order.canceled is False


class TestCancelReplace:
    """Tests for cancel/replace within chase distance."""

    def test_replace_within_chase(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.55, now=1000.0)
        new_order = mgr.cancel_replace(
            order.order_id,
            0.52,
            100.0,
            best_bid=0.49,
            best_ask=0.55,
            toxicity_flag=False,
            spread_regime="NORMAL",
        )
        assert new_order is not None
        assert new_order.price == pytest.approx(0.52, abs=1e-9)
        assert order.canceled is True

    def test_replace_beyond_chase_rejected(self, mgr: OrderManager) -> None:
        """Chase distance > max_chase_ticks * tick_size → only cancel."""
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.60, now=1000.0)
        new_order = mgr.cancel_replace(
            order.order_id,
            0.55,
            100.0,  # 5 ticks > 3 max
            best_bid=0.49,
            best_ask=0.60,
            toxicity_flag=False,
            spread_regime="NORMAL",
        )
        assert new_order is None
        assert order.canceled is True
        assert order.cancel_reason == REASON_ORDER_CHASE_EXCEEDED

    def test_replace_would_cross_rejected(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        new_order = mgr.cancel_replace(
            order.order_id,
            0.52,
            100.0,
            best_bid=0.49,
            best_ask=0.52,  # would cross
            toxicity_flag=False,
            spread_regime="NORMAL",
        )
        assert new_order is None
        assert order.canceled is True

    def test_replace_when_toxic_cancel_only(self, mgr: OrderManager) -> None:
        """Toxic → cancel only, no replacement."""
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.55, now=1000.0)
        new_order = mgr.cancel_replace(
            order.order_id,
            0.51,
            100.0,
            best_bid=0.49,
            best_ask=0.55,
            toxicity_flag=True,
            spread_regime="NORMAL",
        )
        assert new_order is None
        assert order.canceled is True
        assert order.cancel_reason == REASON_ORDER_TOXIC_NO_IMPROVE

    def test_replace_when_wide_cancel_only(self, mgr: OrderManager) -> None:
        """WIDE regime → cancel only, no replacement."""
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.55, now=1000.0)
        new_order = mgr.cancel_replace(
            order.order_id,
            0.51,
            100.0,
            best_bid=0.49,
            best_ask=0.55,
            toxicity_flag=False,
            spread_regime="WIDE",
        )
        assert new_order is None
        assert order.canceled is True
        assert order.cancel_reason == REASON_ORDER_TOXIC_NO_IMPROVE

    def test_replace_when_broken_cancel_only(self, mgr: OrderManager) -> None:
        """BROKEN regime → cancel only."""
        order = mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.55, now=1000.0)
        new_order = mgr.cancel_replace(
            order.order_id,
            0.51,
            100.0,
            best_bid=0.49,
            best_ask=0.55,
            toxicity_flag=False,
            spread_regime="BROKEN",
        )
        assert new_order is None
        assert order.canceled is True


class TestCanImprovePrice:
    """Tests for can_improve_price method."""

    def test_normal_regime_clean(self, mgr: OrderManager) -> None:
        assert mgr.can_improve_price(toxicity_flag=False, spread_regime="NORMAL") is True

    def test_tight_regime_clean(self, mgr: OrderManager) -> None:
        assert mgr.can_improve_price(toxicity_flag=False, spread_regime="TIGHT") is True

    def test_toxic_returns_false(self, mgr: OrderManager) -> None:
        assert mgr.can_improve_price(toxicity_flag=True, spread_regime="NORMAL") is False

    def test_wide_returns_false(self, mgr: OrderManager) -> None:
        assert mgr.can_improve_price(toxicity_flag=False, spread_regime="WIDE") is False

    def test_broken_returns_false(self, mgr: OrderManager) -> None:
        assert mgr.can_improve_price(toxicity_flag=False, spread_regime="BROKEN") is False


class TestPriceRounding:
    """Tests for tick-aligned price rounding away from crossing."""

    def test_buy_rounds_down(self, mgr: OrderManager) -> None:
        order = mgr.submit("m1", "BUY_YES", 0.517, 100.0, best_bid=0.49, best_ask=0.55, now=1000.0)
        assert order.price == pytest.approx(0.51, abs=1e-9)

    def test_post_rounding_still_not_crossing(self, mgr: OrderManager) -> None:
        """After rounding, price still must not cross."""
        # 0.519 rounds to 0.51 for BUY, which is < ask of 0.52 → ok
        order = mgr.submit("m1", "BUY_YES", 0.519, 100.0, best_bid=0.49, best_ask=0.52, now=1000.0)
        assert order.price < 0.52


class TestCancelAll:
    """Tests for cancel_all."""

    def test_cancel_all(self, mgr: OrderManager) -> None:
        mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.55, now=1000.0)
        mgr.submit("m2", "BUY_NO", 0.30, 100.0, best_bid=0.29, best_ask=0.35, now=1000.0)
        count = mgr.cancel_all()
        assert count == 2
        assert mgr.get_active_orders() == []

    def test_cancel_all_by_market(self, mgr: OrderManager) -> None:
        mgr.submit("m1", "BUY_YES", 0.50, 100.0, best_bid=0.49, best_ask=0.55, now=1000.0)
        mgr.submit("m2", "BUY_NO", 0.30, 100.0, best_bid=0.29, best_ask=0.35, now=1000.0)
        count = mgr.cancel_all(market_id="m1")
        assert count == 1
        assert len(mgr.get_active_orders()) == 1
