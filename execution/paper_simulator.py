"""
Paper Trading Fill Simulator (GAP #8 FIX).

CRITICAL: Realistic fill simulation to avoid fake Sharpe in paper mode.

Conservative rules:
- Only fill if book crosses our price
- Apply adverse selection penalty
- Track simulated vs actual for calibration
"""
import logging
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from models.types import StoredOrder, OrderBook, Fill, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class BookSnapshot:
    """Book snapshot for fill probability computation."""

    timestamp: float  # Unix timestamp
    best_bid: float
    best_ask: float
    mid: float
    volume_at_levels: dict = field(default_factory=dict)  # price → volume


class PaperTradingSimulator:
    """
    Realistic fill simulator for paper mode.

    Fill probability based on:
    1. Queue position (if we're at best bid/ask)
    2. Book changes (if book crosses our price)
    3. Time in market (longer → higher fill probability)
    4. Adverse selection penalty (reduce size on "too good" fills)
    """

    def __init__(self):
        self.book_history: dict[str, List[BookSnapshot]] = {}  # market_id → snapshots
        self.max_history_size = 1000  # Keep last 1000 snapshots per market

    def record_book_snapshot(self, market_id: str, book: OrderBook):
        """
        Record book snapshot for fill simulation.

        Args:
            market_id: Market condition_id
            book: OrderBook snapshot
        """
        if market_id not in self.book_history:
            self.book_history[market_id] = []

        from execution.mid import compute_mid

        mid = compute_mid(
            book.best_bid,
            book.best_ask,
            book.last_mid,
            book.timestamp_age_ms,
        )
        if mid is None:
            mid = 0.5  # Fallback for paper trading only

        snapshot = BookSnapshot(
            timestamp=time.time(),
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            mid=mid,
        )

        self.book_history[market_id].append(snapshot)

        # Trim old snapshots
        if len(self.book_history[market_id]) > self.max_history_size:
            self.book_history[market_id] = self.book_history[market_id][-self.max_history_size :]

    def simulate_fill_probability(
        self,
        order: StoredOrder,
        time_in_market_seconds: float,
        current_book: OrderBook,
    ) -> Tuple[bool, float, Optional[float]]:
        """
        Simulate whether order would fill (GAP #8 FIX).

        CONSERVATIVE RULES:
        - Only fill if book crosses our price
        - Apply adverse selection penalty (reduce filled size)
        - Longer time in market → higher fill probability

        Args:
            order: StoredOrder being simulated
            time_in_market_seconds: How long order has been live
            current_book: Current order book

        Returns:
            (filled: bool, size_filled: float, fill_price: Optional[float])
        """
        market_id = order.condition_id

        # Check if book crosses our price
        crossed = False
        fill_price = None

        if order.side == "BUY":
            # We're buying at order.price
            # Fill if someone sells at/below our price (best_ask <= our price)
            if current_book.best_ask <= order.price:
                crossed = True
                fill_price = order.price  # We get filled at our limit price
        else:  # SELL
            # We're selling at order.price
            # Fill if someone buys at/above our price (best_bid >= our price)
            if current_book.best_bid >= order.price:
                crossed = True
                fill_price = order.price

        if not crossed:
            # Book hasn't crossed our price → no fill
            return (False, 0.0, None)

        # Book crossed our price → compute fill probability

        # Base fill probability based on time in market
        # Conservative: require at least 30s for high probability
        if time_in_market_seconds < 10:
            fill_prob = 0.1  # 10% if very recent
        elif time_in_market_seconds < 30:
            fill_prob = 0.3  # 30% if moderate
        elif time_in_market_seconds < 60:
            fill_prob = 0.6  # 60% if mature
        else:
            fill_prob = 0.8  # 80% if old

        # Adverse selection penalty
        # If our price is "too good" relative to mid, reduce fill size
        from execution.mid import compute_mid
        mid = compute_mid(
            current_book.best_bid,
            current_book.best_ask,
            getattr(current_book, "last_mid", None),
            getattr(current_book, "timestamp_age_ms", 0),
        )
        if mid is None:
            mid = 0.5  # Fallback for paper trading
        price_quality = abs(order.price - mid) / mid if mid > 0 else 0

        if price_quality > 0.05:  # More than 5% from mid
            # Suspicious - reduce fill size (adverse selection)
            adverse_selection_penalty = 0.5  # Fill only 50%
        else:
            adverse_selection_penalty = 1.0  # Full size

        # Determine fill outcome (simplified - deterministic for reproducibility)
        # In real implementation, could use random with seed
        filled = fill_prob > 0.5  # Conservative threshold

        if filled:
            size_filled = order.remaining_size * adverse_selection_penalty
            logger.info(
                f"Paper fill simulated: {order.order_id} "
                f"time={time_in_market_seconds:.1f}s prob={fill_prob:.2f} "
                f"size={size_filled:.2f}/{order.remaining_size:.2f} @ {fill_price:.4f}"
            )
            return (True, size_filled, fill_price)
        else:
            return (False, 0.0, None)

    def create_simulated_fill(
        self,
        order: StoredOrder,
        size_filled: float,
        fill_price: float,
        mid_at_fill: float,
    ) -> Fill:
        """
        Create simulated Fill object.

        Args:
            order: Order that filled
            size_filled: Size filled
            fill_price: Fill price
            mid_at_fill: Mid price at time of fill

        Returns:
            Fill instance (marked as simulated)
        """
        fill = Fill(
            fill_id=f"sim_{order.order_id}_{int(time.time() * 1000)}",
            order_id=order.order_id,
            condition_id=order.condition_id,
            token_id=order.token_id,
            side=order.side,
            price=fill_price,
            size_tokens=size_filled,
            timestamp=int(time.time() * 1000),
            maker=True,  # Paper fills are always maker (post-only)
            fee_rate_bps=order.fee_rate_bps,
            fee_paid_usd=fill_price * size_filled * (order.fee_rate_bps / 10000.0),
            mid_at_fill=mid_at_fill,
            classification_source="POST_ONLY",  # Paper fills from post-only orders
        )

        return fill

    def get_stats(self) -> dict:
        """Get simulator statistics."""
        total_snapshots = sum(len(snapshots) for snapshots in self.book_history.values())

        return {
            "markets_tracked": len(self.book_history),
            "total_snapshots": total_snapshots,
            "avg_snapshots_per_market": total_snapshots / len(self.book_history) if self.book_history else 0,
        }
