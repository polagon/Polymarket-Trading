"""
Paper Trading Fill Simulator (GAP #8 FIX).

CRITICAL: Realistic fill simulation to avoid fake Sharpe in paper mode.

Two APIs:
1. simulate_fill_probability() — original deterministic API (kept for backward compat)
2. simulate_fill_detailed()    — new stochastic API using numpy RNG

Conservative rules:
- Only fill if book crosses our price
- Apply adverse selection penalty
- Stochastic Bernoulli draws for fill decisions
- Partial fills with configurable rate
- Track simulated vs actual for calibration

Loop 3 additions (Activities 14-16):
- BookSnapshotRingBuffer: deque(maxlen) per market for O(1) append + auto-eviction
- Latency simulation: exponential latency draws, historical book lookup
- Market impact: sqrt-based impact model with consumed-side liquidity
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from models.types import Fill, OrderBook, OrderStatus, StoredOrder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for the stochastic fill model
# ---------------------------------------------------------------------------
BASE_FILL_RATE = 0.15  # Base probability before factors
MAX_FILL_PROB = 0.70  # Hard cap on fill probability
PARTIAL_FILL_RATE = 0.30  # ~30% of fills are partial
PARTIAL_MIN_FRAC = 0.20  # Minimum partial fill as fraction of remaining
ADVERSE_BASE_BPS = 5.0  # Base adverse move in basis points of mid
ADVERSE_TIGHT_MULT = 3.0  # Multiplier for tight-spread adverse selection

# Activity 16: Market impact constants
MIN_IMPACT_CAP = 0.0001  # Floor cap at 1 bps (in price units)
DEFAULT_LIQUIDITY = 5000.0  # Conservative default when no depth data available


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BookSnapshot:
    """Book snapshot for fill probability computation.

    Note: timestamp must be monotonic per market for lookup_at() early-break.
    """

    timestamp: float  # Unix timestamp
    best_bid: float
    best_ask: float
    mid: float
    volume_at_levels: dict = field(default_factory=dict)  # price -> volume


@dataclass
class FillOutcome:
    """Detailed result of a stochastic fill simulation."""

    filled: bool
    fill_price: Optional[float]
    size_filled: float
    is_partial: bool
    fill_probability: float
    time_factor: float
    spread_factor: float
    queue_factor: float
    adverse_move: float
    # Activity 15: Latency simulation fields
    latency_applied_ms: float = 0.0
    degraded: bool = False
    # Activity 16: Market impact fields
    impact_bps: float = 0.0


# ---------------------------------------------------------------------------
# Activity 14: BookSnapshotRingBuffer
# ---------------------------------------------------------------------------


class BookSnapshotRingBuffer:
    """Per-market ring buffer for book snapshots using deque(maxlen).

    Read-only API: get() returns None for unknown markets (not a mutable empty container).
    Mutation only through append(). lookup_at() uses reverse linear scan from tail.
    """

    def __init__(self, maxlen: int = 1000):
        self._maxlen = maxlen
        self._buffers: dict[str, deque[BookSnapshot]] = {}

    def append(self, market_id: str, snapshot: BookSnapshot) -> None:
        """Append snapshot to market's ring buffer. Creates buffer on first append."""
        if market_id not in self._buffers:
            self._buffers[market_id] = deque(maxlen=self._maxlen)
        self._buffers[market_id].append(snapshot)

    def get(self, market_id: str) -> Optional[deque[BookSnapshot]]:
        """Return buffer for market_id, or None if no snapshots recorded.

        Callers must handle None (not an empty mutable container).
        """
        return self._buffers.get(market_id)

    def latest(self, market_id: str) -> Optional[BookSnapshot]:
        """Return most recent snapshot for market, or None."""
        buf = self._buffers.get(market_id)
        return buf[-1] if buf else None

    def lookup_at(self, market_id: str, target_ts: float) -> Optional[BookSnapshot]:
        """Reverse linear scan from tail to find snapshot nearest to target_ts.

        Rationale: maxlen=1000 is small, and typical latency lookups want
        "near now" so scanning from tail hits the answer in <10 iterations.
        Returns None if no snapshots exist.
        """
        buf = self._buffers.get(market_id)
        if not buf:
            return None
        best: Optional[BookSnapshot] = None
        best_dist = float("inf")
        for snap in reversed(buf):
            dist = abs(snap.timestamp - target_ts)
            if dist < best_dist:
                best = snap
                best_dist = dist
            elif snap.timestamp < target_ts - best_dist:
                # Past the best; timestamps are monotonic, stop early
                break
        return best

    def __len__(self) -> int:
        """Total number of snapshots across all markets."""
        return sum(len(b) for b in self._buffers.values())

    def __contains__(self, market_id: str) -> bool:
        return market_id in self._buffers

    @property
    def markets(self) -> list[str]:
        """List of market IDs with recorded snapshots."""
        return list(self._buffers.keys())


# ---------------------------------------------------------------------------
# Activity 15: Snapshot-to-OrderBook conversion
# ---------------------------------------------------------------------------


def _snapshot_to_orderbook(snap: BookSnapshot, fallback_book: OrderBook) -> OrderBook:
    """Convert BookSnapshot to OrderBook for latency-shifted fill evaluation.

    Deterministic conversion rules:
    - best_bid, best_ask: from snapshot; fall back to fallback_book if zero
    - last_mid: snapshot's mid (the historical fair value)
    - bids/asks depth: empty (snapshot doesn't store depth)
    - token_id, timestamp_age_ms: from fallback_book (unchanged by latency)

    If snapshot has only one side (best_bid==0 or best_ask==0),
    use fallback_book's value for that side.
    """
    return OrderBook(
        token_id=fallback_book.token_id,
        best_bid=snap.best_bid if snap.best_bid > 0 else fallback_book.best_bid,
        best_ask=snap.best_ask if snap.best_ask > 0 else fallback_book.best_ask,
        last_mid=snap.mid,
        timestamp_ms=int(snap.timestamp * 1000),
        timestamp_age_ms=fallback_book.timestamp_age_ms,
    )


# ---------------------------------------------------------------------------
# Activity 16: Consumed-side liquidity helper
# ---------------------------------------------------------------------------


def _get_consumed_side_liquidity(order: StoredOrder, book: OrderBook) -> float:
    """Compute available liquidity from the side the order consumes.

    BUY orders consume ask liquidity; SELL orders consume bid liquidity.
    Uses OrderBook.asks/bids if available, otherwise returns conservative default.
    Units are tokens (matching order_size units).
    """
    if order.side == "BUY":
        # Consume ask side
        if hasattr(book, "asks") and book.asks:
            return sum(size for _price, size in book.asks)
    else:
        # Consume bid side
        if hasattr(book, "bids") and book.bids:
            return sum(size for _price, size in book.bids)

    return DEFAULT_LIQUIDITY


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class PaperTradingSimulator:
    """
    Realistic fill simulator for paper mode.

    Fill probability based on:
    1. Queue position (if we're at best bid/ask)
    2. Book changes (if book crosses our price)
    3. Time in market (longer -> higher fill probability)
    4. Spread width (tighter -> higher fill probability)
    5. Adverse selection penalty (reduce size on "too good" fills)
    6. Latency simulation (Activity 15): exponential delay, historical book lookup
    7. Market impact (Activity 16): sqrt-based price impact

    Stochastic: uses numpy RNG seeded for deterministic replay.
    """

    def __init__(
        self,
        seed: int = 12345,
        max_history_size: int = 1000,
        latency_mean_ms: float = 50.0,
        impact_k: float = 0.02,
    ):
        self._ring_buffer = BookSnapshotRingBuffer(maxlen=max_history_size)
        self.max_history_size = max_history_size

        # Activity 15: Latency simulation
        self._latency_mean_ms = latency_mean_ms

        # Activity 16: Market impact
        self._impact_k = impact_k

        # RNG state
        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Stats counters
        self._total_evaluations: int = 0
        self._total_fills: int = 0
        self._total_no_fills: int = 0

    # ------------------------------------------------------------------
    # Backward-compat property (read-only)
    # ------------------------------------------------------------------

    @property
    def book_history(self) -> dict[str, tuple[BookSnapshot, ...]]:
        """Read-only backward compat. Returns immutable tuples per market.

        Do NOT rely on this for mutation — use record_book_snapshot() to add snapshots.
        """
        return {k: tuple(v) for k, v in self._ring_buffer._buffers.items()}

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------

    def reset_rng(self, seed: int = 12345) -> None:
        """Reset the RNG to replay sequences."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Book snapshot recording
    # ------------------------------------------------------------------

    def record_book_snapshot(self, market_id: str, book: OrderBook) -> None:
        """
        Record book snapshot for fill simulation.

        Args:
            market_id: Market condition_id
            book: OrderBook snapshot
        """
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

        self._ring_buffer.append(market_id, snapshot)

    # ------------------------------------------------------------------
    # Factor calculations (private helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_time_factor(time_in_market_seconds: float) -> float:
        """Logarithmic time factor: longer -> higher, but diminishing returns.

        Returns a value in roughly [0.3, 2.0].
        At 5s  -> ~0.55
        At 30s -> ~1.0
        At 60s -> ~1.3
        At 120s -> ~1.6
        """
        clamped = max(1.0, time_in_market_seconds)
        return 0.2 + 0.35 * math.log(clamped)

    @staticmethod
    def _compute_spread_factor(spread: float) -> float:
        """Tighter spread -> higher fill probability.

        spread is best_ask - best_bid (in price units 0-1).
        Returns a value in roughly [0.5, 2.0].
        """
        if spread <= 0:
            return 2.0
        # Typical Polymarket spread range: 0.01 to 0.10
        # We want tight (0.01) -> high (~1.8), wide (0.10) -> low (~0.6)
        return min(2.0, max(0.5, 1.0 / (1.0 + 10.0 * spread)))

    @staticmethod
    def _compute_queue_factor(order: StoredOrder, book: OrderBook) -> float:
        """Queue position factor: at-best -> higher fill.

        Returns a value in roughly [0.5, 1.5].
        """
        if order.side == "BUY":
            # How close is our price to best ask (crossing level)?
            distance = abs(order.price - book.best_ask)
        else:
            distance = abs(order.price - book.best_bid)

        # At best (distance ~ 0) -> 1.5; far from best (distance > 0.05) -> 0.5
        return max(0.5, 1.5 - distance * 20.0)

    def _compute_activity_factor(self, market_id: str) -> float:
        """Activity factor based on book snapshot history.

        More snapshots -> more active market -> higher fill.
        Returns a value in roughly [0.5, 1.5].
        """
        buf = self._ring_buffer.get(market_id)
        n = len(buf) if buf is not None else 0
        if n == 0:
            return 1.0  # Neutral if no history
        # Scale: 0 snapshots -> 1.0, 100+ snapshots -> 1.5
        return min(1.5, 1.0 + n / 200.0)

    # ------------------------------------------------------------------
    # Activity 15: Latency helpers
    # ------------------------------------------------------------------

    def _draw_latency_ms(self) -> float:
        """Draw latency from exponential distribution. Returns 0.0 if disabled."""
        if self._latency_mean_ms <= 0:
            return 0.0
        return float(self._rng.exponential(self._latency_mean_ms))

    def _resolve_book_at_latency(
        self, market_id: str, latency_ms: float, current_book: OrderBook
    ) -> tuple[OrderBook, bool]:
        """Look up historical book snapshot at (now - latency_ms).

        Returns (book_to_use, degraded).
        If no historical data available, returns (current_book, True).
        """
        if latency_ms <= 0:
            return current_book, False
        target_ts = time.time() - (latency_ms / 1000.0)
        snap = self._ring_buffer.lookup_at(market_id, target_ts)
        if snap is None:
            return current_book, True  # degraded: no historical data
        return _snapshot_to_orderbook(snap, current_book), False

    # ------------------------------------------------------------------
    # Activity 16: Market impact
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_market_impact(
        order_size: float,
        available_liquidity: float,
        spread: float,
        impact_k: float,
    ) -> float:
        """Sqrt-based market impact in price units.

        impact = K * sqrt(order_size / liquidity)
        Capped at max(spread * 2, MIN_IMPACT_CAP).
        Returns 0.0 if liquidity <= 0 or impact_k <= 0.
        """
        if available_liquidity <= 0 or impact_k <= 0:
            return 0.0
        raw = impact_k * math.sqrt(order_size / available_liquidity)
        cap = max(spread * 2.0, MIN_IMPACT_CAP)
        return min(raw, cap)

    # ------------------------------------------------------------------
    # Stochastic fill simulation (NEW API)
    # ------------------------------------------------------------------

    def simulate_fill_detailed(
        self,
        order: StoredOrder,
        time_in_market_seconds: float,
        current_book: OrderBook,
    ) -> FillOutcome:
        """
        Stochastic fill simulation using Bernoulli draws.

        Args:
            order: StoredOrder being simulated
            time_in_market_seconds: How long order has been live
            current_book: Current order book

        Returns:
            FillOutcome with all diagnostic factors
        """
        self._total_evaluations += 1
        market_id = order.condition_id

        # ----------------------------------------------------------
        # Step 1: Check crossing (on CURRENT book — "would the order
        # have been submitted?")
        # ----------------------------------------------------------
        crossed = False
        if order.side == "BUY":
            if current_book.best_ask <= order.price:
                crossed = True
        else:  # SELL
            if current_book.best_bid >= order.price:
                crossed = True

        if not crossed:
            self._total_no_fills += 1
            return FillOutcome(
                filled=False,
                fill_price=None,
                size_filled=0.0,
                is_partial=False,
                fill_probability=0.0,
                time_factor=0.0,
                spread_factor=0.0,
                queue_factor=0.0,
                adverse_move=0.0,
            )

        # ----------------------------------------------------------
        # Step 1b: Latency simulation (Activity 15)
        # Draw latency and resolve effective book from history
        # ----------------------------------------------------------
        latency_ms = self._draw_latency_ms()
        effective_book, degraded = self._resolve_book_at_latency(market_id, latency_ms, current_book)

        # ----------------------------------------------------------
        # Step 2: Compute probability factors (using effective_book)
        # ----------------------------------------------------------
        spread = effective_book.best_ask - effective_book.best_bid
        time_factor = self._compute_time_factor(time_in_market_seconds)
        spread_factor = self._compute_spread_factor(spread)
        queue_factor = self._compute_queue_factor(order, effective_book)
        activity_factor = self._compute_activity_factor(market_id)

        fill_probability = BASE_FILL_RATE * time_factor * spread_factor * queue_factor * activity_factor
        fill_probability = min(fill_probability, MAX_FILL_PROB)
        fill_probability = max(fill_probability, 0.0)

        # ----------------------------------------------------------
        # Step 3: Bernoulli draw
        # ----------------------------------------------------------
        draw = float(self._rng.random())
        filled = draw < fill_probability

        if not filled:
            self._total_no_fills += 1
            return FillOutcome(
                filled=False,
                fill_price=None,
                size_filled=0.0,
                is_partial=False,
                fill_probability=fill_probability,
                time_factor=time_factor,
                spread_factor=spread_factor,
                queue_factor=queue_factor,
                adverse_move=0.0,
                latency_applied_ms=latency_ms,
                degraded=degraded,
            )

        # ----------------------------------------------------------
        # Step 4: Determine fill size (partial vs full)
        # ----------------------------------------------------------
        partial_draw = float(self._rng.random())
        is_partial = partial_draw < PARTIAL_FILL_RATE

        if is_partial:
            # Partial fill: between PARTIAL_MIN_FRAC and 1.0 of remaining
            frac = PARTIAL_MIN_FRAC + (1.0 - PARTIAL_MIN_FRAC) * float(self._rng.random())
            size_filled = order.remaining_size * frac
        else:
            size_filled = order.remaining_size

        # ----------------------------------------------------------
        # Step 5: Fill price = order limit price (maker fill)
        # ----------------------------------------------------------
        fill_price = order.price

        # Activity 16: Market impact
        available_liq = _get_consumed_side_liquidity(order, effective_book)
        impact = self._compute_market_impact(
            order_size=size_filled,
            available_liquidity=available_liq,
            spread=spread,
            impact_k=self._impact_k,
        )
        # Apply impact against the trader (BUY fills at higher price, SELL at lower)
        if order.side == "BUY":
            fill_price += impact
        else:
            fill_price -= impact

        # Clamp to valid range
        fill_price = max(0.01, min(0.99, fill_price))

        impact_bps = impact * 10000.0

        # ----------------------------------------------------------
        # Step 6: Adverse selection (move against the maker)
        # ----------------------------------------------------------
        mid = (effective_book.best_bid + effective_book.best_ask) / 2.0
        # Tighter spread -> worse adverse selection
        adverse_scale = ADVERSE_BASE_BPS / 10000.0
        # Scale: tighter spread means larger adverse move
        tightness_mult = 1.0 + (ADVERSE_TIGHT_MULT - 1.0) * max(0.0, 1.0 - spread / 0.05)
        # Random adverse component (always positive = against maker)
        adverse_move = abs(float(self._rng.exponential(adverse_scale * tightness_mult)))

        self._total_fills += 1

        logger.info(
            f"Paper fill simulated: {order.order_id} "
            f"time={time_in_market_seconds:.1f}s prob={fill_probability:.4f} "
            f"size={size_filled:.2f}/{order.remaining_size:.2f} @ {fill_price:.4f} "
            f"adverse={adverse_move:.6f} partial={is_partial} "
            f"latency={latency_ms:.1f}ms impact={impact_bps:.2f}bps"
        )

        return FillOutcome(
            filled=True,
            fill_price=fill_price,
            size_filled=size_filled,
            is_partial=is_partial,
            fill_probability=fill_probability,
            time_factor=time_factor,
            spread_factor=spread_factor,
            queue_factor=queue_factor,
            adverse_move=adverse_move,
            latency_applied_ms=latency_ms,
            degraded=degraded,
            impact_bps=impact_bps,
        )

    # ------------------------------------------------------------------
    # Original deterministic API (kept for backward compatibility)
    # ------------------------------------------------------------------

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
        - Longer time in market -> higher fill probability

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
            if current_book.best_ask <= order.price:
                crossed = True
                fill_price = order.price
        else:  # SELL
            if current_book.best_bid >= order.price:
                crossed = True
                fill_price = order.price

        if not crossed:
            return (False, 0.0, None)

        # Base fill probability based on time in market
        if time_in_market_seconds < 10:
            fill_prob = 0.1
        elif time_in_market_seconds < 30:
            fill_prob = 0.3
        elif time_in_market_seconds < 60:
            fill_prob = 0.6
        else:
            fill_prob = 0.8

        # Adverse selection penalty
        from execution.mid import compute_mid

        mid = compute_mid(
            current_book.best_bid,
            current_book.best_ask,
            getattr(current_book, "last_mid", None),
            getattr(current_book, "timestamp_age_ms", 0),
        )
        if mid is None:
            mid = 0.5
        price_quality = abs(order.price - mid) / mid if mid > 0 else 0

        if price_quality > 0.05:
            adverse_selection_penalty = 0.5
        else:
            adverse_selection_penalty = 1.0

        filled = fill_prob > 0.5

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

    # ------------------------------------------------------------------
    # Fill creation (updated for maker/taker fee differentiation)
    # ------------------------------------------------------------------

    def create_simulated_fill(
        self,
        order: StoredOrder,
        size_filled: float,
        fill_price: float,
        mid_at_fill: float,
    ) -> Fill:
        """
        Create simulated Fill object.

        Fee logic:
        - post_only=True  -> maker: fee_paid_usd=0.0, classification_source="POST_ONLY"
        - post_only=False -> taker: fee = price * size * (bps / 10000),
                             classification_source="SPREAD_CROSS"

        Args:
            order: Order that filled
            size_filled: Size filled
            fill_price: Fill price
            mid_at_fill: Mid price at time of fill

        Returns:
            Fill instance (marked as simulated)
        """
        if order.post_only:
            maker = True
            fee_paid_usd = 0.0
            classification_source = "POST_ONLY"
        else:
            maker = False
            fee_paid_usd = fill_price * size_filled * (order.fee_rate_bps / 10000.0)
            classification_source = "SPREAD_CROSS"

        fill = Fill(
            fill_id=f"sim_{order.order_id}_{int(time.time() * 1000)}",
            order_id=order.order_id,
            condition_id=order.condition_id,
            token_id=order.token_id,
            side=order.side,
            price=fill_price,
            size_tokens=size_filled,
            timestamp=int(time.time() * 1000),
            maker=maker,
            fee_rate_bps=order.fee_rate_bps,
            fee_paid_usd=fee_paid_usd,
            mid_at_fill=mid_at_fill,
            classification_source=classification_source,  # type: ignore[arg-type]
        )

        return fill

    # ------------------------------------------------------------------
    # Stats (extended with stochastic counters)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get simulator statistics."""
        total_snapshots = len(self._ring_buffer)
        num_markets = len(self._ring_buffer.markets)

        total_decided = self._total_fills + self._total_no_fills
        fill_rate = self._total_fills / total_decided if total_decided > 0 else 0.0

        return {
            # Legacy stats
            "markets_tracked": num_markets,
            "total_snapshots": total_snapshots,
            "avg_snapshots_per_market": total_snapshots / num_markets if num_markets else 0,
            # Stochastic stats
            "total_evaluations": self._total_evaluations,
            "total_fills": self._total_fills,
            "total_no_fills": self._total_no_fills,
            "fill_rate": fill_rate,
            "seed": self._seed,
        }
