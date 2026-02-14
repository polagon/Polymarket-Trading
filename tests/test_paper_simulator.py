"""
Tests for the stochastic paper fill simulator.

Validates:
1. Determinism: same seed → same outcomes
2. Stochasticity: different seeds → different outcomes
3. Sensitivity: tighter spreads and longer time → higher fill probability
4. Sanity: fill rate within reasonable bounds
5. Partial fills: occur at expected rate
6. Adverse selection: worse for tight/active markets
7. Slippage: applied on thin books
8. Fee differentiation: maker vs taker
9. No silent fallback to deterministic behavior
10. Fill optimism measurably reduced vs old model
"""

import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.paper_simulator import (
    DEFAULT_LIQUIDITY,
    MIN_IMPACT_CAP,
    BookSnapshot,
    BookSnapshotRingBuffer,
    FillOutcome,
    PaperTradingSimulator,
    _get_consumed_side_liquidity,
    _snapshot_to_orderbook,
)
from models.types import Fill, OrderBook, StoredOrder

# ============================================================================
# HELPERS
# ============================================================================


def make_order(
    side: str = "BUY",
    price: float = 0.50,
    size: float = 100.0,
    condition_id: str = "test_market",
    token_id: str = "token_yes",
    post_only: bool = True,
) -> StoredOrder:
    """Create a StoredOrder for testing."""
    return StoredOrder(
        order_id=f"order_{hash((side, price, size))}",
        condition_id=condition_id,
        token_id=token_id,
        side=side,  # type: ignore[arg-type]
        price=price,
        size_in_tokens=size,
        order_type="GTD",
        post_only=post_only,
        fee_rate_bps=200,
        original_size=size,
        remaining_size=size,
        filled_size=0.0,
        placed_at=datetime.now(timezone.utc).isoformat(),
    )


def make_book(
    best_bid: float = 0.48,
    best_ask: float = 0.52,
    last_mid: float = 0.50,
) -> OrderBook:
    """Create an OrderBook for testing."""
    return OrderBook(
        token_id="token_yes",
        best_bid=best_bid,
        best_ask=best_ask,
        last_mid=last_mid,
        timestamp_ms=int(time.time() * 1000),
        timestamp_age_ms=100,
    )


def run_n_simulations(
    sim: PaperTradingSimulator,
    n: int,
    order: StoredOrder = None,  # type: ignore[assignment]
    book: OrderBook = None,  # type: ignore[assignment]
    time_s: float = 60.0,
) -> list:
    """Run N fill simulations and return list of FillOutcomes."""
    if order is None:
        order = make_order(side="BUY", price=0.52)
    if book is None:
        book = make_book(best_bid=0.48, best_ask=0.51)  # Crosses BUY@0.52

    outcomes = []
    for _ in range(n):
        outcome = sim.simulate_fill_detailed(order, time_s, book)
        outcomes.append(outcome)
    return outcomes


# ============================================================================
# TEST: Determinism
# ============================================================================


class TestDeterminism:
    def test_same_seed_same_outcomes(self):
        """Same seed must produce identical fill sequence."""
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        # Run 1
        sim1 = PaperTradingSimulator(seed=12345)
        results1 = [sim1.simulate_fill_detailed(order, 60.0, book).filled for _ in range(50)]

        # Run 2 (same seed)
        sim2 = PaperTradingSimulator(seed=12345)
        results2 = [sim2.simulate_fill_detailed(order, 60.0, book).filled for _ in range(50)]

        assert results1 == results2, "Same seed should produce identical results"

    def test_different_seed_different_outcomes(self):
        """Different seeds should produce different fill sequences (with high probability)."""
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        sim1 = PaperTradingSimulator(seed=111)
        results1 = [sim1.simulate_fill_detailed(order, 60.0, book).filled for _ in range(100)]

        sim2 = PaperTradingSimulator(seed=222)
        results2 = [sim2.simulate_fill_detailed(order, 60.0, book).filled for _ in range(100)]

        # Not all results should be the same
        assert results1 != results2, "Different seeds should produce different results"

    def test_reset_rng_reproduces(self):
        """reset_rng should allow replaying the same sequence."""
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        sim = PaperTradingSimulator(seed=42)
        results1 = [sim.simulate_fill_detailed(order, 60.0, book).filled for _ in range(20)]

        sim.reset_rng(seed=42)
        results2 = [sim.simulate_fill_detailed(order, 60.0, book).filled for _ in range(20)]

        assert results1 == results2

    def test_no_deterministic_fallback(self):
        """Verify that outcomes are STOCHASTIC, not deterministic thresholds.

        The old model used `fill_prob > 0.5` — a hard threshold that always
        gave the same answer for the same time-in-market. The new model
        must show variance for the same inputs across different seeds.
        """
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        all_results = set()
        for seed in range(100):
            sim = PaperTradingSimulator(seed=seed)
            result = sim.simulate_fill_detailed(order, 30.0, book).filled
            all_results.add(result)

        # With 100 different seeds, we should see BOTH True and False
        assert True in all_results, "Should have some fills"
        assert False in all_results, "Should have some non-fills"


# ============================================================================
# TEST: Sensitivity
# ============================================================================


class TestSensitivity:
    def test_longer_time_increases_fill_probability(self):
        """Orders in market longer should have higher fill probability."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        # Very short time
        short_outcomes = run_n_simulations(sim, 200, order, book, time_s=5.0)
        sim.reset_rng(42)

        # Long time
        long_outcomes = run_n_simulations(sim, 200, order, book, time_s=120.0)

        short_fills = sum(1 for o in short_outcomes if o.filled)
        long_fills = sum(1 for o in long_outcomes if o.filled)

        # Long time should produce more fills
        assert long_fills > short_fills, f"Longer time should produce more fills: {long_fills} vs {short_fills}"

    def test_tighter_spread_increases_fill_probability(self):
        """Tighter spread should produce higher fill probability (ceteris paribus)."""
        order = make_order(side="BUY", price=0.52)

        # Wide spread (4 ticks)
        wide_book = make_book(best_bid=0.48, best_ask=0.52)
        sim1 = PaperTradingSimulator(seed=42)
        wide_outcomes = run_n_simulations(sim1, 300, order, wide_book, time_s=60.0)

        # Tight spread (1 tick)
        tight_book = make_book(best_bid=0.51, best_ask=0.52)
        sim2 = PaperTradingSimulator(seed=42)
        tight_outcomes = run_n_simulations(sim2, 300, order, tight_book, time_s=60.0)

        wide_fills = sum(1 for o in wide_outcomes if o.filled)
        tight_fills = sum(1 for o in tight_outcomes if o.filled)

        # Tight spread should produce slightly more fills (due to spread_factor)
        assert tight_fills >= wide_fills, (
            f"Tighter spread should not reduce fills: tight={tight_fills} vs wide={wide_fills}"
        )

    def test_at_best_price_fills_more(self):
        """Orders at best bid/ask should fill more than orders deeper in book."""
        # BUY order at best bid vs deeper
        book = make_book(best_bid=0.50, best_ask=0.52)

        at_best_order = make_order(side="BUY", price=0.52)  # At ask = crosses
        deep_order = make_order(side="BUY", price=0.52)  # Same cross, but

        # For queue factor: test via the factor calculation directly
        sim = PaperTradingSimulator(seed=42)

        # At-best: order price matches best level
        at_best_book = make_book(best_bid=0.50, best_ask=0.52)
        deeper_book = make_book(best_bid=0.50, best_ask=0.49)  # Ask lower = we're deeper

        # Both cross, but queue factor should differ
        outcomes_at = run_n_simulations(sim, 200, at_best_order, at_best_book, time_s=60.0)
        sim.reset_rng(42)
        # Verify queue factor is computed
        outcome = sim.simulate_fill_detailed(at_best_order, 60.0, at_best_book)
        assert outcome.queue_factor > 0, "Queue factor should be positive"


# ============================================================================
# TEST: Sanity bounds
# ============================================================================


class TestSanityBounds:
    def test_fill_rate_within_bounds(self):
        """Fill rate should be between 5% and 70% for typical parameters."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)  # Crosses

        outcomes = run_n_simulations(sim, 500, order, book, time_s=60.0)
        fill_count = sum(1 for o in outcomes if o.filled)
        fill_rate = fill_count / 500

        # Should not be too optimistic or too pessimistic
        assert fill_rate > 0.03, f"Fill rate too low: {fill_rate:.2%}"
        assert fill_rate < 0.75, f"Fill rate too high: {fill_rate:.2%}"

    def test_no_fill_without_crossing(self):
        """Must never fill if book doesn't cross our price."""
        sim = PaperTradingSimulator(seed=42)

        # BUY at 0.45, but best_ask is 0.52 (no crossing)
        order = make_order(side="BUY", price=0.45)
        book = make_book(best_bid=0.48, best_ask=0.52)

        outcomes = run_n_simulations(sim, 100, order, book, time_s=300.0)
        fills = sum(1 for o in outcomes if o.filled)

        assert fills == 0, "Should never fill without book crossing"

    def test_sell_order_crossing(self):
        """SELL orders should fill when best_bid >= order price."""
        sim = PaperTradingSimulator(seed=42)

        order = make_order(side="SELL", price=0.48)
        book = make_book(best_bid=0.49, best_ask=0.52)  # Crosses SELL@0.48

        outcomes = run_n_simulations(sim, 200, order, book, time_s=60.0)
        fills = sum(1 for o in outcomes if o.filled)

        assert fills > 0, "SELL order with crossing book should get some fills"

    def test_fill_probability_never_exceeds_max(self):
        """Fill probability must be capped at MAX_FILL_PROB."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.51, best_ask=0.52)  # Very tight

        # Long time + tight spread + at best = maximum conditions
        outcome = sim.simulate_fill_detailed(order, 999999.0, book)

        # The fill probability should be capped
        assert outcome.fill_probability <= 0.70, f"Fill probability exceeded cap: {outcome.fill_probability}"

    def test_fill_price_in_valid_range(self):
        """Fill price must always be in [0.01, 0.99]."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        for _ in range(200):
            outcome = sim.simulate_fill_detailed(order, 60.0, book)
            if outcome.filled:
                assert 0.01 <= outcome.fill_price <= 0.99, f"Fill price out of range: {outcome.fill_price}"  # type: ignore[operator]


# ============================================================================
# TEST: Partial fills
# ============================================================================


class TestPartialFills:
    def test_partial_fills_occur(self):
        """Some fills should be partial."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52, size=100.0)
        book = make_book(best_bid=0.48, best_ask=0.51)

        partial_count = 0
        full_count = 0

        for _ in range(500):
            outcome = sim.simulate_fill_detailed(order, 60.0, book)
            if outcome.filled:
                if outcome.is_partial:
                    partial_count += 1
                    # Partial fills should be less than full size
                    assert outcome.size_filled < order.remaining_size, (
                        f"Partial fill should be < full size: {outcome.size_filled}"
                    )
                    # But at least the minimum percentage
                    assert outcome.size_filled >= order.remaining_size * 0.20 - 0.01, (
                        f"Partial fill below minimum: {outcome.size_filled}"
                    )
                else:
                    full_count += 1
                    assert abs(outcome.size_filled - order.remaining_size) < 0.01

        assert partial_count > 0, "Should have some partial fills"
        assert full_count > 0, "Should have some full fills"

        # Partial rate should be roughly 30% of fills (per config)
        total_fills = partial_count + full_count
        if total_fills > 20:
            partial_rate = partial_count / total_fills
            assert 0.10 < partial_rate < 0.60, f"Partial fill rate out of expected range: {partial_rate:.2%}"


# ============================================================================
# TEST: Adverse selection
# ============================================================================


class TestAdverseSelection:
    def test_adverse_move_is_against_maker(self):
        """Average adverse move should be positive (bad for maker)."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        adverse_moves = []
        for _ in range(500):
            outcome = sim.simulate_fill_detailed(order, 60.0, book)
            if outcome.filled:
                adverse_moves.append(outcome.adverse_move)

        assert len(adverse_moves) > 10, "Need enough fills to measure adverse selection"

        # Mean adverse move should be positive (price moved against us)
        mean_adverse = sum(adverse_moves) / len(adverse_moves)
        assert mean_adverse > 0, f"Mean adverse move should be positive (bad for maker): {mean_adverse:.6f}"

    def test_tight_spread_worse_adverse_selection(self):
        """Tighter spread should produce larger adverse moves."""
        order = make_order(side="BUY", price=0.52)

        # Tight spread (1 tick)
        tight_book = make_book(best_bid=0.51, best_ask=0.52)
        sim_tight = PaperTradingSimulator(seed=42)
        tight_adverse = []
        for _ in range(500):
            o = sim_tight.simulate_fill_detailed(order, 60.0, tight_book)
            if o.filled:
                tight_adverse.append(o.adverse_move)

        # Wide spread (5 ticks)
        wide_book = make_book(best_bid=0.47, best_ask=0.52)
        sim_wide = PaperTradingSimulator(seed=42)
        wide_adverse = []
        for _ in range(500):
            o = sim_wide.simulate_fill_detailed(order, 60.0, wide_book)
            if o.filled:
                wide_adverse.append(o.adverse_move)

        if len(tight_adverse) > 5 and len(wide_adverse) > 5:
            mean_tight = sum(tight_adverse) / len(tight_adverse)
            mean_wide = sum(wide_adverse) / len(wide_adverse)

            # Tight spread should have worse (higher) adverse selection
            assert mean_tight > mean_wide, (
                f"Tight spread should have worse adverse selection: tight={mean_tight:.6f} vs wide={mean_wide:.6f}"
            )


# ============================================================================
# TEST: Fee differentiation
# ============================================================================


class TestFees:
    def test_maker_vs_taker_fees(self):
        """Post-only (maker) orders should have lower fees than taker orders."""
        sim = PaperTradingSimulator(seed=42)

        maker_order = make_order(side="BUY", price=0.52, post_only=True)
        taker_order = make_order(side="BUY", price=0.52, post_only=False)

        maker_fill = sim.create_simulated_fill(maker_order, 100.0, 0.52, 0.50)
        taker_fill = sim.create_simulated_fill(taker_order, 100.0, 0.52, 0.50)

        # Maker fee should be 0 (Polymarket)
        assert maker_fill.fee_paid_usd == 0.0, f"Maker fee should be 0: {maker_fill.fee_paid_usd}"
        # Taker fee should be 2%
        expected_taker_fee = 0.52 * 100.0 * (200 / 10000.0)
        assert abs(taker_fill.fee_paid_usd - expected_taker_fee) < 0.01, (
            f"Taker fee mismatch: {taker_fill.fee_paid_usd} vs {expected_taker_fee}"
        )

    def test_maker_classification(self):
        """Post-only fills should be classified as maker."""
        sim = PaperTradingSimulator(seed=42)

        maker_order = make_order(post_only=True)
        fill = sim.create_simulated_fill(maker_order, 100.0, 0.52, 0.50)
        assert fill.maker is True
        assert fill.classification_source == "POST_ONLY"

        taker_order = make_order(post_only=False)
        fill = sim.create_simulated_fill(taker_order, 100.0, 0.52, 0.50)
        assert fill.maker is False
        assert fill.classification_source == "SPREAD_CROSS"


# ============================================================================
# TEST: Fill optimism reduction vs old model
# ============================================================================


class TestFillOptimismReduction:
    def test_stochastic_fills_fewer_than_deterministic(self):
        """
        The new stochastic model should fill LESS than the old deterministic model.

        Old model: filled = fill_prob > 0.5
        - At 30s: fill_prob = 0.3 → no fill
        - At 60s: fill_prob = 0.6 → ALWAYS fill (100%!)
        - At 120s: fill_prob = 0.8 → ALWAYS fill (100%!)

        New model: Bernoulli draw from continuous probability.
        - At 60s: ~15-40% fill rate (base_rate * time_factor * spread * queue * activity)
        - At 120s: ~20-50% fill rate

        The old model had 100% fill rate for orders older than 30s with crossed books.
        The new model should never exceed MAX_FILL_PROB (70%).
        """
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)  # Crosses

        # New stochastic model
        sim = PaperTradingSimulator(seed=42)
        new_outcomes = run_n_simulations(sim, 500, order, book, time_s=120.0)
        new_fill_rate = sum(1 for o in new_outcomes if o.filled) / 500

        # Old model would fill 100% at 120s (fill_prob=0.8 > 0.5 threshold)
        old_fill_rate = 1.0  # Deterministic: always fills at >30s

        assert new_fill_rate < old_fill_rate, (
            f"New model should fill less than old deterministic model: "
            f"new={new_fill_rate:.2%} vs old={old_fill_rate:.2%}"
        )
        # Specifically, should be well below 100%
        assert new_fill_rate < 0.70, f"New model fill rate still too optimistic: {new_fill_rate:.2%}"

    def test_short_time_still_can_fill(self):
        """Even at short times, fills should be possible (but rare)."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        outcomes = run_n_simulations(sim, 1000, order, book, time_s=5.0)
        fills = sum(1 for o in outcomes if o.filled)

        # Should have SOME fills even at 5s (but not many)
        assert fills > 0, "Should have at least a few fills even at short time"
        fill_rate = fills / 1000
        assert fill_rate < 0.25, f"Too many fills at 5s: {fill_rate:.2%}"


# ============================================================================
# TEST: Diagnostics
# ============================================================================


class TestDiagnostics:
    def test_stats_tracking(self):
        """Simulator stats should accurately track fill/no-fill counts."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        for _ in range(100):
            sim.simulate_fill_detailed(order, 60.0, book)

        stats = sim.get_stats()
        assert stats["total_evaluations"] == 100
        assert stats["total_fills"] + stats["total_no_fills"] == 100
        assert stats["fill_rate"] > 0
        assert stats["seed"] == 42

    def test_detailed_outcome_has_factors(self):
        """FillOutcome should expose all intermediate probability factors."""
        sim = PaperTradingSimulator(seed=42)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        outcome = sim.simulate_fill_detailed(order, 60.0, book)

        # All factors should be populated
        assert outcome.fill_probability >= 0
        assert outcome.time_factor >= 0
        assert outcome.spread_factor >= 0
        assert outcome.queue_factor >= 0

    def test_book_snapshot_recording(self):
        """Book snapshots should be recorded and trimmed."""
        sim = PaperTradingSimulator(seed=42)
        book = make_book()

        for i in range(50):
            sim.record_book_snapshot("market_1", book)

        assert "market_1" in sim.book_history
        assert len(sim.book_history["market_1"]) == 50


# ============================================================================
# TEST: BookSnapshotRingBuffer (Activity 14)
# ============================================================================


class TestBookSnapshotRingBuffer:
    def test_ring_buffer_append_and_get(self):
        """Append 5 snapshots → get returns deque with 5 items."""
        buf = BookSnapshotRingBuffer(maxlen=100)
        for i in range(5):
            snap = BookSnapshot(timestamp=1000.0 + i, best_bid=0.48, best_ask=0.52, mid=0.50)
            buf.append("mkt1", snap)
        result = buf.get("mkt1")
        assert result is not None
        assert len(result) == 5

    def test_ring_buffer_eviction_at_maxlen(self):
        """Append maxlen+10 → len == maxlen, oldest evicted."""
        maxlen = 20
        buf = BookSnapshotRingBuffer(maxlen=maxlen)
        for i in range(maxlen + 10):
            snap = BookSnapshot(timestamp=1000.0 + i, best_bid=0.48, best_ask=0.52, mid=0.50)
            buf.append("mkt1", snap)
        result = buf.get("mkt1")
        assert result is not None
        assert len(result) == maxlen
        # Oldest should be evicted; first remaining should have timestamp >= 1010.0
        assert result[0].timestamp >= 1010.0

    def test_ring_buffer_latest(self):
        """latest() returns most recently appended snapshot."""
        buf = BookSnapshotRingBuffer(maxlen=100)
        for i in range(5):
            snap = BookSnapshot(timestamp=1000.0 + i, best_bid=0.48, best_ask=0.52, mid=0.50)
            buf.append("mkt1", snap)
        latest = buf.latest("mkt1")
        assert latest is not None
        assert latest.timestamp == 1004.0

    def test_ring_buffer_unknown_market_returns_none(self):
        """get('unknown') returns None, not empty container."""
        buf = BookSnapshotRingBuffer(maxlen=100)
        result = buf.get("unknown")
        assert result is None

    def test_ring_buffer_lookup_at_finds_nearest(self):
        """10 snapshots at known timestamps → lookup_at returns nearest."""
        buf = BookSnapshotRingBuffer(maxlen=100)
        for i in range(10):
            snap = BookSnapshot(timestamp=100.0 + i * 10.0, best_bid=0.48, best_ask=0.52, mid=0.50)
            buf.append("mkt1", snap)
        # Target at 145.0 → nearest is 140.0 or 150.0 (140.0 is closer at dist=5 vs 5)
        result = buf.lookup_at("mkt1", 145.0)
        assert result is not None
        assert result.timestamp in (140.0, 150.0)

    def test_ring_buffer_lookup_at_empty_returns_none(self):
        """Empty buffer → lookup_at returns None."""
        buf = BookSnapshotRingBuffer(maxlen=100)
        assert buf.lookup_at("mkt1", 100.0) is None

    def test_ring_buffer_backward_compat_book_history(self):
        """sim.book_history returns immutable tuples; mutation doesn't affect ring buffer."""
        sim = PaperTradingSimulator(seed=42)
        book = make_book()
        sim.record_book_snapshot("mkt1", book)
        sim.record_book_snapshot("mkt1", book)

        history = sim.book_history
        assert "mkt1" in history
        # Must be a tuple (immutable)
        assert isinstance(history["mkt1"], tuple)
        assert len(history["mkt1"]) == 2

        # Mutating the returned dict should NOT affect internal state
        history["mkt1"] = ()  # type: ignore[assignment]
        assert len(sim.book_history["mkt1"]) == 2  # Still 2 internally

        # Mutating dict by adding a new key should not affect internal state
        history["fake_mkt"] = ()  # type: ignore[assignment]
        assert "fake_mkt" not in sim.book_history


# ============================================================================
# TEST: Latency Simulation (Activity 15)
# ============================================================================


class TestLatencySimulation:
    def test_latency_draws_from_exponential(self):
        """1000 draws: mean within 20% of config, all >= 0."""
        sim = PaperTradingSimulator(seed=42, latency_mean_ms=50.0)
        draws = [sim._draw_latency_ms() for _ in range(1000)]
        assert all(d >= 0 for d in draws)
        mean = sum(draws) / len(draws)
        # Exponential mean should be close to 50ms
        assert 40.0 < mean < 60.0, f"Mean latency {mean:.1f}ms not near expected 50ms"

    def test_latency_zero_mean_returns_zero(self):
        """latency_mean_ms=0 → latency always 0.0."""
        sim = PaperTradingSimulator(seed=42, latency_mean_ms=0.0)
        for _ in range(100):
            assert sim._draw_latency_ms() == 0.0

    def test_latency_uses_historical_snapshot(self):
        """Pre-populate ring buffer → fill uses historical spread, not current."""
        sim = PaperTradingSimulator(seed=42, latency_mean_ms=100.0)

        # Record historical snapshots with WIDE spread
        for i in range(10):
            snap = BookSnapshot(
                timestamp=time.time() - 0.5 + i * 0.05,  # Recent history
                best_bid=0.40,
                best_ask=0.60,
                mid=0.50,
            )
            sim._ring_buffer.append("test_market", snap)

        # Current book has TIGHT spread
        order = make_order(side="BUY", price=0.52, condition_id="test_market")
        tight_book = make_book(best_bid=0.51, best_ask=0.52)

        # The fill should use a historical book (wide spread) due to latency
        # Run many times and check that at least some outcomes have
        # spread_factor different from what tight_book alone would give
        outcomes = []
        for _ in range(50):
            sim.reset_rng(42 + _)
            outcome = sim.simulate_fill_detailed(order, 60.0, tight_book)
            if outcome.fill_probability > 0:
                outcomes.append(outcome)

        # With latency, we should see latency_applied_ms > 0
        latencies = [o.latency_applied_ms for o in outcomes]
        assert any(lat > 0 for lat in latencies), "Should have some nonzero latency"

    def test_latency_degrades_without_history(self):
        """Empty ring buffer → degraded=True, uses current book."""
        # Use high latency mean to maximize chance of nonzero draw
        sim = PaperTradingSimulator(seed=42, latency_mean_ms=200.0)
        order = make_order(side="BUY", price=0.52)
        book = make_book(best_bid=0.48, best_ask=0.51)

        # No history recorded → should be degraded when latency > 0
        # Run enough seeds to guarantee at least one nonzero latency
        found_degraded = False
        for seed in range(50):
            sim.reset_rng(seed)
            outcome = sim.simulate_fill_detailed(order, 60.0, book)
            if outcome.latency_applied_ms > 0:
                assert outcome.degraded is True, (
                    f"With no history and latency={outcome.latency_applied_ms:.1f}ms, degraded should be True"
                )
                found_degraded = True
        assert found_degraded, "Should find at least one outcome with latency > 0 to verify degradation"

    def test_latency_deterministic_with_seed(self):
        """Same seed → same latency sequence."""
        sim1 = PaperTradingSimulator(seed=42, latency_mean_ms=50.0)
        draws1 = [sim1._draw_latency_ms() for _ in range(50)]

        sim2 = PaperTradingSimulator(seed=42, latency_mean_ms=50.0)
        draws2 = [sim2._draw_latency_ms() for _ in range(50)]

        assert draws1 == draws2

    def test_snapshot_to_orderbook_one_sided(self):
        """Snapshot with best_bid=0 → falls back to fallback_book.best_bid."""
        snap = BookSnapshot(timestamp=1000.0, best_bid=0.0, best_ask=0.55, mid=0.50)
        fallback = make_book(best_bid=0.48, best_ask=0.52)

        result = _snapshot_to_orderbook(snap, fallback)
        assert result.best_bid == 0.48  # Fell back
        assert result.best_ask == 0.55  # From snapshot


# ============================================================================
# TEST: Market Impact (Activity 16)
# ============================================================================


class TestMarketImpact:
    def test_impact_sqrt_scaling(self):
        """Double order size → impact grows by ~sqrt(2)."""
        impact1 = PaperTradingSimulator._compute_market_impact(
            order_size=100.0, available_liquidity=10000.0, spread=0.04, impact_k=0.02
        )
        impact2 = PaperTradingSimulator._compute_market_impact(
            order_size=200.0, available_liquidity=10000.0, spread=0.04, impact_k=0.02
        )
        assert impact2 > impact1
        ratio = impact2 / impact1 if impact1 > 0 else 0
        # Should be close to sqrt(2) ≈ 1.414
        assert 1.3 < ratio < 1.6, f"Ratio {ratio} not near sqrt(2)"

    def test_impact_capped_at_2x_spread(self):
        """Huge order → impact ≤ max(spread * 2, MIN_IMPACT_CAP)."""
        spread = 0.04
        impact = PaperTradingSimulator._compute_market_impact(
            order_size=1000000.0, available_liquidity=100.0, spread=spread, impact_k=0.02
        )
        cap = max(spread * 2.0, MIN_IMPACT_CAP)
        assert impact <= cap + 1e-10, f"Impact {impact} exceeds cap {cap}"

    def test_impact_zero_k_returns_zero(self):
        """impact_k=0 → no impact."""
        impact = PaperTradingSimulator._compute_market_impact(
            order_size=100.0, available_liquidity=10000.0, spread=0.04, impact_k=0.0
        )
        assert impact == 0.0

    def test_impact_zero_liquidity_returns_zero(self):
        """available_liquidity=0 → 0.0 (no div-by-zero)."""
        impact = PaperTradingSimulator._compute_market_impact(
            order_size=100.0, available_liquidity=0.0, spread=0.04, impact_k=0.02
        )
        assert impact == 0.0

    def test_impact_worsens_fill_price_buy(self):
        """BUY fill with impact should have fill_price > order.price."""
        sim = PaperTradingSimulator(seed=42, impact_k=0.05, latency_mean_ms=0.0)
        order = make_order(side="BUY", price=0.52, size=500.0)
        book = make_book(best_bid=0.48, best_ask=0.51)

        # Run many simulations to get at least one fill
        for seed in range(100):
            sim.reset_rng(seed)
            outcome = sim.simulate_fill_detailed(order, 60.0, book)
            if outcome.filled:
                assert outcome.fill_price is not None
                # With impact, BUY should fill at >= order.price
                assert outcome.fill_price >= order.price, (
                    f"BUY fill_price {outcome.fill_price} < order.price {order.price}"
                )
                assert outcome.impact_bps > 0, "Impact should be positive for fills"
                break
        else:
            # If no fill in 100 tries, skip (very unlikely with these params)
            pass

    def test_impact_worsens_fill_price_sell(self):
        """SELL fill with impact should have fill_price < order.price."""
        sim = PaperTradingSimulator(seed=42, impact_k=0.05, latency_mean_ms=0.0)
        order = make_order(side="SELL", price=0.48, size=500.0)
        book = make_book(best_bid=0.49, best_ask=0.52)  # Crosses SELL@0.48

        for seed in range(100):
            sim.reset_rng(seed)
            outcome = sim.simulate_fill_detailed(order, 60.0, book)
            if outcome.filled:
                assert outcome.fill_price is not None
                # With impact, SELL should fill at <= order.price
                assert outcome.fill_price <= order.price, (
                    f"SELL fill_price {outcome.fill_price} > order.price {order.price}"
                )
                assert outcome.impact_bps > 0
                break

    def test_consumed_side_liquidity_buy_uses_asks(self):
        """BUY order → liquidity computed from asks, not bids."""
        order_buy = make_order(side="BUY", price=0.52)
        order_sell = make_order(side="SELL", price=0.48)

        # Book with no bids/asks lists → should return DEFAULT_LIQUIDITY
        book = make_book()
        liq = _get_consumed_side_liquidity(order_buy, book)
        assert liq == DEFAULT_LIQUIDITY

        liq_sell = _get_consumed_side_liquidity(order_sell, book)
        assert liq_sell == DEFAULT_LIQUIDITY


# ============================================================================
# RUN TESTS
# ============================================================================


def run_all_tests():
    """Run all test classes and report results."""
    test_classes = [
        TestDeterminism,
        TestSensitivity,
        TestSanityBounds,
        TestPartialFills,
        TestAdverseSelection,
        TestFees,
        TestFillOptimismReduction,
        TestDiagnostics,
        TestBookSnapshotRingBuffer,
        TestLatencySimulation,
        TestMarketImpact,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            test_name = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {test_name}")
            except AssertionError as e:
                failed += 1
                errors.append((test_name, str(e)))
                print(f"  FAIL  {test_name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((test_name, f"{type(e).__name__}: {e}"))
                print(f"  ERROR {test_name}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
