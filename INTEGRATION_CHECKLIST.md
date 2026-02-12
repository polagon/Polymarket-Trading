# Integration Checklist - Post-Gap Fixes

**Status**: Ready for integration testing
**Goal**: Validate all 8 gap fixes work together end-to-end

---

## Phase 1: Component Integration (Day 1)

### ✅ GAP #1 + #6: Market Metadata Validation
- [ ] Import `market_fetcher_v2` in main loop
- [ ] Fetch markets with `fetch_markets_with_metadata()`
- [ ] Register markets with CLOBExecutor: `executor.register_market(market)`
- [ ] Verify tick_size, feeRateBps extracted correctly
- [ ] Test: Submit order with non-standard tick → should validate correctly

### ✅ GAP #2: Maker/Taker Classification
- [ ] Update order submission to set `origin="MAKER_QUOTE"` for maker orders
- [ ] Update order submission to set `origin="TAKER_ARB"` for parity arb
- [ ] Test: Fill.classify_maker_taker() with different scenarios
- [ ] Verify: classification_source is set correctly

### ✅ GAP #3 + #4: Partial Fill + Reservation Integration
- [ ] Wire OrderStateStore.update_partial_fill() to WS fill handler
- [ ] Wire PortfolioEngine.update_reservation_partial_fill() to fill handler
- [ ] Test lifecycle:
  - Submit order (size=100) → reservation created
  - Partial fill (size=50) → reservation reduced by 50%
  - Cancel remaining → reservation released
  - Check: available balance = initial balance

### ✅ GAP #5: Reject Storm Handling
- [ ] Wire executor.on_order_reject() to CLOB error responses
- [ ] Wire executor.on_order_success() to successful submissions
- [ ] Wire executor.can_mutate_market() to quote refresh logic
- [ ] Test: Simulate 5+ rejects → verify market paused
- [ ] Test: After pause expires → verify market can trade again

### ✅ GAP #7: Event Refresh Integration
- [ ] Instantiate EventRefresher in main loop
- [ ] Start periodic refresh: `refresher.start_periodic_refresh(get_markets_callback)`
- [ ] Test: Fetch markets with events → verify event.neg_risk populated
- [ ] Test: negRisk market detection → verify parity arb disabled

### ✅ GAP #8: Paper Trading Integration
- [ ] Instantiate PaperTradingSimulator
- [ ] Record book snapshots: `simulator.record_book_snapshot(market_id, book)`
- [ ] On each cycle, check live orders for fills:
  - Call `simulator.simulate_fill_probability(order, time_in_market, current_book)`
  - If filled, call `OrderStateStore.update_partial_fill()`
  - Create simulated Fill object
  - Update PortfolioEngine reservations
- [ ] Test: Order at best bid → book crosses → simulated fill
- [ ] Test: Adverse selection penalty applied on suspicious fills

---

## Phase 2: End-to-End Lifecycle Tests (Day 2)

### Test 1: Full Order Lifecycle
```python
# 1. Fetch markets with metadata
markets = await fetch_markets_with_metadata(limit=10)
executor.register_market(markets[0])

# 2. Submit order
intent = OrderIntent(
    condition_id=markets[0].condition_id,
    token_id=markets[0].yes_token_id,
    side="BUY",
    price=0.50,
    size_in_tokens=100,
    origin="MAKER_QUOTE"
)
await executor.submit_batch_orders([intent])

# 3. Check reservation
assert portfolio_engine.exposure.reserved_usdc_by_market[market_id] > 0

# 4. Simulate partial fill (50 tokens)
order_store.update_partial_fill(order_id, 50, timestamp)
portfolio_engine.update_reservation_partial_fill(order_id, 50, order_dict)

# 5. Check reservation reduced
assert portfolio_engine.exposure.reserved_usdc_by_market[market_id] < initial

# 6. Cancel remaining
await executor.cancel_orders([order_id])
portfolio_engine.release_reservation(order_dict)

# 7. Verify final balance
assert portfolio_engine.get_available_usdc(market_id, wallet_balance) == wallet_balance
```

### Test 2: Reject Storm Recovery
```python
# 1. Submit order to market
# 2. Simulate 5 consecutive rejects
for i in range(5):
    executor.on_order_reject(market_id, "INVALID_TICK_SIZE")

# 3. Verify market paused
allowed, reason = executor.can_mutate_market(market_id)
assert not allowed
assert "paused" in reason.lower()

# 4. Wait for pause expiry
await asyncio.sleep(pause_duration + 1)

# 5. Verify market can trade again
allowed, reason = executor.can_mutate_market(market_id)
assert allowed
```

### Test 3: negRisk Event Detection
```python
# 1. Fetch markets with events
markets = await fetch_markets_with_metadata()

# 2. Refresh event metadata
refresher = EventRefresher()
events = await refresher.refresh_events(markets)

# 3. Find negRisk markets
negRisk_markets = refresher.get_negRisk_markets(markets)

# 4. Verify parity arb disabled
for market in negRisk_markets:
    assert not portfolio_engine.can_trade_parity_arb(market)
```

### Test 4: Paper Trading Fill Simulation
```python
# 1. Submit order at best bid
order = create_test_order(side="BUY", price=0.50)

# 2. Record book crossing our price
book = OrderBook(best_bid=0.49, best_ask=0.50)  # Ask crosses our bid
simulator.record_book_snapshot(market_id, book)

# 3. Simulate fill after 60s
time_in_market = 60
filled, size_filled, fill_price = simulator.simulate_fill_probability(
    order, time_in_market, book
)

# 4. Verify fill occurred
assert filled
assert size_filled > 0
assert fill_price == 0.50
```

---

## Phase 3: Integration Test Suite (Day 3)

### Create: `tests/test_integration_gaps.py`

```python
import pytest
import asyncio
from execution.clob_executor import CLOBExecutor
from execution.order_state_store import OrderStateStore
from risk.portfolio_engine import PortfolioRiskEngine
from scanner.market_fetcher_v2 import fetch_markets_with_metadata
from scanner.event_refresher import EventRefresher
from execution.paper_simulator import PaperTradingSimulator


class TestGapIntegration:
    """Integration tests for all 8 gap fixes."""

    @pytest.fixture
    async def setup(self):
        """Setup all components."""
        order_store = OrderStateStore()
        executor = CLOBExecutor(clob_client=None, order_store=order_store)
        portfolio = PortfolioRiskEngine()
        simulator = PaperTradingSimulator()
        refresher = EventRefresher()

        return {
            "executor": executor,
            "order_store": order_store,
            "portfolio": portfolio,
            "simulator": simulator,
            "refresher": refresher,
        }

    async def test_gap_1_market_constraints(self, setup):
        """GAP #1: Per-market tick size validation."""
        markets = await fetch_markets_with_metadata(limit=10)
        assert len(markets) > 0

        market = markets[0]
        assert market.tick_size > 0
        assert market.fee_rate_bps > 0

        setup["executor"].register_market(market)

        # Test validation
        intent = OrderIntent(
            condition_id=market.condition_id,
            token_id=market.yes_token_id,
            side="BUY",
            price=0.5234,  # Not tick-rounded
            size_in_tokens=10,
        )

        valid, reason = setup["executor"].validate_order_intent(intent, market)
        if market.tick_size == 0.01:
            assert not valid  # Should fail if tick != 0.01

    # ... more tests for GAP #2-8
```

---

## Phase 4: Paper Trading Validation (Days 4-18)

### Setup Paper Trading Environment
```python
# main_paper.py

async def main_paper_trading():
    # 1. Fetch markets with metadata
    markets = await fetch_markets_with_metadata(limit=200)

    # 2. Register with executor
    for market in markets:
        executor.register_market(market)

    # 3. Start event refresher
    refresher = EventRefresher()
    asyncio.create_task(refresher.start_periodic_refresh(lambda: markets))

    # 4. Main loop
    simulator = PaperTradingSimulator()

    while True:
        # Fetch order books
        # Compute quotes
        # Submit orders
        # Simulate fills
        # Update reservations
        # Generate truth report

        await asyncio.sleep(10)
```

### Daily Monitoring Checklist
- [ ] Check reservation accounting: sum(reserved) <= wallet_balance
- [ ] Check reject counts: any markets with 5+ rejects?
- [ ] Check negRisk markets: any parity trades on them? (should be 0)
- [ ] Check paper fills: are fill probabilities reasonable?
- [ ] Check truth report: maker spread > 0? markout not negative?

---

## Gate B Validation (After 7-14 Days)

### Metrics to Check
- [ ] **Duration**: ≥7 days (target 14)
- [ ] **Fills**: ≥3,000 total
- [ ] **Clusters**: ≥8 different clusters
- [ ] **Realized spread**: Median > 0
- [ ] **Markout**: Mean markout_2m not significantly negative (p>0.05)
- [ ] **Fill rates**: Within 20% of assumptions
- [ ] **Cancel/replace ratio**: Healthy (not churning)
- [ ] **Risk violations**: Zero (caps respected 100%)
- [ ] **Sharpe**: ≥1.2

### GO/NO-GO Decision
- ✅ **GO**: All Gate B criteria met → small live ($500)
- ⚠️ **EXTEND**: Close but not all met → extend 7 more days
- ❌ **NO-GO**: Major gaps → debugging + redesign

---

## Quick Start Commands

```bash
# 1. Install dependencies
cd /Users/pads/Claude/Polymarket
./venv/bin/pip install -r requirements.txt

# 2. Run integration tests
./venv/bin/pytest tests/test_integration_gaps.py -v

# 3. Run paper trading (24h test)
./venv/bin/python main_paper.py --duration=24h

# 4. Generate truth report
./venv/bin/python report.py --date=2026-02-13
```

---

## Success Criteria

**Integration Phase Complete When**:
- ✅ All 8 gap fixes tested in isolation
- ✅ End-to-end lifecycle tests passing
- ✅ No reservation drift detected
- ✅ No reject storms in 24h test
- ✅ Paper fills realistic (not all orders fill)
- ✅ Truth report generates daily

**Paper Trading Phase Complete When**:
- ✅ 7-14 days continuous operation
- ✅ 3,000+ simulated fills
- ✅ Gate B metrics all passing
- ✅ Confidence in live deployment

---

Generated: 2026-02-12
Next: Integration testing → Paper trading → Live ($500)
