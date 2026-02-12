# 🚨 PRODUCTION READINESS AUDIT - TRUTH GAPS

**Status:** 65/65 unit tests passing ≠ Paper-trading ready
**Reality:** ~80-90% complete, missing critical runtime truth validation

---

## ✅ What IS Solid (Confirmed)

### Architecture Shape
- ✅ Execution firewall exists and is non-bypassable
- ✅ Risk engine is centralized (cluster/agg caps, reservations, negRisk)
- ✅ Three-clock time model implemented (no hours_to_expiry in code)
- ✅ Toxicity overrides QS (maker survival edge)
- ✅ Maker vs taker separation in Truth Report

### Code Quality
- ✅ 21 ChatGPT fixes claimed as "locked with tests"
- ✅ Clean compilation
- ✅ Proper separation of concerns (feeds/execution/risk/strategy)

---

## 🚨 CRITICAL PRODUCTION GAPS (Must Fix Before Paper)

### **GAP #1: Tick Size / Min Size / Precision**

**Current State:**
- `STANDARD_TICK_SIZE = 0.01` hardcoded in config.py
- No per-market tick_size enforcement
- No min order size validation
- No token decimals handling

**Why This Kills Systems:**
- Markets with non-standard tick (0.001, 0.005) → constant CLOB rejects
- Min size violations → reject storms → mutation budget exhaustion → cancel-all loops
- One bad market kills the entire bot

**Must Fix:**
```python
# execution/clob_executor.py - MISSING
def validate_market_constraints(self, intent: OrderIntent, market: Market):
    """
    CRITICAL: Per-market constraints validation.

    Must check:
    - tick_size from market metadata (NOT config constant)
    - min_order_size from market metadata
    - price within [0.01, 0.99] after rounding
    - size >= min_size
    """
    # Currently assumes tick=0.01 everywhere
    # MUST pull from market.tick_size
```

**Test Gap:**
- No tests with tick_size != 0.01
- No tests with min_size violations
- No tests proving executor rejects invalid orders BEFORE submit

**Fix Priority:** 🔴 CRITICAL - Will cause immediate failure

---

### **GAP #2: Maker vs Taker Classification**

**Current State:**
```python
# feeds/user_ws.py
maker = data.get("maker", True)  # Default to maker if not specified ❌
```

**Why This Is Wrong:**
- Polymarket WS may not always provide `maker` flag
- Partial fills complicate classification
- Post-only orders can still fill quickly
- Misclassified fills poison realized spread metrics

**Must Fix:**
```python
# Need: Order origin tracking
class StoredOrder:
    post_only: bool  # ✅ Already have
    # MISSING:
    initial_intent: str  # "MAKER_QUOTE" vs "TAKER_ARB" vs "SATELLITE"

# Then in fill classification:
def classify_fill_maker_vs_taker(fill: Fill, order: StoredOrder) -> bool:
    """
    CRITICAL: Don't trust WS flag alone.

    Truth sources:
    1. Order origin (post_only maker orders)
    2. WS maker flag (if present)
    3. Fill price vs book at time (if we crossed spread → taker)
    """
    if order.post_only and order.initial_intent == "MAKER_QUOTE":
        return True  # Definitely maker
    # ... more logic
```

**Test Gap:**
- No tests for missing `maker` flag in WS data
- No tests for partial fill classification
- No tests proving realized spread is computed correctly

**Fix Priority:** 🔴 CRITICAL - Will cause incorrect Gate B metrics

---

### **GAP #3: OrderStateStore Reconciliation Details**

**Current State:**
```python
# execution/order_state_store.py
def reconcile_with_clob(self, clob_open_orders):
    # Basic reconciliation exists
    # MISSING: Partial fill handling
    # MISSING: Out-of-order WS events
    # MISSING: Persistent storage durability
```

**Why This Is Wrong:**
- Partial fills: Order stays LIVE but size reduces → reservations wrong
- Out-of-order WS: Event arrives before CLOB response → phantom LIVE orders
- JSON-only storage: Crash mid-write = corrupted state → wrong reservations

**Must Fix:**
```python
class StoredOrder:
    # MISSING:
    original_size: float  # Initial size
    remaining_size: float  # After partial fills
    filled_size: float    # Cumulative filled

def update_partial_fill(self, order_id: str, fill_size: float):
    """CRITICAL: Update reservations on partial fill."""
    order = self.orders[order_id]
    order.filled_size += fill_size
    order.remaining_size -= fill_size

    # RELEASE partial reservation
    if order.side == "BUY":
        release_usdc = fill_size * order.price
        # Update reserved_usdc_by_market

    # If remaining_size == 0 → mark FILLED
    if order.remaining_size <= 0:
        order.status = OrderStatus.FILLED
```

**Storage Fix:**
```python
# MUST use atomic writes
import tempfile
import os

def _save(self):
    # Write to temp file first
    temp_file = f"{ORDER_STORE_FILE}.tmp"
    with open(temp_file, 'w') as f:
        json.dump(data, f)
    # Atomic rename
    os.replace(temp_file, ORDER_STORE_FILE)
```

**Test Gap:**
- No tests for partial fills
- No tests for out-of-order events
- No tests proving reservations correct after partial fill → cancel
- No crash recovery tests

**Fix Priority:** 🔴 CRITICAL - Will cause phantom orders and wrong risk limits

---

### **GAP #4: Reservation Accounting Lifecycle**

**Current State:**
```python
# risk/portfolio_engine.py
def reserve_for_order(self, order):
    # Reserves at submit ✅
    # MISSING: Adjust on partial fill
    # MISSING: Transfer on replace
    # MISSING: Release on cancel
```

**Why This Is Wrong:**
- Submit 100 tokens, fill 50, cancel → 50 tokens still reserved forever
- Replace old order with new → double reservation
- Reservations drift from reality → eventually breach caps incorrectly

**Must Fix:**
```python
# Add to portfolio_engine.py:
def update_reservation_partial_fill(self, order_id: str, fill_size: float):
    """CRITICAL: Reduce reservation on partial fill."""
    order = self.order_store.orders[order_id]

    if order.side == "BUY":
        released_usdc = fill_size * order.price
        self.exposure.reserved_usdc_by_market[order.condition_id] -= released_usdc
    else:
        self.exposure.reserved_tokens_by_token_id[order.token_id] -= fill_size

def transfer_reservation_on_replace(self, old_order_id: str, new_order: OrderIntent):
    """CRITICAL: Transfer reservation from old → new order."""
    # Release old reservation
    old_order = self.order_store.orders[old_order_id]
    self.release_reservation(old_order)
    # Reserve for new
    self.reserve_for_order(new_order)
```

**Test Gap:**
- No integration test: submit → partial fill → replace → cancel → verify final available balance
- No test proving reservations match reality after complex order lifecycle

**Fix Priority:** 🔴 CRITICAL - Will cause slow drift in risk limits

---

### **GAP #5: Mutation Budget - Per-Market Cooldown + Backoff**

**Current State:**
```python
# execution/clob_executor.py
self.mutation_timestamps = deque(maxlen=MUTATION_MAX_PER_MINUTE)
# Global budget only ✅
# MISSING: Per-market cooldown
# MISSING: Exponential backoff on rejects
# MISSING: Pause state after reject storm
```

**Why This Is Wrong:**
- Bursty markets can monopolize mutation budget
- Reject storm → retry immediately → worse reject storm → death spiral
- No "safe pause" state → keeps trying until rate limit ban

**Must Fix:**
```python
class CLOBExecutor:
    def __init__(self):
        # Add:
        self.per_market_last_mutation: Dict[str, float] = {}
        self.per_market_cooldown_seconds = 30  # Min 30s between mutations per market
        self.reject_counts: Dict[str, int] = defaultdict(int)
        self.paused_until: Dict[str, float] = {}  # market_id → unpause_timestamp

    def can_mutate_market(self, market_id: str) -> Tuple[bool, str]:
        """Check per-market cooldown + pause state."""
        now = time.time()

        # Check if paused (after reject storm)
        if market_id in self.paused_until:
            if now < self.paused_until[market_id]:
                return False, f"Market paused until {self.paused_until[market_id]}"
            else:
                del self.paused_until[market_id]
                self.reject_counts[market_id] = 0  # Reset

        # Check per-market cooldown
        last_mut = self.per_market_last_mutation.get(market_id, 0)
        if now - last_mut < self.per_market_cooldown_seconds:
            return False, "Per-market cooldown active"

        return True, "OK"

    def on_order_reject(self, market_id: str, error: str):
        """Handle reject with exponential backoff."""
        self.reject_counts[market_id] += 1

        # Reject storm detection
        if self.reject_counts[market_id] >= 5:
            # Pause market for 5 minutes
            pause_duration = 300 * (2 ** (self.reject_counts[market_id] - 5))
            self.paused_until[market_id] = time.time() + pause_duration
            logger.error(
                f"REJECT STORM: {market_id} paused for {pause_duration}s "
                f"({self.reject_counts[market_id]} rejects)"
            )
```

**Test Gap:**
- No tests for reject storm → pause behavior
- No tests proving exponential backoff works
- No tests for per-market cooldown

**Fix Priority:** 🟠 HIGH - Will cause rate limit bans

---

### **GAP #6: Time Model Data Source**

**Current State:**
```python
# risk/market_state.py
def update_market_state(market: Market, metadata: Optional[dict] = None):
    if market.time_to_close is None:
        return MarketState.RESOLVED  # ❌ Unsafe assumption
```

**Why This Is Wrong:**
- Missing `time_to_close` could mean:
  - Market not yet opened
  - Metadata fetch failed
  - Market permanently closed
- Current code treats all as "RESOLVED" → may quote on markets that shouldn't trade

**Must Fix:**
```python
def update_market_state(market: Market, metadata: Optional[dict] = None) -> MarketState:
    """
    CRITICAL: Conservative handling of missing data.
    """
    if market.time_to_close is None:
        # CONSERVATIVE: Treat as unsafe → CHALLENGE_WINDOW (cancel-all)
        logger.error(
            f"Missing time_to_close for {market.condition_id}. "
            "Treating as CHALLENGE_WINDOW (cancel-all)."
        )
        return MarketState.CHALLENGE_WINDOW

    # ... rest of logic
```

**Data Source Missing:**
```python
# main_maker.py - STUBBED
async def _fetch_markets(self) -> List[Market]:
    # TODO: Implement Gamma API fetch ❌
    return []
```

**Must Implement:**
```python
async def _fetch_markets_from_gamma(self) -> List[Market]:
    """
    Fetch markets from Gamma API with proper time derivation.

    MUST extract:
    - tick_size from market metadata
    - min_size from market metadata
    - feeRateBps from market metadata
    - endDate → time_to_close calculation
    - event_id + negRisk flags
    """
    # Implementation needed
```

**Fix Priority:** 🔴 CRITICAL - Cannot run without real market data

---

### **GAP #7: negRisk Event Binding**

**Current State:**
```python
# risk/portfolio_engine.py
def assign_cluster(self, market: Market) -> str:
    if market.event and (market.event.neg_risk or market.event.augmented_neg_risk):
        # ✅ Logic exists
        # MISSING: How does market get linked to event?
        # MISSING: Periodic event refresh
```

**Why This Is Wrong:**
- Event metadata may arrive after initial market fetch
- Event flags can change (rare but possible)
- Markets not correctly linked → parity trades on negRisk markets → blow up

**Must Fix:**
```python
# Add to main_maker.py:
async def _refresh_events_metadata(self):
    """
    Periodic refresh of event metadata (every 10 minutes).

    MUST:
    - Re-fetch event flags (negRisk, augmentedNegRisk)
    - Update market.event bindings
    - If negRisk flag added → trigger cancel-all for affected markets
    """
    pass  # Implementation needed
```

**Fix Priority:** 🟠 HIGH - negRisk trades are rare but catastrophic

---

### **GAP #8: Paper Trading Fill Simulation**

**Current State:**
```python
# Paper trading not implemented yet
# Assumption: Will naively "assume fill at our price" ❌
```

**Why This Is Wrong:**
- Optimistic fill assumption → fake Sharpe
- Real microstructure matters:
  - Queue position
  - Adverse selection
  - Fill probability based on book changes

**Must Implement:**
```python
class PaperTradingSimulator:
    """
    Realistic fill simulation for paper mode.

    Fill probability based on:
    1. Queue position (if we're at best bid/ask)
    2. Book changes (if book crosses our price)
    3. Time in market (longer → higher fill probability)
    4. Adverse selection penalty (reduce size on "too good" fills)
    """

    def simulate_fill_probability(
        self,
        order: StoredOrder,
        book_updates: List[OrderBook],
        time_in_market_seconds: float,
    ) -> Tuple[bool, float]:
        """
        Returns: (filled: bool, size_filled: float)

        Conservative rules:
        - Only fill if book crosses our price
        - Apply adverse selection penalty
        - Track simulated vs actual for calibration
        """
        pass
```

**Truth Report Flag:**
```python
# reporting/truth_report.py
class TruthReport:
    def add_fill(self, fill: Fill, ...):
        # ADD:
        if fill.paper_simulated:  # ← Need this flag
            # Mark as simulated
            # Exclude from certain metrics
            # Add confidence bounds
```

**Fix Priority:** 🟠 HIGH - Gate A Sharpe will be fake otherwise

---

## 📋 EXACT WORK TO DO (Ordered by Criticality)

### **Phase 1: Pre-Paper Validation (MUST COMPLETE FIRST)**

#### A) Live Integration Sanity Suite
```python
# tests/test_live_integration.py (NEW FILE NEEDED)

def test_gamma_api_market_fetch():
    """Pull real markets and verify metadata."""
    markets = fetch_markets_from_gamma()

    for market in markets[:10]:  # Sample 10
        # MUST have:
        assert market.tick_size > 0
        assert market.fee_rate_bps > 0
        assert market.time_to_close is not None or market.state == MarketState.RESOLVED
        # ... more checks

def test_tick_size_enforcement():
    """Executor rejects orders with wrong tick size."""
    market = Market(..., tick_size=0.001)  # ← Non-standard
    intent = OrderIntent(..., price=0.5234)  # Not divisible by 0.001

    executor = CLOBExecutor(...)
    valid, reason = executor.validate_order_intent(intent)

    assert valid is False
    assert "tick" in reason.lower()

def test_min_size_enforcement():
    """Executor rejects orders below min size."""
    # Implementation needed

def test_clob_dry_run():
    """Submit smallest allowed orders in test mode."""
    # Verify: acceptance rate, reject reasons, WS latency
```

#### B) Partial Fill + Reservation Correctness
```python
# tests/test_reservation_lifecycle.py (NEW FILE NEEDED)

def test_partial_fill_reduces_reservation():
    """Partial fill must reduce reserved amount."""
    # Submit BUY 100 @ 0.50 → reserve $50
    # Fill 50 → release $25 reservation
    # Cancel remaining → release $25 reservation
    # Final: available balance = initial

def test_replace_transfers_reservation():
    """Replace must transfer reservation from old → new."""
    # Submit order 1 → reserve X
    # Replace with order 2 (different price/size) → reserve Y, release X
    # Verify no double reservation

def test_reconciliation_rebuilds_correct_reservations():
    """Restart reconciliation computes correct reserved amounts."""
    # Simulate: 3 LIVE orders in CLOB
    # Reconcile → reserved amounts = sum of LIVE order values
```

---

### **Phase 2: Production Hardening (After Phase 1)**

#### C) Mutation Budget + Reject Storm Handling
```python
# execution/clob_executor.py - ADD:
# - Per-market cooldown (30s)
# - Reject counter + exponential backoff
# - Pause state (5min → 10min → 20min)
# - Global circuit breaker: if total rejects > 50/min → pause ALL quoting

# tests/test_mutation_budget.py - ADD:
def test_reject_storm_triggers_pause():
    """5 rejects → market paused for 5 minutes."""

def test_exponential_backoff():
    """Each subsequent reject storm doubles pause duration."""

def test_per_market_cooldown():
    """Cannot mutate same market within 30s."""
```

#### D) Time Model Data Source Implementation
```python
# scanner/market_fetcher.py (EXISTS) - ENHANCE:
# - Add time_to_close calculation from endDate
# - Add tick_size, min_size, feeRateBps extraction
# - Add event_id binding
# - Add negRisk flag extraction

# main_maker.py - IMPLEMENT:
async def _fetch_markets(self):
    from scanner.market_fetcher import fetch_markets_with_metadata
    markets = await fetch_markets_with_metadata()
    # Validate all required fields present
    for m in markets:
        if m.tick_size is None:
            m.tick_size = 0.01  # Default but log warning
    return markets
```

#### E) Paper Trading Fill Simulation
```python
# execution/paper_simulator.py (NEW FILE NEEDED)
class PaperTradingSimulator:
    # Implement conservative fill model
    # Track simulated vs actual for calibration

# reporting/truth_report.py - ADD:
# - paper_simulated flag on Fill
# - Separate paper metrics from "would-be-live" metrics
# - Confidence bounds on Sharpe
```

---

## 🎯 REVISED READINESS CHECKLIST

### ❌ Current Status: NOT Paper-Trading Ready

**Blockers:**
- [ ] Per-market tick_size enforcement (GAP #1) 🔴
- [ ] Maker vs taker classification robust (GAP #2) 🔴
- [ ] Partial fill + reservation accounting (GAP #3, #4) 🔴
- [ ] Real market data fetch implemented (GAP #6) 🔴

**High Priority:**
- [ ] Reject storm handling + backoff (GAP #5) 🟠
- [ ] negRisk event refresh (GAP #7) 🟠
- [ ] Paper fill simulation realistic (GAP #8) 🟠

### ✅ After Fixes: Paper-Trading Ready Criteria

1. **Live Integration Suite Passing:**
   - ✅ Real markets fetched with correct metadata
   - ✅ Tick/min-size violations rejected pre-submit
   - ✅ Dry-run orders accepted by CLOB

2. **Reservation Accounting Proven:**
   - ✅ Partial fill → replace → cancel lifecycle tested
   - ✅ Reconciliation rebuilds correct reservations
   - ✅ Atomic storage (no corruption on crash)

3. **Reject Storm Resilience:**
   - ✅ Per-market cooldown + pause implemented
   - ✅ Exponential backoff tested
   - ✅ Global circuit breaker at 50 rejects/min

4. **Paper Simulation Honest:**
   - ✅ Fill probability model implemented
   - ✅ Truth Report marks simulated fills
   - ✅ Confidence bounds on metrics

---

## 💡 Bottom Line

**What You Built:** The right architecture (80-90% there)
**What's Missing:** Runtime truth validation (the hard 10-20%)

**ChatGPT's Warning Was Correct:**
> "65/65 tests passing does not mean you're paper-trading ready unless the tests are exercising real integration truths."

**Your tests prove:**
- ✅ Logic correctness (fee math, inventory skew sign, state transitions)
- ✅ Architecture non-bypassability (executor firewall, risk centralization)

**Your tests do NOT prove:**
- ❌ Handles real market metadata correctly (tick_size, min_size)
- ❌ Reservations stay correct through complex lifecycles
- ❌ Reject storms don't kill the bot
- ❌ Paper fills are realistic

---

## 🚀 Recommended Path Forward

**Option A: Quick Paper Mode (2-3 days)**
1. Fix GAP #1 (tick enforcement), #3/#4 (reservations), #6 (real data)
2. Run paper trading with "optimistic fills" flagged
3. Collect initial metrics but don't trust Sharpe yet
4. Fix remaining gaps during paper period

**Option B: Production-Grade First (1 week)**
1. Fix ALL 8 gaps before starting paper
2. Run live integration test suite
3. Paper trading with realistic fills
4. High confidence in Gate A results

**Recommendation: Option B**
- Fixing gaps DURING paper trading is harder (unknown unknowns)
- Better to build trust in metrics from day 1
- Gate A Sharpe needs to be trustworthy for Gate B decision

---

Generated: 2026-02-12
Reality Check: Production gaps identified and scoped
Next: Fix gaps → validate → paper trading
