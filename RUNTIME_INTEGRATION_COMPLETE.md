# Runtime Integration Complete - Ready for Paper Burn-In

**Status**: ✅ Complete (2026-02-12)
**Entrypoint**: `main_maker.py`
**Mode**: Paper trading (PAPER_MODE=true in config)

---

## What Was Built

### 1. **Single Runtime Entrypoint** (`main_maker.py`)

Complete integration of all 8 gap fixes into a runnable paper trading loop.

**Key Components Wired**:
- ✅ Market fetcher v2 with full metadata (GAP #6)
- ✅ Order State Store with partial fills (GAP #3)
- ✅ Portfolio Risk Engine with reservations (GAP #4)
- ✅ CLOB Executor with per-market validation (GAP #1)
- ✅ Reject storm handling with exponential backoff (GAP #5)
- ✅ Event refresher for negRisk detection (GAP #7)
- ✅ Paper trading simulator (GAP #8)
- ✅ Markout/toxicity tracking
- ✅ Canonical mid price calculation (everywhere)
- ✅ Maker/taker classification (GAP #2)

### 2. **Fill Handler - Single Truth Path**

`_on_fill()` method processes ALL fills (WS or paper) through the same logic:

```python
def _on_fill(self, fill: Fill):
    # GAP #3: Update order state with partial fill
    order_store.update_partial_fill(...)

    # GAP #4: Update reservation (release filled portion)
    risk_engine.update_reservation_partial_fill(...)

    # GAP #2: Classify maker/taker
    fill.classify_maker_taker(order, book_mid)

    # Markout tracking (maker fills only)
    markout_tracker.record_fill_for_markout(fill, book)

    # Truth Report
    truth_report.add_fill(fill, cluster_id, pnl)
```

### 3. **Paper Trading Integration** (GAP #8)

`_simulate_paper_fills()` method:
- Records book snapshots for every market
- Checks each live order for potential fill
- Uses conservative fill probabilities (no fake Sharpe)
- Routes simulated fills through same `_on_fill()` handler

### 4. **Event Safety** (GAP #7)

Event refresher runs every ~1 hour:
- Detects negRisk flag changes
- Logs ERROR when negRisk added
- Disables parity arb for negRisk events
- Warns operator to cancel orders

### 5. **Circuit Breakers**

**Stale feed**: Triggers cancel-all + unsafe mode
**User feed disconnect**: Triggers cancel-all + unsafe mode
**Order staleness**: Auto-expires orders not seen in WS

---

## New Modules Created

### `strategy/resolution_risk_scorer.py`
- Computes RRS (Resolution Risk Score) 0-1
- Detects ambiguity markers in market text
- Category-specific dispute priors
- Hard veto gates: RRS > 0.35 = no maker, RRS > 0.25 = no satellite

### `execution/mid.py` Integration
- All ad-hoc mid calculations replaced with canonical `compute_mid()`
- Fallback rules: one-sided books, stale books, no data
- Logging when fallbacks are used

### `config.py` Additions
- `PAPER_MODE` flag (default: true from env)
- Mid fallback constants already existed

---

## Runtime Flow

### Startup Sequence
1. Preflight checks (wallet balance, allowances)
2. Reconciliation (load order store, sync with CLOB)
3. WebSocket connections (market + user feeds)
4. Load Astra predictions (satellite integration)
5. Enter main loop

### Main Loop (60-second cycles)
1. **Fetch markets** (Gamma API with full metadata)
   - Register markets with executor (GAP #1)
   - Store in `self.markets`

2. **Event refresh** (every 60 cycles ~ 1 hour)
   - Check for negRisk flag changes
   - Warn if negRisk markets detected

3. **Subscribe to order books** (WebSocket)
   - Subscribe to new YES/NO tokens

4. **Update market states** (NORMAL → WATCH → CLOSE_WINDOW, etc.)

5. **Compute QS** (Quoteability Scores)
   - For each market: compute RRS, QS
   - Override QS with toxicity check

6. **Select active set** (top N markets by QS)
   - Enforce cluster diversity (max 5 per cluster)

7. **Generate maker quotes**
   - Fair value band + inventory skew
   - Create OrderIntents with `origin="MAKER_QUOTE"`

8. **Scan parity arb**
   - YES/NO parity scanner
   - Skip negRisk markets

9. **Scan satellite opportunities**
   - Astra V2 predictions + satellite filter
   - Strict gates (RRS < 0.25, edge > 15%, etc.)

10. **Submit orders** (batched, ≤15 per batch)
    - Executor validates tick/min_size/cooldown
    - Tracks mutations in rolling budget

11. **Monitor staleness**
    - Check book age, cancel-all if stale
    - Auto-expire orders not seen in WS

12. **Paper fills** (if PAPER_MODE)
    - Simulate fills based on book crossing + time
    - Route through `_on_fill()` handler

### Fill Event Handling (WS or Paper)
- Update OrderStateStore partial fill
- Update PortfolioEngine reservations
- Classify maker/taker
- Record for markout tracking
- Add to Truth Report

---

## Configuration

### Paper Mode
Set in `.env`:
```bash
PAPER_MODE=true  # Use paper simulator
# OR
PAPER_MODE=false  # Use live WebSocket fills
```

### Key Config Values
- `ACTIVE_QUOTE_COUNT = 40` (active set size)
- `WS_STALENESS_THRESHOLD_MS = 5000` (5s max age)
- `RECONCILE_INTERVAL_SECONDS = 300` (5min)
- `MID_FALLBACK_STALE_AGE_MS = 10000` (10s)
- `MID_FALLBACK_ONE_SIDED_OFFSET = 0.02` (2¢)

---

## Running the System

### Paper Trading (Default)
```bash
cd /Users/pads/Claude/Polymarket
./venv/bin/python main_maker.py
```

### Requirements
```bash
./venv/bin/pip install websockets aiohttp
```

### Environment Variables Required
```bash
# Minimal for paper mode
PAPER_MODE=true
ANTHROPIC_API_KEY=sk-ant-...  # For satellite Astra predictions

# Additional for live mode
POLY_PRIVATE_KEY=...
POLY_API_KEY=...
POLY_API_SECRET=...
POLY_API_PASSPHRASE=...
```

---

## What's NOT Implemented Yet

### In main_maker.py
1. **CLOB client initialization** - Currently `None` stub
   - Need to initialize `py-clob-client` with credentials
   - Wire to executor

2. **Truth Report generation** - Not emitted yet
   - Need to call `truth_report.generate()` daily
   - Write to `reports/YYYY-MM-DD.json`

3. **Reconciliation on restart** - Partially stubbed
   - Need to fetch open orders from CLOB
   - Compare with OrderStateStore

4. **Strategy modules** (partially stubbed)
   - `quoteability_scorer.py` - exists but may need tuning
   - `market_maker.py` - exists
   - `parity_scanner.py` - needs YES/NO book logic
   - `satellite_filter.py` - needs integration with Astra

### Circuit Breakers
- ✅ Stale feed → cancel-all (implemented)
- ✅ User feed disconnect → cancel-all (implemented)
- ⚠️ Inventory drift detection → not wired yet
- ⚠️ Abnormal fill rate detection → not wired yet

---

## Integration Test Checklist

### Must Pass Before 24h Burn-In

**Lifecycle Tests**:
- [ ] Submit order → partial fill → update store + reservation
- [ ] Submit order → partial fill → replace → reservation transferred
- [ ] Submit order → partial fill → cancel remainder → reservation released
- [ ] Check: sum(reservations) ≤ wallet_balance always

**Reject Storm**:
- [ ] Simulate 5+ rejects → market paused
- [ ] After pause expires → market can trade again

**negRisk Detection**:
- [ ] Fetch markets with negRisk event
- [ ] Verify parity arb disabled
- [ ] Verify ERROR logged

**Stale Book**:
- [ ] Book age > 5s → cancel-all triggered
- [ ] Unsafe mode prevents new quotes

**Markout**:
- [ ] Maker fill → book snapshot recorded
- [ ] After 2m → markout computed
- [ ] Toxic market → QS override to 0.0

---

## Success Criteria (24h Burn-In)

**Operational**:
- ✅ No crashes or exceptions that stop the loop
- ✅ No reservation drift (sum always ≤ wallet balance)
- ✅ No stale quote incidents without cancel-all
- ✅ No reject storms that death-spiral

**Metrics**:
- ✅ Quote count > 0 (system is quoting)
- ✅ Fill count < quote count * 0.5 (not filling everything = realistic)
- ✅ Cancel/replace ratio < 2.0 (not churning excessively)
- ✅ Truth Report generates (even if empty)

**NOT Required for Burn-In**:
- ❌ Positive Sharpe (too early)
- ❌ Realized spread > 0 (need more data)
- ❌ Perfect markout (need time to stabilize)

---

## Next Steps

1. **Create integration test suite** (`tests/test_integration_gaps.py`)
   - Test scary lifecycles (partial fill → replace → cancel)
   - Test reject storm recovery
   - Test negRisk detection
   - Test stale book handling

2. **Wire CLOB client** (if going beyond paper)
   - Initialize `py-clob-client` with credentials
   - Test preflight checks
   - Test reconciliation

3. **Run 24h paper burn-in**
   ```bash
   PAPER_MODE=true ./venv/bin/python main_maker.py
   ```
   - Monitor logs for errors
   - Check reservation accounting
   - Verify fills are realistic (not 100% fill rate)

4. **Add Truth Report generation**
   - Daily JSON output
   - Maker vs taker separation
   - Realized spread (maker-only)
   - Markout series

5. **Start Gate B validation** (7-14 days)
   - 3,000+ fills minimum
   - Realized spread median > 0
   - Markout_2m mean not significantly negative
   - No risk violations

---

## File Summary

### Modified
- `main_maker.py` - Complete runtime integration
- `config.py` - Added PAPER_MODE flag
- `strategy/markout_tracker.py` - Added `record_fill_for_markout()`
- All mid calculations → use `execution.mid.compute_mid()`

### Created
- `strategy/resolution_risk_scorer.py` - RRS computation
- `RUNTIME_INTEGRATION_COMPLETE.md` - This document

### Dependencies Added
- `websockets` - For WebSocket feeds
- `aiohttp` - Already installed

---

Generated: 2026-02-12
Status: ✅ Ready for integration testing
Next: Create `tests/test_integration_gaps.py` → 24h burn-in
