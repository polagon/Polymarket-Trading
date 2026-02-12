# Canonical Mid Price Implementation

**Status**: ✅ Complete (2026-02-12)
**Purpose**: Single source of truth for "mid" across all system components

---

## Problem Solved

**Before**: Ad-hoc mid calculations scattered across codebase:
- `(bid + ask) / 2` without fallback handling
- No handling of one-sided books
- No staleness checks
- Inconsistent behavior across modules
- Silent failures when book data missing

**After**: Canonical `execution.mid.compute_mid()` with:
- Deterministic fallback rules (one-sided, stale, no data)
- Logging when fallbacks are used (critical for debugging)
- Reliability checking (`is_mid_reliable()`)
- Consistent behavior across all modules

---

## Fallback Hierarchy

```python
1. Both bid and ask exist → (bid + ask) / 2
2. One-sided book → offset from single side by MID_FALLBACK_ONE_SIDED_OFFSET (0.02)
3. Stale book (age > MID_FALLBACK_STALE_AGE_MS) → last_mid if available
4. No data → None (refuse to trade)
```

**Config Constants** (`config.py`):
- `MID_FALLBACK_ONE_SIDED_OFFSET = 0.02` (±2¢ from single side)
- `MID_FALLBACK_STALE_AGE_MS = 10000` (10s max age)

---

## Files Updated

### Core Implementation
- **`execution/mid.py`** (already existed) - Canonical compute_mid() implementation

### Integration Points Updated
1. **`models/types.py`**
   - `Market.mid_price` property → now uses `compute_mid()`
   - Added deprecation notice for direct usage

2. **`execution/paper_simulator.py`**
   - `record_book_snapshot()` → uses `compute_mid()`
   - Adverse selection penalty calculation → uses `compute_mid()`

3. **`feeds/market_ws.py`**
   - Last mid tracking → uses `compute_mid()`

4. **`execution/units.py`**
   - `aggregate_exposure_usd()` docstring → documents requirement

5. **`strategy/markout_tracker.py`**
   - `compute_markout()` docstring → documents requirement

6. **`strategy/market_maker.py`**
   - Already using `mid_module.compute_mid()` ✅ (no changes needed)

---

## Test Coverage

**`tests/test_canonical_mid.py`** - 17 tests, all passing:

### Unit Tests (12 tests)
- ✅ Standard case (both sides present)
- ✅ One-sided book (bid only, ask only)
- ✅ Stale book (with/without last_mid)
- ✅ Crossed book (bid > ask)
- ✅ No data cases
- ✅ Reliability checks

### Integration Tests (5 tests)
- ✅ `Market.mid_price` uses canonical mid
- ✅ `MarkoutTracker` interface documents requirement
- ✅ `aggregate_exposure_usd()` documents requirement
- ✅ `PaperTradingSimulator` uses canonical mid
- ✅ `compute_fair_value_band()` uses canonical mid

---

## Usage Examples

### Standard Usage
```python
from execution.mid import compute_mid

# With OrderBook
mid = compute_mid(
    book.best_bid,
    book.best_ask,
    book.last_mid,
    book.timestamp_age_ms,
)

if mid is None:
    logger.error("Cannot compute mid, refusing to quote")
    return
```

### Reliability Check
```python
from execution.mid import is_mid_reliable

if not is_mid_reliable(book.best_bid, book.best_ask, book.timestamp_age_ms):
    logger.warning("Mid is unreliable (one-sided or stale), widening bands")
    # Adjust strategy accordingly
```

### Inventory Valuation
```python
from execution.mid import compute_mid
from execution.units import aggregate_exposure_usd

# Compute mids for all tokens
mid_prices = {}
for token_id, book in books.items():
    mid = compute_mid(book.best_bid, book.best_ask, book.last_mid, book.timestamp_age_ms)
    if mid is not None:
        mid_prices[token_id] = mid

# Aggregate exposure
total_exposure = aggregate_exposure_usd(token_positions, mid_prices)
```

---

## Why This Matters for Sharpe 3.0

1. **Mark-to-mid inventory tracking**: If mid is wrong, inventory exposure is wrong
2. **Markout calculation**: If mid is inconsistent, markout metrics are meaningless
3. **Spread capture stats**: Realized spread = (fill_price - mid) depends on canonical mid
4. **Satellite edge comparison**: "market price vs our estimate" needs canonical mid
5. **Fair value bands**: Market maker quotes around mid need consistent mid

**Without canonical mid**: Silent lies in Truth Reports, fake Sharpe from bad accounting.
**With canonical mid**: Every metric uses the same truth, enabling real learning.

---

## Next Steps

Integration checklist item completed:
- ✅ **Canonical compute_mid() implemented and tested**

Ready for:
- Wire end-to-end paper loop (markets → executor → fills → markout → report)
- All components now use consistent mid calculation
- Markout/inventory/spread metrics will be correct

---

## Logging Behavior

**Important**: `compute_mid()` logs when using fallbacks:
```
WARNING: One-sided book (bid only): bid=0.50, using mid=0.52 (offset +0.02)
WARNING: Book stale (12000ms > 10000ms), using last_mid=0.51
ERROR: Book stale (12000ms) and no last_mid available. Refusing to compute mid.
ERROR: Crossed book: bid=0.52 > ask=0.50. Refusing to compute mid.
```

These logs are CRITICAL for post-trade debugging. If markout looks wrong, check logs for fallback usage.

---

Generated: 2026-02-12
Next: Wire end-to-end paper loop
