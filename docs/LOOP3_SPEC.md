# Loop 3: Activities 14-17 -- Fill Realism & Burn-in Safety

## Summary

Loop 3 adds three layers of fill realism to the paper trading simulator and a
fail-closed startup sequence with monitor-only burn-in. Starting from Loop 2
baseline (292 tests, pre-commit clean), Loop 3 delivers 328 tests.

## Activities

### Activity 14: BookSnapshotRingBuffer

**File:** `execution/paper_simulator.py`

Replaces unbounded `dict[str, list[BookSnapshot]]` with a per-market
`collections.deque(maxlen=N)` ring buffer. Key design decisions:

- `get(market_id)` returns `Optional[deque]` -- `None` for unknown markets
  (not a mutable empty container).
- `lookup_at(market_id, target_ts)` uses reverse linear scan from tail.
  For maxlen=1000 and "near now" lookups this hits the answer in <10 iterations.
- `book_history` backward-compat property returns
  `dict[str, tuple[BookSnapshot, ...]]` -- truly immutable (tuples, not deque
  references).

### Activity 15: Latency Simulation

**Files:** `execution/paper_simulator.py`, `config.py`

Simulates order-to-fill latency using exponential distribution
(`rng.exponential(mean_ms)`). Fill evaluation uses the historical book snapshot
from `now - latency_ms` via the ring buffer's `lookup_at()`.

- `PAPER_SIM_LATENCY_MEAN_MS` (default 50ms) controls mean latency.
- `FillOutcome` extended with `latency_applied_ms` and `degraded` fields.
- When no historical snapshot exists, degrades to current book with
  `degraded=True`.

### Activity 16: Market Impact Model

**Files:** `execution/paper_simulator.py`, `config.py`

Applies sqrt-based market impact to fill prices:

    impact = K * sqrt(order_size / available_liquidity)

Capped at `max(spread * 2, MIN_IMPACT_CAP)` where `MIN_IMPACT_CAP = 0.0001`
(1 bps in price units).

- `MARKET_IMPACT_K` (default 0.02) controls impact severity.
- Liquidity computed from the **consumed side** (BUY -> asks, SELL -> bids).
- `FillOutcome` extended with `impact_bps` field.
- BUY fills shift up (worse), SELL fills shift down (worse).

### Activity 17: Startup Checklist + Monitor-Only Phase

**Files:** `ops/startup_checklist.py` (new), `paper_trader.py`, `config.py`

#### Startup Checklist (17 checks, fail-closed)

1. run_id valid UUID4
2. DB connection alive (SELECT 1)
3. DB schema_version matches CURRENT_SCHEMA_VERSION
4. Config hash is 64-hex SHA-256
5. Config snapshot exists in DB for this run
6. Prompt bundle hash is 64-hex SHA-256
7. Prompt registry artifact exists on disk
8. Artifacts directory writable
9. Manifest written successfully
10. Heartbeat path writable
11. DB backup exists (.db.bak1/.bak2/.bak3)
12. Gate engine dry-run evaluates successfully
13. Alert manager instantiated
14. Signal handlers registered
15. Memory directory exists and writable
16. Disk free > DISK_FAIL_MB
17. API key set (warn-only in paper mode)

Any non-warn failure aborts startup with `SystemExit(1)`.

#### Monitor-Only Burn-in

- `BURN_IN_MONITOR_CYCLES` (default 3) consecutive clean gate cycles required.
- Degraded gate status resets counter; `BURN_IN_MAX_FAILED_MONITORS` (default 10)
  total failures triggers halt.
- **Belt-and-suspenders enforcement:**
  1. `run_paper_scan(allow_new_positions=False)` skips position opening loop
  2. `PaperPortfolio.monitor_only` property + `set_monitor_only()` blocks
     `open_position()` at the portfolio level
- Monitor state written to gate_status and decision_report artifacts.

## Config Constants Added

| Constant | Default | Purpose |
|----------|---------|---------|
| `PAPER_SIM_LATENCY_MEAN_MS` | 50 | Mean latency for exponential draw (ms) |
| `MARKET_IMPACT_K` | 0.02 | Sqrt impact coefficient |
| `BURN_IN_MONITOR_CYCLES` | 3 | Clean cycles before positions allowed |
| `BURN_IN_MAX_FAILED_MONITORS` | 10 | Max degraded cycles before halt |

All added to `_CONFIG_ALLOWLIST`.

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `execution/paper_simulator.py` | Modified | Ring buffer, latency, impact model |
| `config.py` | Modified | 4 new constants + allowlist |
| `ops/startup_checklist.py` | **Created** | 17-point fail-closed checklist |
| `paper_trader.py` | Modified | Checklist call, monitor-only phase, artifacts |
| `tests/test_paper_simulator.py` | Modified | 20 new tests (3 classes) |
| `tests/test_ops.py` | Modified | 16 new tests (3 classes) |

## Test Summary

| Test Class | File | Count |
|------------|------|-------|
| `TestBookSnapshotRingBuffer` | test_paper_simulator.py | 7 |
| `TestLatencySimulation` | test_paper_simulator.py | 6 |
| `TestMarketImpact` | test_paper_simulator.py | 7 |
| `TestStartupChecklist` | test_ops.py | 8 |
| `TestMonitorOnly` | test_ops.py | 6 |
| `TestMonitorArtifacts` | test_ops.py | 2 |
| **Total new** | | **36** |

## Backward Compatibility

- `FillOutcome`: 3 new fields all have defaults
- `PaperTradingSimulator.__init__`: new params all optional with defaults
- `sim.book_history`: returns immutable tuples (existing read patterns work)
- `run_paper_scan`: new `allow_new_positions=True` default
- `PaperPortfolio.monitor_only`: explicit property API
- Seeded RNG: latency draws consume RNG state, so fill sequences change with
  same seed (intentional)

## Verification

```
pytest:      328 passed, 2 skipped
pre-commit:  ruff lint passed, ruff format passed, mypy passed
```
