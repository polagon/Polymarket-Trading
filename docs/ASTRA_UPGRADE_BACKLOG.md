# ASTRA UPGRADE BACKLOG
**Date: 2026-02-13 | Merged from 7-Persona Review**

**Priority Key**: P0 = Blocks paper burn-in. P1 = Blocks live trading. P2 = Blocks allocator-grade judgment. P3 = Operational excellence.

---

## P0: BLOCKS PAPER BURN-IN

These must be completed before the 30-day paper burn-in can start producing meaningful data.

### P0-1: Run Unattended Paper Burn-In (30 days)

**Source Personas**: A (Allocator PM), F (Strategy Director), B (Execution Scientist)

**What**: Start paper_trader.py in continuous mode. Let it scan, estimate, size, simulate fills, resolve, learn. Do not intervene.

**DoD**:
- 200+ resolved positions in `trades` table with `resolution_outcome IS NOT NULL`
- Trades span at least 3 market categories
- Learning agent has processed all resolutions
- Metrics engine produces non-NaN Sharpe/Sortino with gate=PASS

**Tests**:
```sql
-- Minimum resolved trades
SELECT COUNT(*) FROM trades WHERE resolution_outcome IS NOT NULL; -- >= 200

-- Category diversity
SELECT COUNT(DISTINCT category) FROM trades WHERE resolution_outcome IS NOT NULL; -- >= 3

-- Metrics gate
-- PerformanceEngine.compute("all_time").sharpe_gate == "PASS"
```

**Metric Gate**: Sharpe per-trade > 0.05 AND Brier < 0.25 AND win_rate > 0.52

**Depends On**: Nothing (system is ready for paper burn-in NOW)

---

### P0-2: Add SIGTERM Graceful Shutdown Handler

**Source Personas**: G (SRE)

**What**: Handle SIGTERM/SIGINT to cleanly shut down: write final truth report, close WS connections, save state.

**DoD**:
- `signal.signal(SIGTERM, handler)` registered in main_maker.py and paper_trader.py
- Handler writes final truth report JSON
- Handler closes all WS connections
- Process exits cleanly (exit code 0)

**Tests**:
```python
def test_graceful_shutdown():
    # Send SIGTERM to subprocess, verify truth_report written and clean exit
```

**Metric Gate**: None (operational requirement)

---

### P0-3: Add Health Check Heartbeat

**Source Personas**: G (SRE)

**What**: Write heartbeat file (`/tmp/astra_health`) every 60 seconds with JSON status (timestamp, cycle_count, error_count, memory_mb).

**DoD**:
- Heartbeat file written every 60s
- Contains: `{"timestamp": "...", "cycle": N, "errors": N, "memory_mb": N, "feed_connected": bool}`
- External monitoring can check file age

**Tests**:
```python
def test_health_check_written():
    # Run 2 cycles, verify heartbeat file exists and is recent
```

**Metric Gate**: None (operational requirement)

---

## P1: BLOCKS LIVE TRADING

These must be completed before any real orders can be placed.

### P1-1: Initialize py-clob-client

**Source Personas**: B (Execution Scientist), G (SRE)

**What**: Replace `clob_client = None` in main_maker.py with properly initialized py-clob-client using credentials from config.py.

**DoD**:
- `clob_client` is initialized with POLY_PRIVATE_KEY, POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE
- Can call `clob_client.get_order()` without error
- Credential validation on startup (fail fast if missing)

**Tests**:
```python
def test_clob_client_initialized():
    # Verify client is not None and can make API call
```

**Metric Gate**: None (binary: works or doesn't)

---

### P1-2: Implement Order Submission + Cancellation

**Source Personas**: B (Execution Scientist)

**What**: Replace stubs in `execution/clob_executor.py` with actual py-clob-client calls for `create_order()` and `cancel_order()`.

**DoD**:
- `submit_batch_orders()` calls `clob_client.create_order()` for each order
- `cancel_order()` calls `clob_client.cancel_order()`
- Error handling for API errors, timeouts, rate limits
- Verified with 1 real order submit + cancel roundtrip on testnet/small size

**Tests**:
```python
def test_submit_cancel_roundtrip():
    # Submit small order, verify on CLOB, cancel, verify cancelled
```

**Metric Gate**: None (binary: works or doesn't)

---

### P1-3: Implement Order Reconciliation on Startup

**Source Personas**: B (Execution Scientist), G (SRE)

**What**: Replace `clob_orders = []` stub in `reconcile_with_clob()` with actual CLOB order fetch. Cancel stale orders. Rebuild order_state_store.

**DoD**:
- On startup, fetch all open orders from CLOB API
- Match against order_state_store
- Cancel orders not in store (orphans from crash)
- Add CLOB orders to store (missed during downtime)
- Log reconciliation summary

**Tests**:
```python
def test_reconcile_stale_orders():
    # Create mock stale orders, verify they're cancelled on startup
```

**Metric Gate**: Zero orphaned orders after 10 restart cycles

---

### P1-4: Wire User WebSocket Feed API Key

**Source Personas**: B (Execution Scientist)

**What**: Replace `api_key=None` in main_maker.py user_feed initialization with actual API key from config.

**DoD**:
- User feed connects with valid API key
- Fill events arrive via WebSocket callback
- Order update events arrive
- Disconnect/reconnect works

**Tests**:
```python
def test_user_ws_receives_fills():
    # Connect user feed, submit order, verify fill event arrives
```

**Metric Gate**: None (binary: works or doesn't)

---

### P1-5: Implement Realized P&L Tracking

**Source Personas**: B (Execution Scientist), D (Data Truth Architect)

**What**: Replace `pnl = 0.0` TODO in main_maker.py with actual realized P&L computation from position tracking (fills → position → mark-to-mid → realized on close).

**DoD**:
- P&L computed from fill history per market
- Truth report shows non-zero P&L
- P&L matches manual calculation for 10 test positions

**Tests**:
```python
def test_realized_pnl_computed():
    # Create fills, verify P&L is correct
```

**Metric Gate**: P&L within 1bps of manual calculation

---

### P1-6: Implement Semantic/NER-Based Clustering

**Source Personas**: C (Risk Engineer), A (Allocator PM)

**What**: Replace hash-based cluster assignment with entity-based clustering. Markets mentioning the same entity (e.g., "Trump", "Bitcoin", "Fed") should share a cluster.

**DoD**:
- Extract entities from market question text using NER or keyword matching
- Markets with overlapping key entities → same cluster
- negRisk event override still applies
- Cluster assignments logged
- Verified: Trump-related markets share a cluster

**Tests**:
```python
def test_trump_markets_same_cluster():
    # Create 3 Trump-related markets, verify same cluster_id

def test_unrelated_markets_different_clusters():
    # Create unrelated markets, verify different cluster_ids
```

**Metric Gate**: 90%+ of obviously-correlated markets share a cluster (manual validation on 50 markets)

---

### P1-7: Add SIGTERM Cancel-All for Live Mode

**Source Personas**: G (SRE), C (Risk Engineer)

**What**: In live mode, graceful shutdown must cancel ALL open orders on CLOB before exiting. This prevents orphaned orders from executing after shutdown.

**DoD**:
- SIGTERM handler cancels all open orders via CLOB API
- Waits for cancellation confirmation (with timeout)
- Logs all cancellations
- Then proceeds with shutdown (write report, close feeds)

**Tests**:
```python
def test_shutdown_cancels_all_orders():
    # Submit orders, send SIGTERM, verify all cancelled on CLOB
```

**Metric Gate**: Zero orphaned orders after SIGTERM

---

## P2: BLOCKS ALLOCATOR-GRADE JUDGMENT

These improve the quality of data and analysis needed to assess whether Astra meets allocator targets.

### P2-1: Build Closed-Loop Brier Pipeline

**Source Personas**: D (Data Truth Architect), A (Allocator PM), F (Strategy Director)

**What**: Automatically match estimates in the `estimates` table with resolution outcomes in the `trades` table. Compute running Brier score.

**DoD**:
- On resolution, find matching estimate by condition_id + closest timestamp
- Compute Brier: `(probability - outcome)^2`
- Store Brier score in trades table
- Running Brier available via `PerformanceEngine`
- Alert if Brier > 0.25 over rolling 50 predictions

**Tests**:
```python
def test_brier_pipeline_end_to_end():
    # Insert estimate + trade with known outcome, verify Brier computed
```

**Metric Gate**: Brier < 0.20 over 50+ resolved predictions

---

### P2-2: Add Regime Tagging to All Trades

**Source Personas**: A (Allocator PM), F (Strategy Director)

**What**: Tag every trade with regime metadata: vol_bucket (low/med/high), category, event_type (election/crypto/sports/other), time_to_resolution_bucket.

**DoD**:
- Every trade in SQLite has regime fields populated
- Metrics can be sliced by regime
- PerformanceEngine supports `compute(window, regime_filter=...)`

**Tests**:
```python
def test_regime_tags_present():
    # After paper trades, verify all have non-null regime tags
```

**Metric Gate**: 100% of trades have regime tags

---

### P2-3: Build Fill Rate Comparison Dashboard

**Source Personas**: B (Execution Scientist)

**What**: When live orders start, compare sim fill rate vs actual fill rate. Track the ratio over time.

**DoD**:
- For each market, track: sim_fill_rate (from paper_simulator stats) vs live_fill_rate (from actual fills / orders)
- Alert if ratio > 2.0 (sim too optimistic) or < 0.5 (sim too pessimistic)
- Daily summary in truth report

**Tests**:
```python
def test_fill_rate_comparison_computed():
    # Verify sim vs live fill rate ratio is computed and logged
```

**Metric Gate**: |sim_fill_rate / live_fill_rate - 1.0| < 0.5 (within 50% of reality)

---

### P2-4: Build Strategy Attribution Dashboard

**Source Personas**: A (Allocator PM), F (Strategy Director)

**What**: Extend metrics to compute per-strategy Sharpe, P&L, win rate after sufficient trades.

**DoD**:
- Every trade tagged with source strategy
- PerformanceEngine computes per-strategy metrics
- Strategies ranked by risk-adjusted return
- Gate: each strategy needs 100+ trades before judgment

**Tests**:
```python
def test_strategy_attribution_nonzero():
    # After paper trades, verify per-strategy P&L is non-zero
```

**Metric Gate**: At least 2 strategies with Sharpe > 0.5 per-trade

---

### P2-5: Add Sharpe Precision Tracker

**Source Personas**: A (Allocator PM)

**What**: Track CI width on Sharpe ratio. Show how many more trades needed for CI width < 0.5.

**DoD**:
- After each resolved trade, compute: current Sharpe CI width, trades_needed for target precision
- Display in truth report
- Alert when Sharpe CI width drops below 1.0 (first meaningful precision)

**Tests**:
```python
def test_sharpe_ci_width_tracking():
    # Verify CI width decreases as N increases
```

**Metric Gate**: CI width < 1.0 after 200 trades

---

### P2-6: Implement WS Uptime Tracking

**Source Personas**: D (Data Truth Architect), G (SRE)

**What**: Replace placeholder uptime values in truth report with actual connection uptime tracking.

**DoD**:
- Track connection time, disconnection events, reconnection time
- Compute: `uptime_pct = connected_seconds / wall_clock_seconds`
- Report in truth report per feed (market + user)
- Alert if uptime < 95%

**Tests**:
```python
def test_ws_uptime_nonplaceholder():
    # Verify uptime values differ from 86000.0 placeholder
```

**Metric Gate**: Market feed uptime > 99% over 24h

---

### P2-7: Add Config Snapshot to Audit Trail

**Source Personas**: D (Data Truth Architect), E (Model Governance Lead)

**What**: Log all config values as a JSON snapshot to SQLite on every startup. This enables tracing parameter changes that affect results.

**DoD**:
- New SQLite table: `config_snapshots (session_id TEXT, timestamp TEXT, config_json TEXT)`
- All PAPER_SIM_*, risk cap, QS, and RRS parameters captured
- Config diff between consecutive sessions displayed

**Tests**:
```python
def test_config_snapshot_logged():
    # Start session, verify config_snapshot row exists
```

**Metric Gate**: None (audit requirement)

---

## P3: OPERATIONAL EXCELLENCE

These improve robustness and operational quality for sustained production use.

### P3-1: Build Monte Carlo Ruin Simulator

**Source Personas**: C (Risk Engineer)

**What**: Simulate 10,000 paths of portfolio returns using observed return distribution. Compute P(drawdown > 50%), P(ruin).

**DoD**:
- Monte Carlo engine using observed trade returns (or synthetic if insufficient)
- 10,000 paths, 1000 trades each
- Report: P(MDD > 20%), P(MDD > 50%), expected max drawdown, 95% worst-case drawdown
- Current risk caps as constraints

**Tests**:
```python
def test_monte_carlo_ruin_prob():
    # With current caps, P(ruin) < 0.1%
```

**Metric Gate**: P(MDD > 50%) < 1%

---

### P3-2: Build Stress Testing Framework

**Source Personas**: C (Risk Engineer)

**What**: Simulate correlated adverse scenarios: 5 markets in same cluster all resolving against position simultaneously.

**DoD**:
- Define stress scenarios (correlated resolution, liquidity drain, feed outage)
- Compute P&L impact under each scenario
- Verify risk caps contain the damage
- Report worst-case loss per scenario

**Tests**:
```python
def test_correlated_stress_scenario():
    # 5 correlated markets resolve against, verify loss < 12% (cluster cap)
```

**Metric Gate**: No scenario produces loss > aggregate cap (40%)

---

### P3-3: Create Model Cards

**Source Personas**: E (Model Governance Lead)

**What**: Formal documentation for each model: fill simulator, estimation pipeline, Kelly sizer.

**DoD**:
- Model card per model: purpose, assumptions, limitations, parameters, validation status, failure modes
- Stored in `docs/model_cards/`
- Updated when parameters change

**Tests**: Manual review

**Metric Gate**: None (documentation requirement)

---

### P3-4: Version Estimation Prompts

**Source Personas**: E (Model Governance Lead)

**What**: Move AI estimation prompts from inline code to versioned config. Log prompt version hash with each estimate.

**DoD**:
- Prompts in `config/estimation_prompts.yaml` with version hash
- Each estimate in SQLite includes `prompt_version` field
- Schema migration to add column

**Tests**:
```python
def test_prompt_version_logged():
    # Run estimation, verify prompt_version is non-null in estimates table
```

**Metric Gate**: None (governance requirement)

---

### P3-5: Add Dynamic VaR-Based Position Scaling

**Source Personas**: C (Risk Engineer)

**What**: Scale position sizes based on realized volatility. When vol is 2x target, halve position sizes.

**DoD**:
- Compute rolling 30-trade realized vol
- Compare to target vol (derived from Calmar target)
- Scale `MAX_MARKET_INVENTORY_PCT` dynamically: `base_cap * (target_vol / realized_vol)`
- Floor at 0.25x base cap, ceiling at 1.5x base cap

**Tests**:
```python
def test_dynamic_var_scaling():
    # High vol period → smaller positions. Low vol → larger positions.
```

**Metric Gate**: Position sizes respond to vol changes within 5 trades

---

### P3-6: Create Docker Deployment

**Source Personas**: G (SRE)

**What**: Dockerfile + docker-compose for reproducible deployment with restart policy.

**DoD**:
- Dockerfile with pinned Python version, requirements.txt
- docker-compose.yml with restart: unless-stopped
- Volume mounts for SQLite data and config
- Health check in docker-compose (check heartbeat file age)

**Tests**:
```bash
docker-compose up -d && sleep 120 && docker-compose ps  # Should show "healthy"
```

**Metric Gate**: 99.9% uptime over 7 days

---

### P3-7: Add Alerting Integration

**Source Personas**: G (SRE)

**What**: Send alerts to Slack/Discord on: circuit break, kill switch activation, error rate spike, daily summary.

**DoD**:
- Webhook-based alerting (Slack or Discord)
- Alert on: feed disconnect, reject storm, kill switch, daily Sharpe update
- Configurable via env var (ALERT_WEBHOOK_URL)

**Tests**:
```python
def test_alert_sent_on_circuit_break():
    # Trigger circuit break, verify webhook called
```

**Metric Gate**: Alert delivery < 30s from event

---

### P3-8: Add DB Schema Versioning + Integrity Checks

**Source Personas**: D (Data Truth Architect)

**What**: Use SQLite `PRAGMA user_version` for schema versioning. Run integrity checks on startup.

**DoD**:
- Schema version tracked in `PRAGMA user_version`
- Startup checks: schema version, row counts, last_write timestamp
- Migration runner for schema changes
- Alert on corruption detected

**Tests**:
```python
def test_schema_version_check():
    # Verify user_version is set and migration runner works
```

**Metric Gate**: Zero data integrity failures over 30 days

---

### P3-9: Build Learning Agent Holdout Validation

**Source Personas**: E (Model Governance Lead)

**What**: Reserve 20% of resolved outcomes as holdout set. Alert if training Brier diverges from holdout Brier by > 0.05 (overfitting signal).

**DoD**:
- 20% of resolutions held out
- Training Brier vs holdout Brier tracked
- Alert if gap > 0.05
- Rollback to previous calibration if overfitting detected

**Tests**:
```python
def test_learning_agent_holdout():
    # Verify holdout set exists and overfitting detection works
```

**Metric Gate**: |train_brier - holdout_brier| < 0.05

---

### P3-10: Build Aggregate Drawdown Circuit Breaker

**Source Personas**: C (Risk Engineer)

**What**: Halt all new entries when portfolio equity drops >10% from peak.

**DoD**:
- Drawdown tracker feeds circuit breaker
- When DD > 10%, `can_enter_position()` returns False for all markets
- Resumes when DD recovers to < 5% (hysteresis)
- Alert sent on activation/deactivation

**Tests**:
```python
def test_aggregate_drawdown_circuit_breaker():
    # Simulate 10% drawdown, verify all entries blocked
```

**Metric Gate**: Circuit breaker activates within 1 trade of threshold breach

---

## BACKLOG SUMMARY

| Priority | Count | Status |
|----------|-------|--------|
| **P0** (Paper burn-in) | 3 | Ready to start NOW |
| **P1** (Live trading) | 7 | Blocked on CLOB integration |
| **P2** (Allocator judgment) | 7 | Blocked on resolved trades |
| **P3** (Operational excellence) | 10 | Can be parallelized |

### Recommended Execution Order

```
Week 1-4:  P0-1 (paper burn-in starts immediately)
           P0-2, P0-3 (operational basics, can run in parallel)

Week 1-2:  P1-1 through P1-5 (CLOB integration, parallel with burn-in)
           P2-7 (config audit, quick win)

Week 2-4:  P1-6 (NER clustering, needs research)
           P2-1 (Brier pipeline)
           P2-2 (regime tagging)
           P2-6 (WS uptime)

Week 4+:   P2-3 through P2-5 (needs resolved trade data)
           P1-7 (needs live CLOB)
           P3-* (as bandwidth allows)
```

**Critical Path**: P0-1 (paper burn-in) is the single most important activity. Everything else waits on resolved trade data.
