# Polymarket Allocator-Grade Trading System

A production-ready market-making system for Polymarket with portfolio risk management, resolution risk scoring, and paper mode optimization.

## System Architecture

### Core Components

1. **Market-Making Engine** (`strategy/market_maker.py`)
   - Inventory-aware quoting with fair value bands
   - Post-only limit orders (GTC/GTD)
   - Tick rounding enforcement

2. **Quoteability Scorer (QS)** (`strategy/quoteability_scorer.py`)
   - Selects top N markets from watchlist for active quoting
   - Hard veto gates: RRS, market state, spread, staleness, liquidity
   - Mode-aware staleness thresholds (5s PROD / 120s PAPER)

3. **Portfolio Risk Engine** (`risk/portfolio_engine.py`)
   - Cluster exposure caps (12% per cluster, 40% aggregate)
   - Token-level inventory tracking (mark-to-mid)
   - NegRisk event detection

4. **Resolution Risk Score (RRS)** (`strategy/resolution_risk_scorer.py`)
   - Ambiguity detection in market rules
   - Dispute susceptibility scoring
   - Hard vetoes: >0.35 refuse maker, >0.25 refuse satellite

5. **Markout/Toxicity Tracker** (`strategy/markout_tracker.py`)
   - 30s/2m/10m post-fill markout measurement
   - Toxic market detection (mean markout < -0.2¢)
   - QS override for adverse selection

6. **WebSocket Feeds** (`feeds/`)
   - Market feed: L2 order books, PING keepalive
   - User feed: Order/fill/balance updates (PROD only)

7. **Truth Report** (`reporting/truth_report.py`)
   - Daily JSON output with maker/taker separation
   - Realized spread, markout, fill rate tracking
   - Component attribution (maker, taker, arb, satellite)

## Paper Mode Optimization

**Problem**: Initial burn-in had Active Set = 0 due to QS vetoes.

**Solution**: Mode-aware thresholds and prefiltering for paper mode stability.

### Changes (Paper Mode Only)

#### A) Mode-Aware Staleness Threshold
```python
# config.py
QS_BOOK_STALENESS_S_PROD = 5     # 5s max in production
QS_BOOK_STALENESS_S_PAPER = 120  # 120s max in paper mode (2x cycle time)
```

Dynamically selected based on `PAPER_MODE` environment variable.

#### B) WS Warmup (Cycle 1)
- Waits up to 10s OR until 25 books received before first QS computation
- Processes WS messages during warmup (500ms check interval)
- Logs: `warmup_s=X books_received=Y msg_types={...}`

#### C) Market Prefiltering
- Fetch limit: 200 → 500 markets in paper mode
- Excludes `close_window` markets (time_to_close < 24h)
- Excludes `negRisk` cluster markets
- Ranks by liquidity, subscribes to top 250 markets (500 assets)

#### D) Enhanced QS Veto Logging
- Renamed `veto_spread` → `veto_crossed` (spread < tick_size only)
- Added `stale_threshold_s` to veto summary
- 5 QS_SAMPLE lines per cycle with `primary_veto` field

### Results (Cycle 1)

```
Prefilter: 32/597 markets (excluded: close_window=23, negRisk=542)
Warmup complete: 64 books received in 0.5s
QS veto summary: ok=16 stale=4 crossed=11 rrs=1 state=0 liquidity=0 stale_threshold_s=120
Active set selected: 10 markets across 3 clusters
```

**✅ Acceptance Criteria Met:**
- Active set non-zero: ✅ 10 markets
- QS ok >= 1: ✅ 16 markets passed all gates
- Warmup functional: ✅ 64 books in 0.5s
- Mode-aware staleness: ✅ 120s threshold logged
- Enhanced logging: ✅ 5 samples with primary_veto

### Production Safety

**All paper mode relaxations are strictly gated by `PAPER_MODE=true` flag.**

Production behavior unchanged:
- 5s staleness threshold
- 200 market fetch limit
- No prefiltering (subscribes to all markets)

## Running the System

### Paper Mode (Testing)
```bash
PAPER_MODE=true ./venv/bin/python main_maker.py
```

### Production Mode
```bash
# Requires CLOB credentials in .env
PAPER_MODE=false ./venv/bin/python main_maker.py
```

## Environment Variables

Required in `.env`:
```
PAPER_MODE=true                    # true for paper trading, false for live
POLY_PRIVATE_KEY=<wallet_key>      # Production only
POLY_API_KEY=<api_key>             # Production only
POLY_API_SECRET=<api_secret>       # Production only
POLY_API_PASSPHRASE=<passphrase>   # Production only
TRADING_ADDRESS=<polygon_address>  # Production only
```

## Configuration

Key settings in `config.py`:

### Risk Limits
- `MAX_CLUSTER_EXPOSURE_PCT = 0.12` (12% per cluster)
- `MAX_AGG_EXPOSURE_PCT = 0.40` (40% total)
- `MAX_MARKET_INVENTORY_PCT = 0.01` (1% per market)

### QS Thresholds
- `ACTIVE_QUOTE_COUNT = 40` (markets to actively quote)
- `QS_MIN_LIQUIDITY = 500.0` ($500 minimum)
- `QS_BOOK_STALENESS_S_PROD = 5` (production)
- `QS_BOOK_STALENESS_S_PAPER = 120` (paper mode)

### RRS Gates
- `RRS_VETO_MAKER = 0.35` (no maker quoting above)
- `RRS_VETO_SATELLITE = 0.25` (no satellite trades above)

### Market State Thresholds
- `STATE_WATCH_THRESHOLD_HOURS = 72` (NORMAL → WATCH)
- `STATE_CLOSE_WINDOW_THRESHOLD_HOURS = 24` (WATCH → CLOSE_WINDOW)

## Testing

```bash
# Run all tests
./venv/bin/python -m pytest tests/

# Run specific test file
./venv/bin/python -m pytest tests/test_quoteability_scorer.py -v
```

## Monitoring

### Truth Reports
Daily JSON reports written to `reports/YYYY-MM-DD.json` with:
- Portfolio metrics (Sharpe, Calmar, drawdown)
- Component attribution (maker/taker PnL)
- Maker truth metrics (realized spread, markout, fill rate)
- Health metrics (WS uptime, cancel-all triggers)

### Real-Time Logs
- `logs/paper_burnin.log` - Full burn-in output
- QS veto summaries every cycle
- Feed health metrics
- Active set selection

## Architecture Principles

1. **Maker-First**: 80-90% risk budget in market-making, not prediction edge
2. **Risk Engine as Single Source of Truth**: No strategy bypasses caps
3. **Mode-Aware Thresholds**: Relaxed for testing, strict for production
4. **Observability Over Guessing**: Surgical debug counters, not speculation
5. **Tail Control**: Hard veto gates prevent catastrophic losses

## Credits

Built with guidance from ChatGPT (Anthropic's Claude Sonnet 4.5).

**ChatGPT Review Status**: All 21 critical fixes applied.
- ✅ TTL → GTD/GTC with unix timestamps
- ✅ Fees → feeRateBps per market
- ✅ Time model → 3 clocks + state machine
- ✅ Markout/toxicity module
- ✅ Inventory → token-level + mark-to-mid
- ✅ Tick rounding enforcement
- ✅ Parity arb execution mode + leg risk
- ✅ Gates → maker sample sizes + microstructure KPIs
- ✅ And 13 more...

## License

Private repository - not for distribution.
