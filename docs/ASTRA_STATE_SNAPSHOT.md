# ASTRA SYSTEM TRUTH SNAPSHOT
**Date: 2026-02-13 | Post-Metrics-Engine + Post-Stochastic-Fill-Model**

---

## WHAT IS PRESENT

| Layer | LOC | Status |
|-------|-----|--------|
| Strategy (maker, QS, RRS, parity, satellite, longshot, arb) | 1,512 | WORKING |
| Scanner (market fetch, probability estimation, clustering, learning) | 4,275 | WORKING |
| Execution (CLOB executor, paper simulator, order state store) | 2,000+ | PARTIAL (CLOB client = None) |
| Risk (portfolio engine, state machine, caps enforcement) | 280 | WORKING |
| Data sources (crypto, sports, weather, whale, calendar, signals) | 8 modules | WORKING |
| Metrics engine (Sharpe/Sortino/Calmar/PF/VaR/CVaR/Brier/CI) | 400 | WORKING |
| Drawdown tracker (real-time, monthly grouping, time-under-water) | 250 | WORKING |
| Confidence intervals (Mertens Sharpe CI, Wilson win-rate CI, gates) | 200 | WORKING |
| Paper simulator (stochastic fills, adverse selection, partial fills) | 530 | WORKING |
| Reporting (truth report, daily JSON, maker/taker separation) | 500 | WORKING |
| Learning agent (Brier, calibration buckets, evolution, overrides) | 450 | WORKING |
| Trade logger (SQLite, WAL, 3 tables, indexed) | 200 | WORKING |
| Tests (13 files, ~4,000 LOC, all green) | 4,000 | WORKING |

## WHAT IS WORKING

1. **Paper trading loop** (`paper_trader.py`): Continuous scan → estimate → size → simulate → resolve → learn. Delegates metrics to centralized engine.
2. **Market-maker runtime** (`main_maker.py`): Full cycle with QS scoring, order generation, paper fill simulation, truth reporting. Kill switch + pause mode.
3. **Centralized metrics**: Annualized Sharpe/Sortino/Calmar with correct `sqrt(trades/year)` scaling. Sample-size gates prevent premature conclusions. CI bands on Sharpe and win rate.
4. **Stochastic fill model**: Replaces old deterministic fills. Bernoulli draw from `base_rate * time_factor * spread_factor * queue_factor * activity_factor`. Capped at 70%. Adverse selection biased against maker in tight markets.
5. **Risk caps**: Per-market (1%), per-cluster (12%), aggregate (40%) enforced in portfolio engine. Near-resolution halving (<48h). negRisk single-cluster.
6. **21 critical fixes**: All implemented, tested, verified.

## WHAT IS FAILING

1. **CLOB client**: `clob_client = None` — cannot submit/cancel real orders.
2. **Order reconciliation**: `reconcile_with_clob()` has `clob_orders = []` stub.
3. **User feed API key**: Hardcoded `None` — live user WS feed cannot authenticate.
4. **Realized P&L in maker mode**: `TODO: Compute realized P&L from position tracking` in main_maker.py.

## WHAT CANNOT BE TRUSTED YET

| Metric / Claim | Why Untrusted | Evidence Required |
|----------------|---------------|-------------------|
| **Paper Sharpe / Sortino** | Zero resolved trades in paper mode so far; metrics engine returns NaN / gate=FAIL | ≥30 resolved trades with known outcome |
| **Fill realism calibration** | Stochastic model parameters (base_rate=0.15, half_life=45s, adverse=30bps) are theoretical, not fitted to Polymarket data | Compare sim fill rate vs live fill rate on 1000+ orders |
| **Brier score** | Learning agent has calibration infrastructure but no resolved predictions yet | ≥50 resolved predictions with Brier < 0.20 |
| **Maker edge** | Markout tracker exists but no fills to analyze | ≥200 maker fills with markout_30s mean > 0 |
| **Cluster correlation** | Portfolio engine uses hash-based clustering, not NER/semantic | Correlation matrix from ≥100 trades across ≥8 clusters |
| **Adverse selection model accuracy** | Parameters are educated guesses, not fitted | Compare sim adverse_move distribution vs live markout distribution |
| **Strategy attribution** | No live trades to attribute — all metrics are prospective | ≥100 resolved trades per strategy with positive expectancy |

## EXPLICIT UNKNOWNS BLOCKING JUDGMENT

1. **No resolved paper trades**: The metrics engine reports NaN for all risk-adjusted metrics because there are zero resolved positions. Cannot assess any target.
2. **Fill model calibration gap**: The stochastic fill model uses config-driven parameters, not empirically fitted from Polymarket data. Real fill rates may differ materially.
3. **Estimation accuracy unknown**: The adversarial AI estimation pipeline has not produced enough resolved predictions to compute Brier scores or measure actual edge.
4. **Market regime sensitivity**: No data on how Astra performs across different regimes (bull, bear, low-vol, high-vol, event-driven). All claims are architectural, not empirical.
5. **Tail risk untested**: VaR/CVaR formulas are implemented but cannot produce meaningful values without trade history.
6. **Correlation structure unknown**: Current cluster assignment is hash-based. Real market correlations (e.g., multiple Trump markets moving together) are not measured.

## MEASUREMENT ILLUSIONS TO GUARD AGAINST

1. **Paper fill optimism**: Even with the stochastic model, sim fills may be easier to obtain than real fills. The `PAPER_SIM_BASE_FILL_RATE=0.15` is a guess.
2. **Survivorship bias in market selection**: QS scorer selects the "best" markets, but these may be the ones where adverse selection is highest.
3. **Selection bias in estimation**: Markets where Astra is most confident may be the ones where the market is also confident (no edge, just agreement).
4. **Overfitting calibration**: The learning agent's evolution loop may overfit to recent resolved trades rather than identifying genuine systematic errors.
5. **Annualization amplification**: With `sqrt(trades/year)` scaling, a per-trade Sharpe of 0.15 looks like 2.9 if you trade once per day. This is mathematically correct but assumes IID returns, which binary markets may not have.
