# ASTRA SYSTEM SCORECARD
**Date: 2026-02-13 | Evidence-Based Rating Framework**

---

## SCORING METHOD

Each dimension rated **1-10** based on codebase evidence, not aspirations.

| Score | Meaning |
|-------|---------|
| 1-2 | Absent or fundamentally broken |
| 3-4 | Stubbed or deeply incomplete |
| 5-6 | Functional but significant gaps |
| 7-8 | Strong with known limitations |
| 9-10 | Institutional/allocator-grade |

---

## 1. ESTIMATION QUALITY — 8/10

*Does the system produce probability estimates that could generate edge?*

### Strong

- **Adversarial PRO/CON pipeline**: Two simultaneous Claude agents argue opposite sides, then a Synthesizer Judge aggregates. This is structurally superior to single-model estimation and mirrors the "Polyseer" multi-agent research design. The architecture forces genuine debate rather than anchoring to a prior.
- **Evidence tier weighting**: 4-tier system (A=2.0, B=1.6, C=0.8, D=0.3) with examples per tier. Correlation collapse rule prevents double-counting when multiple sources cite the same underlying fact.
- **Research-backed debiasing**: Acquiescence bias correction (arXiv:2402.19379) addresses the known LLM tendency to overestimate YES outcomes. Extremizing aggregation (arXiv:1406.2148) sharpens independent estimates. Market timing filter (arXiv:2510.17638) vetoes sub-48h markets.
- **Multi-module pipeline**: Classifier → data enrichment → PRO/CON adversarial → synthesis → validation loop. Markets that fail validation get a second pass.
- **Astra V2 Laws**: Hard no-trade rules enforced at the prompt level. "No trade is success when edge isn't strong after fees/slippage" — correct epistemic stance.

### Weak

- **Zero empirical validation**: Not one Brier score has been computed from a real resolved prediction. The pipeline looks right but has never been measured.
- **No backtesting harness**: Cannot run the estimation pipeline on historical markets to compute out-of-sample accuracy.
- **Prompt brittleness**: Prompts are embedded as Python string constants. A typo or wording change silently alters the model. No version control on prompts.

### Opportunities

- Run the estimator against 100+ resolved historical Polymarket markets. This would produce the first empirical Brier score and immediately reveal whether the pipeline generates edge.
- A/B test prompt variants. The adversarial structure makes this natural: swap one agent for a different prompt and measure whether Brier improves.

### Gaps

- No automated prompt versioning or regression testing.
- Sports hallucination guard (rejects >5x market price on longshots) is a patch, not a principled solution.
- Stake-based confidence filter uses a hardcoded 30k threshold.

### Potential

If Brier < 0.20 on 100+ historical markets, this pipeline would be genuinely competitive with human superforecasters. The adversarial structure is the kind of thing research papers write about but few trading systems actually implement.

---

## 2. EXECUTION REALISM — 7/10

*Does the paper trading simulator produce results that would survive contact with a real exchange?*

### Strong

- **Stochastic fill model**: Bernoulli draw from `base_rate × time_factor × spread_factor × queue_factor × activity_factor`, capped at 70%. This is fundamentally more honest than the old deterministic 4-bucket model (which gave 100% fill rate after 30 seconds).
- **Adverse selection modeled**: 30bps base move against maker on fill, 2x multiplier for tight spreads, Gaussian noise. This is the single most important realism feature — most paper trading systems ignore adverse selection entirely.
- **Partial fills**: 30% probability, uniform draw from [20%, 100%] of remaining size.
- **CLOB executor firewall**: Validation (postOnly + GTC/GTD, tick rounding before clamping, price bounds), rate limiting (60/min, 30s per-market), reject storm handling (5 rejects → exponential backoff). This is production-grade even though the actual submission is stubbed.
- **Maker/taker fee differentiation**: 0bps maker vs 200bps taker. Correctly penalizes crossing the spread.

### Weak

- **CLOB client = None**: The entire live execution path is a stub. Four TODO comments block real order submission, cancellation, reconciliation, and user feed authentication. This is the single largest gap in the system.
- **No latency simulation**: Paper fills are evaluated instantly. Real exchanges have 50-200ms roundtrip. Latency creates adverse selection that the paper sim doesn't model.
- **Fill model parameters are guesses**: `base_rate=0.15`, `half_life=45s`, `adverse=30bps` are educated estimates, not fitted to Polymarket order book data.
- **Queue model is simplified**: Exponential decay from best price. Real CLOB queues are FIFO with complex cancellation dynamics.

### Opportunities

- Place 1,000 small live orders and compare actual vs simulated fill rates. This would immediately calibrate the model.
- Add latency simulation (50-200ms delay before fill evaluation) to paper mode.

### Gaps

- No market impact model. System assumes its orders don't move the book. At scale, this breaks.
- Realized P&L in maker mode is hardcoded to 0.0.
- Order reconciliation on restart returns empty list.

### Potential

The stochastic model framework is correct — it just needs calibration against real data. Once `base_rate` and `adverse_move` are fitted from live observations, this simulator would produce honest paper results that transfer to live.

---

## 3. RISK MANAGEMENT — 9/10

*Can this system prevent catastrophic loss? How many layers of protection exist?*

### Strong

- **4-layer position caps**: Per-market (1%), per-cluster (12%), aggregate (40%), satellite budget (15%). All enforced in a single function (`can_enter_position()`) that cannot be bypassed by strategy modules.
- **State machine gates**: 7 market states (NORMAL → WATCH → CLOSE_WINDOW → POST_CLOSE → PROPOSED → CHALLENGE_WINDOW → RESOLVED). Zero-tolerance for quoting in dangerous states.
- **Near-resolution ratchet**: Caps halved when `time_to_close < 48h`. This prevents the common failure of holding full size into a binary resolution event.
- **Balance reservations**: USDC and token reservations tracked per-order, released on fill/cancel, adjusted on partial fill. Prevents double-spending.
- **negRisk event handling**: All markets in a negRisk event assigned to a single cluster. Parity arbitrage disabled for negRisk. This prevents the specific Polymarket failure mode where correlated negRisk markets breach cluster limits.
- **Kill switch + pause mode**: File-based emergency controls. `/tmp/astra_kill` halts everything. `/tmp/astra_pause` stops new orders but continues managing existing positions.
- **Daily loss limit**: Paper trader halts new entries when daily P&L drops below -5% of bankroll.

### Weak

- **Hash-based clustering**: `md5(category|resolution_source)[:8]` is the cluster assignment. 10 Trump-related markets could land in 10 different clusters if they have different resolution sources. The aggregate cap (40%) is the only real protection against correlated drawdowns.
- **Static caps**: No dynamic adjustment based on realized volatility, regime, or portfolio VaR. Position sizes are the same in a calm market and a crisis.

### Opportunities

- NER/keyword-based clustering would catch correlated markets (same entity = same cluster). This is the single highest-impact risk improvement.
- Dynamic VaR-based scaling: `position_cap = base_cap × (target_vol / realized_vol)` would automatically shrink exposure when volatility spikes.
- Aggregate drawdown circuit breaker: halt all new entries when equity drops >10% from peak.

### Gaps

- No stress testing framework. Cannot simulate correlated adverse scenarios.
- No Monte Carlo ruin analysis.
- VaR/CVaR formulas exist in the metrics engine but produce NaN (no data).

### Potential

With NER clustering and dynamic VaR scaling, this risk framework would be genuinely institutional. The architecture is already correct — the caps are hard, layered, and non-bypassable. The gaps are about sophistication, not structural soundness.

---

## 4. STRATEGY DEPTH — 7/10

*How many independent edges does the system pursue? Are they theoretically sound?*

### Strong

- **7 strategy types**: Market-making (spread capture), QS-driven selection (best market identification), RRS-gated avoidance (resolution risk), parity arbitrage (YES+NO mispricing), satellite (AI directional), longshot screening (bias exploitation), and cross-market arbitrage.
- **Market maker is theoretically correct**: Fair value band with churn/jump/time-aware widening. Inventory skew correctly implemented (long → lower both bid/ask). Tick rounding before clamping. Quote sizing reduces with inventory.
- **QS scoring is multi-factor**: Spread, liquidity, activity, RRS, state, book staleness — each with hard vetoes. Active set capped at 40 markets with max 5 per cluster. This prevents concentration.
- **Toxicity override**: Markout tracker can veto a market that scores well on QS but bleeds on fills. This is the key defense against "looks great in paper, dies live."
- **Kelly sizing is conservative**: Half-Kelly with confidence weighting. Limits the damage from estimation errors.

### Weak

- **Zero empirical edge evidence**: No strategy has produced a single resolved trade. All assessments are theoretical.
- **No strategy comparison**: Cannot measure which strategy generates the most edge because no data exists.
- **Selection bias risk**: QS selects active markets with tight spreads — exactly the markets where informed flow (adverse selection) is highest.
- **No regime conditioning**: Same parameters in all market conditions. The system doesn't adapt to high-vol vs low-vol environments.

### Opportunities

- Per-strategy attribution after paper burn-in. The SQLite schema already has a `source` field — it just needs trades to populate.
- Regime tagging (vol bucket, category, event type) on every trade. Then compute Sharpe by regime to identify where edge concentrates.
- Adverse selection dashboard: plot markout by market, spread bucket, and time-of-day.

### Gaps

- No options/conditional market strategy.
- No cross-market correlation exploitation (e.g., arbitraging slow-adjusting markets after a related market moves).
- Strategy interaction effects untested (maker quotes + satellite directional in same market could conflict).

### Potential

7 strategies is good breadth. If even 2-3 produce positive per-trade Sharpe, the system has a diversified edge. The parity arbitrage and satellite strategies are the most likely to produce immediate measurable edge.

---

## 5. DATA INFRASTRUCTURE — 8/10

*Is data honest, auditable, and traceable? Can you reconstruct any number?*

### Strong

- **SQLite audit trail**: 3 tables (market_snapshots, estimates, trades) with WAL mode, indexed timestamps, and insert-on-conflict idempotency. Every estimate and trade is logged.
- **Truth report**: Daily JSON with maker/taker separation, markout tracking, fill rate diagnostics, reconciliation error counting, stale quote incidents. Atomic writes via temp file + `os.replace()`.
- **Metrics honesty**: Sample-size gates prevent premature claims. NaN reported when insufficient data. Confidence intervals on Sharpe (Mertens 2002) and win rate (Wilson score). `to_dict()` maps NaN and inf to None for JSON safety. This level of metrics honesty is rare.
- **Estimation audit trail**: Every AI estimate logged with probability, confidence, robustness score, edge, EV, Kelly size, truth state, and no-trade reason.
- **Simulator diagnostics**: `FillOutcome` captures all intermediate factors (time, spread, queue, activity, adverse move). Full traceability for fill model debugging.

### Weak

- **No schema versioning**: Schema embedded in code. If table structure changes, old data becomes unreadable without manual migration.
- **WS uptime placeholders**: Truth report claims 99.9% uptime but uses hardcoded `86000.0` seconds.
- **No config audit trail**: Changing `PAPER_SIM_BASE_FILL_RATE` from 0.15 to 0.30 silently changes all subsequent results with no record of the change.
- **No data retention policy**: SQLite grows unbounded.

### Opportunities

- Config snapshot table: log all parameters as JSON on every startup. Enables "what changed?" analysis.
- Schema versioning via `PRAGMA user_version` + migration runner.
- Closed-loop Brier pipeline: automatically match estimates to resolution outcomes.

### Gaps

- No real-time dashboard (only JSON files + SQLite for post-hoc analysis).
- No data integrity checks on startup (corrupt DB = silent failures).
- No backup mechanism.

### Potential

The data infrastructure is already honest — which is the hard part. Adding schema versioning, config snapshots, and a closed-loop Brier pipeline would make it fully auditable.

---

## 6. TESTING & VALIDATION — 6/10

*How confident can you be that the code does what it claims?*

### Strong

- **166 tests across 12 files**: Covers execution primitives, risk controls, strategy logic, paper simulator, metrics engine, truth reporting, and integration.
- **Metrics engine well-tested**: 37 tests covering Sharpe, Sortino, Calmar, Brier, VaR, CVaR, profit factor, drawdown, confidence intervals, sample gates, and JSON serialization edge cases.
- **Paper simulator well-tested**: 22 tests covering determinism, sensitivity, sanity bounds, partial fills, adverse selection, fees, fill optimism reduction, and diagnostics.
- **Risk layer well-tested**: 15 tests covering cluster caps, aggregate caps, negRisk, reservations, partial fills, and near-resolution ratchet.
- **All 166 tests green**: Zero failures as of latest run.

### Weak

- **Zero tests for estimation pipeline**: `probability_estimator.py` (1400+ lines, the most complex module) has no tests. The adversarial PRO/CON pipeline, evidence tier weighting, Bayesian aggregation, validation loop — all untested.
- **Zero tests for learning agent**: `learning_agent.py` (450 lines) has no tests. Calibration tracking, evolution cycle, override generation — all untested.
- **No integration test for full paper trading cycle**: No test that runs scan → estimate → size → simulate → resolve → learn as a single chain.
- **No stress tests**: No test for WS reconnection, reject storms under load, or concurrent order handling.
- **No backtesting framework**: Cannot validate strategies against historical data.

### Opportunities

- Mock-based tests for estimation pipeline (mock Claude API responses, verify parsing and aggregation logic).
- End-to-end paper trading cycle test with mock markets.
- Property-based testing for risk caps (fuzz with random positions, verify caps never breached).

### Gaps

- The most critical and complex module (estimation) has zero tests.
- No mutation testing to verify test quality.
- No continuous integration pipeline.

### Potential

The test infrastructure (pytest, fixtures, assertions) is solid. The gap is coverage of the AI-dependent modules. Adding 20-30 tests for the estimation pipeline and learning agent would push this to 8/10.

---

## 7. CODE ARCHITECTURE & QUALITY — 8/10

*Is the codebase maintainable, well-structured, and professional?*

### Strong

- **Clean separation of concerns**: scanner/ → strategy/ → execution/ → risk/ → metrics/ → reporting/. Each layer has a clear responsibility and interacts through defined interfaces.
- **Firewall pattern**: CLOBExecutor centralizes all validation, rate limiting, and error handling. Strategy modules cannot submit orders directly.
- **Builder pattern**: TruthReportBuilder accumulates data throughout the day and produces atomic reports.
- **Research citations**: arXiv papers cited in code comments (acquiescence bias, extremizing aggregation, debate overconfidence). Unusual for a trading system and valuable for review.
- **Consistent logging**: Module-level loggers (`logging.getLogger(__name__)`), appropriate log levels, contextual messages.
- **Comprehensive docstrings**: Most functions have Args/Returns documentation. Critical invariants marked with `CRITICAL FIX #N` comments.
- **Dataclass discipline**: `TradeRecord`, `MetricsSnapshot`, `FillOutcome`, `DrawdownState`, `Fill`, `StoredOrder` — well-typed data containers.

### Weak

- **Inconsistent type hints**: Some functions fully typed, others partially. No `mypy` or `pyright` enforcement.
- **Config.py is 407 lines**: Flat namespace with 100+ constants. Could benefit from structured groups (risk config, execution config, estimation config).
- **Some magic numbers**: Stake threshold 30k, tick size 0.01 assumed in places, 86000.0 uptime placeholder.

### Opportunities

- Type checking with mypy/pyright would catch a class of bugs at zero runtime cost.
- Config dataclasses (RiskConfig, ExecutionConfig, etc.) would make parameter groups explicit.
- Pre-commit hooks for linting, formatting, type checking.

### Gaps

- No CI/CD pipeline.
- No code coverage measurement.
- No linting configuration (ruff, flake8).

### Potential

The architecture is already at the level where a new developer could understand the system quickly. The layered design means changes in one module are unlikely to break others.

---

## 8. OBSERVABILITY & OPERATIONS — 5/10

*Can you tell what the system is doing right now? Will you know when it breaks?*

### Strong

- **Circuit breakers**: Stale feed detection (5s), feed disconnect handler, reject storm exponential backoff. These correctly halt the system when inputs are unreliable.
- **Truth report**: Daily JSON capturing fills, quotes, health, markouts, fill rate. This is the post-hoc audit tool.
- **SQLite persistence**: All trades, estimates, and market snapshots persist across restarts.
- **Veto counters**: QS tracks why each market was rejected (RRS, state, staleness, liquidity, crossed book). Good for debugging strategy behavior.
- **Simulator diagnostics**: Full factor breakdown per fill decision (time, spread, queue, activity, adverse).

### Weak

- **No health check endpoint**: No HTTP server, no `/health`, no heartbeat file. A stuck process looks identical to a running one from outside.
- **No alerting**: Circuit breakers fire, kill switch activates, error rates spike — and nobody is notified. Silent failures are the most dangerous kind.
- **No process supervision**: No systemd, no Docker, no restart policy. Process crash = system stops.
- **No graceful shutdown**: SIGTERM kills the process. Open orders remain on CLOB (when live). No final truth report written.
- **No metrics export**: No Prometheus, no Grafana, no StatsD. Cannot build real-time dashboards.
- **WS uptime is fake**: Hardcoded placeholder values in truth report.

### Opportunities

- Heartbeat file + Slack webhook would provide basic monitoring at minimal effort.
- Docker + restart policy would prevent crash-related outages.
- SIGTERM handler would prevent orphaned orders on shutdown.
- Prometheus exporter for real-time dashboards (fill rate, exposure, markout, error rate).

### Gaps

- No log rotation for long-running processes.
- No memory/CPU monitoring.
- No WS reconnection integration test.
- No deployment automation.

### Potential

This is the lowest-hanging-fruit improvement area. A heartbeat file (10 lines of code), a SIGTERM handler (20 lines), and a Docker compose file (30 lines) would push this from 5/10 to 7/10 in a day's work.

---

## 9. LEARNING & ADAPTATION — 7/10

*Does the system improve from experience? How fast can it correct mistakes?*

### Strong

- **Calibration buckets**: Learning agent tracks predicted vs actual resolution rate by probability range (0.0-0.1, 0.1-0.2, etc.). Computes bias and overconfidence index per bucket.
- **Evolution cycle**: After N resolved predictions, calls Claude to analyze performance, identify systematic errors, and propose rule updates. Extracts machine-readable parameter overrides (MISPRICING_THRESHOLD, MIN_CONFIDENCE, ACQUIESCENCE_CORRECTION, EXTREMIZE_K).
- **Few-shot learning**: Successful predictions injected into estimation prompts as examples. The system teaches itself by showing its own best work.
- **Anti-hallucination rule**: "If you can't measure it from the data, label it Unmeasured and propose a measurement method." Applied in the evolution prompt.
- **Override persistence**: Strategy overrides written to `memory/strategy_overrides.json`, read by config.py at startup. Parameter changes persist across restarts.
- **30-day expiry**: Overrides auto-expire, forcing the system to re-earn its parameter changes. Prevents stale calibration from persisting.

### Weak

- **No holdout validation**: All resolved outcomes feed the same calibration. No 20% holdout to detect overfitting.
- **No A/B testing**: Cannot compare prompt variants or parameter settings on the same markets.
- **Override mechanism is coarse**: 4 parameters (mispricing threshold, min confidence, acquiescence correction, extremize K) is a narrow control surface. Cannot adjust per-category or per-strategy.
- **Evolution cycle depends on Claude**: If Claude produces a bad analysis, the system adopts bad parameters. No sanity check on proposed overrides.

### Opportunities

- Holdout validation (20% of resolutions withheld) would immediately detect overfitting.
- Bounded overrides: reject any proposed parameter change that exceeds ±30% of default.
- Per-category calibration: separate overrides for crypto, sports, politics, etc.
- Override A/B: run 50% of estimates with old params, 50% with new, compare.

### Gaps

- No meta-learning (which prompt structures work best? which categories are most predictable?).
- No automated rollback if override degrades performance.
- Evolution cycle has run zero times (no resolved predictions yet).

### Potential

The learning architecture is sound. The feedback loop (predict → resolve → calibrate → update → predict) is the right structure. Once it has 50+ resolved predictions to learn from, this becomes the system's most valuable differentiator.

---

## 10. MARKET MICROSTRUCTURE — 8/10

*Does the system understand how markets actually work at the order-book level?*

### Strong

- **Multi-interval markout**: Tracks price drift at 30s, 2m, and 10m after each fill. This is the gold-standard measure of adverse selection in market-making.
- **Toxicity detection**: Market-level (mean markout < -20bps over 20 samples) and cluster-level (mean markout < -15bps over 30 samples). Toxic markets get their QS overridden to zero — they cannot be quoted regardless of spread or liquidity.
- **Adaptive toxicity response**: Before full veto, mildly toxic markets get wider fair value bands (1.5x) and reduced quote sizes (0.5x). Graduated response, not binary.
- **Inventory skew correctly implemented**: Long position → lower both bid and ask (encourages selling, discourages buying). Short → reverse. The sign was initially wrong and was caught and fixed.
- **Fair value band dynamics**: Base width + churn multiplier (high churn → wider) + jump risk multiplier (recent large moves → wider) + near-close multiplier (approaching resolution → wider). This is textbook adaptive quoting.
- **Tick rounding before clamping**: Rounds to valid tick sizes before applying [0.01, 0.99] bounds. This prevents the subtle bug where rounding after clamping produces invalid prices.
- **Favorite-longshot bias**: Mispricing detector accounts for the well-documented bias where longshots are overpriced and favorites are underpriced.
- **Maker/taker classification hierarchy**: POST_ONLY > SPREAD_CROSS > WS_FLAG. Multiple signals used to correctly classify fills.

### Weak

- **No market impact model**: System assumes its orders don't move the book. At scale ($500+ per market), its own quotes affect the prices it quotes against.
- **Queue model is simplified**: Exponential decay from best price. Real CLOB queues are FIFO with time priority, cancellation dynamics, and size priority at each level.
- **No intraday spread modeling**: System doesn't track how spreads vary by time of day, event proximity, or volatility regime.

### Opportunities

- Track empirical spread patterns by time-of-day and day-of-week. Quote tighter when spreads are historically wide (more edge per trade).
- Market impact estimation after live deployment: measure how book depth changes after placing a large order.
- Cross-market microstructure: detect when a related market moves and the current market hasn't repriced yet (latency arbitrage within Polymarket).

### Gaps

- No cross-market correlation at the book level.
- No queue position estimation from order book depth.
- Markout tracker exists but has zero observations.

### Potential

The markout/toxicity framework is the kind of infrastructure that separates serious market-makers from hobbyists. Once it has 200+ fills to analyze, it will automatically identify and avoid the markets where adverse selection is killing the book.

---

## 11. LIVE DEPLOYMENT READINESS — 3/10

*How close is this system to placing real orders with real money?*

### Strong

- **CLOB executor firewall is complete**: Validation, rate limiting, batch slicing, reject storm handling — all implemented and tested. The wall around the exchange interface is solid.
- **Config-driven mode switching**: `PAPER_MODE` toggles between paper simulator and live execution paths.
- **Kill switch and pause mode**: File-based emergency controls exist and are checked every cycle.

### Weak

- **`clob_client = None`**: Cannot submit or cancel real orders. This is the single largest gap.
- **Order reconciliation stub**: `reconcile_with_clob()` returns empty list. Crash recovery will orphan orders on the exchange.
- **User WS feed unauthenticated**: `api_key=None` means no live fill events, no balance updates.
- **Realized P&L = 0.0**: Hardcoded TODO. Truth report has no idea how much money is being made or lost.
- **No graceful shutdown**: SIGTERM kills the process, potentially leaving open orders.
- **No Docker/systemd**: Manual Python execution only.
- **No monitoring**: Process could crash silently.

### Opportunities

- py-clob-client integration is well-scoped. The executor interface is designed for it — just replace stubs with real calls.
- Testnet deployment possible before real money.

### Gaps

- 4 critical TODO stubs block all live trading.
- Token balance query not implemented.
- No wallet preflight validation with real balances.

### Potential

The architecture is 85% complete for live trading. The remaining 15% is integration work (py-clob-client, user feed auth, reconciliation, P&L tracking). This is days of focused work, not weeks, but it's the difference between a simulation and a trading system.

---

## 12. COMPETITIVE MOAT & UNIQUENESS — 7/10

*What does this system do that off-the-shelf tools don't?*

### Strong

- **Adversarial multi-agent estimation**: PRO/CON/Synthesizer architecture with evidence tier weighting and Bayesian aggregation. This is not available in any standard trading framework.
- **Polymarket-specific risk**: negRisk event handling, resolution risk scoring, market state machine, parity arb disabling — all tailored to Polymarket's unique mechanics.
- **Self-improving calibration**: Learning agent with few-shot injection and machine-readable overrides. The system literally teaches itself from its own resolved predictions.
- **Markout-driven toxicity**: Automatic detection and avoidance of adversely-selected markets. Most retail market-making bots have no concept of markout.
- **Metrics honesty**: Sample-size gates, confidence intervals, NaN-on-insufficient-data. This is anti-marketing — the system refuses to report metrics it can't justify statistically.

### Weak

- **Not battle-tested**: No live trades, no resolved paper trades. Uniqueness without evidence is theory.
- **Single-exchange**: Polymarket only. Cannot cross-exchange arbitrage or hedge.
- **Claude dependency**: Estimation pipeline requires Anthropic API. Model changes, rate limits, or pricing changes are external risks.

### Opportunities

- Multi-exchange expansion: same estimation pipeline could price Kalshi, Metaculus, or PredictIt markets for cross-exchange arbitrage.
- Model fallback: add a local model (e.g., quantized LLM) as a backup estimator when API is unavailable.

### Gaps

- No intellectual property protection (open codebase).
- No competitive benchmarking against other Polymarket bots.

### Potential

The adversarial estimation pipeline + markout toxicity system + self-improving learning agent is a combination that would be difficult to replicate. If the estimation pipeline produces Brier < 0.20, this system has a genuine structural advantage.

---

## SUMMARY SCORECARD

| # | Dimension | Score | One-Line |
|---|-----------|-------|----------|
| 1 | Estimation Quality | **8** | Adversarial multi-agent pipeline with research-backed debiasing — needs empirical validation |
| 2 | Execution Realism | **7** | Stochastic fill model with adverse selection — needs calibration against real data |
| 3 | Risk Management | **9** | 4-layer hard caps with state machine and balance reservations — needs semantic clustering |
| 4 | Strategy Depth | **7** | 7 strategies with sound theory — zero empirical evidence of edge |
| 5 | Data Infrastructure | **8** | Honest metrics with sample gates and CI — needs schema versioning and config audit |
| 6 | Testing & Validation | **6** | 166 tests across execution/risk/metrics — zero tests for estimation pipeline |
| 7 | Code Architecture | **8** | Clean layered design with firewall pattern — needs type checking and CI/CD |
| 8 | Observability & Ops | **5** | Circuit breakers and truth reports — no health check, no alerting, no Docker |
| 9 | Learning & Adaptation | **7** | Calibration tracking with evolution cycle — no holdout validation, never run |
| 10 | Market Microstructure | **8** | Multi-interval markout with toxicity veto — no market impact model |
| 11 | Live Deployment | **3** | Executor firewall complete — clob_client=None, 4 critical stubs |
| 12 | Competitive Moat | **7** | Unique adversarial estimation + markout toxicity — unproven |

---

## COMPOSITE SCORE: 6.9 / 10

### Score Distribution

```
9  ██████████████████████████████  Risk Management
8  ████████████████████████████    Estimation, Data, Architecture, Microstructure
7  ██████████████████████████      Execution, Strategy, Learning, Moat
6  ████████████████████████        Testing
5  ██████████████████████          Observability
3  ██████████████████              Live Deployment
```

### What Moves the Needle Most

| Action | Score Impact | Effort |
|--------|-------------|--------|
| **Start paper burn-in (200+ resolved trades)** | +1.0 to composite (validates 5 dimensions at once) | Low (start the system, wait) |
| **CLOB integration (py-clob-client)** | Live Deployment 3→7 | Medium (days) |
| **Health check + SIGTERM + Docker** | Observability 5→7 | Low (hours) |
| **Estimation pipeline tests** | Testing 6→8 | Medium (days) |
| **NER-based clustering** | Risk 9→10 | Medium (days) |
| **Historical backtesting** | Estimation 8→9, Strategy 7→8 | High (week+) |

### The Honest Assessment

Astra is an **architecturally excellent system with zero empirical evidence**. It scores 8-9 in design dimensions (risk, estimation, data, microstructure) and 3-6 in operational dimensions (live deployment, observability, testing). The gap between design quality and operational readiness is the defining characteristic of the system right now.

The single most important thing is to start the paper burn-in. Every dimension that scores below 8 is blocked by the same root cause: no resolved trades. Architecture without data is decoration.
