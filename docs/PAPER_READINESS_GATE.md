# ASTRA PAPER READINESS GATE
**Date: 2026-02-13 | Fail-Closed Checklist**

**Rule**: All items must be PASS to declare paper phase complete. Any FAIL blocks progression to live.

---

## GATE PHILOSOPHY

Paper phase exists to answer one question: **Does this system produce positive risk-adjusted returns when trading simulated orders against real market data?**

The paper phase is NOT about:
- Proving the system can handle live CLOB integration (that's the live gate)
- Producing beautiful metrics (metrics without data are decoration)
- Running for a specific duration (30 days is guidance, not a hard gate)

The paper phase IS about:
- Accumulating enough resolved trades for statistical significance
- Measuring actual estimation accuracy (Brier score)
- Observing at least one drawdown and recovery
- Confirming the stochastic fill model doesn't wildly distort results

---

## SECTION 1: INFRASTRUCTURE READINESS (Pre-Burn-In)

These must be PASS before starting the paper burn-in.

| # | Gate | Current | Evidence | PASS Criteria |
|---|------|---------|----------|---------------|
| 1.1 | Paper trading loop runs without crash for 24h | UNTESTED | Run paper_trader.py for 24h, check exit code | Zero crashes, zero unhandled exceptions |
| 1.2 | SIGTERM handler works | FAIL | P0-2 not implemented | Clean shutdown with truth report written |
| 1.3 | Health check heartbeat works | FAIL | P0-3 not implemented | `/tmp/astra_health` updated every 60s |
| 1.4 | SQLite WAL mode confirmed | PASS | trade_logger.py uses `PRAGMA journal_mode=WAL` | WAL mode active |
| 1.5 | Metrics engine returns NaN (not errors) on empty | PASS | 37/37 tests passing | `compute("all_time")` returns NaN, not exception |
| 1.6 | Stochastic fill model deterministic | PASS | 22/22 tests passing | Same seed produces identical fill sequence |
| 1.7 | Config loaded from .env with override=True | PASS | config.py uses `load_dotenv(override=True)` | Fresh env vars always used |
| 1.8 | Anthropic API key validated | PASS | config.py validates min 20 chars + strip | Invalid key detected on startup |

**Section 1 Status**: 6/8 PASS. Items 1.2, 1.3 need implementation (P0-2, P0-3).

---

## SECTION 2: DATA ACCUMULATION (During Burn-In)

These accumulate during the paper burn-in and must reach threshold before gate clears.

| # | Gate | Current | Threshold | How to Check |
|---|------|---------|-----------|--------------|
| 2.1 | Resolved trades count | 0 | >= 200 | `SELECT COUNT(*) FROM trades WHERE resolution_outcome IS NOT NULL` |
| 2.2 | Category diversity | 0 | >= 3 distinct categories | `SELECT COUNT(DISTINCT category) FROM trades WHERE resolution_outcome IS NOT NULL` |
| 2.3 | Strategy diversity | 0 | >= 2 strategies with 30+ trades | `SELECT source, COUNT(*) FROM trades GROUP BY source HAVING COUNT(*) >= 30` |
| 2.4 | Time span | 0 days | >= 14 days between first and last resolved trade | `SELECT julianday(MAX(exit_time)) - julianday(MIN(exit_time)) FROM trades` |
| 2.5 | Estimation count | 0 | >= 300 estimates logged | `SELECT COUNT(*) FROM estimates` |
| 2.6 | Brier observations | 0 | >= 50 with computed Brier | `SELECT COUNT(*) FROM trades WHERE brier_score IS NOT NULL` |

**Section 2 Status**: 0/6 PASS. Paper burn-in has not started.

---

## SECTION 3: PERFORMANCE METRICS (Post-Burn-In)

These are evaluated after sufficient data accumulation (Section 2 all PASS).

| # | Gate | Target | Fail Condition | Evidence |
|---|------|--------|----------------|----------|
| 3.1 | Sharpe ratio (per-trade) | > 0.05 | <= 0.05 OR gate=FAIL (N<30) | `PerformanceEngine.compute("all_time").sharpe_per_trade` |
| 3.2 | Sharpe CI lower bound | > 0.0 | Lower CI bound <= 0 (includes zero) | `PerformanceEngine.compute("all_time").sharpe_ci_lower` |
| 3.3 | Win rate | > 52% | <= 52% OR gate=FAIL (N<20) | `PerformanceEngine.compute("all_time").win_rate` |
| 3.4 | Win rate CI lower bound | > 50% | Lower CI bound <= 50% | Wilson CI from `compute("all_time")` |
| 3.5 | Profit factor | > 1.1 | <= 1.1 | `PerformanceEngine.compute("all_time").profit_factor` |
| 3.6 | Expectancy (per trade) | > $0.50 | <= $0.50 | `PerformanceEngine.compute("all_time").expectancy` |
| 3.7 | Brier score | < 0.22 | >= 0.22 | `PerformanceEngine.compute("all_time").brier_score` |
| 3.8 | Max drawdown | < 20% | >= 20% of paper bankroll | `DrawdownTracker.get_state().max_drawdown_pct` |
| 3.9 | Worst month | > -8% | <= -8% | `DrawdownTracker.get_state().worst_month_pnl_pct` |
| 3.10 | Paper simulator fill rate | 5%-40% | < 5% (too conservative) or > 40% (too optimistic) | `PaperTradingSimulator.get_stats()["fill_rate"]` |

**Section 3 Status**: CANNOT EVALUATE (no data). All gates return NaN or gate=FAIL.

---

## SECTION 4: SYSTEM INTEGRITY (Post-Burn-In)

These verify the system operated correctly during the burn-in.

| # | Gate | Target | Fail Condition | How to Check |
|---|------|--------|----------------|--------------|
| 4.1 | Zero unhandled exceptions | 0 | Any unhandled exception in logs | `grep -c "Traceback" logs/*.log` |
| 4.2 | Feed uptime > 90% | > 90% | < 90% | Health check heartbeat gaps |
| 4.3 | Memory stable | < 500MB | Growing unbounded | Compare memory at start vs end |
| 4.4 | All trades have P&L | 100% | Any trade with NULL P&L and resolved=true | `SELECT COUNT(*) FROM trades WHERE resolution_outcome IS NOT NULL AND profit_loss IS NULL` |
| 4.5 | No duplicate trades | 0 duplicates | Any duplicate trade_id | `SELECT trade_id, COUNT(*) FROM trades GROUP BY trade_id HAVING COUNT(*) > 1` |
| 4.6 | Learning agent processed all resolutions | 100% | Unprocessed resolutions exist | Compare resolved count vs learning agent update count |
| 4.7 | Metrics engine non-NaN at end | All gates PASS | Any NaN metric with sufficient data | `compute("all_time")` has no NaN where gate=PASS |

**Section 4 Status**: CANNOT EVALUATE (no burn-in data).

---

## SECTION 5: RISK VALIDATION (Post-Burn-In)

These verify risk controls worked correctly during burn-in.

| # | Gate | Target | Fail Condition | How to Check |
|---|------|--------|----------------|--------------|
| 5.1 | No market exposure > 1% bankroll | 0 violations | Any violation logged | `grep "Market cap exceeded" logs/*.log` should be non-zero (vetoes happened) |
| 5.2 | No cluster exposure > 12% bankroll | 0 violations | Any actual breach (veto failed) | Check max cluster exposure in truth reports |
| 5.3 | No aggregate exposure > 40% bankroll | 0 violations | Any actual breach | Check max aggregate exposure in truth reports |
| 5.4 | Near-resolution ratchet fired | > 0 times | Never fired (either no markets approached resolution, or ratchet broken) | `grep "Near-close cap" logs/*.log` |
| 5.5 | RRS veto fired | > 0 times | Never fired (either no risky markets, or veto broken) | QS veto summary shows RRS vetoes > 0 |
| 5.6 | Drawdown tracker consistent | Yes | Tracker state diverges from manual calculation | Compare tracker MDD vs recalculated MDD from trades table |

**Section 5 Status**: CANNOT EVALUATE (no burn-in data).

---

## OVERALL PAPER GATE STATUS

```
Section 1 (Infrastructure):  6/8 PASS    ← Need P0-2, P0-3
Section 2 (Data):             0/6 PASS    ← Need paper burn-in
Section 3 (Performance):      N/A         ← Blocked on Section 2
Section 4 (Integrity):        N/A         ← Blocked on burn-in
Section 5 (Risk):             N/A         ← Blocked on burn-in

OVERALL: FAIL (BLOCKED ON BURN-IN)
```

### To Unblock

1. Implement P0-2 (SIGTERM handler) — ~1h work
2. Implement P0-3 (health check) — ~1h work
3. Start paper burn-in: `python paper_trader.py`
4. Wait for 200+ resolved trades (estimated 14-30 days depending on market resolution cadence)
5. Evaluate Sections 3-5

### Decision Criteria

When all 5 sections are PASS:
- **GREEN**: Proceed to live gate evaluation
- **YELLOW** (3.1-3.10 marginal): Extend burn-in for another 100 trades, re-evaluate
- **RED** (any Section 3 metric FAIL): Investigate root cause before proceeding. Do NOT proceed to live.

### Escalation

If paper gate remains FAIL after 60 days:
- Re-examine estimation pipeline accuracy (is AI producing edge?)
- Re-examine market selection (is QS selecting adversely?)
- Consider: the strategy may not have edge. This is a valid outcome. Knowing has value.
