# ASTRA LIVE READINESS GATE
**Date: 2026-02-13 | Fail-Closed Checklist**

**Rule**: ALL items must be PASS. Any single FAIL blocks live deployment. No exceptions.

**Prerequisite**: Paper Readiness Gate must be fully PASS before evaluating this gate.

---

## GATE PHILOSOPHY

Going live means real money at risk. This gate exists to prevent premature deployment driven by architectural enthusiasm rather than empirical evidence.

The live gate asks: **Has the system demonstrated, with statistical confidence, that it produces positive risk-adjusted returns in paper mode — AND is the infrastructure robust enough to handle real money without catastrophic failure?**

The bar is intentionally high. False negatives (staying in paper too long) cost opportunity. False positives (going live too early) cost capital. Capital loss is worse.

---

## SECTION 1: PAPER GATE PREREQUISITE

| # | Gate | Current | PASS Criteria |
|---|------|---------|---------------|
| 1.1 | Paper Readiness Gate complete | FAIL | All 5 sections of PAPER_READINESS_GATE.md are PASS |
| 1.2 | Paper phase duration | 0 days | >= 21 days of continuous paper trading |
| 1.3 | No manual interventions during paper | N/A | Zero config changes, zero restarts (except planned), zero trade overrides during paper phase |

**Section 1 Status**: FAIL (paper gate not started)

---

## SECTION 2: STATISTICAL EVIDENCE (from Paper Phase)

These require the paper phase data to meet allocator-grade thresholds with statistical confidence.

| # | Gate | Target | Fail Condition | Evidence |
|---|------|--------|----------------|----------|
| 2.1 | Annualized Sharpe ratio | >= 2.0 | < 2.0 OR CI lower bound < 1.0 | `compute("all_time").sharpe_annualized` with Mertens CI |
| 2.2 | Annualized Sortino ratio | >= 3.0 | < 3.0 | `compute("all_time").sortino_annualized` |
| 2.3 | Win rate | >= 55% | < 55% OR Wilson CI lower < 50% | `compute("all_time").win_rate` |
| 2.4 | Profit factor | >= 1.5 | < 1.5 | `compute("all_time").profit_factor` |
| 2.5 | Brier score | < 0.20 | >= 0.20 on 50+ observations | `compute("all_time").brier_score` |
| 2.6 | Max drawdown (paper) | < 15% | >= 15% | `DrawdownTracker.get_state().max_drawdown_pct` |
| 2.7 | Calmar ratio | >= 2.0 | < 2.0 | `compute("all_time").calmar_ratio` |
| 2.8 | Worst month | > -5% | <= -5% | `DrawdownTracker.get_state().worst_month_pnl_pct` |
| 2.9 | Expectancy per trade | > $1.00 | <= $1.00 | `compute("all_time").expectancy` |
| 2.10 | Resolved trades | >= 300 | < 300 | Total resolved count |
| 2.11 | VaR 95% | < 3% of bankroll per trade | >= 3% | `compute("all_time").var_95` |
| 2.12 | Sharpe CI width | < 1.5 | >= 1.5 (too uncertain) | `sharpe_ci_upper - sharpe_ci_lower` |

**Section 2 Status**: FAIL (no paper data)

**Note on targets**: These are stretch targets derived from the user's allocator-grade goals (Sharpe >= 3, Sortino >= 5, CAGR >= 50%). The live gate uses lower thresholds (Sharpe >= 2.0 not 3.0) because paper mode introduces noise (fill model imprecision, simulation artifacts). Live performance may differ materially from paper.

---

## SECTION 3: INFRASTRUCTURE READINESS

These verify the system can operate safely with real money.

| # | Gate | Current | PASS Criteria | Evidence |
|---|------|---------|---------------|----------|
| 3.1 | py-clob-client initialized | FAIL | `clob_client is not None`, authenticated API call succeeds | P1-1 |
| 3.2 | Order submit works | FAIL | Successfully submit + cancel 1 small order on live CLOB | P1-2 |
| 3.3 | Order reconciliation works | FAIL | On restart, stale orders detected and cancelled | P1-3 |
| 3.4 | User WS feed authenticated | FAIL | Live fill events arrive via WebSocket | P1-4 |
| 3.5 | Realized P&L tracking works | FAIL | Non-zero P&L computed from position tracking | P1-5 |
| 3.6 | NER/semantic clustering deployed | FAIL | Correlated markets share clusters | P1-6 |
| 3.7 | SIGTERM cancels all orders | FAIL | Graceful shutdown cancels all open orders before exit | P1-7 |
| 3.8 | Docker deployment | FAIL | Containerized with restart policy | P3-6 |
| 3.9 | Alerting active | FAIL | Alerts fire on circuit break, disconnect, error spike | P3-7 |
| 3.10 | WS uptime tracking real | FAIL | Non-placeholder uptime values in truth report | P2-6 |

**Section 3 Status**: 0/10 PASS

---

## SECTION 4: RISK CONTROLS VALIDATION

These verify risk controls are operational and have been tested.

| # | Gate | Current | PASS Criteria | Evidence |
|---|------|---------|---------------|----------|
| 4.1 | Per-market cap enforced live | UNTESTED | Submit order exceeding 1% cap, verify rejected | Integration test |
| 4.2 | Per-cluster cap enforced live | UNTESTED | Fill markets to 12% cluster cap, verify next entry rejected | Integration test |
| 4.3 | Aggregate cap enforced live | UNTESTED | Fill to 40% aggregate, verify next entry rejected | Integration test |
| 4.4 | Kill switch works live | UNTESTED | Create `/tmp/astra_kill`, verify cancel-all and halt | Manual test |
| 4.5 | Pause mode works live | UNTESTED | Create `/tmp/astra_pause`, verify no new orders but existing managed | Manual test |
| 4.6 | Rate limiting works live | UNTESTED | Verify no more than 60 mutations/min sent to CLOB | Rate limit counter in logs |
| 4.7 | Reject storm handling works | UNTESTED | 5 consecutive rejects trigger backoff | Integration test |
| 4.8 | Stale feed circuit breaker works | UNTESTED | Simulate feed stale > 5s, verify cancel-all | Integration test |
| 4.9 | Monte Carlo P(ruin) < 0.1% | FAIL | P3-1 not built | Run simulator with observed returns |
| 4.10 | Aggregate drawdown circuit breaker | FAIL | P3-10 not built | 10% DD halts all new entries |

**Section 4 Status**: 0/10 PASS

---

## SECTION 5: LIVE WARM-UP PROTOCOL

Even after all gates pass, live deployment follows a graduated warm-up:

### Phase A: Observation Only (Days 1-3)
- Connect to live CLOB (read-only)
- Compare real book data to what paper simulator assumed
- Log theoretical fills vs what would have actually filled
- **Gate**: Sim fill rate within 2x of theoretical live fill rate

### Phase B: Micro-Orders (Days 4-7)
- Deploy with `BANKROLL = 100` (total $100 at risk)
- Maximum 5 markets at a time
- Maximum $5 per market
- **Gate**: No losses > $10 total. No system errors. Orders submit/cancel correctly.

### Phase C: Small Book (Days 8-14)
- Scale to `BANKROLL = 500`
- Maximum 20 markets
- Maximum $25 per market
- **Gate**: Sharpe per-trade > 0 after 50+ live trades. Fill rate within 2x of paper sim.

### Phase D: Target Book (Day 15+)
- Scale to `BANKROLL = 5000` (or configured target)
- Full market count (40 markets)
- Full risk caps active
- **Gate**: Performance within 50% of paper metrics after 200+ live trades.

### Phase E: Steady State
- Full deployment
- Daily truth reports
- Weekly review of live vs paper divergence
- Monthly model validation (Brier, fill rate, markout)

---

## SECTION 6: LIVE KILL CRITERIA

Automatic halt if ANY of these conditions occur during live trading:

| # | Condition | Action | Recovery |
|---|-----------|--------|----------|
| 6.1 | Total loss > 5% in one day | Cancel all orders, halt new entries | Manual review required |
| 6.2 | Total loss > 10% from peak | Cancel all orders, halt system | Full review + paper re-validation |
| 6.3 | 3+ consecutive failed order submissions | Halt new orders for 10 minutes | Auto-resume after 10 min |
| 6.4 | WS feed down > 5 minutes | Cancel all orders | Auto-resume on reconnect |
| 6.5 | Kill switch activated | Cancel all orders, halt system | Manual restart required |
| 6.6 | Sharpe per-trade drops below 0 over 100+ trades | Reduce exposure by 50% | Review strategy performance |
| 6.7 | Fill rate diverges > 3x from paper sim | Alert + reduce exposure | Re-calibrate fill model |
| 6.8 | Brier > 0.30 over 50+ live predictions | Halt satellite strategy | Re-validate estimation pipeline |

---

## OVERALL LIVE GATE STATUS

```
Section 1 (Paper prerequisite):  FAIL     ← Paper gate not passed
Section 2 (Statistical evidence): FAIL     ← No paper data
Section 3 (Infrastructure):      0/10 PASS ← CLOB integration needed
Section 4 (Risk controls):       0/10 PASS ← Integration testing needed
Section 5 (Warm-up protocol):    NOT STARTED
Section 6 (Kill criteria):       DEFINED ← Ready when deployed

OVERALL: FAIL (BLOCKED ON PAPER GATE)
```

### Critical Path to Live

```
1. Complete P0-2, P0-3 (operational basics)     → Section 1 infrastructure
2. Start paper burn-in (P0-1)                    → 14-30 days
3. In parallel: P1-1 through P1-7 (CLOB work)   → Section 3
4. In parallel: P3-1, P3-6, P3-7 (ops work)     → Section 4
5. Paper gate passes                              → Section 1, 2
6. Integration testing on live CLOB               → Section 4
7. Live warm-up Phase A                           → Section 5
8. Live warm-up Phase B-D                         → Graduated deployment
```

### Estimated Path

Paper burn-in is the bottleneck. CLOB integration and operational work can proceed in parallel. The earliest possible live Phase A start is after the paper gate passes and CLOB integration is complete.

### Risk Acceptance

If an operator chooses to override any gate item, they must:
1. Document which gate was overridden and why
2. Set tighter kill criteria for the overridden area
3. Reduce initial bankroll by 50% as a safety margin
4. Log the override in the config audit trail

**No gate override should be taken lightly. Each gate exists because a persona identified a specific failure mode that the gate prevents.**

---

## APPENDIX: ALLOCATOR-GRADE TARGET RECONCILIATION

| Metric | User Target | Paper Gate | Live Gate | Rationale |
|--------|-------------|------------|-----------|-----------|
| Sharpe | >= 3 | per-trade > 0.05 | annualized >= 2.0 | Paper gate is per-trade (raw edge). Live gate is annualized (full picture). Target 3.0 is aspirational; 2.0 is the minimum for live. |
| Sortino | >= 5 | N/A (not gated in paper) | >= 3.0 | Sortino depends on downside vol, which needs more data. Paper uses Sharpe as proxy. |
| CAGR | >= 50% | N/A | N/A (depends on bankroll + turnover) | CAGR is a function of capital deployed and trade frequency, not directly gatable from per-trade metrics. |
| MDD | <= 15% | < 20% (paper) | < 15% (live) | Paper allows wider MDD because fill model introduces noise. |
| Calmar | >= 3 | N/A | >= 2.0 | Conservative gate; Calmar is volatile with small N. |
| Hit rate | >= 60% | > 52% (paper) | > 55% (live) | Paper minimum is lower because we need to observe sufficient losses for risk metrics. |
| PF | >= 2 | > 1.1 (paper) | > 1.5 (live) | Paper gate is low to avoid false rejection. Live gate tighter. |
| Worst month | >= -5% | > -8% (paper) | > -5% (live) | Paper allows wider because stochastic fill model can create clustered losses. |
| TUW | <= 1-2 months | N/A | Observed during live | Time-under-water requires multi-month observation. |
| CVaR | tight | N/A | VaR 95% < 3% per trade | Formal CVaR gating requires 200+ trades. |
