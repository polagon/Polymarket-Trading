# 10 Polymarket Opportunities + Astra Capability Scoring
## February 14, 2026 — Post Loop 3 (fill realism + startup safety)

**Scoring Legend:**
- **Data Edge** (0-5): Does Astra have Tier A/B data sources for this category?
- **Execution Edge** (0-5): Can Astra's fill simulator (latency + impact + partial) handle it?
- **Risk Containable** (0-5): Does the gate engine + portfolio caps manage downside?
- **Calibration History** (0-5): Has the learning agent seen enough of this category?
- **Composite Score** = weighted avg (Data 30%, Execution 25%, Risk 25%, Calibration 20%)

---

## Opportunity 1: Bitcoin Monthly Price Markets
**Market:** "Will BTC hit $85K in February?" / "$100K by March?"
**Category:** Crypto | **Volume:** $3M+ monthly | **Current:** 71% ($85K), 10% ($100K)

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **5** | Tier 1 algorithmic: CoinGecko price + lognormal distribution model, real-time |
| Execution Edge | **4** | High liquidity → tight spreads, but 200ms arb windows too fast for paper sim |
| Risk Containable | **5** | Crypto category has 15% dispute prior (lowest), clean binary resolution |
| Calibration History | **4** | Lognormal model well-calibrated for strike prices; vol regime matters |
| **Composite** | **4.55** | ★★★★★ **TOP PICK** — Astra's strongest category by far |

**Strategy:** BUY YES on monthly strikes where lognormal p_hat > market + 8¢. Current BTC ~$75K post-crash, mean-reverting → $85K recovery likely mispriced. AVOID <48h expiry (market timing degradation).

---

## Opportunity 2: NHL Stanley Cup Champion
**Market:** "Colorado Avalanche to win Stanley Cup" at 22%
**Category:** Sports | **Volume:** $26M+ | **98 active NHL markets**

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **4** | Tier 1 algorithmic: The Odds API devigged probabilities, consensus bookmaker lines |
| Execution Edge | **4** | Deep liquidity on top teams; thinner on longshots |
| Risk Containable | **4** | Sports: 20% dispute prior; resolution typically clean for championships |
| Calibration History | **3** | Seasonal — need game-by-game data; few-shot examples sparse for hockey |
| **Composite** | **3.85** | ★★★★ Cross-reference sportsbook odds vs Polymarket for systematic arbitrage |

**Strategy:** Devig consensus bookmaker lines → find markets where Polymarket deviates >8¢. Avalanche at 22% vs sportsbook devigged ~18% = potential SHORT YES. Focus on top-6 teams with >$1M volume each.

---

## Opportunity 3: Fed Rate Cuts in 2026
**Market:** Multi-outcome: 0/1/2/3/4+ cuts | **3 cuts at 24%, 4+ at 19%**
**Category:** Finance | **Volume:** Multi-million

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **4** | FRED API (Tier A), economic calendar integration, yield curve data |
| Execution Edge | **3** | Multi-outcome markets = wider spreads, harder to fill optimally |
| Risk Containable | **4** | Macro events well-understood; correlated across Fed sub-markets |
| Calibration History | **3** | Limited history — new category emphasis in V2 |
| **Composite** | **3.55** | ★★★½ Good edge via FRED data; combinatorial arb across FOMC meeting markets |

**Strategy:** FOMC meeting binary markets (e.g., "March: 93% no change") are near-certain → use as anchor. If individual meeting probabilities don't sum to year-end distribution, arb the inconsistency. Use parity checker: sum of meeting-level "cut" probabilities should match "total cuts" market.

---

## Opportunity 4: 2026 Midterms — House + Senate
**Market:** Democrats win House (84%) / Republicans hold Senate (63%)
**Category:** Politics | **Volume:** $7.1M+ across 91 markets

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **3** | Tier 2 Astra AI (adversarial PRO/CON); no proprietary polling data |
| Execution Edge | **4** | High liquidity on main binary markets; thin on individual races |
| Risk Containable | **3** | Politics: 30% dispute prior (highest category); resolution ambiguity |
| Calibration History | **3** | 2024 election data in learning agent, but midterms are different |
| **Composite** | **3.25** | ★★★ **Combinatorial arb opportunity**: D House (84%) × R Senate (63%) = 53%, but "R Senate + D House" market at 47% → 6pt mispricing |

**Strategy:** Focus on combinatorial consistency across the 91 markets — not individual race prediction. If House + Senate + Balance-of-Power markets are inconsistent, trade the mispriced leg. Belt-and-suspenders: cluster exposure cap (12%) prevents over-concentration in politics.

---

## Opportunity 5: Oscars 2026 — Best Picture
**Market:** "One Battle After Another" at 80% | **Ceremony: March 15**
**Category:** Entertainment | **Volume:** $8M

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **2** | Tier 2 only — no structured entertainment data source; relies on Claude reasoning |
| Execution Edge | **3** | Decent volume but single-event resolution = lumpy risk |
| Risk Containable | **3** | Low dispute risk (Academy winner is unambiguous), but single-event = binary blowup |
| Calibration History | **2** | Minimal entertainment category history in learning agent |
| **Composite** | **2.50** | ★★½ Low data edge; only trade if precursor awards create >10¢ shift vs market |

**Strategy:** AVOID unless guild/precursor awards create a >10¢ market dislocation. At 80% favorite with ceremony in 30 days, the risk/reward is thin. If Dark Horse emerges from guild season, the remaining 20% could be split inefficiently.

---

## Opportunity 6: Claude 5 Release Date
**Market:** 86% by March 31, 2026 | **Volume:** $8.5M across 13 markets
**Category:** Technology / AI

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **2** | Tier 2 only — Claude reasoning about its own parent company; potential bias |
| Execution Edge | **4** | Good volume, binary resolution, clear criteria |
| Risk Containable | **3** | Tech: 25% dispute prior (medium); timeline markets can slip |
| Calibration History | **2** | Tech launch prediction has high variance; few training examples |
| **Composite** | **2.75** | ★★★ Interesting but **conflict of interest** — Astra uses Claude, predicting Claude 5 release is reflexive. The "best AI model" sub-market at 77% Anthropic is more tradeable. |

**Strategy:** AVOID direct Claude 5 timing markets (reflexive bias). The cross-company "best AI model by end of month" market is more interesting — use web-scraped benchmark data as evidence.

---

## Opportunity 7: Supreme Court Tariff Ruling
**Market:** 30% rules in Trump's favor (up from 21%) | **Volume:** $5.5M+
**Category:** Politics / Legal

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **3** | Tier 2 Astra AI; can reference SCOTUS prediction base rates + legal analysis |
| Execution Edge | **3** | Moderate volume; price moves on oral arguments / leak days |
| Risk Containable | **3** | Resolution can be ambiguous (partial rulings, narrow vs broad) |
| Calibration History | **2** | Legal category is thin in training data |
| **Composite** | **2.80** | ★★★ Volatile (9pt move in 2 days) → good for mean-reversion plays. Cross-reference with tariff refund market for consistency. |

**Strategy:** If "Rules in favor" and "Force tariff refund" markets are logically inconsistent, arb the pair. The 9pt recent move suggests overreaction to news → wait for regression. Use RRS (Resolution Risk Score) to gate — legal ambiguity may be high.

---

## Opportunity 8: FIFA World Cup 2026 Winner
**Market:** Spain at 16% (favorite) | **Volume:** $131M (largest on platform)
**Category:** Sports | **Tournament:** June-July 2026

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **4** | The Odds API covers World Cup; devigged team odds are high quality |
| Execution Edge | **5** | Deepest liquidity on platform → tight spreads, fast fills |
| Risk Containable | **3** | Multi-outcome market; must check all outcomes sum to ~100% |
| Calibration History | **3** | Long-dated (4+ months) = high uncertainty; sports model calibrates better near event |
| **Composite** | **3.80** | ★★★★ **Massive liquidity** — focus on multi-outcome parity (do all teams sum to 100%?). Event is 4 months away → gradual information incorporation creates persistent mispricings. |

**Strategy:** Check if YES prices across all 32+ teams sum to >$1.00 or <$1.00. If >$1.00, sell the most overpriced team. Compare each team's Polymarket price vs sportsbook devigged line — systematic deviation = edge. The Odds API gives direct comparison data. Volume ($131M) means fills are nearly guaranteed.

---

## Opportunity 9: Gold / Commodity Price Markets
**Market:** Gold at $5,300+ (resolved for Feb); longer-dated strikes active
**Category:** Commodities | **Volume:** $11.6M across 30 markets

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **3** | FRED API has gold spot price; no dedicated commodities model yet |
| Execution Edge | **2** | Thin order books on commodities (newer category) → high impact |
| Risk Containable | **3** | Binary resolution is clean; but thin books = slippage |
| Calibration History | **1** | Commodities is brand new on Polymarket; zero training data |
| **Composite** | **2.30** | ★★ **Skip for now** — thin liquidity + no calibration history. Loop 3 market impact model would penalize heavily (sqrt(size/liq) with low liq = big impact). |

**Strategy:** WAIT until Astra accumulates 20+ commodity trades in paper mode. The Loop 3 market impact model correctly flags this: DEFAULT_LIQUIDITY = 5000 tokens, but real commodity books may be <1000 → impact_bps would be massive.

---

## Opportunity 10: SpaceX Starship Launches
**Market:** 7-8 launches at 30% (most likely); Flight Test 12 by March at 55%
**Category:** Technology / Space | **Volume:** Multi-million

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Edge | **2** | No structured space/launch data source; pure Tier 2 reasoning |
| Execution Edge | **3** | Moderate volume; event-driven binary resolution |
| Risk Containable | **4** | Single-event markets with clear binary outcomes (launched or didn't) |
| Calibration History | **2** | Tech/space is thin; but base rates for SpaceX launches exist |
| **Composite** | **2.70** | ★★★ Individual launch markets (binary, near-dated) are better than year-total. Track FAA approval + weather as leading indicators. |

**Strategy:** Focus on individual Flight Test markets (binary, 2-4 week horizon) rather than year-total (multi-outcome, high variance). Use web search for FAA status + SpaceX announcements as Tier B evidence. The 55% for FT12 by March is tradeable if FAA signals readiness.

---

## RANKED SUMMARY

| Rank | Market | Composite | Primary Edge |
|------|--------|-----------|-------------|
| 1 | **Bitcoin Monthly Prices** | 4.55 | Tier 1 lognormal model + deep liquidity |
| 2 | **NHL Stanley Cup** | 3.85 | Sportsbook devig arbitrage |
| 3 | **FIFA World Cup** | 3.80 | Deepest liquidity + multi-outcome parity check |
| 4 | **Fed Rate Cuts** | 3.55 | FRED Tier A data + combinatorial arb |
| 5 | **Midterms Balance of Power** | 3.25 | Combinatorial mispricing (6pt observed gap) |
| 6 | **SCOTUS Tariffs** | 2.80 | Cross-market consistency + volatility |
| 7 | **Claude 5 / AI Models** | 2.75 | Reflexive bias concern; sub-markets tradeable |
| 8 | **SpaceX Launches** | 2.70 | Binary near-term launches only |
| 9 | **Oscars 2026** | 2.50 | Only if precursor-driven dislocation |
| 10 | **Gold / Commodities** | 2.30 | Skip — thin liquidity, zero calibration |

---

## LOOP 3 REALISM IMPACT ON SCORING

The fill realism improvements from Loop 3 directly affect opportunity viability:

1. **Latency simulation** (Activity 15, mean=50ms): Markets with 200ms arb windows (crypto HFT) are NOT capturable in paper mode — Astra's exponential latency distribution would miss most. But 5-minute+ mispricings in sports/politics are fine.

2. **Market impact** (Activity 16, K=0.02): With `impact = 0.02 * sqrt(size/liquidity)`:
   - $100 order on $131M FIFA market: impact ≈ 0 bps (negligible)
   - $100 order on $500K commodity market: impact ≈ 8.9 bps (significant)
   - This correctly penalizes Opportunity 9 (gold) and rewards Opportunity 8 (FIFA)

3. **Ring buffer** (Activity 14): Historical book snapshots enable latency-shifted fills. Markets with consistent book depth (crypto, FIFA) get accurate fills. Markets with sporadic updates (commodities) degrade more often.

4. **Startup checklist** (Activity 17): Monitor-only burn-in (3 cycles) prevents premature position-taking on first run. This is critical for live deployment.

---

## RECOMMENDED PAPER TRADING FOCUS

**Phase 1 (Weeks 1-4):** Run paper trader on Top 3 only
- Bitcoin monthly prices (crypto Tier 1)
- NHL/FIFA sports markets (sports Tier 1)
- Fed rate markets (finance FRED Tier A)

**Phase 2 (Weeks 5-8):** Add combinatorial arb
- Midterms cross-market consistency
- Fed meeting-level vs year-total arb

**Phase 3 (Weeks 9+):** Expand to Tier 2 categories
- SCOTUS, tech launches (after 30+ resolved trades in Tier 1)

---

*Generated: 2026-02-14 | Loop 3 merged (328 tests, pre-commit green) | Main @ 64fc823*
