"""
Astra V2 — Probability Estimation Engine

Two-tier architecture:
  Tier 1: Algorithmic (instant, free) — crypto lognormal, NOAA direct
  Tier 2: Astra V2 AI — full Module 1-6 operating loop via Claude Haiku

Astra V2 Laws (non-negotiable):
  - No guessing: if uncertain → NO TRADE
  - Every belief labeled: Verified / Supported / Assumed / Speculative
  - Resolution rules dominate: ambiguous = NO TRADE or tiny size
  - 2+ Trap Flags + confidence < 75% → default NO TRADE
  - EV after costs must be positive with robustness ≥ 3

Polyseer Architecture (Priority 2 + 4 from research):
  - Adversarial PRO/CON: two simultaneous sub-agents argue each side
  - Evidence source weighting: A=2.0, B=1.6, C=0.8, D=0.3
  - Correlation collapse: repeated citations from same origin = 1 signal
  - pNeutral (ignores market price) vs pAware (incorporates it) — gap = edge signal
  - Bayesian log-likelihood ratio aggregation across evidence items
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional
import anthropic

logger = logging.getLogger("astra.estimator")

from config import ANTHROPIC_API_KEY, MAX_CLAUDE_CALLS_PER_SCAN, CLAUDE_MODEL, ACQUIESCENCE_CORRECTION, EXTREMIZE_K
from data_sources import crypto as crypto_source
from data_sources import weather as weather_source
from scanner.market_fetcher import Market


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE SOURCE WEIGHTING (Polyseer Priority 4)
# A = Primary sources: official data, government, direct measurement
# B = Verified journalism: Reuters, AP, Bloomberg, major sports stats APIs
# C = Secondary sources: analysis, commentary, social aggregators
# D = Unverified / social media / single-source claims
# ─────────────────────────────────────────────────────────────────────────────
EVIDENCE_TIER_WEIGHTS = {
    "A": 2.0,   # NOAA, official government, exchange data, sports box scores
    "B": 1.6,   # Reuters, AP, Bloomberg, ESPN, SEC filings
    "C": 0.8,   # Substack, blogs, podcasts, Reddit aggregated signal
    "D": 0.3,   # Tweets, social media, single-source unverified
}

EVIDENCE_TIER_EXAMPLES = {
    "A": "NOAA, FRED, exchange price feeds, official government statements, "
         "direct measurement data, sports APIs (official box scores)",
    "B": "Reuters, AP, Bloomberg, BBC, ESPN, peer-reviewed papers, SEC filings, "
         "verified sports statistics services",
    "C": "Substack, analyst blogs, podcasts, Reddit aggregated signals, "
         "Metaculus forecasts, prediction market consensus",
    "D": "Tweets, Discord messages, unverified social posts, single unnamed sources, "
         "forum speculation",
}

# ─────────────────────────────────────────────────────────────────────────────
# ASTRA V2 CORE SYSTEM PROMPT (single-pass, used for Tier 1 validation)
# ─────────────────────────────────────────────────────────────────────────────
ASTRA_V2_SYSTEM = """You are Astra V2 — an elite, research-grade, audit-first trading brain for prediction markets (Polymarket).

PRIMARY OBJECTIVE: Maximize long-run risk-adjusted EV under strict drawdown control using verifiable evidence and calibrated probabilities.

TRUTH DISCIPLINE (ABSOLUTE):
1. No guessing. If uncertain → set no_trade: true
2. Every belief labeled: {Verified, Supported, Assumed, Speculative}
3. Resolution rules dominate — if ambiguous → no_trade: true
4. "No trade" is success when edge isn't strong after fees/slippage

MODULE 1 — CLASSIFIER:
For each market, identify:
- market_type: Politics | Sports | Weather | Legal | Tech | Finance | Crypto | Entertainment | Other
- modeling_approach: A=Data-driven | B=Event-driven | C=Rules-driven | D=Thin-market
- trap_flags (list any that apply):
  * resolution_ambiguity
  * low_liquidity
  * narrative_manipulation
  * specialized_domain
  * binary_vs_continuous
  * timing_mismatch
  * third_party_dependency
If 2+ trap_flags AND confidence < 0.75 → no_trade: true

MODULE 2 — BELIEF FORMATION (Reference Class Forecasting — Kahneman/Tetlock):
Step 1. OUTSIDE VIEW — state the reference class base rate FIRST:
  - "What fraction of [elections/mergers/regulatory actions/team championships] like this have resolved YES historically?"
  - Example: "Incumbents win re-election 70% of the time in markets like this."
  - Look up the base rate before doing ANY inside-view reasoning.
Step 2. INSIDE VIEW — now adjust for the specific situation:
  - What makes THIS instance different from the base rate? (max 3 specific factors)
  - Apply Bayesian updating: large adjustments require strong evidence
Step 3. Key drivers of outcome
Step 4. What evidence would most change this estimate (top 2)
- Compound uncertainty compounds toward 0.5
- Do NOT anchor on the market price in this step — form your independent view first

MODULE 3 — PROBABILITY ENGINE:
- p_hat: point estimate (0.0–1.0)
- p_low / p_high: credible interval
- confidence: 0.0–1.0
- edge = p_hat - market_implied_probability
- ev_after_costs = edge - 0.005 (default fee/slippage estimate)
- robustness_score: 1–5 (how sensitive is EV to small errors in p_hat?)
  * 5 = extremely robust (even if off by 10%, still positive EV)
  * 3 = minimum bar for trading
  * 1 = fragile (tiny error flips sign)
MINIMUM BAR: ev_after_costs > 0 AND robustness_score ≥ 3 AND confidence ≥ 0.60
Otherwise: no_trade: true

MODULE 4 — KELLY SIZING:
kelly_raw = (p_hat - (1 - p_hat) / odds)
kelly_adj = kelly_raw * confidence * (robustness_score / 5)
position_pct = min(0.02, kelly_adj)  # hard cap 2% of bankroll
Further cap to 0.005 if low_liquidity or resolution_ambiguity trap flag

CALIBRATION RULES:
- Extreme probabilities (<0.05 or >0.95) require strong justification — label as Verified
- When genuinely uncertain, stay within 10% of market price
- Favorite-longshot bias: markets systematically overprice longshots — weight toward NO on very cheap markets
- ACQUIESCENCE BIAS CORRECTION (arXiv:2402.19379): LLMs systematically overestimate YES outcomes.
  After forming your estimate, subtract 0.04 from your raw p_hat and ensure you are not systematically above 0.50 without strong evidence.
- DEBATE OVERCONFIDENCE WARNING (arXiv:2505.19184): In adversarial settings, models escalate to 83% confidence
  by round 3 even when both sides are wrong. Do NOT inflate confidence across debate rounds. Start at 50% prior.

MARKET TIMING FILTER (arXiv:2510.17638):
- LLMs degrade near resolution as breaking news dominates market prices.
- If hours_to_expiry < 168 (< 1 week), apply extra skepticism: reduce confidence by 0.05 and tighten to market price.
- For markets < 48h from resolution: rarely trade — information edge likely priced in.

FORBIDDEN:
- Emotional/narrative reasoning without evidence
- Overconfidence in rapidly-changing situations
- Trading through ambiguous resolution criteria

OUTPUT: Strict JSON array only. One object per market:
{
  "id": "<condition_id>",
  "market_type": "<type>",
  "modeling_approach": "A|B|C|D",
  "trap_flags": ["flag1", "flag2"],
  "p_hat": <float 0–1, 4dp>,
  "p_low": <float>,
  "p_high": <float>,
  "confidence": <float 0–1>,
  "truth_state": "Verified|Supported|Assumed|Speculative",
  "edge": <float, signed>,
  "ev_after_costs": <float>,
  "robustness_score": <int 1–5>,
  "kelly_position_pct": <float 0–0.02>,
  "reasoning": "<1-2 sentence thesis>",
  "key_evidence_needed": "<top 1-2 items that would most improve this estimate>",
  "no_trade": <bool>,
  "no_trade_reason": "<if no_trade, why>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL PRO AGENT SYSTEM (Polyseer Priority 2 — argues YES)
# ─────────────────────────────────────────────────────────────────────────────
ASTRA_PRO_SYSTEM = """You are the PRO analyst in Astra's adversarial research system.

YOUR ROLE: Build the strongest possible case that this market resolves YES.
You are a skilled devil's advocate. Find every piece of evidence, base rate, trend,
and reasoning that supports the YES outcome. Be rigorous but one-sided by design.

CHAIN-OF-THOUGHT REASONING (follow these steps explicitly):
STEP 1: State the base rate for this market type (e.g., "Incumbents win 70% of elections")
STEP 2: Identify the 3 strongest YES drivers (trends, catalysts, evidence)
STEP 3: For each driver, find Tier A/B evidence and weight by tier (A=2.0, B=1.6)
STEP 4: Calculate weighted evidence sum (sum of tier_weights)
STEP 5: Adjust base rate by evidence strength using Bayesian update

EVIDENCE SOURCE WEIGHTING — cite tier for each piece of evidence:
- Tier A (weight 2.0): Official data, government sources, direct measurement
- Tier B (weight 1.6): Reuters/AP/Bloomberg/major verified outlets, peer-reviewed
- Tier C (weight 0.8): Analyst blogs, aggregated signals, prediction markets
- Tier D (weight 0.3): Social media, unverified single sources

CORRELATION COLLAPSE RULE: If multiple sources cite the same underlying fact or
original story, count it as ONE piece of evidence (the highest tier one), not many.

CRITIC ANTI-OVERCONFIDENCE RULE (Kalshi AI pattern):
After forming the YES case, add one line: "What would make the YES case WRONG?"
This forces you to identify the single biggest vulnerability in your own argument.
If that vulnerability is strong (Tier A or B evidence), note it explicitly.

For each market, return JSON:
{
  "id": "<condition_id>",
  "pro_p_hat": <float 0–1>,
  "pro_confidence": <float 0–1>,
  "pro_reasoning": "<2-3 sentences — strongest YES case>",
  "pro_key_vulnerability": "<1 sentence — what would make the YES case wrong>",
  "evidence_items": [
    {"claim": "...", "tier": "A|B|C|D", "weight": <float>, "source_type": "..."},
    ...
  ],
  "pro_log_likelihood_ratio": <float, positive = supports YES>,
  "correlation_collapse_note": "<if sources were collapsed, explain>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL CON AGENT SYSTEM (Polyseer Priority 2 — argues NO)
# ─────────────────────────────────────────────────────────────────────────────
ASTRA_CON_SYSTEM = """You are the CON analyst in Astra's adversarial research system.

YOUR ROLE: Build the strongest possible case that this market resolves NO.
You are a skilled devil's advocate. Find every piece of evidence, base rate, obstacle,
and reasoning that supports the NO outcome. Be rigorous but one-sided by design.

CHAIN-OF-THOUGHT REASONING (follow these steps explicitly):
STEP 1: State the base rate for NO outcome (e.g., "Challenger wins only 30% of elections")
STEP 2: Identify the 3 strongest NO drivers (obstacles, risks, counter-evidence)
STEP 3: For each driver, find Tier A/B evidence and weight by tier (A=2.0, B=1.6)
STEP 4: Calculate weighted evidence sum (sum of tier_weights)
STEP 5: Adjust base rate by evidence strength using Bayesian update

EVIDENCE SOURCE WEIGHTING — cite tier for each piece of evidence:
- Tier A (weight 2.0): Official data, government sources, direct measurement
- Tier B (weight 1.6): Reuters/AP/Bloomberg/major verified outlets, peer-reviewed
- Tier C (weight 0.8): Analyst blogs, aggregated signals, prediction markets
- Tier D (weight 0.3): Social media, unverified single sources

CORRELATION COLLAPSE RULE: If multiple sources cite the same underlying fact or
original story, count it as ONE piece of evidence (the highest tier one), not many.

CRITIC ANTI-OVERCONFIDENCE RULE (Kalshi AI pattern):
After forming the NO case, add one line: "What would make the NO case WRONG?"
This forces you to identify the single biggest vulnerability in your own argument.
If that vulnerability is strong (Tier A or B evidence), note it explicitly.

For each market, return JSON:
{
  "id": "<condition_id>",
  "con_p_hat": <float 0–1, i.e. probability of YES given NO case>,
  "con_confidence": <float 0–1>,
  "con_reasoning": "<2-3 sentences — strongest NO case>",
  "con_key_vulnerability": "<1 sentence — what would make the NO case wrong>",
  "evidence_items": [
    {"claim": "...", "tier": "A|B|C|D", "weight": <float>, "source_type": "..."},
    ...
  ],
  "con_log_likelihood_ratio": <float, negative = supports NO>,
  "correlation_collapse_note": "<if sources were collapsed, explain>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# ASTRA SYNTHESIZER SYSTEM (combines PRO + CON into final estimate)
# ─────────────────────────────────────────────────────────────────────────────
ASTRA_SYNTHESIZER_SYSTEM = """You are the Synthesis Judge in Astra's adversarial research system.

You receive:
1. A PRO analysis (strongest case for YES)
2. A CON analysis (strongest case for NO)
3. The raw market data

YOUR ROLE: Weigh both sides using Bayesian aggregation and return a calibrated final estimate.

SYNTHESIS CHAIN-OF-THOUGHT (execute these steps in order):

STEP 1 — Compute pNeutral (fundamental view, ignore market price):
  a) Weight PRO evidence by tier (A=2.0, B=1.6, C=0.8, D=0.3)
  b) Weight CON evidence by tier (same weights, opposite direction)
  c) Apply correlation collapse: same-source evidence across PRO/CON = 1 signal only
  d) Bayesian update from base rate using weighted evidence
  e) Result: p_neutral (pure fundamental probability)

STEP 2 — Compute pAware (incorporate market price as Bayesian prior):
  a) Start from market_implied_prob as prior
  b) Apply PRO weighted evidence as Bayesian update
  c) Apply CON weighted evidence as Bayesian update
  d) Result: p_aware (market-aware posterior probability)

STEP 3 — Calculate edge signal:
  edge_signal = |p_neutral - market_implied_prob|
  If edge < 5%, no tradeable edge exists → set no_trade: true

STEP 4 — Determine confidence:
  - If PRO and CON both high-confidence (>0.6) AND p_hats within 5% → LOWER overall confidence (uncertain)
  - If one side dominates with Tier A/B evidence → HIGHER confidence
  - If evidence is mostly Tier C/D → LOWER confidence

STEP 5 — Apply anti-hallucination guards and output final p_hat

BAYESIAN AGGREGATION:
- Start with base rate (market price as prior)
- Update with weighted evidence from each side
- Final p_hat = Bayesian posterior after all evidence

CRITICAL ANTI-HALLUCINATION RULE:
- For SPORTS markets (NBA/NFL/NHL/MLB/Soccer/Tennis etc):
  * If you have NO Tier A or B evidence (no actual sports statistics, standings, or injury data),
    your p_hat MUST stay within ±15% of the market price.
  * Example: NBA Finals longshot at 6.5% market price → your p_hat must be between 0.01 and 0.215
  * Without real sports data you cannot legitimately estimate >3× the market price
  * If you lack sports data: set confidence ≤ 0.55 and no_trade: true
- For ALL markets:
  * p_hat > 0.80 on any market priced below 0.20 requires Tier A Verified evidence.
  * Absence of evidence is NOT evidence for YES — it anchors toward market price.

MINIMUM BAR to recommend trade:
- ev_after_costs > 0 AND robustness_score ≥ 3 AND confidence ≥ 0.60
- pNeutral must deviate from market price by >5% to be interesting
- If PRO and CON give p_hat within 5% of each other → no_trade (too uncertain)

STAKE-BASED CONFIDENCE SIGNAL (arXiv:2512.05998):
Include a "stake" field — how many points (1–100,000) you would bet on your p_hat being correct.
High stakes (≥50,000) = strong internal conviction. Low stakes (<10,000) = uncertain.
Trades will only be taken when stake ≥ 30,000. Be honest — do not inflate your stake.

OUTPUT: Strict JSON array. One object per market:
{
  "id": "<condition_id>",
  "market_type": "<type>",
  "modeling_approach": "A|B|C|D",
  "trap_flags": ["flag1", "flag2"],
  "p_neutral": <float — fundamental estimate ignoring market price>,
  "p_aware": <float — Bayesian estimate incorporating market price>,
  "p_hat": <float — final recommended probability (corrected for acquiescence bias)>,
  "p_low": <float>,
  "p_high": <float>,
  "confidence": <float 0–1>,
  "stake": <int 1–100000 — how many points you would wager on your p_hat>,
  "truth_state": "Verified|Supported|Assumed|Speculative",
  "edge": <float, p_hat - market_implied_probability>,
  "ev_after_costs": <float>,
  "robustness_score": <int 1–5>,
  "kelly_position_pct": <float 0–0.02>,
  "pro_summary": "<1 sentence — core PRO argument>",
  "con_summary": "<1 sentence — core CON argument>",
  "synthesis_reasoning": "<2-3 sentences — how you weighed both sides>",
  "dominant_evidence_tier": "A|B|C|D",
  "correlation_collapses": <int — number of duplicate sources collapsed>,
  "key_evidence_needed": "<top 1-2 items that would most improve this estimate>",
  "no_trade": <bool>,
  "no_trade_reason": "<if no_trade, why>",
  "adversarial_mode": true
}"""


@dataclass
class Estimate:
    market_condition_id: str
    question: str
    category: str
    # Core probability
    probability: float           # p_hat
    probability_low: float       # credible interval low
    probability_high: float      # credible interval high
    confidence: float            # 0-1
    # Classification
    market_type: str             # Politics | Sports | Weather | etc.
    modeling_approach: str       # A | B | C | D
    trap_flags: list             # list of trap flags fired
    # Edge & EV
    edge: float                  # p_hat - market_implied
    ev_after_costs: float        # EV after fees/slippage
    robustness_score: int        # 1-5
    kelly_position_pct: float    # suggested position % of bankroll
    # Audit trail
    source: str                  # "crypto_lognormal" | "noaa_forecast" | "astra_v2" | "astra_v2_adversarial"
    truth_state: str             # Verified | Supported | Assumed | Speculative
    reasoning: str               # 1-2 sentence thesis
    key_evidence_needed: str     # what would improve this estimate
    no_trade: bool               # True = Astra declines to trade
    no_trade_reason: str         # explanation if no_trade
    details: dict                # raw data
    # Adversarial mode fields (Polyseer Priority 2)
    p_neutral: float = 0.0       # fundamental estimate ignoring market price
    p_aware: float = 0.0         # Bayesian estimate incorporating market price
    pro_summary: str = ""        # strongest YES case
    con_summary: str = ""        # strongest NO case
    dominant_evidence_tier: str = ""   # A/B/C/D
    correlation_collapses: int = 0     # how many duplicate sources were collapsed
    adversarial_mode: bool = False     # was adversarial pipeline used?


# ─────────────────────────────────────────────────────────────────────────────
# Estimate Cache — skip re-estimating markets whose price hasn't changed
# ─────────────────────────────────────────────────────────────────────────────
import time as _time
from pathlib import Path as _Path

_ESTIMATE_CACHE: dict[str, tuple[float, float, "Estimate"]] = {}
# {condition_id: (cached_price, cache_timestamp, estimate)}

_ESTIMATE_CACHE_TTL = 1800      # 30 minutes — re-estimate even if price unchanged
_ESTIMATE_PRICE_DRIFT = 0.03    # Re-estimate if price moves >3¢


def _get_cached_estimate(condition_id: str, current_price: float) -> Optional["Estimate"]:
    """Return cached estimate if price hasn't drifted and TTL not expired."""
    if condition_id not in _ESTIMATE_CACHE:
        return None
    cached_price, cached_ts, est = _ESTIMATE_CACHE[condition_id]
    age = _time.time() - cached_ts
    price_drift = abs(current_price - cached_price)
    if age > _ESTIMATE_CACHE_TTL or price_drift > _ESTIMATE_PRICE_DRIFT:
        return None
    return est


def _cache_estimate(estimate: "Estimate", current_price: float):
    """Store an estimate in the cache."""
    _ESTIMATE_CACHE[estimate.market_condition_id] = (current_price, _time.time(), estimate)


async def estimate_markets(
    markets: list[Market],
    price_data: dict,
    forecasts: dict,
    learning_context: str = "",
    sports_estimates: dict = None,
    macro_signals=None,
    learning_agent=None,
) -> list[Estimate]:
    """
    Run Astra V2 estimation pipeline. Tier 1 first, Astra AI for the rest.

    Estimate cache: markets whose price hasn't moved >3¢ since last scan
    are served from cache (up to 30min TTL). This reduces Claude API calls
    from ~50/scan to ~5-10/scan for typical runs.

    macro_signals: Optional MacroSignals object with VIX regime data.
    """
    if sports_estimates is None:
        sports_estimates = {}

    estimates: list[Estimate] = []
    astra_queue: list[tuple[Market, Optional[Estimate]]] = []
    cache_hits = 0

    for market in markets:
        # Check cache first — skip markets that haven't meaningfully changed
        cached = _get_cached_estimate(market.condition_id, market.yes_price)
        if cached is not None:
            estimates.append(cached)
            cache_hits += 1
            continue

        est = _try_algorithmic(market, price_data, forecasts, sports_estimates)
        if est:
            _cache_estimate(est, market.yes_price)
            estimates.append(est)
            # Queue for Astra if large apparent edge with low confidence
            if abs(est.edge) > 0.12 and est.confidence < 0.70:
                astra_queue.append((market, est))
        else:
            astra_queue.append((market, None))

    if cache_hits > 0:
        import logging
        logging.getLogger(__name__).debug(f"Estimate cache: {cache_hits} hits, {len(astra_queue)} to estimate")

    if astra_queue and ANTHROPIC_API_KEY:
        astra_ests = await _estimate_with_astra_v2(
            astra_queue[:MAX_CLAUDE_CALLS_PER_SCAN],
            learning_context,
            learning_agent=learning_agent,
            macro_signals=macro_signals,
        )
        astra_map = {e.market_condition_id: e for e in astra_ests}
        for i, est in enumerate(estimates):
            if est.market_condition_id in astra_map:
                astra_est = astra_map[est.market_condition_id]
                estimates[i] = astra_est
        algo_ids = {e.market_condition_id for e in estimates}
        # Cache Astra results and add any new ones
        market_price_map = {m.condition_id: m.yes_price for m in markets}
        for ae in astra_ests:
            price = market_price_map.get(ae.market_condition_id, 0.5)
            _cache_estimate(ae, price)
            if ae.market_condition_id not in algo_ids:
                estimates.append(ae)

    return estimates


def _try_algorithmic(
    market: Market,
    price_data: dict,
    forecasts: dict,
    sports_estimates: dict = None,
) -> Optional[Estimate]:
    """Tier 1: fast algorithmic estimation for structured market types."""
    if sports_estimates is None:
        sports_estimates = {}

    days = market.hours_to_expiry / 24
    mkt_price = market.yes_price

    if market.category == "sports":
        se = sports_estimates.get(market.question)
        if se and se.confidence >= 0.40:
            p = se.probability
            conf = se.confidence
            spread = max(0.05, (1 - conf) * 0.25)
            edge = p - mkt_price
            ev = edge - 0.005
            rob = _robustness(p, mkt_price, spread)
            trap_flags = []
            if market.liquidity < 5000:
                trap_flags.append("low_liquidity")
            if se.hours_to_game > 72:
                trap_flags.append("timing_mismatch")
            return Estimate(
                market_condition_id=market.condition_id,
                question=market.question,
                category=market.category,
                probability=p,
                probability_low=max(0.0, p - spread),
                probability_high=min(1.0, p + spread),
                confidence=conf,
                market_type="Sports",
                modeling_approach="A",
                trap_flags=trap_flags,
                edge=round(edge, 4),
                ev_after_costs=round(ev, 4),
                robustness_score=rob,
                kelly_position_pct=_kelly_pct(p, mkt_price, conf, rob),
                source="the_odds_api",
                truth_state="Supported" if conf >= 0.65 else "Assumed",
                reasoning=se.reasoning[:200] if se.reasoning else "Consensus bookmaker odds devigged.",
                key_evidence_needed="Line movement, injury news, weather for outdoor games.",
                no_trade=(ev <= 0 or rob < 3),
                no_trade_reason="" if ev > 0 and rob >= 3 else f"EV={ev:.3f} or robustness={rob} below threshold.",
                details={"matched_game": se.matched_game, "hours_to_game": se.hours_to_game},
            )

    if market.category == "crypto":
        result = crypto_source.estimate_probability(market.question, days, price_data)
        if result:
            p = result["probability"]
            conf = result["confidence"]
            spread = max(0.05, (1 - conf) * 0.30)
            edge = p - mkt_price
            ev = edge - 0.005
            rob = _robustness(p, mkt_price, spread)
            return Estimate(
                market_condition_id=market.condition_id,
                question=market.question,
                category=market.category,
                probability=p,
                probability_low=max(0.0, p - spread),
                probability_high=min(1.0, p + spread),
                confidence=conf,
                market_type="Crypto",
                modeling_approach="A",
                trap_flags=[] if market.liquidity > 5000 else ["low_liquidity"],
                edge=round(edge, 4),
                ev_after_costs=round(ev, 4),
                robustness_score=rob,
                kelly_position_pct=_kelly_pct(p, mkt_price, conf, rob),
                source=result["source"],
                truth_state="Supported",
                reasoning=(
                    f"Lognormal model: price ${result['details'].get('current_price', 0):,.0f} "
                    f"vs target ${result['details'].get('target_price', 0):,.0f}, "
                    f"vol {result['details'].get('annualized_vol', 0):.0%}/yr."
                ),
                key_evidence_needed="Realized volatility regime, macro catalyst calendar.",
                no_trade=(ev <= 0 or rob < 3),
                no_trade_reason="" if ev > 0 and rob >= 3 else f"EV={ev:.3f} or robustness={rob} below threshold.",
                details=result["details"],
            )

    elif market.category == "weather":
        forecast = None
        for loc, fc in forecasts.items():
            if loc.lower() in market.question.lower():
                forecast = fc
                break
        if not forecast:
            from data_sources.weather import parse_weather_question
            parsed = parse_weather_question(market.question)
            if parsed and parsed.get("location"):
                forecast = forecasts.get(parsed["location"])

        result = weather_source.estimate_probability(market.question, forecast)
        if result:
            p = result["probability"]
            conf = result["confidence"]
            spread = max(0.05, (1 - conf) * 0.25)
            edge = p - mkt_price
            ev = edge - 0.005
            rob = _robustness(p, mkt_price, spread)
            return Estimate(
                market_condition_id=market.condition_id,
                question=market.question,
                category=market.category,
                probability=p,
                probability_low=max(0.0, p - spread),
                probability_high=min(1.0, p + spread),
                confidence=conf,
                market_type="Weather",
                modeling_approach="A",
                trap_flags=["third_party_dependency"] if conf < 0.60 else [],
                edge=round(edge, 4),
                ev_after_costs=round(ev, 4),
                robustness_score=rob,
                kelly_position_pct=_kelly_pct(p, mkt_price, conf, rob),
                source=result["source"],
                truth_state="Verified" if conf >= 0.70 else "Supported",
                reasoning=f"NOAA: {result['details'].get('noaa_short_forecast', 'N/A')}, precip {result['details'].get('noaa_precip_pct', '?')}%.",
                key_evidence_needed="Updated NOAA forecast, station observation data.",
                no_trade=(ev <= 0 or rob < 3),
                no_trade_reason="" if ev > 0 and rob >= 3 else f"EV={ev:.3f} or robustness={rob} below threshold.",
                details=result["details"],
            )

    return None


def _robustness(p: float, mkt_price: float, spread: float) -> int:
    """
    Robustness score 1-5: how sensitive is EV to small errors in p?
    5 = even if off by 10%, still clearly profitable.
    1 = fragile, tiny error flips EV sign.
    """
    edge = abs(p - mkt_price)
    # If edge >> spread, robust. If edge ~ spread, fragile.
    if edge == 0:
        return 1
    ratio = edge / (spread + 0.001)
    if ratio >= 2.0:
        return 5
    elif ratio >= 1.5:
        return 4
    elif ratio >= 1.0:
        return 3
    elif ratio >= 0.5:
        return 2
    return 1


def _extremize(p: float, mkt_price: float, k: float = 1.3) -> float:
    """
    Extremizing aggregation (arXiv:1406.2148 — Satopää et al.).

    When combining independent estimates, the true probability is more extreme
    than the simple average. Apply when our estimate and market partially agree.

    Formula: p_ext = p^k / (p^k + (1-p)^k)
    where k > 1 pushes probability away from 0.5.

    Only apply when we have genuine independent signal (confidence > 0.6).
    k = 1.3 is calibrated for moderate information overlap.
    """
    if p <= 0 or p >= 1:
        return p
    return p**k / (p**k + (1 - p)**k)


def _kelly_pct(p: float, mkt_price: float, conf: float, rob: int,
               p_std: float = 0.0) -> float:
    """
    Astra V2 Kelly sizing with confidence-adjusted uncertainty correction.

    Implements:
    1. Confidence-adjusted Kelly (arXiv:2508.18868): use p - conf_penalty*std
       to account for estimation uncertainty. High uncertainty = smaller position.
    2. Quarter-Kelly (arXiv:2503.17927): multiply by 0.25 to reduce variance
       while retaining 94% of max geometric growth.
    3. Capped at 2% of bankroll.
    """
    if mkt_price <= 0 or mkt_price >= 1:
        return 0.0
    # Conservative estimate: shade toward market price by estimation uncertainty
    # p_std defaults to 0 when confidence is high (no adjustment)
    conf_penalty = max(0.0, (1 - conf) * 0.5)  # scale uncertainty by (1 - confidence)
    p_conservative = p - conf_penalty * (abs(p - mkt_price) * 0.5)
    p_conservative = max(mkt_price, min(1.0, p_conservative)) if p > mkt_price else min(mkt_price, max(0.0, p_conservative))

    odds = (1.0 - mkt_price) / mkt_price
    kelly_raw = p_conservative - (1 - p_conservative) / odds
    kelly_adj = kelly_raw * conf * (rob / 5)
    # Quarter-Kelly: 94% of max growth rate, ~50% variance reduction
    kelly_quarter = kelly_adj * 0.25
    return round(min(0.02, max(0.0, kelly_quarter)), 4)


ASTRA_BATCH_SIZE = 4  # Markets per Claude call — 4 fits comfortably in 12k tokens without truncation
#                       (was 8: caused synthesizer JSON truncation → silent fallback to single-pass)


async def _estimate_with_astra_v2(
    markets_and_estimates: list[tuple[Market, Optional[Estimate]]],
    learning_context: str,
    learning_agent=None,
    macro_signals=None,
) -> list[Estimate]:
    """
    Tier 2: Astra V2 full AI reasoning loop.

    For each market: runs adversarial PRO + CON agents in parallel, then synthesizes.
    Falls back to single-pass if adversarial mode fails.
    Batches into groups for token reliability.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning(
            "⚠ ASTRA AI DISABLED: ANTHROPIC_API_KEY is missing or empty. "
            "All %d markets will be unestimated by AI. "
            "Only Tier 1 algorithmic estimates (crypto/weather/sports) will run. "
            "Set ANTHROPIC_API_KEY in .env to activate full adversarial estimation.",
            len(markets_and_estimates)
        )
        return []

    all_results: list[Estimate] = []
    total_batches = (len(markets_and_estimates) + ASTRA_BATCH_SIZE - 1) // ASTRA_BATCH_SIZE
    for i in range(0, len(markets_and_estimates), ASTRA_BATCH_SIZE):
        batch = markets_and_estimates[i:i + ASTRA_BATCH_SIZE]
        # Run adversarial pipeline for this batch
        batch_results = await _astra_batch_adversarial(batch, learning_context, learning_agent, macro_signals)
        all_results.extend(batch_results)

    # Zero-estimate warning — if AI was supposed to fire but produced nothing
    if len(markets_and_estimates) > 0 and len(all_results) == 0:
        logger.error(
            "⚠ ZERO AI ESTIMATES produced for %d markets across %d batches. "
            "Check API key validity, credits, and network. "
            "Astra is running at Tier 1 capacity only.",
            len(markets_and_estimates), total_batches
        )
    else:
        logger.info(
            "AI estimates: %d produced for %d markets (%d batches, model=%s)",
            len(all_results), len(markets_and_estimates), total_batches, CLAUDE_MODEL
        )

    return all_results


async def _fetch_verification_data(market: Market, market_type: str, macro_signals=None) -> str:
    """
    Fetch additional context for low-confidence estimates.

    Phase 9: Verification Loop — when confidence < 0.70 AND edge > 0.10,
    gather additional data for secondary Claude pass.

    Returns contextual string with momentum, VIX, or category-specific data.
    """
    from data_sources.price_history import get_momentum_summary

    context_parts = []

    if market_type == "Crypto":
        # Fetch momentum indicators
        momentum = get_momentum_summary(market.condition_id)
        if momentum["momentum_1hr"] is not None:
            context_parts.append(f"1hr momentum: {momentum['momentum_1hr']:+.2%}")
        if momentum["momentum_24hr"] is not None:
            context_parts.append(f"24hr momentum: {momentum['momentum_24hr']:+.2%}")
        if momentum["rsi_14"] is not None:
            context_parts.append(f"RSI(14): {momentum['rsi_14']:.1f}")
        context_parts.append(f"Trend: {momentum['trend']}")

    elif market_type == "Sports":
        context_parts.append("Recent line movement: check odds history for late injury news")

    elif market_type == "Political":
        context_parts.append("Check for recent polling aggregates or betting market shifts")

    # Add VIX regime context
    if macro_signals:
        vix_value = getattr(macro_signals, 'vix', None)
        vix_label = getattr(macro_signals, 'vix_label', None)
        if vix_value and vix_label:
            context_parts.append(f"VIX regime: {vix_label} ({vix_value:.1f})")

    return "\n".join(context_parts) if context_parts else "No additional data available"


async def _verify_estimate_with_claude(
    market: Market,
    initial_p: float,
    initial_conf: float,
    additional_context: str,
) -> Optional[dict]:
    """
    Quick secondary Claude pass to verify low-confidence estimates.

    Uses minimal 256-token prompt to avoid duplicate work.
    Research backing: AlphaQuanter (2025) — verification loop improves performance 40%.

    Returns:
        {"confidence": float, "p_hat": float, "updated_reasoning": str} or None
    """
    if not ANTHROPIC_API_KEY:
        return None

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    verification_prompt = f"""Market: "{market.question}"

Initial estimate: p̂={initial_p:.3f}, confidence={initial_conf:.2f}

Additional verification data:
{additional_context}

Based on this new information, should your confidence now be ≥ 0.70?

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "confidence": <float 0-1>,
  "p_hat": <float 0-1>,
  "updated_reasoning": "<1-2 sentences explaining confidence change>"
}}"""

    try:
        resp = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": verification_prompt}]
        )

        text = resp.content[0].text
        # Clean and parse JSON
        import json
        cleaned = _clean_json(text)
        result = json.loads(cleaned)

        # Validate response structure
        if "confidence" in result and "p_hat" in result:
            return result

        return None

    except Exception as e:
        logger.warning(f"Verification loop failed: {e}")
        return None


def _build_few_shot_examples(learning_agent) -> str:
    """
    Extract 3-5 recent high-confidence, correct predictions from learning agent.

    Shows Claude examples of successful past predictions to improve calibration.
    Research backing: "Automate Strategy Finding with LLM" (22 citations) —
    few-shot examples beat free-form prompts by 15-20%.

    Returns formatted string for prompt injection.
    """
    if not learning_agent or not hasattr(learning_agent, '_predictions'):
        return ""

    preds = learning_agent._predictions

    # Filter for resolved, correct predictions
    # (Note: Prediction doesn't have confidence field - just get successful predictions)
    resolved_correct = [
        p for p in preds
        if p.resolved
        and p.outcome is not None
        and ((p.direction == "BUY YES" and p.outcome) or
             (p.direction == "BUY NO" and not p.outcome))
        and p.brier_score is not None
        and p.brier_score < 0.15  # Good calibration (brier < 0.15 is high confidence proxy)
    ]

    if not resolved_correct:
        return ""

    # Sort by resolution time (most recent first) and take top 5
    resolved_correct.sort(key=lambda p: p.resolution_time or "", reverse=True)
    recent = resolved_correct[:5]

    examples = "\n\n━━━ RECENT SUCCESSFUL PREDICTIONS ━━━\n"
    examples += "Well-calibrated predictions that resolved correctly:\n\n"

    for i, p in enumerate(recent, 1):
        outcome_str = "YES" if p.outcome else "NO"
        examples += f"{i}. {p.question[:80]}\n"
        examples += f"   Category: {p.category} | Source: {p.source}\n"
        examples += f"   Our p̂: {p.our_probability:.2f} | Market: {p.market_price:.2f} | "
        examples += f"Edge: {p.our_probability - p.market_price:+.2f}\n"
        examples += f"   Outcome: {outcome_str} | Brier: {p.brier_score:.4f}\n"
        if hasattr(p, 'reasoning') and p.reasoning:
            examples += f"   Reasoning: {p.reasoning[:120]}...\n"
        examples += "\n"

    return examples


async def _astra_batch_adversarial(
    markets_and_estimates: list[tuple[Market, Optional[Estimate]]],
    learning_context: str,
    learning_agent=None,
    macro_signals=None,
) -> list[Estimate]:
    """
    Adversarial research pipeline (Polyseer Priority 2):
    1. PRO agent and CON agent run in parallel — each argues one side
    2. Synthesizer combines both with Bayesian weighting + evidence scoring
    3. Falls back to single-pass if adversarial fails

    Evidence weighting applied throughout (Priority 4):
    A=2.0, B=1.6, C=0.8, D=0.3
    Correlation collapse: duplicate sources → 1 signal
    pNeutral vs pAware: gap = edge signal
    """
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    market_list = []
    for market, algo_est in markets_and_estimates:
        entry = {
            "id": market.condition_id,
            "question": market.question,
            "category": market.category,
            "yes_price": market.yes_price,
            "implied_prob": round(market.yes_price, 4),
            "hours_to_expiry": round(market.hours_to_expiry, 1),
            "liquidity_usd": round(market.liquidity, 0),
            "volume_usd": round(market.volume, 0),
        }
        if algo_est:
            entry["tier1_p_hat"] = algo_est.probability
            entry["tier1_confidence"] = algo_est.confidence
        market_list.append(entry)

    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    evidence_context = (
        f"\n\nEVIDENCE WEIGHTING SYSTEM:\n"
        f"Tier A (weight 2.0): {EVIDENCE_TIER_EXAMPLES['A']}\n"
        f"Tier B (weight 1.6): {EVIDENCE_TIER_EXAMPLES['B']}\n"
        f"Tier C (weight 0.8): {EVIDENCE_TIER_EXAMPLES['C']}\n"
        f"Tier D (weight 0.3): {EVIDENCE_TIER_EXAMPLES['D']}\n"
        f"\nCORRELATION COLLAPSE: If multiple evidence items cite the same underlying "
        f"fact/story/source, count only the highest-tier instance.\n"
    )

    # Build VIX regime context (Phase 7: VIX Regime Conditioning)
    vix_context = ""
    if macro_signals:
        vix_value = getattr(macro_signals, 'vix', None)
        vix_label = getattr(macro_signals, 'vix_label', None)
        vix_kelly_mult = getattr(macro_signals, 'vix_kelly_mult', 1.0)

        if vix_value is not None and vix_label:
            vix_context = f"\n\n━━━ MARKET REGIME (VIX) ━━━\n"
            vix_context += f"VIX: {vix_value:.1f} — Regime: {vix_label.upper()}\n"
            vix_context += f"Kelly multiplier: {vix_kelly_mult:.2f}x\n"

            if vix_value > 30:
                vix_context += "⚠️  HIGH VOLATILITY: Reduce confidence on tail-risk markets (p̂ < 0.10 or > 0.90)\n"
                vix_context += "   Markets in crisis regimes exhibit larger mispricings but higher noise.\n"
            elif vix_value > 25:
                vix_context += "⚠️  ELEVATED VOLATILITY: Exercise caution on extreme edge signals (>20%).\n"
            else:
                vix_context += "✅  NORMAL VOLATILITY: Standard confidence thresholds apply.\n"

    calibration = ""
    if learning_context:
        calibration = f"\n\n━━━ ASTRA CALIBRATION MEMORY ━━━\n{learning_context}"

    market_json = json.dumps(market_list, indent=2)
    base_msg = (
        f"Date: {today}\n\n"
        f"MARKETS:\n{market_json}\n\n"
        f"Return ONLY a valid JSON array. No markdown. No explanation outside JSON."
    )

    # ── Step 1: Inject few-shot examples and prepare systems ────────────────
    few_shot = _build_few_shot_examples(learning_agent)

    # ── Step 2: Run PRO and CON agents in parallel ──────────────────────────
    pro_system = ASTRA_PRO_SYSTEM + evidence_context + vix_context + few_shot + calibration
    con_system = ASTRA_CON_SYSTEM + evidence_context + vix_context + few_shot + calibration

    pro_msg = (
        f"Date: {today}\n\n"
        f"Build the strongest possible YES case for each market.\n\n"
        f"MARKETS:\n{market_json}\n\n"
        f"Return ONLY a valid JSON array."
    )
    con_msg = (
        f"Date: {today}\n\n"
        f"Build the strongest possible NO case for each market.\n\n"
        f"MARKETS:\n{market_json}\n\n"
        f"Return ONLY a valid JSON array."
    )

    try:
        pro_response, con_response = await asyncio.gather(
            client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": pro_msg}],
                system=pro_system,
            ),
            client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": con_msg}],
                system=con_system,
            ),
        )

        pro_text = _clean_json(pro_response.content[0].text)
        con_text = _clean_json(con_response.content[0].text)
        pro_data = json.loads(pro_text)
        con_data = json.loads(con_text)

        # ── Step 3: Synthesize PRO + CON into final estimate ────────────────
        synth_system = ASTRA_SYNTHESIZER_SYSTEM + evidence_context + vix_context + few_shot + calibration

        # Build synthesis input: merge PRO and CON side by side
        synthesis_input = []
        pro_by_id = {item.get("id"): item for item in pro_data}
        con_by_id = {item.get("id"): item for item in con_data}

        for entry in market_list:
            cid = entry["id"]
            synthesis_input.append({
                "market": entry,
                "pro_analysis": pro_by_id.get(cid, {}),
                "con_analysis": con_by_id.get(cid, {}),
            })

        synth_msg = (
            f"Date: {today}\n\n"
            f"Synthesize the PRO and CON analyses into calibrated final estimates.\n\n"
            f"PRO_CON_DATA:\n{json.dumps(synthesis_input, indent=2)}\n\n"
            f"Return ONLY a valid JSON array."
        )

        synth_response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=12288,  # 12k: batch of 4 markets fits comfortably without truncation
            messages=[{"role": "user", "content": synth_msg}],
            system=synth_system,
        )

        synth_text = _clean_json(synth_response.content[0].text)
        try:
            synth_parsed = json.loads(synth_text)
        except json.JSONDecodeError as json_err:
            # Response was truncated or malformed — log char count and retry with smaller batch
            char_count = len(synth_response.content[0].text)
            logger.warning(
                "Synthesizer JSON parse failed (%s) — response was %d chars (max_tokens=12288). "
                "Batch of %d markets may be too large. Falling back to single-pass.",
                json_err, char_count, len(markets_and_estimates)
            )
            return await _astra_batch_single_pass(markets_and_estimates, learning_context, learning_agent, macro_signals)

        # ── Schema validator (S1 quality gate) ───────────────────────────────
        # Validates that synthesizer actually applied the adversarial framework.
        # Logs quality failures so we can track synthesis degradation over time.
        valid_items, quality_failures = _validate_adversarial_schema(synth_parsed)
        if quality_failures:
            logger.warning(
                "Adversarial schema failures in %d/%d items: %s — using valid items only",
                len(quality_failures), len(synth_parsed),
                quality_failures[:3]   # Log first 3 failures
            )
        synth_parsed = valid_items  # Only pass schema-valid items downstream

        return await _build_estimates_from_adversarial(
            synth_parsed, markets_and_estimates, pro_by_id, con_by_id, macro_signals
        )

    except Exception as e:
        # Fallback: single-pass Astra V2 if adversarial pipeline fails
        logger.warning("Adversarial pipeline failed (%s: %s) — falling back to single-pass", type(e).__name__, e)
        return await _astra_batch_single_pass(markets_and_estimates, learning_context, learning_agent, macro_signals)


async def _build_estimates_from_adversarial(
    synth_parsed: list,
    markets_and_estimates: list[tuple[Market, Optional[Estimate]]],
    pro_by_id: dict,
    con_by_id: dict,
    macro_signals=None,
) -> list[Estimate]:
    """Convert synthesizer JSON output into Estimate objects."""
    market_map = {m.condition_id: m for m, _ in markets_and_estimates}
    results: list[Estimate] = []

    for item in synth_parsed:
        cid = item.get("id")
        if not cid or cid not in market_map:
            continue
        market = market_map[cid]

        p = float(item.get("p_hat", 0.5))
        p_low = float(item.get("p_low", max(0.0, p - 0.1)))
        p_high = float(item.get("p_high", min(1.0, p + 0.1)))
        conf = float(item.get("confidence", 0.5))
        edge = float(item.get("edge", p - market.yes_price))
        ev = float(item.get("ev_after_costs", edge - 0.005))
        rob = int(item.get("robustness_score", 2))
        trap_flags = item.get("trap_flags", [])
        no_trade = bool(item.get("no_trade", False))
        kelly = float(item.get("kelly_position_pct", 0.0))
        stake = int(item.get("stake", 50000))   # Default to high stake if not provided

        # ── PHASE 9: Verification Loop for Low-Confidence Estimates ───────────
        # Research backing: AlphaQuanter (2025) — verification loop improves
        # performance 40% by catching false positives on low-liquidity markets.
        # If confidence < 0.70 AND edge > 0.10, trigger secondary Claude pass.
        if conf < 0.70 and abs(edge) > 0.10 and not no_trade:
            market_type = item.get("market_type", "Unknown")
            additional_context = await _fetch_verification_data(market, market_type, macro_signals)
            refined = await _verify_estimate_with_claude(market, p, conf, additional_context)

            if refined and refined.get("confidence") is not None:
                new_conf = float(refined["confidence"])
                new_p = float(refined.get("p_hat", p))

                # If verification boosts confidence above threshold, use it
                if new_conf >= 0.70:
                    logger.info(
                        f"Verification passed for {market.question[:60]}: "
                        f"conf {conf:.2f}→{new_conf:.2f}, p̂ {p:.3f}→{new_p:.3f}"
                    )
                    p = new_p
                    conf = new_conf
                else:
                    # Confidence still low after verification
                    # If edge is now tiny (<5%), veto the trade
                    new_edge = new_p - market.yes_price
                    if abs(new_edge) < 0.05:
                        no_trade = True
                        if not item.get("no_trade_reason"):
                            item["no_trade_reason"] = (
                                f"Low confidence ({new_conf:.2f}) + small edge "
                                f"({new_edge:+.2%}) after verification"
                            )
                        logger.info(
                            f"Verification failed for {market.question[:60]}: "
                            f"vetoing trade (conf={new_conf:.2f}, edge={new_edge:+.2%})"
                        )

        # ── Stake-based confidence filter (arXiv:2512.05998) ─────────────────
        # Only trade when model's implied stake exceeds threshold.
        # High stakes (≥30k) correlate with ~99% accuracy; low stakes with ~74%.
        STAKE_THRESHOLD = 30000
        if stake < STAKE_THRESHOLD:
            no_trade = True
            if not item.get("no_trade_reason"):
                item["no_trade_reason"] = f"Low stake signal ({stake:,} < {STAKE_THRESHOLD:,}) — model lacks conviction"

        # ── Market timing filter (arXiv:2510.17638) ───────────────────────────
        # LLMs degrade near resolution as real-time news dominates prices.
        if market.hours_to_expiry < 48:
            no_trade = True
            if not item.get("no_trade_reason"):
                item["no_trade_reason"] = f"Market resolves in {market.hours_to_expiry:.0f}h — LLM info edge likely priced in"
        elif market.hours_to_expiry < 168:   # < 1 week
            conf = max(0.0, conf - 0.05)     # Reduce confidence near resolution

        # Enforce V2 rule: 2+ trap flags + conf < 0.75 → no_trade
        if len(trap_flags) >= 2 and conf < 0.75:
            no_trade = True

        # Code-level hallucination guard: sports markets with no real data.
        # If our p_hat is >5× the market price on a sub-20¢ sports market,
        # the model hallucinated (no Tier A sports data means it cannot
        # legitimately give >3× the market price).
        if (market.category == "sports"
                and market.yes_price < 0.20
                and p > market.yes_price * 5.0):
            # Anchor to market price with a small structural adjustment
            p = round(market.yes_price * 1.05, 4)  # allow only 5% adjustment
            conf = min(conf, 0.52)                  # very low confidence
            no_trade = True
            item["no_trade_reason"] = (
                f"Sports hallucination guard: p_hat={item.get('p_hat', 0):.3f} "
                f"vs market={market.yes_price:.3f} (>{5}× — no real sports data)"
            )

        # ── Extremizing aggregation (arXiv:1406.2148) ────────────────────────
        # When PRO and CON agents provide independent estimates that are then
        # averaged, the true probability is more extreme than the average.
        # Apply only when confidence > 0.6 (genuine independent signal).
        if conf > 0.60 and 0.05 < p < 0.95:
            p = _extremize(p, market.yes_price, k=EXTREMIZE_K)  # from config (default 1.25, tunable by learning agent)
            # Recalculate edge and EV after extremizing
            edge = p - market.yes_price
            ev = edge - 0.005

        # pNeutral and pAware from adversarial synthesis
        p_neutral = float(item.get("p_neutral", p))
        p_aware = float(item.get("p_aware", p))

        # Evidence tier from adversarial analysis
        dominant_tier = item.get("dominant_evidence_tier", "C")
        collapses = int(item.get("correlation_collapses", 0))

        # Combine PRO + CON reasoning for audit trail
        pro_summ = item.get("pro_summary", "")
        con_summ = item.get("con_summary", "")
        synthesis = item.get("synthesis_reasoning", "")
        full_reasoning = synthesis if synthesis else f"PRO: {pro_summ} | CON: {con_summ}"

        results.append(Estimate(
            market_condition_id=cid,
            question=market.question,
            category=market.category,
            probability=p,
            probability_low=p_low,
            probability_high=p_high,
            confidence=conf,
            market_type=item.get("market_type", market.category.title()),
            modeling_approach=item.get("modeling_approach", "B"),
            trap_flags=trap_flags,
            edge=round(edge, 4),
            ev_after_costs=round(ev, 4),
            robustness_score=rob,
            kelly_position_pct=kelly if not no_trade else 0.0,
            source="astra_v2_adversarial",
            truth_state=item.get("truth_state", "Assumed"),
            reasoning=full_reasoning[:300],
            key_evidence_needed=item.get("key_evidence_needed", ""),
            no_trade=no_trade,
            no_trade_reason=item.get("no_trade_reason", ""),
            details={
                "model": CLAUDE_MODEL,
                "adversarial": True,
                "dominant_evidence_tier": dominant_tier,
                "correlation_collapses": collapses,
                "p_neutral": p_neutral,
                "p_aware": p_aware,
                "edge_signal": round(abs(p_neutral - market.yes_price), 4),
                "raw": item,
            },
            # Adversarial-specific fields
            p_neutral=p_neutral,
            p_aware=p_aware,
            pro_summary=pro_summ,
            con_summary=con_summ,
            dominant_evidence_tier=dominant_tier,
            correlation_collapses=collapses,
            adversarial_mode=True,
        ))

    return results


async def _astra_batch_single_pass(
    markets_and_estimates: list[tuple[Market, Optional[Estimate]]],
    learning_context: str,
    learning_agent=None,
    macro_signals=None,
) -> list[Estimate]:
    """Fallback: single-pass Astra V2 (original behaviour)."""
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    market_list = []
    for market, algo_est in markets_and_estimates:
        entry = {
            "id": market.condition_id,
            "question": market.question,
            "category": market.category,
            "yes_price": market.yes_price,
            "implied_prob": round(market.yes_price, 4),
            "hours_to_expiry": round(market.hours_to_expiry, 1),
            "liquidity_usd": round(market.liquidity, 0),
            "volume_usd": round(market.volume, 0),
        }
        if algo_est:
            entry["tier1_p_hat"] = algo_est.probability
            entry["tier1_confidence"] = algo_est.confidence
        market_list.append(entry)

    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    system = ASTRA_V2_SYSTEM
    if learning_context:
        system += f"\n\n━━━ ASTRA CALIBRATION MEMORY ━━━\n{learning_context}"

    user_msg = (
        f"Date: {today}\n\n"
        f"Evaluate these Polymarket markets using Astra V2's full operating loop.\n\n"
        f"MARKETS:\n{json.dumps(market_list, indent=2)}\n\n"
        f"Return ONLY a valid JSON array. No markdown. No explanation outside JSON."
    )

    results: list[Estimate] = []
    try:
        response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": user_msg}],
            system=system,
        )

        text = _clean_json(response.content[0].text)
        parsed = json.loads(text)

        market_map = {m.condition_id: m for m, _ in markets_and_estimates}

        for item in parsed:
            cid = item.get("id")
            if not cid or cid not in market_map:
                continue
            market = market_map[cid]
            p = float(item.get("p_hat", 0.5))
            p_low = float(item.get("p_low", max(0.0, p - 0.1)))
            p_high = float(item.get("p_high", min(1.0, p + 0.1)))
            conf = float(item.get("confidence", 0.5))
            edge = float(item.get("edge", p - market.yes_price))
            ev = float(item.get("ev_after_costs", edge - 0.005))
            rob = int(item.get("robustness_score", 2))
            trap_flags = item.get("trap_flags", [])
            no_trade = bool(item.get("no_trade", False))
            kelly = float(item.get("kelly_position_pct", 0.0))

            if len(trap_flags) >= 2 and conf < 0.75:
                no_trade = True

            # Sports hallucination guard (same as adversarial path)
            if (market.category == "sports"
                    and market.yes_price < 0.20
                    and p > market.yes_price * 5.0):
                p = round(market.yes_price * 1.05, 4)
                conf = min(conf, 0.52)
                no_trade = True

            results.append(Estimate(
                market_condition_id=cid,
                question=market.question,
                category=market.category,
                probability=p,
                probability_low=p_low,
                probability_high=p_high,
                confidence=conf,
                market_type=item.get("market_type", market.category.title()),
                modeling_approach=item.get("modeling_approach", "B"),
                trap_flags=trap_flags,
                edge=round(edge, 4),
                ev_after_costs=round(ev, 4),
                robustness_score=rob,
                kelly_position_pct=kelly if not no_trade else 0.0,
                source="astra_v2",
                truth_state=item.get("truth_state", "Assumed"),
                reasoning=item.get("reasoning", ""),
                key_evidence_needed=item.get("key_evidence_needed", ""),
                no_trade=no_trade,
                no_trade_reason=item.get("no_trade_reason", ""),
                details={"model": CLAUDE_MODEL, "raw": item},
            ))

    except Exception as e:
        logger.error("Single-pass estimation failed (%s: %s) — %d markets unestimated",
                     type(e).__name__, e, len(markets_and_estimates))

    return results


def _validate_adversarial_schema(items: list) -> tuple[list, list]:
    """
    S1 quality gate: validate synthesizer output contains required adversarial fields.

    Returns (valid_items, failure_descriptions).
    Logs which fields are missing so we can track synthesis quality degradation.
    """
    REQUIRED_FIELDS = {"id", "p_hat", "confidence", "dominant_evidence_tier"}
    VALID_TIERS = {"A", "B", "C", "D"}

    valid = []
    failures = []

    for item in items:
        if not isinstance(item, dict):
            failures.append(f"non-dict item: {type(item).__name__}")
            continue

        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            failures.append(f"id={item.get('id', '?')[:12]} missing: {missing}")
            continue

        # Validate dominant_evidence_tier is a real tier
        tier = item.get("dominant_evidence_tier", "")
        if tier not in VALID_TIERS:
            # Don't reject — just fix the tier to "C" (default)
            item["dominant_evidence_tier"] = "C"
            failures.append(f"id={item.get('id', '?')[:12]} invalid tier '{tier}' → defaulted to C")

        # Validate p_hat is in range
        p = item.get("p_hat", None)
        if p is None or not (0.0 <= float(p) <= 1.0):
            failures.append(f"id={item.get('id', '?')[:12]} invalid p_hat={p}")
            continue

        # Validate stake is present (default to high if missing to avoid false vetoes)
        if "stake" not in item:
            item["stake"] = 50000  # Default to high stake — model omitted it

        valid.append(item)

    return valid, failures


def _clean_json(text: str) -> str:
    """
    Strip markdown code fences and repair common JSON issues from Claude responses.

    Handles:
    - ```json ... ``` code fences
    - Trailing commas before ] or } (common LLM error)
    - Control characters in string values
    - Truncated responses (extract valid JSON array prefix)
    """
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Remove trailing commas before closing brackets (common LLM JSON error)
    text = re.sub(r",\s*([\]\}])", r"\1", text)

    # Remove control characters that break JSON parsing
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # If response is a JSON array that got truncated, try to recover
    # by finding the last complete object and closing the array
    if text.startswith("[") and not text.endswith("]"):
        # Find last complete object boundary
        depth = 0
        last_complete = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_complete = i + 1
        if last_complete > 0:
            text = text[:last_complete] + "]"
            logger.debug("Recovered truncated JSON array at position %d", last_complete)

    return text
