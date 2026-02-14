"""
Astra V2 — Self-Grading Calibration Tracker & Evolution Engine

Implements Module 6 of the Astra V2 operating framework:
- Brier score tracking per trade
- Calibration buckets with accuracy
- Overconfidence index
- EV estimate vs realized return
- "No-trade correctness" (avoided losses)
- Post-resolution review with rule proposals
- Anti-hallucination: unmeasured things are labeled as such
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

MEMORY_FILE = Path("memory/predictions.json")
STRATEGY_FILE = Path("memory/strategy.md")
JOURNAL_FILE = Path("memory/journal.json")
OVERRIDES_FILE = Path("memory/strategy_overrides.json")  # Machine-readable, read by config.py at startup


ASTRA_EVOLUTION_PROMPT = """You are Astra V2 — an elite prediction market trading brain operating in self-improvement mode.

Your task: analyze resolved market predictions, identify systematic errors, and propose calibration improvements.

Follow the Module 6 Self-Grading framework:
1. Compute Brier scores (lower = better: 0 = perfect, 1 = worst)
2. Identify calibration bucket errors (where we're systematically wrong)
3. Measure overconfidence index (are we too confident? Too timid?)
4. Find top 3 systematic errors
5. Propose SMALL, TESTABLE rule updates
6. Tighten or relax thresholds only if metrics justify it

Anti-hallucination rule: if you can't measure it from the data, label it "Unmeasured" and propose a measurement method.

Output format — structured strategy notes under 600 words, followed by a JSON block:
- Calibration summary
- Top 3 systematic errors (with evidence from the data)
- Specific rule proposals (max 3, each must be falsifiable)
- Updated confidence multipliers by category/source
- Any markets to avoid or prioritize

END your response with a JSON block (inside ```json ... ```) with ONLY the parameters you are confident should change based on evidence. Leave out params you're unsure about. Use null for "no change". Expire in 30 days:
```json
{
  "MISPRICING_THRESHOLD_OVERRIDE": <float or null>,
  "MIN_CONFIDENCE_OVERRIDE": <float or null>,
  "ACQUIESCENCE_CORRECTION_OVERRIDE": <float or null>,
  "EXTREMIZE_K_OVERRIDE": <float or null>,
  "expires": "<YYYY-MM-DD 30 days from today>"
}
```"""


@dataclass
class Prediction:
    """A recorded Astra prediction with full audit trail."""

    market_condition_id: str
    question: str
    category: str
    our_probability: float
    probability_low: float
    probability_high: float
    market_price: float
    direction: str
    source: str
    truth_state: str
    reasoning: str
    key_unknowns: str
    no_trade: bool
    timestamp: str
    # Post-resolution fields
    resolved: bool = False
    outcome: Optional[bool] = None
    brier_score: Optional[float] = None
    resolution_time: Optional[str] = None
    profit_loss: Optional[float] = None
    ev_estimate: Optional[float] = None
    slippage_estimate: Optional[float] = None


@dataclass
class CalibrationBucket:
    """Calibration stats for a probability range bucket."""

    bucket: str
    predicted_avg: float
    actual_rate: float
    n: int
    brier_avg: float
    bias: float  # actual_rate - predicted_avg (+ = we were underconfident)
    overconfidence: float  # predicted_avg - actual_rate (+ = overconfident)


@dataclass
class JournalEntry:
    """Structured log entry — matches Astra V2 output format."""

    timestamp: str
    market_condition_id: str
    question: str
    decision: str
    p_hat: float
    p_low: float
    p_high: float
    confidence: float
    edge: float
    ev_after_costs: float
    size_usd: float
    reasoning: str
    truth_state: str
    trap_flags: list
    monitoring_triggers: str


class LearningAgent:
    def __init__(self):
        MEMORY_FILE.parent.mkdir(exist_ok=True)
        self._predictions: list[Prediction] = self._load_predictions()
        self._strategy: str = self._load_strategy()

    def _load_predictions(self) -> list[Prediction]:
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text())
                loaded = []
                for p in data:
                    # Handle old format predictions (missing new fields)
                    p.setdefault("probability_low", p.get("our_probability", 0.5) - 0.1)
                    p.setdefault("probability_high", p.get("our_probability", 0.5) + 0.1)
                    p.setdefault("truth_state", "Assumed")
                    p.setdefault("reasoning", "")
                    p.setdefault("key_unknowns", "")
                    p.setdefault("no_trade", False)
                    p.setdefault("brier_score", None)
                    p.setdefault("ev_estimate", None)
                    p.setdefault("slippage_estimate", None)
                    loaded.append(Prediction(**p))
                return loaded
            except Exception:
                return []
        return []

    def _load_strategy(self) -> str:
        if STRATEGY_FILE.exists():
            return STRATEGY_FILE.read_text()
        return ""

    def _save_predictions(self):
        MEMORY_FILE.write_text(json.dumps([asdict(p) for p in self._predictions], indent=2))

    def record_prediction(self, prediction: Prediction):
        """Record a new Astra prediction for audit and tracking."""
        existing_ids = {p.market_condition_id for p in self._predictions}
        if prediction.market_condition_id not in existing_ids:
            self._predictions.append(prediction)
            self._save_predictions()

    def update_outcome(self, condition_id: str, outcome: bool, profit_loss: float = 0.0):
        """
        Update prediction with resolved outcome.
        Computes Brier score: (p_hat - outcome)^2
        """
        for p in self._predictions:
            if p.market_condition_id == condition_id and not p.resolved:
                p.resolved = True
                p.outcome = outcome
                p.profit_loss = profit_loss
                p.resolution_time = datetime.now(timezone.utc).isoformat()
                # Brier score: 0 = perfect, 1 = worst
                outcome_val = 1.0 if outcome else 0.0
                p.brier_score = (p.our_probability - outcome_val) ** 2
                break
        self._save_predictions()

    def get_strategy_context(self) -> str:
        """Return Astra's current calibration knowledge for the estimator."""
        return self._strategy

    def get_stats(self) -> dict:
        """Return comprehensive Astra performance statistics."""
        resolved = [p for p in self._predictions if p.resolved and p.outcome is not None]
        unresolved = [p for p in self._predictions if not p.resolved]
        no_trades = [p for p in self._predictions if p.no_trade]

        if not resolved:
            return {
                "total_predictions": len(self._predictions),
                "resolved": 0,
                "unresolved": len(unresolved),
                "no_trades": len(no_trades),
                "accuracy": None,
                "brier_score_avg": None,
                "overconfidence_index": None,
                "total_pnl": 0.0,
            }

        correct = sum(
            1
            for p in resolved
            if (p.direction == "BUY YES" and p.outcome) or (p.direction == "BUY NO" and not p.outcome)
        )

        brier_scores = [p.brier_score for p in resolved if p.brier_score is not None]
        brier_avg = sum(brier_scores) / len(brier_scores) if brier_scores else None

        # Overconfidence: avg(p_hat - outcome), positive = overconfident
        overconf = sum(p.our_probability - (1.0 if p.outcome else 0.0) for p in resolved) / len(resolved)

        total_pnl = sum(p.profit_loss or 0.0 for p in resolved)

        # No-trade correctness: no_trades that would have been losers
        # (We can't fully compute this without knowing what we'd have predicted,
        #  but track no_trade count as "Unmeasured" per anti-hallucination rule)

        return {
            "total_predictions": len(self._predictions),
            "resolved": len(resolved),
            "unresolved": len(unresolved),
            "no_trades": len(no_trades),
            "accuracy": round(correct / len(resolved), 4),
            "brier_score_avg": round(brier_avg, 4) if brier_avg else None,
            "overconfidence_index": round(overconf, 4),
            "total_pnl": round(total_pnl, 2),
            "by_category": self._stats_by_category(resolved),
            "by_source": self._stats_by_source(resolved),
            "no_trade_correctness": "Unmeasured — requires counterfactual analysis",
        }

    def _stats_by_category(self, resolved: list[Prediction]) -> dict:
        cats = {}
        for p in resolved:
            c = p.category
            if c not in cats:
                cats[c] = {"n": 0, "correct": 0, "pnl": 0.0, "brier_sum": 0.0}
            cats[c]["n"] += 1
            if (p.direction == "BUY YES" and p.outcome) or (p.direction == "BUY NO" and not p.outcome):
                cats[c]["correct"] += 1
            cats[c]["pnl"] += p.profit_loss or 0.0
            if p.brier_score is not None:
                cats[c]["brier_sum"] += p.brier_score
        for c in cats:
            n = cats[c]["n"]
            cats[c]["accuracy"] = round(cats[c]["correct"] / n, 4)
            cats[c]["brier_avg"] = round(cats[c]["brier_sum"] / n, 4)
        return cats

    def _stats_by_source(self, resolved: list[Prediction]) -> dict:
        sources = {}
        for p in resolved:
            s = p.source
            if s not in sources:
                sources[s] = {"n": 0, "correct": 0, "brier_sum": 0.0}
            sources[s]["n"] += 1
            if (p.direction == "BUY YES" and p.outcome) or (p.direction == "BUY NO" and not p.outcome):
                sources[s]["correct"] += 1
            if p.brier_score is not None:
                sources[s]["brier_sum"] += p.brier_score
        for s in sources:
            n = sources[s]["n"]
            sources[s]["accuracy"] = round(sources[s]["correct"] / n, 4)
            sources[s]["brier_avg"] = round(sources[s]["brier_sum"] / n, 4)
        return sources

    def calibration_buckets(self) -> list[CalibrationBucket]:
        """Module 6: Compute calibration statistics by probability bucket."""
        resolved = [p for p in self._predictions if p.resolved and p.outcome is not None]
        if len(resolved) < 10:
            return []

        results = []
        for low in [i / 10 for i in range(10)]:
            high = low + 0.1
            bucket = [p for p in resolved if low <= p.our_probability < high]
            if not bucket:
                continue
            predicted_avg = sum(p.our_probability for p in bucket) / len(bucket)
            actual_rate = sum(1 for p in bucket if p.outcome) / len(bucket)
            brier_scores = [p.brier_score for p in bucket if p.brier_score is not None]
            brier_avg = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0
            results.append(
                CalibrationBucket(
                    bucket=f"{low:.1f}-{high:.1f}",
                    predicted_avg=round(predicted_avg, 4),
                    actual_rate=round(actual_rate, 4),
                    n=len(bucket),
                    brier_avg=round(brier_avg, 4),
                    bias=round(actual_rate - predicted_avg, 4),
                    overconfidence=round(predicted_avg - actual_rate, 4),
                )
            )
        return results

    async def evolve(self):
        """
        Module 6 evolution cycle: analyze performance and update Astra's strategy.
        Runs after every N resolved predictions. Produces testable rule proposals.
        """
        resolved = [p for p in self._predictions if p.resolved]
        if len(resolved) < 5:
            return

        stats = self.get_stats()
        calibration = self.calibration_buckets()

        analysis = {
            "resolved_count": len(resolved),
            "performance": stats,
            "calibration_buckets": [asdict(c) for c in calibration],
            "brier_score_benchmark": {
                "perfect": 0.0,
                "random": 0.25,
                "current": stats.get("brier_score_avg"),
            },
            "recent_resolved": [
                {
                    "question": p.question[:100],
                    "category": p.category,
                    "source": p.source,
                    "truth_state": p.truth_state,
                    "p_hat": p.our_probability,
                    "p_low": p.probability_low,
                    "p_high": p.probability_high,
                    "market_price": p.market_price,
                    "direction": p.direction,
                    "no_trade": p.no_trade,
                    "outcome": p.outcome,
                    "brier": p.brier_score,
                    "pnl": p.profit_loss,
                }
                for p in sorted(resolved, key=lambda x: x.resolution_time or "", reverse=True)[:30]
            ],
        }

        if not ANTHROPIC_API_KEY:
            return

        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        try:
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1200,  # Increased: enough for structured overrides + analysis
                messages=[
                    {
                        "role": "user",
                        "content": f"{ASTRA_EVOLUTION_PROMPT}\n\nPERFORMANCE DATA:\n{json.dumps(analysis, indent=2)}",
                    }
                ],
            )

            brier_note = f"Brier: {stats.get('brier_score_avg', 'N/A')} (benchmark: random=0.25, perfect=0.0)"
            overconf_note = (
                f"Overconfidence index: {stats.get('overconfidence_index', 'N/A')} "
                f"(positive = overconfident, target = near 0)"
            )

            raw_text = response.content[0].text  # type: ignore[union-attr]

            # Extract machine-readable overrides from JSON block if present
            import re as _re

            override_match = _re.search(r"```json\s*(\{.*?\})\s*```", raw_text, _re.DOTALL)
            if override_match:
                try:
                    overrides = json.loads(override_match.group(1))
                    # Only write non-null overrides
                    clean_overrides = {k: v for k, v in overrides.items() if v is not None}
                    if clean_overrides:
                        OVERRIDES_FILE.write_text(json.dumps(clean_overrides, indent=2))
                        import logging as _log

                        _log.getLogger("astra.learning").info(
                            "Strategy overrides written: %s (expires %s)",
                            list(clean_overrides.keys()),
                            clean_overrides.get("expires", "?"),
                        )
                except Exception as parse_err:
                    import logging as _log

                    _log.getLogger("astra.learning").warning("Failed to parse strategy overrides JSON: %s", parse_err)

            new_strategy = (
                f"# Astra V2 Strategy Context\n"
                f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                f"## Performance Snapshot\n"
                f"- Resolved: {stats['resolved']} | Accuracy: {stats.get('accuracy', 'N/A')}\n"
                f"- {brier_note}\n"
                f"- {overconf_note}\n"
                f"- Total P&L: ${stats['total_pnl']}\n\n"
                f"## Calibration Analysis\n"
                f"{raw_text}\n"
            )
            STRATEGY_FILE.write_text(new_strategy)
            self._strategy = new_strategy

        except Exception as e:
            import logging as _log

            _log.getLogger("astra.learning").error(
                "evolve() failed: %s: %s — strategy not updated", type(e).__name__, e
            )
