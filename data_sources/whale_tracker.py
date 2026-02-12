"""
Whale / Large Order Signal — Tier B evidence for Astra V2.

Detects unusual volume spikes on Polymarket markets using the volume data
already returned by the Gamma API (no extra API calls needed).

Strategy:
  - Maintain a rolling per-market volume baseline in memory/volume_baseline.json
  - Each scan: compare current market.volume to the baseline
  - Spike threshold: current volume > SPIKE_MULTIPLIER × baseline (default 3×)
  - On spike: flag as whale_signal with magnitude (how many × above baseline)
  - Inject as Tier B evidence into Astra context: informed money has moved

Why this works:
  - Thin markets (the ones we trade) have predictable low volume
  - A sudden 3× spike almost always means an informed actor took a position
  - This is the copy-trading / whale-following signal used by AdrienAlvarez's bot
    and confirmed by the polymarket-intelligence 6-agent debate dashboard

From research:
  - AdrienAlvarez: monitors top Binance leaderboard → mirrors positions
  - polymarket-intelligence: whale tracking with >$100 orders as signal threshold
  - jaredzwick BaseStrategy: tracks fills and PnL per strategy with metrics

Integration:
  - Call track_volume_and_detect_whales(markets) in run_paper_scan()
  - Returns a dict: {condition_id: WhaleSignal}
  - Inject summary into learning_context before Astra calls
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scanner.market_fetcher import Market


BASELINE_FILE = Path("memory/volume_baseline.json")
SPIKE_MULTIPLIER = 3.0       # Volume must be 3× baseline to flag
MIN_VOLUME_FOR_SIGNAL = 500  # Don't flag markets with <$500 volume (noise)
MIN_BASELINE_SCANS = 3       # Need at least 3 data points to establish baseline
MAX_BASELINE_HISTORY = 20    # Rolling window of volume observations per market


@dataclass
class WhaleSignal:
    condition_id: str
    question: str
    current_volume: float
    baseline_volume: float
    spike_ratio: float         # current / baseline
    evidence_tier: str = "B"   # Tier B: verified data, not primary source
    timestamp: str = ""

    @property
    def summary(self) -> str:
        return (
            f"[WHALE TIER-B] Unusual volume spike on: {self.question[:60]}\n"
            f"  Volume: ${self.current_volume:,.0f} = {self.spike_ratio:.1f}× "
            f"baseline (${self.baseline_volume:,.0f}). "
            f"Informed money may have moved on this market."
        )


def _load_baseline() -> dict:
    """Load per-market volume history from disk."""
    if BASELINE_FILE.exists():
        try:
            return json.loads(BASELINE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_baseline(data: dict):
    """Persist volume baseline to disk."""
    BASELINE_FILE.parent.mkdir(exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(data, indent=2))


def track_volume_and_detect_whales(markets: list[Market]) -> dict[str, WhaleSignal]:
    """
    Main entry point. Call once per scan with the current markets list.

    1. Load existing volume baseline
    2. Update baseline with current volumes
    3. Detect spikes (current > SPIKE_MULTIPLIER × rolling average)
    4. Save updated baseline
    5. Return dict of {condition_id: WhaleSignal} for any detected spikes
    """
    baseline = _load_baseline()
    now_str = datetime.now(timezone.utc).isoformat()
    whale_signals: dict[str, WhaleSignal] = {}

    for market in markets:
        cid = market.condition_id
        vol = market.volume

        if vol <= 0:
            continue

        # Update rolling history for this market
        if cid not in baseline:
            baseline[cid] = {"history": [], "question": market.question[:80]}

        history: list[float] = baseline[cid].get("history", [])
        history.append(vol)

        # Keep only the last N observations
        if len(history) > MAX_BASELINE_HISTORY:
            history = history[-MAX_BASELINE_HISTORY:]
        baseline[cid]["history"] = history
        baseline[cid]["last_updated"] = now_str

        # Need enough history to establish a reliable baseline
        if len(history) < MIN_BASELINE_SCANS:
            continue

        # Baseline = median of all but the last observation
        # (using median avoids previous spikes inflating the baseline)
        previous = sorted(history[:-1])
        mid = len(previous) // 2
        if len(previous) % 2 == 0:
            baseline_vol = (previous[mid - 1] + previous[mid]) / 2
        else:
            baseline_vol = previous[mid]

        if baseline_vol < 1.0:
            continue  # Avoid division by zero on very thin markets

        spike_ratio = vol / baseline_vol

        if spike_ratio >= SPIKE_MULTIPLIER and vol >= MIN_VOLUME_FOR_SIGNAL:
            whale_signals[cid] = WhaleSignal(
                condition_id=cid,
                question=market.question,
                current_volume=vol,
                baseline_volume=round(baseline_vol, 2),
                spike_ratio=round(spike_ratio, 2),
                timestamp=now_str,
            )

    _save_baseline(baseline)
    return whale_signals


def format_whale_context(signals: dict[str, WhaleSignal]) -> str:
    """
    Format whale signals for injection into Astra V2 learning context.
    Returns a compact string ready for prepending to the Astra prompt.
    Labelled as Tier B evidence (verified data, programmatic detection).
    """
    if not signals:
        return ""

    lines = ["=== WHALE SIGNALS (Tier B — Volume Spikes) ==="]
    for sig in sorted(signals.values(), key=lambda s: s.spike_ratio, reverse=True)[:5]:
        lines.append(
            f"  • {sig.question[:70]}\n"
            f"    Volume ${sig.current_volume:,.0f} = {sig.spike_ratio:.1f}× "
            f"baseline — potential large-order activity"
        )
    lines.append(
        "Evidence tier B: volume spike detected from on-chain/API data. "
        "Treat as moderate signal that informed money may have moved."
    )
    return "\n".join(lines)
