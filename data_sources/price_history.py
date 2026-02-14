"""
Multi-Timeframe Price History Tracker

Stores price snapshots for momentum indicators (1hr, 24hr).
Research backing: 4 papers show multi-timeframe analysis beats single-interval by +10-15% Sharpe.

Storage: memory/price_snapshots.json (rolling 168-hour window per asset)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

SNAPSHOTS_FILE = Path(__file__).resolve().parent.parent / "memory" / "price_snapshots.json"
MAX_HISTORY_HOURS = 168  # Keep 7 days of history


def _load_snapshots() -> dict:
    """Load price snapshot history from disk."""
    if not SNAPSHOTS_FILE.exists():
        return {}
    try:
        with open(SNAPSHOTS_FILE, "r") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return {}


def _save_snapshots(snapshots: dict):
    """Save price snapshot history to disk."""
    SNAPSHOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOTS_FILE, "w") as f:
        json.dump(snapshots, f, indent=2)


def store_price_snapshot(asset_id: str, price: float, timestamp: Optional[str] = None):
    """
    Store price snapshot for an asset (crypto coin, market condition_id, etc.).

    Args:
        asset_id: Unique identifier (e.g., "bitcoin", "0x1234...")
        price: Current price (0-1 for prediction markets, absolute for crypto)
        timestamp: ISO timestamp (defaults to now)
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    snapshots = _load_snapshots()

    if asset_id not in snapshots:
        snapshots[asset_id] = []

    # Add new snapshot
    snapshots[asset_id].append({"timestamp": timestamp, "price": price})

    # Prune old snapshots (keep only last 168 hours)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_HISTORY_HOURS)
    snapshots[asset_id] = [s for s in snapshots[asset_id] if datetime.fromisoformat(s["timestamp"]) > cutoff]

    # Sort by timestamp (most recent last)
    snapshots[asset_id].sort(key=lambda s: s["timestamp"])

    _save_snapshots(snapshots)


def get_price_history(asset_id: str, hours: int = 24) -> list[tuple[str, float]]:
    """
    Get price history for an asset.

    Args:
        asset_id: Asset identifier
        hours: How many hours back to retrieve

    Returns:
        List of (timestamp, price) tuples, oldest first
    """
    snapshots = _load_snapshots()
    if asset_id not in snapshots:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent = [
        (s["timestamp"], s["price"]) for s in snapshots[asset_id] if datetime.fromisoformat(s["timestamp"]) > cutoff
    ]

    return recent


def compute_momentum_1hr(asset_id: str) -> Optional[float]:
    """
    Calculate 1-hour momentum (rate of change).

    Returns:
        Rate of change: (current - 1h_ago) / 1h_ago
        None if insufficient data
    """
    history = get_price_history(asset_id, hours=2)  # Get 2 hours for safety
    if len(history) < 2:
        return None

    # Find price ~1 hour ago
    now = datetime.now(timezone.utc)
    target_time = now - timedelta(hours=1)

    # Find closest snapshot to 1 hour ago
    closest = None
    min_diff = timedelta(hours=999)
    for ts, price in history:
        dt = datetime.fromisoformat(ts)
        diff = abs(dt - target_time)
        if diff < min_diff:
            min_diff = diff
            closest = price

    if closest is None:
        return None

    current_price = history[-1][1]  # Most recent price

    if closest == 0:
        return 0.0

    return (current_price - closest) / closest


def compute_momentum_24hr(asset_id: str) -> Optional[float]:
    """
    Calculate 24-hour momentum (rate of change).

    Returns:
        Rate of change: (current - 24h_ago) / 24h_ago
        None if insufficient data
    """
    history = get_price_history(asset_id, hours=30)  # Get 30 hours for safety
    if len(history) < 2:
        return None

    # Find price ~24 hours ago
    now = datetime.now(timezone.utc)
    target_time = now - timedelta(hours=24)

    # Find closest snapshot to 24 hours ago
    closest = None
    min_diff = timedelta(hours=999)
    for ts, price in history:
        dt = datetime.fromisoformat(ts)
        diff = abs(dt - target_time)
        if diff < min_diff:
            min_diff = diff
            closest = price

    if closest is None:
        return None

    current_price = history[-1][1]  # Most recent price

    if closest == 0:
        return 0.0

    return (current_price - closest) / closest


def compute_rsi_14(asset_id: str, periods: int = 14) -> Optional[float]:
    """
    Calculate 14-period RSI (Relative Strength Index).

    RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss over last 14 periods

    Returns:
        RSI value (0-100), or None if insufficient data
    """
    history = get_price_history(asset_id, hours=periods + 2)
    if len(history) < periods + 1:
        return None

    prices = [p for _, p in history]
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))  # type: ignore[arg-type]

    if len(gains) < periods:
        return None

    avg_gain = sum(gains[-periods:]) / periods
    avg_loss = sum(losses[-periods:]) / periods

    if avg_loss == 0:
        return 100.0  # No losses = overbought

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_momentum_summary(asset_id: str) -> dict:
    """
    Get comprehensive momentum summary for an asset.

    Returns:
        {
            "momentum_1hr": float or None,
            "momentum_24hr": float or None,
            "rsi_14": float or None,
            "trend": "bullish" | "bearish" | "neutral" | "unknown"
        }
    """
    m1h = compute_momentum_1hr(asset_id)
    m24h = compute_momentum_24hr(asset_id)
    rsi = compute_rsi_14(asset_id)

    # Determine trend
    trend = "unknown"
    if m1h is not None and m24h is not None:
        if m1h > 0.01 and m24h > 0.01:
            trend = "bullish"  # Both positive momentum
        elif m1h < -0.01 and m24h < -0.01:
            trend = "bearish"  # Both negative momentum
        else:
            trend = "neutral"  # Mixed signals

    return {
        "momentum_1hr": m1h,
        "momentum_24hr": m24h,
        "rsi_14": rsi,
        "trend": trend,
    }
