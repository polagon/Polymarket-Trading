"""
Markout/Toxicity Tracker - Maker survival module.

CRITICAL FIX #4: This is the BIGGEST maker survival edge.
QS can look great while you're being picked off slowly.

"This single module is often the difference between 'looks amazing in paper' and 'dies live.'"
- ChatGPT
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from config import (
    MARKOUT_INTERVALS,
    TOXIC_FV_BAND_MULTIPLIER,
    TOXIC_MARKOUT_MEAN_VETO,
    TOXIC_MARKOUT_MILD,
    TOXIC_MARKOUT_WINDOW_SIZE,
    TOXIC_SIZE_MULTIPLIER,
)
from models.types import Fill, OrderBook

logger = logging.getLogger(__name__)


class MarkoutTracker:
    """
    Tracks post-fill price drift to detect adverse selection.

    Markout = sign(fill) * (mid_after - fill_price)

    Positive markout = good (price moved in our favor)
    Negative markout = toxic (we bought at top, sold at bottom)
    """

    def __init__(self):
        # Markout history per market
        self.markout_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=TOXIC_MARKOUT_WINDOW_SIZE))

        # Cluster-level markout
        self.cluster_markout_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=TOXIC_MARKOUT_WINDOW_SIZE))

        # Fill tracking for markout computation
        self.pending_fills: List[dict] = []

    def compute_markout(
        self,
        fill: Fill,
        book_snapshots: Dict[int, float],
        intervals: List[int] = MARKOUT_INTERVALS,
    ) -> dict:
        """
        Compute markout at multiple time intervals after fill.

        Markout = sign(fill) * (mid_after - fill_price)

        CRITICAL: book_snapshots must use mid from execution.mid.compute_mid(), not ad-hoc.

        Args:
            fill: Fill object
            book_snapshots: {timestamp_ms: mid_price} (from execution.mid.compute_mid)
            intervals: Seconds after fill to measure [30s, 2m, 10m]

        Returns:
            {
                "markout_30s": float or None,
                "markout_2m": float or None,
                "markout_10m": float or None,
            }
        """
        fill_side = 1 if fill.side == "BUY" else -1  # Buy = +1, Sell = -1
        fill_price = fill.price
        fill_timestamp = fill.timestamp

        markouts = {}  # type: ignore[var-annotated]

        for interval in intervals:
            target_ts = fill_timestamp + (interval * 1000)  # Convert to ms

            # Find closest book snapshot
            closest_mid = self._find_closest_snapshot(book_snapshots, target_ts)

            if closest_mid is None:
                markouts[f"markout_{interval}s"] = None  # type: ignore[assignment]
                continue

            # Compute markout
            markout = fill_side * (closest_mid - fill_price)
            markouts[f"markout_{interval}s"] = markout

        return markouts

    def _find_closest_snapshot(self, book_snapshots: Dict[int, float], target_ts: int) -> Optional[float]:
        """
        Find closest book snapshot to target timestamp.

        Args:
            book_snapshots: {timestamp_ms: mid_price}
            target_ts: Target timestamp (ms)

        Returns:
            Mid price or None if no snapshot found
        """
        if not book_snapshots:
            return None

        closest_ts = min(book_snapshots.keys(), key=lambda ts: abs(ts - target_ts))

        # Only accept snapshots within 30s of target
        if abs(closest_ts - target_ts) > 30000:
            return None

        return book_snapshots[closest_ts]

    def track_fill_markout(
        self,
        market_id: str,
        cluster_id: str,
        markout_2m: float,
    ):
        """
        Track rolling markout for market and cluster.

        Args:
            market_id: Market condition_id
            cluster_id: Cluster ID
            markout_2m: 2-minute markout value
        """
        self.markout_history[market_id].append(markout_2m)
        self.cluster_markout_history[cluster_id].append(markout_2m)

        logger.debug(
            f"Markout tracked: {market_id} markout_2m={markout_2m:.4f} "
            f"(market_count={len(self.markout_history[market_id])}, "
            f"cluster_count={len(self.cluster_markout_history[cluster_id])})"
        )

    def get_markout_stats(self, market_id: str) -> dict:
        """
        Get rolling markout statistics for a market.

        Args:
            market_id: Market condition_id

        Returns:
            {"mean": float, "std": float, "count": int}
        """
        history = self.markout_history[market_id]

        if len(history) < 2:
            return {"mean": 0.0, "std": 0.0, "count": len(history)}

        mean = np.mean(history)
        std = np.std(history)

        return {
            "mean": float(mean),
            "std": float(std),
            "count": len(history),
        }

    def get_cluster_markout_stats(self, cluster_id: str) -> dict:
        """Get cluster-level markout statistics."""
        history = self.cluster_markout_history[cluster_id]

        if len(history) < 2:
            return {"mean": 0.0, "std": 0.0, "count": len(history)}

        mean = np.mean(history)
        std = np.std(history)

        return {
            "mean": float(mean),
            "std": float(std),
            "count": len(history),
        }

    def is_toxic_market(self, market_id: str, cluster_id: str) -> bool:
        """
        Determine if market should be paused due to adverse selection.

        CRITICAL: This OVERRIDES QS for active set selection.

        Triggers:
        - Rolling mean markout < -0.002 (losing 0.2¢ per fill on average)
        - Cluster-level markout deteriorating

        Args:
            market_id: Market condition_id
            cluster_id: Cluster ID

        Returns:
            True if toxic (pause quoting), False if healthy
        """
        market_stats = self.get_markout_stats(market_id)

        # Market-level toxicity
        if market_stats["mean"] < TOXIC_MARKOUT_MEAN_VETO and market_stats["count"] >= 20:
            logger.warning(
                f"TOXIC MARKET DETECTED: {market_id} "
                f"mean_markout={market_stats['mean']:.4f} < {TOXIC_MARKOUT_MEAN_VETO} "
                f"(n={market_stats['count']})"
            )
            return True

        # Cluster-level toxicity
        cluster_stats = self.get_cluster_markout_stats(cluster_id)

        if cluster_stats["mean"] < -0.0015 and cluster_stats["count"] >= 30:
            logger.warning(
                f"TOXIC CLUSTER DETECTED: {cluster_id} "
                f"mean_markout={cluster_stats['mean']:.4f} < -0.0015 "
                f"(n={cluster_stats['count']})"
            )
            return True

        return False

    def is_mildly_toxic(self, market_id: str) -> bool:
        """
        Check for mild toxicity (adjust but don't veto).

        Args:
            market_id: Market condition_id

        Returns:
            True if mildly toxic, False otherwise
        """
        stats = self.get_markout_stats(market_id)

        if stats["mean"] < TOXIC_MARKOUT_MILD and stats["count"] >= 10:
            return True

        return False

    def record_fill_for_markout(self, fill: Fill, book: OrderBook):
        """
        Record fill with current book snapshot for later markout calculation.

        Args:
            fill: Fill object
            book: Current order book snapshot
        """
        from execution.mid import compute_mid

        mid = compute_mid(
            book.best_bid,
            book.best_ask,
            book.last_mid,
            book.timestamp_age_ms,
        )

        if mid is None:
            logger.warning(f"Cannot record fill for markout: no mid for {fill.fill_id}")
            return

        self.pending_fills.append(
            {
                "fill": fill,
                "mid_at_fill": mid,
                "timestamp_ms": fill.timestamp,
            }
        )

        logger.debug(f"Recorded fill for markout: {fill.fill_id} mid={mid:.4f} @ {fill.timestamp}")

    def override_quoteability(
        self,
        market_id: str,
        cluster_id: str,
        qs: float,
    ) -> float:
        """
        Override QS based on toxicity.

        CRITICAL: Toxicity OVERRIDES QS for active set selection.

        Even if QS looks great, if markout is toxic, refuse to quote.

        Args:
            market_id: Market condition_id
            cluster_id: Cluster ID
            qs: Original QS score

        Returns:
            Adjusted QS (0.0 if toxic)
        """
        if self.is_toxic_market(market_id, cluster_id):
            logger.warning(f"QS override: {market_id} toxic → QS=0.0 (was {qs:.2f})")
            return 0.0

        return qs

    def adjust_for_toxicity(
        self,
        market_id: str,
        fv_band: tuple,
        base_size: float,
    ) -> tuple:
        """
        Adjust FV band and size if market shows early toxicity signs.

        If mildly toxic (but not full veto):
        - Widen fair value band by 50% (reduce quote frequency)
        - Reduce size by 50% (limit exposure)

        Args:
            market_id: Market condition_id
            fv_band: (fv_low, fv_high)
            base_size: Base quote size

        Returns:
            ((fv_low, fv_high), size)
        """
        if not self.is_mildly_toxic(market_id):
            return (fv_band, base_size)

        # Widen band
        fv_low, fv_high = fv_band
        mid = (fv_low + fv_high) / 2.0
        half_width = (fv_high - fv_low) / 2.0

        new_half_width = half_width * TOXIC_FV_BAND_MULTIPLIER
        fv_low = mid - new_half_width
        fv_high = mid + new_half_width

        # Reduce size
        new_size = base_size * TOXIC_SIZE_MULTIPLIER

        logger.info(
            f"Toxicity adjustment: {market_id} "
            f"band widened by {TOXIC_FV_BAND_MULTIPLIER}x, size reduced by {TOXIC_SIZE_MULTIPLIER}x"
        )

        return ((fv_low, fv_high), new_size)

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        total_markets_tracked = len(self.markout_history)
        total_clusters_tracked = len(self.cluster_markout_history)

        toxic_markets = sum(
            1
            for market_id in self.markout_history.keys()
            if self.is_toxic_market(market_id, "unknown")  # Cluster unknown here
        )

        return {
            "markets_tracked": total_markets_tracked,
            "clusters_tracked": total_clusters_tracked,
            "toxic_markets_count": toxic_markets,
            "pending_fills": len(self.pending_fills),
        }
