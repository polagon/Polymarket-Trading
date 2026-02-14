"""
Truth Report - Daily JSON output with maker/taker separation.

CRITICAL: Maker vs taker fill separation for accurate performance attribution.
"""

import json
import logging
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from models.types import Fill

logger = logging.getLogger(__name__)


class TruthReportBuilder:
    """
    In-memory aggregation for daily truth report.

    Tracks:
    - Maker vs taker fills (CRITICAL for Sharpe attribution)
    - Quote events (placements, cancels, replaces)
    - Health metrics (WS uptime, reconciliation errors)
    - Portfolio snapshots
    """

    def __init__(self, date: str):
        """
        Initialize builder for a specific date.

        Args:
            date: ISO date string (YYYY-MM-DD)
        """
        self.date = date

        # Fill tracking
        self.maker_fills: List[Dict] = []
        self.taker_fills: List[Dict] = []

        # Quote events
        self.quote_count = 0
        self.cancel_count = 0
        self.replace_count = 0

        # Health tracking
        self.ws_market_uptime_seconds = 0.0
        self.ws_user_uptime_seconds = 0.0
        self.cancel_all_triggers = 0
        self.reconciliation_errors = 0
        self.stale_quote_incidents = 0
        self.toxic_market_pauses = 0

        # Portfolio snapshot (set once at end of day)
        self.portfolio_snapshot: Optional[Dict] = None

        # Cluster/market tracking
        self.cluster_pnl: Dict[str, float] = defaultdict(float)
        self.market_pnl: Dict[str, float] = defaultdict(float)

    def record_fill(
        self,
        fill: Fill,
        cluster_id: str,
        pnl: float,
        is_maker: bool,
        realized_spread: Optional[float] = None,
        markout_30s: Optional[float] = None,
        markout_2m: Optional[float] = None,
        markout_10m: Optional[float] = None,
    ):
        """
        Record a fill for daily report.

        CRITICAL: Separate maker vs taker.

        Args:
            fill: Fill object
            cluster_id: Cluster ID
            pnl: Realized P&L for this fill
            is_maker: True if maker fill, False if taker
            realized_spread: Realized spread (maker only)
            markout_30s: 30-second markout
            markout_2m: 2-minute markout
            markout_10m: 10-minute markout
        """
        fill_record = {
            "fill_id": fill.fill_id,
            "condition_id": fill.condition_id,
            "token_id": fill.token_id,
            "side": fill.side,
            "price": fill.price,
            "size_tokens": fill.size_tokens,
            "timestamp": fill.timestamp.isoformat() if hasattr(fill.timestamp, "isoformat") else str(fill.timestamp),
            "pnl": pnl,
            "cluster_id": cluster_id,
            "realized_spread": realized_spread,
            "markout_30s": markout_30s,
            "markout_2m": markout_2m,
            "markout_10m": markout_10m,
        }

        if is_maker:
            self.maker_fills.append(fill_record)
        else:
            self.taker_fills.append(fill_record)

        # Aggregate P&L
        self.cluster_pnl[cluster_id] += pnl
        self.market_pnl[fill.condition_id] += pnl

        logger.debug(f"Fill recorded: {fill.fill_id} {'MAKER' if is_maker else 'TAKER'} pnl=${pnl:.2f}")

    def record_quote_event(self, event_type: str, count: int = 1):
        """
        Record quote event (placement, cancel, replace).

        Args:
            event_type: "quote", "cancel", or "replace"
            count: Number of events (default 1)
        """
        if event_type == "quote":
            self.quote_count += count
        elif event_type == "cancel":
            self.cancel_count += count
        elif event_type == "replace":
            self.replace_count += count
        else:
            logger.warning(f"Unknown quote event type: {event_type}")

    def record_health(
        self,
        ws_market_uptime_seconds: Optional[float] = None,
        ws_user_uptime_seconds: Optional[float] = None,
        cancel_all_trigger: bool = False,
        reconciliation_error: bool = False,
        stale_quote_incident: bool = False,
        toxic_market_pause: bool = False,
    ):
        """
        Record health metrics.

        Args:
            ws_market_uptime_seconds: WS market feed uptime
            ws_user_uptime_seconds: WS user feed uptime
            cancel_all_trigger: Cancel-all triggered
            reconciliation_error: Reconciliation error occurred
            stale_quote_incident: Stale quote incident
            toxic_market_pause: Toxic market paused
        """
        if ws_market_uptime_seconds is not None:
            self.ws_market_uptime_seconds = ws_market_uptime_seconds
        if ws_user_uptime_seconds is not None:
            self.ws_user_uptime_seconds = ws_user_uptime_seconds

        if cancel_all_trigger:
            self.cancel_all_triggers += 1
        if reconciliation_error:
            self.reconciliation_errors += 1
        if stale_quote_incident:
            self.stale_quote_incidents += 1
        if toxic_market_pause:
            self.toxic_market_pauses += 1

    def set_portfolio_snapshot(
        self,
        daily_return: float,
        weekly_return: float,
        monthly_return: float,
        sharpe_90d: float,
        calmar_90d: float,
        max_drawdown: float,
        cluster_exposures: Dict[str, float],
        aggregate_exposure: float,
        max_market_inventory: float,
    ):
        """
        Set portfolio snapshot (call once at end of day).

        Args:
            daily_return: Daily return
            weekly_return: Weekly return
            monthly_return: Monthly return
            sharpe_90d: 90-day Sharpe ratio
            calmar_90d: 90-day Calmar ratio
            max_drawdown: Maximum drawdown
            cluster_exposures: Cluster exposures dict
            aggregate_exposure: Aggregate exposure
            max_market_inventory: Max market inventory
        """
        self.portfolio_snapshot = {
            "daily_return": daily_return,
            "weekly_return": weekly_return,
            "monthly_return": monthly_return,
            "sharpe_90d": sharpe_90d,
            "calmar_90d": calmar_90d,
            "max_drawdown": max_drawdown,
            "cluster_exposures": cluster_exposures,
            "aggregate_exposure": aggregate_exposure,
            "max_market_inventory": max_market_inventory,
        }

    def _compute_maker_truth_metrics(self) -> Dict:
        """
        Compute maker truth metrics from recorded fills.

        Returns:
            Maker truth metrics dict
        """
        if not self.maker_fills:
            return {
                "quote_count": self.quote_count,
                "cancel_count": self.cancel_count,
                "replace_count": self.replace_count,
                "fill_count": 0,
                "realized_spread_bps": 0.0,
                "realized_spread_per_quote_bps": 0.0,
                "markout_30s_bps": 0.0,
                "markout_2m_bps": 0.0,
                "markout_10m_bps": 0.0,
                "fill_rate_per_quote": 0.0,
                "cancel_replace_rate": 0.0,
                "toxic_market_pauses": self.toxic_market_pauses,
            }

        # Extract metrics from maker fills
        realized_spreads = [f["realized_spread"] for f in self.maker_fills if f["realized_spread"] is not None]
        markouts_30s = [f["markout_30s"] for f in self.maker_fills if f["markout_30s"] is not None]
        markouts_2m = [f["markout_2m"] for f in self.maker_fills if f["markout_2m"] is not None]
        markouts_10m = [f["markout_10m"] for f in self.maker_fills if f["markout_10m"] is not None]

        # Compute means (in basis points)
        realized_spread_bps = np.mean(realized_spreads) * 10000 if realized_spreads else 0.0
        markout_30s_bps = np.mean(markouts_30s) * 10000 if markouts_30s else 0.0
        markout_2m_bps = np.mean(markouts_2m) * 10000 if markouts_2m else 0.0
        markout_10m_bps = np.mean(markouts_10m) * 10000 if markouts_10m else 0.0

        # Per-quote metrics
        fill_rate = len(self.maker_fills) / self.quote_count if self.quote_count > 0 else 0.0
        cancel_replace_rate = (
            (self.cancel_count + self.replace_count) / self.quote_count if self.quote_count > 0 else 0.0
        )
        realized_spread_per_quote_bps = realized_spread_bps * fill_rate if self.quote_count > 0 else 0.0

        return {
            "quote_count": self.quote_count,
            "cancel_count": self.cancel_count,
            "replace_count": self.replace_count,
            "fill_count": len(self.maker_fills),
            "realized_spread_bps": float(realized_spread_bps),
            "realized_spread_per_quote_bps": float(realized_spread_per_quote_bps),
            "markout_30s_bps": float(markout_30s_bps),
            "markout_2m_bps": float(markout_2m_bps),
            "markout_10m_bps": float(markout_10m_bps),
            "fill_rate_per_quote": float(fill_rate),
            "cancel_replace_rate": float(cancel_replace_rate),
            "toxic_market_pauses": self.toxic_market_pauses,
        }

    def finalize(self) -> Dict:
        """
        Finalize and return complete daily report.

        Returns:
            Complete report dict matching schema
        """
        # Component attribution
        maker_pnl = sum(f["pnl"] for f in self.maker_fills)
        taker_pnl = sum(f["pnl"] for f in self.taker_fills)
        total_pnl = maker_pnl + taker_pnl

        # Compute WS uptime ratios (24 hours = 86400 seconds)
        total_seconds_in_day = 86400.0
        ws_market_uptime = self.ws_market_uptime_seconds / total_seconds_in_day if total_seconds_in_day > 0 else 0.0
        ws_user_uptime = self.ws_user_uptime_seconds / total_seconds_in_day if total_seconds_in_day > 0 else 0.0

        report = {
            "date": self.date,
            "portfolio_metrics": self.portfolio_snapshot or {},
            "component_attribution": {
                "maker_pnl": maker_pnl,
                "taker_pnl": taker_pnl,
                "total_pnl": total_pnl,
            },
            "risk_metrics": {
                "cluster_pnl": dict(self.cluster_pnl),
                "top_5_markets_pnl": dict(sorted(self.market_pnl.items(), key=lambda x: abs(x[1]), reverse=True)[:5]),
            },
            "gate_outcomes": {
                # These would be computed by gate evaluation logic
                # Placeholder for now
            },
            "health_metrics": {
                "ws_market_uptime": ws_market_uptime,
                "ws_user_uptime": ws_user_uptime,
                "cancel_all_triggers": self.cancel_all_triggers,
                "reconciliation_errors": self.reconciliation_errors,
                "stale_quote_incidents": self.stale_quote_incidents,
            },
            "maker_truth_metrics": self._compute_maker_truth_metrics(),
        }

        return report


def write_daily_report(report: Dict, reports_dir: Path = None) -> Path:  # type: ignore[assignment]
    """
    Write daily report to JSON file with atomic write.

    Uses temp file + os.replace() to ensure atomicity.

    Args:
        report: Report dict from TruthReportBuilder.finalize()
        reports_dir: Directory to write reports (default: repo_root/reports)

    Returns:
        Path to written report file
    """
    if reports_dir is None:
        # Default: repo_root/reports
        repo_root = Path(__file__).resolve().parent.parent
        reports_dir = repo_root / "reports"

    # Ensure reports directory exists
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Target file path
    date = report["date"]
    target_path = reports_dir / f"{date}.json"

    # Atomic write: temp file + os.replace()
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=reports_dir,
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        json.dump(report, tmp_file, indent=2)
        tmp_path = tmp_file.name

    # Atomic replace
    os.replace(tmp_path, target_path)

    logger.info(f"Daily report written: {target_path}")

    return target_path
