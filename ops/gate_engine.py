"""
Astra Gate Engine — single source of truth for system health status.

Status: ok | degraded | halted
- ok: all infrastructure gates pass, system operates normally
- degraded: warning conditions detected, block new entries but allow monitoring
- halted: critical failure, cancel all orders and block everything

Sample-size gates do NOT cause halted — they block readiness transitions
and emit "insufficient_data" in decision_report blockers.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("astra.gate_engine")


@dataclass
class GateContext:
    """Input context for gate evaluation — populated each cycle."""

    ws_connected: bool = True
    feed_age_s: float = 0.0
    daily_pnl_pct: float = 0.0
    cumulative_dd_pct: float = 0.0
    kill_switch_active: bool = False
    db_integrity_ok: bool = True
    artifacts_writable: bool = True
    error_count: int = 0
    memory_mb: Optional[float] = None  # None = unmeasurable
    disk_free_mb: Optional[float] = None  # None = unmeasurable


@dataclass
class GateResult:
    """Result of a single gate check."""

    name: str
    status: str  # "pass", "warn", "fail"
    message: str = ""
    value: Any = None


@dataclass
class GateStatus:
    """Aggregate gate evaluation result."""

    status: str  # "ok", "degraded", "halted"
    gates: list = field(default_factory=list)
    run_id: str = ""
    cycle_id: int = 0
    timestamp: str = ""
    previous_status: str = ""
    transitions: list = field(default_factory=list)
    blockers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "gates": [{"name": g.name, "status": g.status, "message": g.message, "value": g.value} for g in self.gates],
            "run_id": self.run_id,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "previous_status": self.previous_status,
            "transitions": self.transitions,
            "blockers": self.blockers,
        }


class GateEngine:
    """Evaluates system health gates each cycle."""

    def __init__(
        self,
        feed_max_age_s: float = 60.0,
        max_daily_loss_pct: float = 0.05,
        max_cumulative_dd_pct: float = 0.15,
        max_memory_mb: float = 500.0,
        min_disk_free_mb: float = 100.0,
        max_errors_per_cycle: int = 5,
        # Warn/fail split thresholds (Fix 5)
        feed_warn_age_s: float = 30.0,
        feed_fail_age_s: float = 120.0,
        memory_warn_mb: float = 400.0,
        memory_fail_mb: float = 600.0,
        disk_warn_mb: float = 200.0,
        disk_fail_mb: float = 50.0,
        errors_warn: int = 3,
        errors_fail: int = 10,
    ):
        # Backward compat
        self.feed_max_age_s = feed_max_age_s
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_cumulative_dd_pct = max_cumulative_dd_pct
        self.max_memory_mb = max_memory_mb
        self.min_disk_free_mb = min_disk_free_mb
        self.max_errors_per_cycle = max_errors_per_cycle
        # Warn/fail split
        self.feed_warn_age_s = feed_warn_age_s
        self.feed_fail_age_s = feed_fail_age_s
        self.memory_warn_mb = memory_warn_mb
        self.memory_fail_mb = memory_fail_mb
        self.disk_warn_mb = disk_warn_mb
        self.disk_fail_mb = disk_fail_mb
        self.errors_warn = errors_warn
        self.errors_fail = errors_fail
        self._previous_status = "ok"

    def evaluate(
        self,
        ctx: GateContext,
        run_id: str = "",
        cycle_id: int = 0,
        metrics_snapshot: dict = None,  # type: ignore[assignment]
    ) -> GateStatus:
        """
        Evaluate all gates and return aggregate status.

        Args:
            ctx: Current cycle context
            run_id: Current run ID
            cycle_id: Current cycle number
            metrics_snapshot: Optional metrics with gate fields for sample-size checks
        """
        gates = []

        # Infrastructure gates
        gates.append(self._check_ws_feed(ctx))
        gates.append(self._check_kill_switch(ctx))
        gates.append(self._check_daily_loss(ctx))
        gates.append(self._check_cumulative_dd(ctx))
        gates.append(self._check_db_integrity(ctx))
        gates.append(self._check_artifacts(ctx))
        gates.append(self._check_error_rate(ctx))
        gates.append(self._check_memory(ctx))
        gates.append(self._check_disk(ctx))

        # Determine aggregate status
        has_fail = any(g.status == "fail" for g in gates)
        has_warn = any(g.status == "warn" for g in gates)

        if has_fail:
            status = "halted"
        elif has_warn:
            status = "degraded"
        else:
            status = "ok"

        # Track transitions
        transitions = []
        if self._previous_status != status:
            transitions.append(
                {
                    "from": self._previous_status,
                    "to": status,
                    "cycle_id": cycle_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            logger.warning(f"Gate status transition: {self._previous_status} -> {status}")

        # Sample-size gate blockers (do NOT affect halted/degraded, just block readiness)
        blockers = []
        if metrics_snapshot:
            blockers = self._check_sample_gates(metrics_snapshot)

        previous_status = self._previous_status
        self._previous_status = status

        return GateStatus(
            status=status,
            gates=gates,
            run_id=run_id,
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            previous_status=previous_status,
            transitions=transitions,
            blockers=blockers,
        )

    def _check_ws_feed(self, ctx: GateContext) -> GateResult:
        if not ctx.ws_connected:
            return GateResult("ws_feed", "warn", "WebSocket disconnected")
        if ctx.feed_age_s > self.feed_fail_age_s:
            return GateResult(
                "ws_feed", "fail", f"Feed stale: {ctx.feed_age_s:.0f}s > {self.feed_fail_age_s:.0f}s", ctx.feed_age_s
            )
        if ctx.feed_age_s > self.feed_warn_age_s:
            return GateResult(
                "ws_feed", "warn", f"Feed stale: {ctx.feed_age_s:.0f}s > {self.feed_warn_age_s:.0f}s", ctx.feed_age_s
            )
        return GateResult("ws_feed", "pass", "Feed connected and fresh", ctx.feed_age_s)

    def _check_kill_switch(self, ctx: GateContext) -> GateResult:
        # Also check file-based kill switch
        kill_file = os.path.exists("/tmp/astra_kill")
        if ctx.kill_switch_active or kill_file:
            return GateResult("kill_switch", "fail", "Kill switch active")
        return GateResult("kill_switch", "pass", "Kill switch inactive")

    def _check_daily_loss(self, ctx: GateContext) -> GateResult:
        # daily_pnl_pct is negative for losses
        if ctx.daily_pnl_pct < -self.max_daily_loss_pct:
            return GateResult(
                "daily_loss",
                "fail",
                f"Daily loss {ctx.daily_pnl_pct:.2%} exceeds -{self.max_daily_loss_pct:.2%}",
                ctx.daily_pnl_pct,
            )
        return GateResult("daily_loss", "pass", f"Daily P&L: {ctx.daily_pnl_pct:.2%}", ctx.daily_pnl_pct)

    def _check_cumulative_dd(self, ctx: GateContext) -> GateResult:
        if ctx.cumulative_dd_pct < -self.max_cumulative_dd_pct:
            return GateResult(
                "cumulative_dd",
                "fail",
                f"Drawdown {ctx.cumulative_dd_pct:.2%} exceeds -{self.max_cumulative_dd_pct:.2%}",
                ctx.cumulative_dd_pct,
            )
        return GateResult("cumulative_dd", "pass", f"Drawdown: {ctx.cumulative_dd_pct:.2%}", ctx.cumulative_dd_pct)

    def _check_db_integrity(self, ctx: GateContext) -> GateResult:
        if not ctx.db_integrity_ok:
            return GateResult("db_integrity", "fail", "Database integrity check failed")
        return GateResult("db_integrity", "pass", "Database integrity OK")

    def _check_artifacts(self, ctx: GateContext) -> GateResult:
        if not ctx.artifacts_writable:
            return GateResult("artifacts", "fail", "Artifacts directory not writable")
        return GateResult("artifacts", "pass", "Artifacts directory writable")

    def _check_error_rate(self, ctx: GateContext) -> GateResult:
        if ctx.error_count >= self.errors_fail:
            return GateResult(
                "error_rate", "fail", f"Error count {ctx.error_count} >= {self.errors_fail}", ctx.error_count
            )
        if ctx.error_count >= self.errors_warn:
            return GateResult(
                "error_rate", "warn", f"Error count {ctx.error_count} >= {self.errors_warn}", ctx.error_count
            )
        return GateResult("error_rate", "pass", f"Errors: {ctx.error_count}", ctx.error_count)

    def _check_memory(self, ctx: GateContext) -> GateResult:
        if ctx.memory_mb is None:
            return GateResult("memory", "warn", "Memory unmeasurable", None)
        if ctx.memory_mb > self.memory_fail_mb:
            return GateResult(
                "memory", "fail", f"Memory {ctx.memory_mb:.0f}MB > {self.memory_fail_mb:.0f}MB", ctx.memory_mb
            )
        if ctx.memory_mb > self.memory_warn_mb:
            return GateResult(
                "memory", "warn", f"Memory {ctx.memory_mb:.0f}MB > {self.memory_warn_mb:.0f}MB", ctx.memory_mb
            )
        return GateResult("memory", "pass", f"Memory: {ctx.memory_mb:.0f}MB", ctx.memory_mb)

    def _check_disk(self, ctx: GateContext) -> GateResult:
        if ctx.disk_free_mb is None:
            return GateResult("disk", "warn", "Disk free unmeasurable", None)
        if ctx.disk_free_mb < self.disk_fail_mb:
            return GateResult(
                "disk", "fail", f"Disk free {ctx.disk_free_mb:.0f}MB < {self.disk_fail_mb:.0f}MB", ctx.disk_free_mb
            )
        if ctx.disk_free_mb < self.disk_warn_mb:
            return GateResult(
                "disk", "warn", f"Disk free {ctx.disk_free_mb:.0f}MB < {self.disk_warn_mb:.0f}MB", ctx.disk_free_mb
            )
        return GateResult("disk", "pass", f"Disk free: {ctx.disk_free_mb:.0f}MB", ctx.disk_free_mb)

    def _check_sample_gates(self, metrics: dict) -> list:
        """Check sample-size gates from metrics snapshot. Returns list of blocker strings."""
        blockers = []
        # Metrics engine uses 'gate' field: "PASS" or "FAIL (reason)"
        gate_checks = {
            "sharpe_per_trade": "Sharpe",
            "win_rate": "WinRate",
            "brier_score": "Brier",
            "sortino_per_trade": "Sortino",
            "calmar_ratio": "Calmar",
        }
        for metric_key, label in gate_checks.items():
            gate_val = metrics.get(f"{metric_key}_gate") or metrics.get("gate", "")
            # Check if metrics has a nested structure
            if isinstance(metrics.get("all_time"), dict):
                sub = metrics["all_time"]
                gate_val = sub.get(f"{metric_key}_gate", sub.get("gate", ""))
            if isinstance(gate_val, str) and "FAIL" in gate_val.upper():
                blockers.append(f"SampleGate_{label}: insufficient data")
        return blockers
