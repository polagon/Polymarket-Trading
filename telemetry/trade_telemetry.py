"""
Trade Telemetry — schema-stable decision and order lifecycle artifacts.

Decision artifacts are emitted for EVERY market evaluation (SKIP is first-class).
Order lifecycle is event-sourced as JSONL (one line per event).

Loop 4: Allocator-grade audit trail.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DECISION_SCHEMA_VERSION = "1.0"
ORDER_SCHEMA_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomic JSON write: tempfile + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), delete=False, suffix=".tmp") as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, str(path))


def _append_jsonl(path: Path, data: dict) -> None:
    """Append a JSON line to a file with flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")
        f.flush()


class TradeTelemetry:
    """Emits schema-stable decision and order lifecycle artifacts.

    Args:
        artifacts_dir: Root directory for artifacts.
        run_id: Current run identifier.
    """

    def __init__(self, artifacts_dir: Path, run_id: str) -> None:
        self._artifacts_dir = artifacts_dir
        self._run_id = run_id

    def emit_decision(
        self,
        cycle_id: int,
        market_id: str,
        definition_present: bool,
        definition_hash: Optional[str],
        inputs: dict[str, Any],
        estimator: dict[str, Any],
        frictions: dict[str, Any],
        ev_net_lb: float,
        gates: dict[str, dict[str, Any]],
        action: str,
        order: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Emit a decision artifact (always, including SKIPs).

        Args:
            cycle_id: Current cycle number.
            market_id: Market condition_id.
            definition_present: Whether a DefinitionContract exists.
            definition_hash: Hash of the definition (None if no contract).
            inputs: Market inputs (best_bid, best_ask, mid, spread_frac, depth_proxy_usd).
            estimator: Estimator outputs (p_hat, p_low, p_high, calibration_bucket, estimator_version).
            frictions: Friction breakdown (fee_est_frac, slippage_est_frac, adverse_buffer_frac, etc).
            ev_net_lb: Net EV lower bound.
            gates: Gate outcomes {definition: {ok, reason}, ev: {ok, reason}, risk: {ok, reason}}.
            action: "SKIP" or "PLACE_ORDER".
            order: Order params if PLACE_ORDER, else None.

        Returns:
            Path to the written artifact.
        """
        artifact = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "run_id": self._run_id,
            "cycle_id": cycle_id,
            "market_id": market_id,
            "timestamp": _now_iso(),
            "definition_present": definition_present,
            "definition_hash": definition_hash,
            "inputs": inputs,
            "estimator": estimator,
            "frictions": frictions,
            "ev_net_lb": ev_net_lb,
            "gates": gates,
            "action": action,
            "order": order,
        }

        path = self._artifacts_dir / "decisions" / str(cycle_id) / f"{market_id}.json"
        _atomic_write_json(path, artifact)
        logger.debug(f"Decision artifact: {path} action={action}")
        return path

    def emit_order_event(
        self,
        order_id: str,
        market_id: str,
        event_type: str,
        side: str,
        price: float,
        size: float,
        remaining_size: float,
        reason: str = "",
    ) -> Path:
        """Emit an order lifecycle event (append to JSONL).

        Args:
            order_id: Unique order identifier.
            market_id: Market condition_id.
            event_type: One of: submit, cancel, replace, fill, expire, stale_cancel.
            side: Order side (BUY_YES, BUY_NO).
            price: Order price.
            size: Order size.
            remaining_size: Remaining unfilled size.
            reason: Reason for event (cancel/replace/expire/stale_cancel).

        Returns:
            Path to the JSONL file.
        """
        event = {
            "schema_version": ORDER_SCHEMA_VERSION,
            "timestamp": _now_iso(),
            "order_id": order_id,
            "market_id": market_id,
            "event_type": event_type,
            "side": side,
            "price": price,
            "size": size,
            "remaining_size": remaining_size,
            "reason": reason,
        }

        path = self._artifacts_dir / "orders" / f"{order_id}.jsonl"
        _append_jsonl(path, event)
        logger.debug(f"Order event: {order_id} type={event_type}")
        return path
