"""
Astra Run Artifacts — atomic JSON writers for run provenance and per-cycle state.

Artifacts directory: memory/runs/{run_id}/
- manifest.json: immutable run identity (write once at start, update once at shutdown)
- gate_status.json: per-cycle gate engine state
- metrics_all_windows.json: per-cycle metrics snapshots
- drawdown_state.json: per-cycle drawdown tracker state
- decision_report.json: per-cycle gate transitions and blockers
- edge_snapshot.json: edge proof (stub until sufficient data)
- fill_realism_report.json: sim vs observed (stub until realism layer)
- prompt_registry.json: prompt hash → name/intent mapping
- anomalies.ndjson: append-only anomaly log

All artifacts include run_id, cycle_id, timestamp at top level.
"""

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("astra.artifacts")

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"

# ---------------------------------------------------------------------------
# Manifest V1 schema contract (fail-closed)
# ---------------------------------------------------------------------------
MANIFEST_SCHEMA_VERSION = 1

ALLOWED_KEYS_V1 = frozenset(
    {
        "manifest_schema_version",
        "run_id",
        "git_sha",
        "started_at",
        "ended_at",
        "exit_reason",
        "python_version",
        "config_hash",
        "prompt_bundle_hash",
        "schema_version",
        "paper_mode",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_manifest_v1(manifest: dict) -> None:
    """Validate manifest dict against the V1 schema contract.

    Raises RuntimeError on any violation:
    - Extra keys not in ALLOWED_KEYS_V1
    - Missing required keys
    - config_hash or prompt_bundle_hash not 64-hex sha256
    - manifest_schema_version != 1
    - schema_version not int >= 1
    - run_id not uuid4-shaped
    """
    keys = set(manifest.keys())

    # No extra keys
    extra = keys - ALLOWED_KEYS_V1
    if extra:
        raise RuntimeError(f"Manifest V1 schema violation: unexpected keys {extra}. Allowed: {sorted(ALLOWED_KEYS_V1)}")

    # All required keys present
    missing = ALLOWED_KEYS_V1 - keys
    if missing:
        raise RuntimeError(f"Manifest V1 schema violation: missing keys {missing}")

    # manifest_schema_version must be 1
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Manifest V1 schema violation: manifest_schema_version is "
            f"{manifest['manifest_schema_version']}, expected {MANIFEST_SCHEMA_VERSION}"
        )

    # config_hash must be 64-hex sha256
    ch = manifest["config_hash"]
    if not isinstance(ch, str) or not _SHA256_RE.match(ch):
        raise RuntimeError(f"Manifest V1 schema violation: config_hash must be 64-hex sha256, got {ch!r}")

    # prompt_bundle_hash must be 64-hex sha256
    pbh = manifest["prompt_bundle_hash"]
    if not isinstance(pbh, str) or not _SHA256_RE.match(pbh):
        raise RuntimeError(f"Manifest V1 schema violation: prompt_bundle_hash must be 64-hex sha256, got {pbh!r}")

    # schema_version must be int >= 1
    sv = manifest["schema_version"]
    if not isinstance(sv, int) or sv < 1:
        raise RuntimeError(f"Manifest V1 schema violation: schema_version must be int >= 1, got {sv!r}")

    # run_id should be uuid4-shaped (8-4-4-4-12 hex)
    rid = manifest["run_id"]
    if not isinstance(rid, str) or not re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        rid,
    ):
        raise RuntimeError(f"Manifest V1 schema violation: run_id must be UUID4, got {rid!r}")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomic JSON write: tempfile + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), delete=False, suffix=".tmp") as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, str(path))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactWriter:
    """Manages per-run artifact directory and writes."""

    def __init__(self, run_id: str, base_dir: Path = None):  # type: ignore[assignment]
        self.run_id = run_id
        self.base_dir = base_dir or (MEMORY_DIR / "runs")
        self.run_dir = self.base_dir / run_id
        self._manifest_written = False

    def ensure_writable(self) -> None:
        """Create artifacts dir and verify writability. Raises RuntimeError if not writable."""
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.run_dir / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
        except OSError as e:
            raise RuntimeError(f"Artifacts directory not writable: {self.run_dir}: {e}")

    def write_manifest_start(self, manifest: dict) -> None:
        """Write initial manifest.json. Must be called exactly once at startup.

        Validates manifest against V1 schema before writing.
        Raises RuntimeError if validation fails (fail-closed).
        """
        if self._manifest_written:
            raise RuntimeError("Manifest already written for this run — immutable after startup")
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            raise RuntimeError(f"Manifest already exists at {manifest_path} — refusing to overwrite")
        # Fail-closed: validate before writing
        validate_manifest_v1(manifest)
        _atomic_write_json(manifest_path, manifest)
        self._manifest_written = True
        logger.info(f"Manifest written: {manifest_path}")

    def write_manifest_end(self, exit_reason: str) -> None:
        """Update manifest with shutdown info. Only updates ended_at and exit_reason."""
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error("Cannot update manifest — file does not exist")
            return
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Validate immutable fields unchanged
            if manifest.get("run_id") != self.run_id:
                raise RuntimeError("Manifest run_id mismatch — file may be corrupt")
            if manifest.get("ended_at") is not None:
                logger.warning("Manifest already has ended_at — skipping duplicate shutdown update")
                return
            # Only update end fields — nothing else
            manifest["ended_at"] = _now_iso()
            manifest["exit_reason"] = exit_reason
            _atomic_write_json(manifest_path, manifest)
            logger.info(f"Manifest updated with ended_at: {exit_reason}")
        except Exception as e:
            logger.error(f"Failed to update manifest end: {e}")

    def _enrich(self, data: dict, cycle_id: int) -> dict:
        """Add standard provenance fields to artifact data."""
        data["run_id"] = self.run_id
        data["cycle_id"] = cycle_id
        data["timestamp"] = _now_iso()
        return data

    def write_gate_status(self, cycle_id: int, gate_status: dict) -> None:
        """Write gate_status.json for current cycle."""
        _atomic_write_json(
            self.run_dir / "gate_status.json",
            self._enrich(gate_status, cycle_id),
        )

    def write_metrics(self, cycle_id: int, metrics: dict) -> None:
        """Write metrics_all_windows.json for current cycle."""
        _atomic_write_json(
            self.run_dir / "metrics_all_windows.json",
            self._enrich(metrics, cycle_id),
        )

    def write_drawdown_state(self, cycle_id: int, drawdown: dict) -> None:
        """Write drawdown_state.json for current cycle."""
        _atomic_write_json(
            self.run_dir / "drawdown_state.json",
            self._enrich(drawdown, cycle_id),
        )

    def write_decision_report(self, cycle_id: int, report: dict) -> None:
        """Write decision_report.json for current cycle."""
        _atomic_write_json(
            self.run_dir / "decision_report.json",
            self._enrich(report, cycle_id),
        )

    def write_edge_snapshot(self, cycle_id: int, snapshot: dict = None) -> None:  # type: ignore[assignment]
        """Write edge_snapshot.json. Default: stub with insufficient_data."""
        if snapshot is None:
            snapshot = {"status": "insufficient_data", "resolved_trades": 0}
        _atomic_write_json(
            self.run_dir / "edge_snapshot.json",
            self._enrich(snapshot, cycle_id),
        )

    def write_fill_realism_report(self, cycle_id: int, report: dict = None) -> None:  # type: ignore[assignment]
        """Write fill_realism_report.json. Default: stub."""
        if report is None:
            report = {"status": "not_implemented", "note": "Reserved for realism layer (Activities 14-16)"}
        _atomic_write_json(
            self.run_dir / "fill_realism_report.json",
            self._enrich(report, cycle_id),
        )

    def write_prompt_registry(self, registry: dict) -> None:
        """Write prompt_registry.json — prompt hash to name/intent mapping."""
        data = {
            "run_id": self.run_id,
            "timestamp": _now_iso(),
            "prompts": registry,
        }
        _atomic_write_json(self.run_dir / "prompt_registry.json", data)

    def append_anomaly(self, cycle_id: int, anomaly_type: str, details: dict) -> None:
        """Append a JSON line to anomalies.ndjson."""
        entry = {
            "run_id": self.run_id,
            "cycle_id": cycle_id,
            "timestamp": _now_iso(),
            "type": anomaly_type,
            **details,
        }
        anomaly_path = self.run_dir / "anomalies.ndjson"
        try:
            with open(anomaly_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to append anomaly: {e}")

    def write_all_cycle_artifacts(
        self,
        cycle_id: int,
        gate_status: dict,
        metrics: dict = None,  # type: ignore[assignment]
        drawdown: dict = None,  # type: ignore[assignment]
        decision_report: dict = None,  # type: ignore[assignment]
    ) -> None:
        """Convenience: write all per-cycle artifacts in one call."""
        self.write_gate_status(cycle_id, gate_status)
        self.write_metrics(cycle_id, metrics or {"status": "not_available"})
        self.write_drawdown_state(cycle_id, drawdown or {"status": "not_available"})
        self.write_decision_report(
            cycle_id, decision_report or {"gates_evaluated": 0, "transitions": [], "blocked_by": []}
        )
        self.write_edge_snapshot(cycle_id)
        self.write_fill_realism_report(cycle_id)
