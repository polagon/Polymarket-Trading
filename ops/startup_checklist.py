"""
Astra Startup Checklist — fail-closed validation before main loop.

Activity 17 (Loop 3): 17-point checklist verifying the full provenance chain,
infrastructure readiness, and backup integrity before allowing trading.

Usage:
    results = run_startup_checklist(...)
    if not all_passed(results):
        sys.exit(1)
"""

import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("astra.startup_checklist")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class CheckResult:
    """Result of a single startup check."""

    name: str
    passed: bool
    message: str
    warn_only: bool = False  # True = logged as warning, does not block startup


def _check_uuid4(run_id: str) -> bool:
    """Validate run_id is UUID4-shaped."""
    try:
        parsed = uuid.UUID(run_id, version=4)
        return str(parsed) == run_id
    except (ValueError, AttributeError):
        return False


def run_startup_checklist(  # noqa: C901
    run_ctx: Any,
    db_conn: Any,
    artifact_writer: Any,
    gate_engine: Any,
    alert_mgr: Any,
    config_hash: str,
    prompt_bundle_hash: str,
    api_key_set: bool,
    paper_mode: bool,
    heartbeat_path: str,
    signal_handlers_registered: bool,
    db_path: Optional[Path] = None,
) -> list[CheckResult]:
    """Fail-closed startup checklist.

    Returns list of CheckResults. Caller must abort if any non-warn check fails.

    Checks:
     1. run_id is valid UUID4
     2. DB connection is live (SELECT 1)
     3. DB schema_version matches CURRENT_SCHEMA_VERSION
     4. Config hash is 64-hex sha256
     5. Config snapshot exists in DB for this run
     6. Prompt bundle hash is 64-hex sha256
     7. Prompt registry artifact exists on disk
     8. Artifacts directory writable
     9. Manifest written (file exists)
    10. Heartbeat path writable
    11. DB backup exists (bak1/bak2/bak3 rotation per Activity 5)
    12. Gate engine evaluates successfully once (dry-run)
    13. Alert manager instantiated
    14. Signal handlers registered
    15. Memory directory exists and writable
    16. Disk free > DISK_FAIL_MB
    17. API key set (warn-only in paper mode, fail in live mode)
    """
    results: list[CheckResult] = []

    # 1. run_id is valid UUID4
    try:
        ok = _check_uuid4(run_ctx.run_id)
        results.append(CheckResult("run_id_uuid4", ok, f"run_id={run_ctx.run_id}" if ok else "Invalid UUID4"))
    except Exception as e:
        results.append(CheckResult("run_id_uuid4", False, f"Error: {e}"))

    # 2. DB connection is live
    try:
        if db_conn is None:
            results.append(CheckResult("db_connection", False, "db_conn is None"))
        else:
            row = db_conn.execute("SELECT 1").fetchone()
            results.append(CheckResult("db_connection", row is not None and row[0] == 1, "DB alive"))
    except Exception as e:
        results.append(CheckResult("db_connection", False, f"DB query failed: {e}"))

    # 3. DB schema_version matches CURRENT_SCHEMA_VERSION
    try:
        from scanner.trade_logger import CURRENT_SCHEMA_VERSION

        if db_conn is not None:
            version = db_conn.execute("PRAGMA user_version").fetchone()[0]
            ok = version == CURRENT_SCHEMA_VERSION
            results.append(
                CheckResult(
                    "schema_version",
                    ok,
                    f"DB v{version} == expected v{CURRENT_SCHEMA_VERSION}"
                    if ok
                    else f"Mismatch: DB v{version} != expected v{CURRENT_SCHEMA_VERSION}",
                )
            )
        else:
            results.append(CheckResult("schema_version", False, "No DB connection"))
    except Exception as e:
        results.append(CheckResult("schema_version", False, f"Error: {e}"))

    # 4. Config hash is 64-hex sha256
    ok = isinstance(config_hash, str) and bool(_SHA256_RE.match(config_hash))
    results.append(CheckResult("config_hash", ok, f"len={len(config_hash) if config_hash else 0}"))

    # 5. Config snapshot exists in DB for this run
    try:
        if db_conn is not None:
            row = db_conn.execute(
                "SELECT COUNT(*) FROM config_snapshots WHERE run_id = ?",
                (run_ctx.run_id,),
            ).fetchone()
            ok = row is not None and row[0] > 0
            results.append(CheckResult("config_snapshot_exists", ok, f"snapshots={row[0] if row else 0}"))
        else:
            results.append(CheckResult("config_snapshot_exists", False, "No DB connection"))
    except Exception as e:
        results.append(CheckResult("config_snapshot_exists", False, f"Error: {e}"))

    # 6. Prompt bundle hash is 64-hex sha256
    ok = isinstance(prompt_bundle_hash, str) and bool(_SHA256_RE.match(prompt_bundle_hash))
    results.append(CheckResult("prompt_bundle_hash", ok, f"len={len(prompt_bundle_hash) if prompt_bundle_hash else 0}"))

    # 7. Prompt registry artifact exists on disk
    try:
        registry_path = artifact_writer.run_dir / "prompt_registry.json"
        ok = registry_path.exists()
        results.append(CheckResult("prompt_registry_artifact", ok, str(registry_path)))
    except Exception as e:
        results.append(CheckResult("prompt_registry_artifact", False, f"Error: {e}"))

    # 8. Artifacts directory writable
    try:
        test_file = artifact_writer.run_dir / ".startup_check"
        test_file.write_text("ok")
        test_file.unlink()
        results.append(CheckResult("artifacts_writable", True, str(artifact_writer.run_dir)))
    except Exception as e:
        results.append(CheckResult("artifacts_writable", False, f"Not writable: {e}"))

    # 9. Manifest written (file exists)
    try:
        manifest_path = artifact_writer.run_dir / "manifest.json"
        ok = manifest_path.exists()
        results.append(CheckResult("manifest_exists", ok, str(manifest_path)))
    except Exception as e:
        results.append(CheckResult("manifest_exists", False, f"Error: {e}"))

    # 10. Heartbeat path writable
    try:
        hp = Path(heartbeat_path)
        hp.parent.mkdir(parents=True, exist_ok=True)
        # Test write to a temp file in same dir
        test_path = hp.parent / ".heartbeat_check"
        test_path.write_text("ok")
        test_path.unlink()
        results.append(CheckResult("heartbeat_writable", True, str(hp)))
    except Exception as e:
        results.append(CheckResult("heartbeat_writable", False, f"Not writable: {e}"))

    # 11. DB backup exists (bak1/bak2/bak3 rotation per Activity 5)
    try:
        if db_path is None:
            from scanner.trade_logger import DB_PATH

            effective_db_path = DB_PATH
        else:
            effective_db_path = db_path
        backup_found = False
        for suffix in (".db.bak1", ".db.bak2", ".db.bak3"):
            bak = effective_db_path.with_suffix(suffix)
            if bak.exists():
                backup_found = True
                break
        results.append(
            CheckResult(
                "db_backup_exists",
                backup_found,
                f"Backup found near {effective_db_path.name}"
                if backup_found
                else f"No .bak1/.bak2/.bak3 near {effective_db_path}",
            )
        )
    except Exception as e:
        results.append(CheckResult("db_backup_exists", False, f"Error: {e}"))

    # 12. Gate engine evaluates successfully once (dry-run)
    try:
        from ops.gate_engine import GateContext

        neutral_ctx = GateContext(
            ws_connected=True,
            feed_age_s=0.0,
            daily_pnl_pct=0.0,
            cumulative_dd_pct=0.0,
            error_count=0,
            memory_mb=100.0,
            disk_free_mb=1000.0,
        )
        dry_run_result = gate_engine.evaluate(neutral_ctx, run_id=run_ctx.run_id, cycle_id=0)
        ok = dry_run_result.status in ("ok", "degraded", "halted")
        results.append(CheckResult("gate_engine_dry_run", ok, f"status={dry_run_result.status}"))
    except Exception as e:
        results.append(CheckResult("gate_engine_dry_run", False, f"Error: {e}"))

    # 13. Alert manager instantiated
    ok = alert_mgr is not None
    results.append(CheckResult("alert_manager", ok, "Instantiated" if ok else "None"))

    # 14. Signal handlers registered
    results.append(
        CheckResult(
            "signal_handlers",
            signal_handlers_registered,
            "Registered" if signal_handlers_registered else "Not registered",
        )
    )

    # 15. Memory directory exists and writable
    try:
        mem_dir = Path(__file__).resolve().parent.parent / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        test_file = mem_dir / ".startup_check"
        test_file.write_text("ok")
        test_file.unlink()
        results.append(CheckResult("memory_dir_writable", True, str(mem_dir)))
    except Exception as e:
        results.append(CheckResult("memory_dir_writable", False, f"Not writable: {e}"))

    # 16. Disk free > DISK_FAIL_MB
    try:
        from config import DISK_FAIL_MB
        from ops.health import _get_disk_free_mb

        disk_free = _get_disk_free_mb("memory")
        if disk_free is not None:
            ok = disk_free > DISK_FAIL_MB
            results.append(CheckResult("disk_free", ok, f"{disk_free:.0f}MB free (min {DISK_FAIL_MB}MB)"))
        else:
            results.append(CheckResult("disk_free", False, "Cannot measure disk free"))
    except Exception as e:
        results.append(CheckResult("disk_free", False, f"Error: {e}"))

    # 17. API key set (warn-only in paper mode, fail in live)
    if api_key_set:
        results.append(CheckResult("api_key", True, "API key set"))
    elif paper_mode:
        results.append(CheckResult("api_key", False, "API key not set (paper mode — warn only)", warn_only=True))
    else:
        results.append(CheckResult("api_key", False, "API key not set (LIVE mode — required)"))

    return results


def all_passed(results: list[CheckResult]) -> bool:
    """Returns True if all non-warn checks passed."""
    return all(r.passed for r in results if not r.warn_only)
