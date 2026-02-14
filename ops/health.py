"""
Astra Health Heartbeat — atomic JSON file written each cycle.

External watchdogs (Docker HEALTHCHECK, systemd, cron) check file age.
If file age > max_age_seconds, the process is considered dead.
"""

import json
import logging
import os
import resource
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("astra.health")


def get_default_health_path() -> str:
    """Return the health file path, reading ASTRA_HEALTH_PATH env var."""
    return os.getenv("ASTRA_HEALTH_PATH", "/tmp/astra_health.json")


def _get_memory_mb() -> Optional[float]:
    """Get current process memory usage in MB.

    Uses resource.getrusage on Unix. macOS returns bytes, Linux returns KB.
    Returns None if memory is unmeasurable (gate should WARN, not PASS).
    """
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = usage.ru_maxrss
        if maxrss <= 0:
            return None  # Unmeasurable
        import sys

        if sys.platform == "darwin":
            return maxrss / (1024 * 1024)  # macOS: bytes -> MB
        else:
            return maxrss / 1024  # Linux: KB -> MB
    except Exception:
        return None


def _get_disk_free_mb(path: str = ".") -> Optional[float]:
    """Get free disk space in MB for the given path."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 * 1024)
    except Exception:
        return None


def write_heartbeat(
    path: Optional[str] = None,
    run_id: str = "",
    cycle_id: int = 0,
    status: str = "ok",
    cycle_duration_s: float = 0.0,
    ws_connected: bool = True,
    active_markets: int = 0,
    open_orders: int = 0,
    daily_pnl_usd: float = 0.0,
    memory_dir: str = ".",
    max_age_seconds: int = 300,
) -> None:
    """Write atomic heartbeat JSON to the specified path."""
    if path is None:
        path = get_default_health_path()
    memory = _get_memory_mb()
    disk = _get_disk_free_mb(memory_dir)

    heartbeat = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "cycle_id": cycle_id,
        "status": status,
        "cycle_duration_s": round(cycle_duration_s, 3),
        "ws_connected": ws_connected,
        "active_markets": active_markets,
        "open_orders": open_orders,
        "daily_pnl_usd": round(daily_pnl_usd, 4),
        "memory_mb": round(memory, 1) if memory is not None else None,
        "disk_free_mb": round(disk, 1) if disk is not None else None,
        "max_age_seconds": max_age_seconds,
    }

    try:
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=str(parent), delete=False, suffix=".tmp") as tmp:
            json.dump(heartbeat, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"Failed to write heartbeat to {path}: {e}")
