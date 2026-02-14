"""
Astra Structured Logging — JSON lines file handler with run context injection.

File handler always writes JSON lines to memory/astra.log with rotation.
Console handler defaults to text format, optional JSON via JSON_LOGGING env var.
Run context (run_id, cycle_id, component) injected via contextvars.
"""

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from ops.run_context import get_component, get_cycle_id


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines with run context."""

    def __init__(self, run_id: str = ""):
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": self.run_id,
            "cycle_id": get_cycle_id(),
            "component": get_component(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        # Add any extra fields passed via extra= kwarg
        for key in ("condition_id", "alert_type", "gate_status"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """Standard text formatter with run context prefix."""

    def __init__(self, run_id: str = ""):
        self.run_id = run_id
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        super().__init__(fmt=fmt)


def setup_logging(
    run_id: str = "",
    json_mode: bool = False,
    log_dir: str = "memory",
    log_file: str = "astra.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure Astra logging.

    - File handler: always JSON format to {log_dir}/{log_file} with rotation
    - Console handler: text by default, JSON if json_mode=True

    Args:
        run_id: Current run ID for context injection
        json_mode: Use JSON format for console output
        log_dir: Directory for log files
        log_file: Log file name
        max_bytes: Max log file size before rotation
        backup_count: Number of rotated log files to keep
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("astra")

    # Prevent duplicate handlers on re-init
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    # File handler — always JSON lines for machine parsing
    fh = RotatingFileHandler(
        str(Path(log_dir) / log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter(run_id=run_id))
    root.addHandler(fh)

    # Console handler — text by default, JSON if requested
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    if json_mode:
        ch.setFormatter(JSONFormatter(run_id=run_id))
    else:
        ch.setFormatter(TextFormatter(run_id=run_id))
    root.addHandler(ch)

    root.info(
        f"Astra logging initialized — run_id={run_id}, json_mode={json_mode}, log_file={Path(log_dir) / log_file}"
    )
