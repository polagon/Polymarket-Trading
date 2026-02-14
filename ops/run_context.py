"""
Run context — immutable identity for the current process run.

Every Astra run has:
- run_id: UUID4 generated once at process start
- started_at: ISO-8601 UTC timestamp
- git_sha: short commit hash (None allowed in paper mode)
- paper_mode: True/False
- cycle_id: monotonic counter, incremented each main loop iteration

cycle_id is stored in a contextvars.ContextVar for injection into structured logs.
"""

import platform
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Context variables for logging injection
_cycle_id_var: ContextVar[int] = ContextVar("cycle_id", default=0)
_component_var: ContextVar[str] = ContextVar("component", default="")


def set_cycle_context(cycle_id: int, component: str = "") -> None:
    """Set cycle context for structured logging injection."""
    _cycle_id_var.set(cycle_id)
    if component:
        _component_var.set(component)


def get_cycle_id() -> int:
    """Get current cycle_id from context."""
    return _cycle_id_var.get()


def get_component() -> str:
    """Get current component from context."""
    return _component_var.get()


@dataclass
class RunContext:
    """Immutable identity for a single Astra process run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: Optional[str] = None
    paper_mode: bool = True
    schema_version: int = 0
    config_hash: str = ""
    prompt_bundle_hash: str = ""
    python_version: str = field(default_factory=platform.python_version)
    cycle_id: int = 0

    def next_cycle(self) -> int:
        """Increment and return the new cycle_id. Also updates contextvars."""
        self.cycle_id += 1
        set_cycle_context(self.cycle_id)
        return self.cycle_id

    def to_manifest_dict(self) -> dict:
        """Return dict for manifest.json (V1 schema — exactly 11 keys)."""
        return {
            "manifest_schema_version": 1,
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "started_at": self.started_at,
            "ended_at": None,
            "exit_reason": None,
            "python_version": self.python_version,
            "config_hash": self.config_hash,
            "prompt_bundle_hash": self.prompt_bundle_hash,
            "schema_version": self.schema_version,
            "paper_mode": self.paper_mode,
        }
