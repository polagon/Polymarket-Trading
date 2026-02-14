"""
Definition lint — category-aware semantic completeness validation.

Fail-closed: unknown keys are FAIL, not warn.
Returns named missing semantics, never boolean-only.

Loop 4: Only crypto_threshold category supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from models.definition_contract import DefinitionContract
from models.reasons import (
    ALLOWED_CONDITION_OPS,
    ALLOWED_ROUNDING_RULES,
    LINT_INVALID_CONDITION_LEVEL_TYPE,
    LINT_INVALID_CONDITION_WINDOW,
    LINT_INVALID_CUTOFF,
    LINT_INVALID_ORACLE_ROUNDING,
    LINT_MISSING_CONDITION_LEVEL,
    LINT_MISSING_CONDITION_OP,
    LINT_MISSING_MEASUREMENT_TIME,
    LINT_MISSING_ORACLE_FEED,
    LINT_MISSING_ORACLE_FINALITY,
    LINT_MISSING_ORACLE_ROUNDING,
    LINT_UNKNOWN_CONDITION_KEY,
    LINT_UNKNOWN_ORACLE_KEY,
)


@dataclass
class LintResult:
    """Result of linting a DefinitionContract.

    Attributes:
        ok: True if all required semantics are present and valid.
        missing: Required fields/semantics that are absent or invalid (stable strings).
        warnings: Non-blocking concerns (never for unknown keys or type errors).
    """

    ok: bool
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Allowed keys per category+resolution_type (strictly enforced)
_CONDITION_KEYS_TOUCH = frozenset({"op", "level", "window"})
_CONDITION_KEYS_CLOSE = frozenset({"op", "level", "window", "measurement_time"})

_ORACLE_KEYS_BASE = frozenset({"feed", "rounding"})
_ORACLE_KEYS_TOUCH = _ORACLE_KEYS_BASE | frozenset({"finality"})
_ORACLE_KEYS_CLOSE = _ORACLE_KEYS_BASE


def _check_cutoff(cutoff_str: str) -> bool:
    """Check if cutoff_ts_utc is parseable as ISO-8601."""
    try:
        dt = datetime.fromisoformat(cutoff_str)
        # Must have timezone info or be parseable
        if dt.tzinfo is None:
            # Try adding Z
            datetime.fromisoformat(cutoff_str.replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False


def lint(contract: DefinitionContract) -> LintResult:
    """Lint a DefinitionContract for category-aware semantic completeness.

    Args:
        contract: The DefinitionContract to validate.

    Returns:
        LintResult with ok=False and named missing semantics if incomplete.
    """
    missing: list[str] = []
    warnings: list[str] = []

    if contract.category != "crypto_threshold":
        missing.append(f"unsupported_category: {contract.category}")
        return LintResult(ok=False, missing=missing, warnings=warnings)

    if contract.resolution_type not in ("touch", "close"):
        missing.append(f"invalid_resolution_type: {contract.resolution_type}")
        return LintResult(ok=False, missing=missing, warnings=warnings)

    # Determine allowed keys based on resolution_type
    is_touch = contract.resolution_type == "touch"
    allowed_condition_keys = _CONDITION_KEYS_TOUCH if is_touch else _CONDITION_KEYS_CLOSE
    allowed_oracle_keys = _ORACLE_KEYS_TOUCH if is_touch else _ORACLE_KEYS_CLOSE

    # ── Condition validation ──
    cond = contract.condition

    # Unknown keys → FAIL
    for key in cond:
        if key not in allowed_condition_keys:
            missing.append(f"{LINT_UNKNOWN_CONDITION_KEY}: {key}")

    # Required: op
    if "op" not in cond:
        missing.append(LINT_MISSING_CONDITION_OP)
    elif cond["op"] not in ALLOWED_CONDITION_OPS:
        missing.append(f"invalid_condition_op: {cond['op']}")

    # Required: level (must be int for crypto_threshold)
    if "level" not in cond:
        missing.append(LINT_MISSING_CONDITION_LEVEL)
    elif not isinstance(cond["level"], int):
        missing.append(LINT_INVALID_CONDITION_LEVEL_TYPE)

    # Required: window
    if "window" not in cond:
        missing.append(LINT_INVALID_CONDITION_WINDOW)
    elif is_touch and cond["window"] != "any_time":
        missing.append(LINT_INVALID_CONDITION_WINDOW)
    elif not is_touch and cond["window"] != "at_close":
        missing.append(LINT_INVALID_CONDITION_WINDOW)

    # Close-specific: measurement_time
    if not is_touch:
        if "measurement_time" not in cond:
            missing.append(LINT_MISSING_MEASUREMENT_TIME)
        elif not _check_cutoff(str(cond["measurement_time"])):
            missing.append(f"invalid_measurement_time: {cond.get('measurement_time')}")

    # ── Oracle details validation ──
    oracle = contract.oracle_details

    # Unknown keys → FAIL
    for key in oracle:
        if key not in allowed_oracle_keys:
            missing.append(f"{LINT_UNKNOWN_ORACLE_KEY}: {key}")

    # Required: feed
    if "feed" not in oracle:
        missing.append(LINT_MISSING_ORACLE_FEED)

    # Required: rounding (must be from allowlist)
    if "rounding" not in oracle:
        missing.append(LINT_MISSING_ORACLE_ROUNDING)
    elif oracle["rounding"] not in ALLOWED_ROUNDING_RULES:
        missing.append(LINT_INVALID_ORACLE_ROUNDING)

    # Touch-specific: finality
    if is_touch and "finality" not in oracle:
        missing.append(LINT_MISSING_ORACLE_FINALITY)

    # ── Cutoff validation ──
    if not _check_cutoff(contract.cutoff_ts_utc):
        missing.append(LINT_INVALID_CUTOFF)

    return LintResult(ok=len(missing) == 0, missing=missing, warnings=warnings)
