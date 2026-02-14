"""
Definition loader — reads DefinitionContracts from JSON file into DefinitionRegistry.

Design constraints (Loop 4 consistent):
  - File path via env: DEFINITION_CONTRACTS_PATH
  - If file missing: log once, proceed with empty registry (fail-closed is correct)
  - If file exists but has invalid contracts: raise ValueError (hard fail startup)
  - Each contract is lint-validated on register (DefinitionRegistry enforces this)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from definitions.registry import DefinitionRegistry
from models.definition_contract import DefinitionContract

logger = logging.getLogger(__name__)

# Fields that go into DefinitionContract constructor (excluding definition_hash)
_CONTRACT_FIELDS = frozenset(
    {
        "market_id",
        "category",
        "resolution_type",
        "underlying",
        "quote_ccy",
        "cutoff_ts_utc",
        "oracle_source",
        "oracle_details",
        "condition",
        "venue_rules_version",
    }
)


def load_definitions(
    path: Path,
    registry: DefinitionRegistry,
) -> int:
    """Load DefinitionContracts from a JSON file into the registry.

    Args:
        path: Path to JSON file containing a list of contract dicts.
        registry: DefinitionRegistry to populate.

    Returns:
        Number of contracts successfully loaded.

    Raises:
        ValueError: If the file exists but contains invalid contracts.
        FileNotFoundError: Not raised — missing file logs a warning and returns 0.
    """
    if not path.exists():
        logger.warning("Definition contracts file not found: %s — proceeding with empty registry", path)
        return 0

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}")

    loaded = 0
    errors: list[str] = []

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            errors.append(f"Entry {i}: expected dict, got {type(entry).__name__}")
            continue

        # Reject unknown keys (fail-closed)
        unknown_keys = set(entry.keys()) - _CONTRACT_FIELDS
        if unknown_keys:
            errors.append(f"Entry {i} (market_id={entry.get('market_id', '?')[:16]}): unknown keys {unknown_keys}")
            continue

        # Reject missing required keys
        missing_keys = _CONTRACT_FIELDS - set(entry.keys())
        if missing_keys:
            errors.append(f"Entry {i} (market_id={entry.get('market_id', '?')[:16]}): missing keys {missing_keys}")
            continue

        try:
            contract = DefinitionContract(
                market_id=entry["market_id"],
                category=entry["category"],
                resolution_type=entry["resolution_type"],
                underlying=entry["underlying"],
                quote_ccy=entry["quote_ccy"],
                cutoff_ts_utc=entry["cutoff_ts_utc"],
                oracle_source=entry["oracle_source"],
                oracle_details=entry["oracle_details"],
                condition=entry["condition"],
                venue_rules_version=entry["venue_rules_version"],
            )
            # register() calls lint — raises ValueError if lint fails
            registry.register(contract)
            loaded += 1
        except ValueError as e:
            errors.append(f"Entry {i} (market_id={entry.get('market_id', '?')[:16]}): {e}")
            continue

    if errors:
        error_msg = f"Definition loader found {len(errors)} invalid contract(s) in {path}:\n" + "\n".join(
            f"  • {e}" for e in errors
        )
        raise ValueError(error_msg)

    logger.info("Loaded %d definition contracts from %s", loaded, path)
    return loaded


def dump_definitions(registry: DefinitionRegistry) -> str:
    """Format loaded definitions for display.

    Returns:
        Human-readable summary string.
    """
    contracts = registry.all_contracts()
    if not contracts:
        return "No definitions loaded."

    lines = [f"Loaded {len(contracts)} definition contract(s):"]
    for c in contracts:
        lines.append(
            f"  {c.market_id[:16]}... | {c.underlying}/{c.quote_ccy} | "
            f"{c.resolution_type} {c.condition.get('op', '?')} {c.condition.get('level', '?')} | "
            f"hash={c.definition_hash[:12]}..."
        )
    return "\n".join(lines)
