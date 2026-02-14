"""
DefinitionContract — frozen, hash-guaranteed market definition.

Loop 4: Every trade requires a valid, lint-checked DefinitionContract.
No definition = no trade (fail-closed).

Canonicalization rules:
  - condition.level must be int for crypto_threshold
  - Allowed keys are strictly validated by lint (unknown keys FAIL)
  - JSON serialization: sort_keys=True, separators=(",", ":") — no spaces
  - definition_hash = SHA-256 of canonical JSON excluding hash field
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def compute_definition_hash(fields: dict) -> str:
    """Compute SHA-256 of canonical JSON of definition fields.

    Args:
        fields: Dict of all DefinitionContract fields EXCEPT definition_hash.

    Returns:
        Hex digest of SHA-256 hash.
    """
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _fields_for_hash(contract: DefinitionContract) -> dict:
    """Extract hashable fields from a DefinitionContract (excludes definition_hash)."""
    return {
        "market_id": contract.market_id,
        "category": contract.category,
        "resolution_type": contract.resolution_type,
        "underlying": contract.underlying,
        "quote_ccy": contract.quote_ccy,
        "cutoff_ts_utc": contract.cutoff_ts_utc,
        "oracle_source": contract.oracle_source,
        "oracle_details": contract.oracle_details,
        "condition": contract.condition,
        "venue_rules_version": contract.venue_rules_version,
    }


@dataclass(frozen=True)
class DefinitionContract:
    """Immutable market definition contract with computed hash.

    Construction: Pass definition_hash="" (empty string). The hash is computed
    automatically in __post_init__. Passing a non-empty hash raises ValueError.

    Attributes:
        market_id: Unique market identifier (condition_id).
        category: Market category. Only "crypto_threshold" in Loop 4.
        resolution_type: "touch" or "close".
        underlying: Asset symbol, e.g. "BTC", "ETH".
        quote_ccy: Quote currency, e.g. "USD".
        cutoff_ts_utc: ISO-8601 UTC cutoff timestamp.
        oracle_source: Canonical oracle identifier, e.g. "coingecko_v3".
        oracle_details: Source-specific details (validated keys only).
        condition: Resolution condition semantics (validated keys only).
        venue_rules_version: Venue rules version, e.g. "polymarket_v2".
        definition_hash: SHA-256 of canonical JSON (computed, never user-set).
    """

    market_id: str
    category: str
    resolution_type: str

    underlying: str
    quote_ccy: str

    cutoff_ts_utc: str

    oracle_source: str
    oracle_details: dict  # type: ignore[type-arg]

    condition: dict  # type: ignore[type-arg]

    venue_rules_version: str
    definition_hash: str = ""

    def __post_init__(self) -> None:
        if self.definition_hash != "":
            raise ValueError("definition_hash_must_be_computed")
        computed = compute_definition_hash(_fields_for_hash(self))
        object.__setattr__(self, "definition_hash", computed)
