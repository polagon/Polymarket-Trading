"""
Definition Gate — fail-closed definition check.

No DefinitionContract = no trade.
Reason strings are stable enums from models.reasons.

Loop 4: First gate in the evaluation chain.
"""

from __future__ import annotations

from definitions.lint import lint
from definitions.registry import DefinitionRegistry
from models.reasons import (
    REASON_DEFINITION_INCOMPLETE,
    REASON_DEFINITION_OK,
    REASON_NO_DEFINITION,
)


def check(market_id: str, registry: DefinitionRegistry) -> tuple[bool, str]:
    """Check if a market has a valid, lint-passing DefinitionContract.

    Args:
        market_id: The market condition_id to check.
        registry: The DefinitionRegistry to look up.

    Returns:
        (ok, reason) tuple. ok=True means definition is valid.
        Reason strings are stable enums for analytics.
    """
    contract = registry.get(market_id)
    if contract is None:
        return (False, REASON_NO_DEFINITION)

    result = lint(contract)
    if not result.ok:
        reason = f"{REASON_DEFINITION_INCOMPLETE}: {'|'.join(result.missing)}"
        return (False, reason)

    return (True, REASON_DEFINITION_OK)
