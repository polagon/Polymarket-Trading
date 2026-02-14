"""
DefinitionRegistry — market_id → DefinitionContract mapping.

Only lint-ok contracts can be registered.
build_from_market() returns None if incomplete (no partial objects ever).

Loop 4: Fail-closed contract storage.
"""

from __future__ import annotations

import logging
from typing import Optional

from definitions.lint import lint
from models.definition_contract import DefinitionContract

logger = logging.getLogger(__name__)


class DefinitionRegistry:
    """Registry of validated DefinitionContracts.

    Only contracts that pass lint can be stored.
    """

    def __init__(self) -> None:
        self._contracts: dict[str, DefinitionContract] = {}

    def register(self, contract: DefinitionContract) -> None:
        """Register a lint-validated contract.

        Args:
            contract: The DefinitionContract to register.

        Raises:
            ValueError: If the contract fails lint (with stable missing semantics).
        """
        result = lint(contract)
        if not result.ok:
            raise ValueError(f"DefinitionContract lint failed for {contract.market_id}: {', '.join(result.missing)}")
        self._contracts[contract.market_id] = contract
        logger.info(
            f"Registered definition: {contract.market_id} "
            f"({contract.category}/{contract.resolution_type}) "
            f"hash={contract.definition_hash[:12]}..."
        )

    def get(self, market_id: str) -> Optional[DefinitionContract]:
        """Get a registered contract by market_id.

        Returns:
            The DefinitionContract or None if not registered.
        """
        return self._contracts.get(market_id)

    def has(self, market_id: str) -> bool:
        """Check if a contract is registered for market_id."""
        return market_id in self._contracts

    def all_contracts(self) -> list[DefinitionContract]:
        """Return all registered contracts."""
        return list(self._contracts.values())

    def __len__(self) -> int:
        return len(self._contracts)

    def __contains__(self, market_id: str) -> bool:
        return market_id in self._contracts


def build_from_market(
    market_id: str,
    category: str,
    resolution_type: str,
    underlying: str,
    quote_ccy: str,
    cutoff_ts_utc: str,
    oracle_source: str,
    oracle_details: dict,  # type: ignore[type-arg]
    condition: dict,  # type: ignore[type-arg]
    venue_rules_version: str = "polymarket_v2",
) -> Optional[DefinitionContract]:
    """Build a DefinitionContract from explicit fields.

    Returns None if the contract cannot be constructed (any required field missing).
    Never returns partial objects.
    """
    try:
        contract = DefinitionContract(
            market_id=market_id,
            category=category,
            resolution_type=resolution_type,
            underlying=underlying,
            quote_ccy=quote_ccy,
            cutoff_ts_utc=cutoff_ts_utc,
            oracle_source=oracle_source,
            oracle_details=oracle_details,
            condition=condition,
            venue_rules_version=venue_rules_version,
        )
        # Validate via lint
        result = lint(contract)
        if not result.ok:
            logger.debug(f"build_from_market failed lint for {market_id}: {', '.join(result.missing)}")
            return None
        return contract
    except (TypeError, ValueError) as e:
        logger.debug(f"build_from_market failed for {market_id}: {e}")
        return None
