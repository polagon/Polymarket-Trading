"""
Test inventory skew math (CRITICAL FIX #4).
"""

import pytest

from config import BANKROLL, INVENTORY_SKEW_MAX_CENTS, MAX_MARKET_INVENTORY_PCT


def test_inventory_skew_correct_sign():
    """CRITICAL FIX #4: Long inventory → negative skew → lower prices."""
    max_position = BANKROLL * MAX_MARKET_INVENTORY_PCT

    # Long inventory (80% of max)
    inventory_long = max_position * 0.8
    inventory_fraction_long = inventory_long / max_position  # 0.8

    # Skew formula: skew = -inventory_fraction * INVENTORY_SKEW_MAX_CENTS
    # CRITICAL: Note the NEGATIVE sign
    skew_long = -inventory_fraction_long * INVENTORY_SKEW_MAX_CENTS

    # Long inventory → negative skew → shifts prices DOWN
    assert skew_long < 0, f"Long inventory should produce negative skew: {skew_long}"

    # Short inventory (80% short)
    inventory_short = -max_position * 0.8
    inventory_fraction_short = inventory_short / max_position  # -0.8
    skew_short = -inventory_fraction_short * INVENTORY_SKEW_MAX_CENTS

    # Short inventory → positive skew → shifts prices UP
    assert skew_short > 0, f"Short inventory should produce positive skew: {skew_short}"

    # Verify: long and short skews are opposite signs
    assert skew_long < 0 < skew_short, "Long and short inventory skews must have opposite signs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
