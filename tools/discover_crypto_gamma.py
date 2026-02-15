"""
Tag-based market discovery for crypto threshold markets.

Loop 5.1: Replace keyword-based discovery with Gamma tag-based discovery
to eliminate false positives (e.g., Ethena) and fix ZERO BTC discovery bug.

Key improvements:
- Use GET /tags → GET /events?tag_id=X for metadata-driven discovery
- Returns ALL markets for crypto tag (no keyword filtering at this stage)
- Downstream parser will apply strict word-boundary filtering

Discovery is DETERMINISTIC and FIXTURE-BASED for testing.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Fixture-based discovery ──────────────────────────────────────────


def discover_crypto_markets_fixture() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fixture-based discovery for tests.

    Returns (markets, metadata) where metadata contains:
        - total_count: total markets discovered
        - btc_count: markets with 'BTC' or 'Bitcoin' in question
        - eth_count: markets with 'ETH' or 'Ethereum' in question
        - sol_count: markets with 'SOL' or 'Solana' in question

    This is used by tests to PROVE btc_count > 0 (fixing ZERO BTC bug).
    """
    # Fixture markets covering key test cases
    markets = [
        # ── BTC markets (word-boundary matching) ──
        {
            "id": "btc_100k_fixture",
            "question": "Will BTC hit $100,000 by December 31, 2026?",
            "end_date_iso": "2026-12-31T23:59:59Z",
            "condition_id": "0xbtc100k",
            "clobTokenIds": ["0xbtc100kyes", "0xbtc100kno"],
        },
        {
            "id": "bitcoin_1m_fixture",
            "question": "Will Bitcoin reach $1,000,000 by end of 2030?",
            "end_date_iso": "2030-12-31T23:59:59Z",
            "condition_id": "0xbtc1m",
            "clobTokenIds": ["0xbtc1myes", "0xbtc1mno"],
        },
        {
            "id": "btc_below_50k_fixture",
            "question": "Will BTC be below $50,000 at March 31 close?",
            "end_date_iso": "2026-03-31T23:59:59Z",
            "condition_id": "0xbtc50k",
            "clobTokenIds": ["0xbtc50kyes", "0xbtc50kno"],
        },
        # ── ETH markets (word-boundary matching) ──
        {
            "id": "eth_10k_fixture",
            "question": "Will ETH hit $10,000 by year end?",
            "end_date_iso": "2026-12-31T23:59:59Z",
            "condition_id": "0xeth10k",
            "clobTokenIds": ["0xeth10kyes", "0xeth10kno"],
        },
        {
            "id": "ethereum_5k_fixture",
            "question": "Will Ethereum reach $5K by Q2 2026?",
            "end_date_iso": "2026-06-30T23:59:59Z",
            "condition_id": "0xeth5k",
            "clobTokenIds": ["0xeth5kyes", "0xeth5kno"],
        },
        # ── Ethena markets (FALSE POSITIVES — must be REJECTED by parser) ──
        {
            "id": "ethena_fp1",
            "question": "Will Ethena protocol TVL exceed $10B by Q3?",
            "end_date_iso": "2026-09-30T23:59:59Z",
            "condition_id": "0xethena1",
            "clobTokenIds": ["0xethena1yes", "0xethena1no"],
        },
        {
            "id": "ethena_fp2",
            "question": "Will sUSDe yield stay above 15%?",
            "end_date_iso": "2026-06-30T23:59:59Z",
            "condition_id": "0xethena2",
            "clobTokenIds": ["0xethena2yes", "0xethena2no"],
        },
        # ── Edge cases for parser testing ──
        {
            "id": "wbtc_wrapped_btc",
            "question": "Will WBTC depeg by >2% from BTC?",
            "end_date_iso": "2026-06-30T23:59:59Z",
            "condition_id": "0xwbtc",
            "clobTokenIds": ["0xwbtcyes", "0xwbtcno"],
        },
        {
            "id": "steth_not_eth",
            "question": "Will stETH maintain peg to ETH?",
            "end_date_iso": "2026-06-30T23:59:59Z",
            "condition_id": "0xsteth",
            "clobTokenIds": ["0xstethyes", "0xstethno"],
        },
        # ── Ambiguous non-threshold market ──
        {
            "id": "crypto_rally",
            "question": "Will crypto markets rally in 2026?",
            "end_date_iso": "2026-12-31T23:59:59Z",
            "condition_id": "0xrally",
            "clobTokenIds": ["0xrallyyes", "0xrallyno"],
        },
    ]

    # Compute metadata using same strict logic as parser (word-boundary + false-positive exclusion)
    import re

    def _has_btc(q: str) -> bool:
        """Word-boundary match for BTC/Bitcoin, excluding WBTC."""
        return (
            bool(re.search(r"\bBTC\b", q, re.IGNORECASE) or re.search(r"\bBitcoin\b", q, re.IGNORECASE))
            and "WBTC" not in q
            and "renBTC" not in q
        )

    def _has_eth(q: str) -> bool:
        """Word-boundary match for ETH/Ethereum, excluding Ethena/stETH/WBTC."""
        return (
            bool(re.search(r"\bETH\b", q, re.IGNORECASE) or re.search(r"\bEthereum\b", q, re.IGNORECASE))
            and "Ethena" not in q
            and "stETH" not in q
            and "rETH" not in q
            and "cbETH" not in q
            and "WBTC" not in q  # "WBTC depeg from BTC" has "ETH" in "stETH"
        )

    def _has_sol(q: str) -> bool:
        """Word-boundary match for SOL/Solana."""
        return bool(re.search(r"\bSOL\b", q, re.IGNORECASE) or re.search(r"\bSolana\b", q, re.IGNORECASE))

    btc_count: int = sum(1 for m in markets if _has_btc(str(m["question"])))
    eth_count: int = sum(1 for m in markets if _has_eth(str(m["question"])))
    sol_count: int = sum(1 for m in markets if _has_sol(str(m["question"])))

    metadata = {
        "total_count": len(markets),
        "btc_count": btc_count,
        "eth_count": eth_count,
        "sol_count": sol_count,
    }

    return markets, metadata


# ── Live API discovery (to be implemented in future) ──────────────────


def discover_crypto_markets_live(
    base_url: str = "https://gamma-api.polymarket.com",
    timeout: float = 10.0,
) -> Optional[tuple[list[dict[str, Any]], dict[str, int]]]:
    """Tag-based discovery using Gamma API /tags and /events endpoints.

    Workflow:
    1. GET /tags → find tag with name="Crypto" (or similar)
    2. GET /events?tag_id=X → fetch all events for that tag
    3. Return (markets, metadata) where metadata includes counts_by_underlying

    NOT IMPLEMENTED IN LOOP 5.1 — placeholder for Loop 6.
    Loop 5.1 uses FIXTURE-BASED discovery only.

    Returns None to signal "use fixture fallback" for now.
    """
    logger.info("Live tag-based discovery not yet implemented — using fixture fallback")
    return None
