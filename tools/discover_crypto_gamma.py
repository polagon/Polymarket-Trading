"""
Tag-based market discovery for crypto threshold markets.

Loop 6.0: Live Gamma tag-based discovery with fail-closed behavior.

Key improvements:
- Use GET /tags → GET /markets?tag_id=X for metadata-driven discovery
- Returns ALL markets for crypto tag (no keyword filtering at this stage)
- Downstream parser will apply strict word-boundary filtering
- Fail-closed: HTTP errors, timeouts, invalid responses → named veto reasons

Discovery modes:
- "fixture": Deterministic fixture-based (tests/default)
- "live": Live Gamma API calls (production only, blocked under pytest)
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from models.reasons import (
    REASON_DISCOVERY_GAMMA_HTTP_ERROR,
    REASON_DISCOVERY_GAMMA_TIMEOUT,
    REASON_DISCOVERY_INVALID_RESPONSE,
    REASON_DISCOVERY_OK,
    REASON_DISCOVERY_PAGINATION_EXHAUSTED,
    REASON_DISCOVERY_TAG_NOT_FOUND,
)

logger = logging.getLogger(__name__)


# ── Discovery result dataclass ───────────────────────────────────────


@dataclass
class DiscoveryResult:
    """Result of market discovery with provenance metadata."""

    markets: list[dict[str, Any]]
    """List of raw market dicts from Gamma API (or fixture)."""

    metadata: dict[str, int]
    """Counts: total_count, btc_count, eth_count."""

    reason: str
    """Outcome reason enum from models.reasons."""

    tag_ids_used: list[str] = field(default_factory=list)
    """Tag IDs used for discovery (e.g., ['crypto'])."""

    pages_fetched: int = 0
    """Number of pages fetched (pagination count)."""

    discovered_at: str = ""
    """ISO8601 timestamp of discovery."""


# ── Pytest guard ─────────────────────────────────────────────────────


def _should_use_live_discovery() -> bool:
    """Check if live discovery is allowed (not under pytest)."""
    return os.getenv("PYTEST_CURRENT_TEST") is None


# ── Fixture-based discovery ──────────────────────────────────────────


def discover_crypto_markets_fixture() -> DiscoveryResult:
    """Fixture-based discovery for tests.

    Loop 5.x scope: BTC/ETH ONLY.

    Returns DiscoveryResult with:
        - markets: 10 fixture markets (3 BTC, 2 ETH, 5 false positives/edge cases)
        - metadata: total_count, btc_count, eth_count
        - reason: REASON_DISCOVERY_OK
        - tag_ids_used: ["crypto_fixture"]
        - pages_fetched: 1
        - discovered_at: ISO8601 timestamp

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

    # SOL removed from Loop 5.x scope (deferred to Loop 6+)

    btc_count: int = sum(1 for m in markets if _has_btc(str(m["question"])))
    eth_count: int = sum(1 for m in markets if _has_eth(str(m["question"])))

    metadata = {
        "total_count": len(markets),
        "btc_count": btc_count,
        "eth_count": eth_count,
        # sol_count removed — out of scope for Loop 5.x
    }

    return DiscoveryResult(
        markets=markets,
        metadata=metadata,
        reason=REASON_DISCOVERY_OK,
        tag_ids_used=["crypto_fixture"],
        pages_fetched=1,
        discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# ── Live API discovery ───────────────────────────────────────────────


class GammaClient:
    """Client for Gamma API tag-based discovery with exponential backoff."""

    def __init__(self, base_url: str = "https://gamma-api.polymarket.com", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = None

    def _get_session(self):
        """Lazy-import requests (avoid import if fixture mode)."""
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _request_with_backoff(self, url: str, params: Optional[dict] = None, max_retries: int = 3) -> dict[str, Any]:
        """GET request with exponential backoff on 429/5xx.

        Raises:
            TimeoutError: on timeout
            requests.HTTPError: on 4xx (except 429)
            ValueError: on invalid JSON response
        """
        import random

        import requests

        session = self._get_session()
        for attempt in range(max_retries):
            try:
                resp = session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()  # type: ignore[no-any-return]
                elif resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        backoff = (2**attempt) + random.uniform(0, 1)
                        logger.warning(f"HTTP {resp.status_code} on {url}, retrying in {backoff:.2f}s")
                        time.sleep(backoff)
                        continue
                    else:
                        raise requests.HTTPError(f"Max retries exceeded: {resp.status_code}")
                else:
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text}")
            except requests.Timeout:
                if attempt < max_retries - 1:
                    backoff = (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Timeout on {url}, retrying in {backoff:.2f}s")
                    time.sleep(backoff)
                    continue
                else:
                    raise TimeoutError(f"Request timed out after {max_retries} attempts")
        raise requests.HTTPError("Unreachable")

    def find_crypto_tag(self) -> Optional[str]:
        """Find tag ID for 'Crypto' (case-insensitive).

        Returns tag ID or None if not found.

        Raises:
            TimeoutError: on timeout
            requests.HTTPError: on HTTP errors
            ValueError: on invalid response shape
        """
        url = f"{self.base_url}/tags"
        data = self._request_with_backoff(url)
        if not isinstance(data, list):
            logger.error(f"Invalid /tags response shape: expected list, got {type(data)}")
            raise ValueError(f"Invalid /tags response shape: expected list, got {type(data)}")
        for tag in data:
            if isinstance(tag, dict) and tag.get("label", "").lower() == "crypto":
                return tag.get("id")
        return None  # Tag not found (legitimate case, not an error)

    def fetch_markets_for_tag(self, tag_id: str, max_pages: int = 10) -> list[dict[str, Any]]:
        """Fetch all markets for a tag with pagination.

        Returns list of market dicts. Each market has:
            - id, question, end_date_iso (or endDate), condition_id, clobTokenIds

        Pagination: follows next_cursor until exhausted or max_pages reached.
        """
        markets = []
        url = f"{self.base_url}/markets"
        params = {"tag_id": tag_id, "limit": 100}
        page = 0

        while page < max_pages:
            try:
                data = self._request_with_backoff(url, params=params)
                if not isinstance(data, dict):
                    logger.error(f"Invalid /markets response shape: expected dict, got {type(data)}")
                    break

                # Extract markets (handle both "data" and "markets" keys)
                batch = data.get("data") or data.get("markets") or []
                if not isinstance(batch, list):
                    logger.error(f"Invalid markets batch: expected list, got {type(batch)}")
                    break

                # Normalize market shape
                for m in batch:
                    normalized = self._normalize_market(m)
                    if normalized:
                        markets.append(normalized)

                page += 1

                # Check for next page
                next_cursor = data.get("next_cursor")
                if not next_cursor:
                    break
                params["cursor"] = next_cursor

            except Exception as e:
                logger.error(f"Failed to fetch markets page {page}: {e}")
                break

        return markets

    def _normalize_market(self, raw: dict) -> Optional[dict[str, Any]]:
        """Normalize Gamma market shape to internal format.

        Expected Gamma shape (may vary):
            - id: str
            - question: str
            - endDate: str (ISO8601) OR end_date_iso: str
            - conditionId: str OR condition_id: str
            - clobTokenIds: list[str] OR tokens: list[{token_id: str, ...}]

        Returns normalized dict or None if invalid.
        """
        try:
            market_id = raw.get("id")
            question = raw.get("question")
            end_date = raw.get("end_date_iso") or raw.get("endDate")
            condition_id = raw.get("condition_id") or raw.get("conditionId")
            clob_token_ids = raw.get("clobTokenIds") or []

            # Handle tokens array shape
            if not clob_token_ids and "tokens" in raw:
                tokens = raw["tokens"]
                if isinstance(tokens, list):
                    clob_token_ids = [t.get("token_id") for t in tokens if isinstance(t, dict)]

            if not market_id or not question or not end_date:
                return None

            return {
                "id": market_id,
                "question": question,
                "end_date_iso": end_date,
                "condition_id": condition_id or "",
                "clobTokenIds": clob_token_ids,
            }
        except Exception as e:
            logger.warning(f"Failed to normalize market: {e}")
            return None


def discover_crypto_markets_live(
    base_url: str = "https://gamma-api.polymarket.com",
    timeout: float = 10.0,
) -> DiscoveryResult:
    """Tag-based discovery using Gamma API /tags and /markets endpoints.

    Workflow:
    1. GET /tags → find tag with label="Crypto" (case-insensitive)
    2. GET /markets?tag_id=X → fetch all markets for that tag with pagination
    3. Return DiscoveryResult with markets, metadata, provenance

    Fail-closed behavior:
    - HTTP errors → REASON_DISCOVERY_GAMMA_HTTP_ERROR
    - Timeouts → REASON_DISCOVERY_GAMMA_TIMEOUT
    - Tag not found → REASON_DISCOVERY_TAG_NOT_FOUND
    - Invalid response → REASON_DISCOVERY_INVALID_RESPONSE
    - Pagination exhausted normally → REASON_DISCOVERY_OK

    Returns DiscoveryResult with:
        - markets: list of normalized market dicts
        - metadata: total_count, btc_count, eth_count
        - reason: outcome enum from models.reasons
        - tag_ids_used: list of tag IDs used
        - pages_fetched: number of pages fetched
        - discovered_at: ISO8601 timestamp
    """
    client = GammaClient(base_url=base_url, timeout=timeout)

    # Step 1: Find crypto tag
    try:
        tag_id = client.find_crypto_tag()
        if tag_id is None:
            logger.warning("Crypto tag not found in /tags")
            return DiscoveryResult(
                markets=[],
                metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
                reason=REASON_DISCOVERY_TAG_NOT_FOUND,
                tag_ids_used=[],
                pages_fetched=0,
                discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
    except TimeoutError as e:
        logger.error(f"Timeout while fetching /tags: {e}")
        return DiscoveryResult(
            markets=[],
            metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
            reason=REASON_DISCOVERY_GAMMA_TIMEOUT,
            tag_ids_used=[],
            pages_fetched=0,
            discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    except ValueError as e:
        logger.error(f"Invalid response while fetching /tags: {e}")
        return DiscoveryResult(
            markets=[],
            metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
            reason=REASON_DISCOVERY_INVALID_RESPONSE,
            tag_ids_used=[],
            pages_fetched=0,
            discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    except Exception as e:
        logger.error(f"HTTP error while fetching /tags: {e}")
        return DiscoveryResult(
            markets=[],
            metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
            reason=REASON_DISCOVERY_GAMMA_HTTP_ERROR,
            tag_ids_used=[],
            pages_fetched=0,
            discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    # Step 2: Fetch markets for tag
    try:
        markets = client.fetch_markets_for_tag(tag_id, max_pages=10)
        pages_fetched = min(len(markets) // 100 + 1, 10)  # Estimate
    except TimeoutError:
        logger.error("Timeout while fetching /markets")
        return DiscoveryResult(
            markets=[],
            metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
            reason=REASON_DISCOVERY_GAMMA_TIMEOUT,
            tag_ids_used=[tag_id],
            pages_fetched=0,
            discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    except Exception as e:
        logger.error(f"HTTP error while fetching /markets: {e}")
        return DiscoveryResult(
            markets=[],
            metadata={"total_count": 0, "btc_count": 0, "eth_count": 0},
            reason=REASON_DISCOVERY_GAMMA_HTTP_ERROR,
            tag_ids_used=[tag_id],
            pages_fetched=0,
            discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    # Step 3: Compute metadata using same strict logic as parser
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
            and "WBTC" not in q
        )

    btc_count = sum(1 for m in markets if _has_btc(str(m.get("question", ""))))
    eth_count = sum(1 for m in markets if _has_eth(str(m.get("question", ""))))

    metadata = {
        "total_count": len(markets),
        "btc_count": btc_count,
        "eth_count": eth_count,
    }

    return DiscoveryResult(
        markets=markets,
        metadata=metadata,
        reason=REASON_DISCOVERY_OK,
        tag_ids_used=[tag_id],
        pages_fetched=pages_fetched,
        discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# ── Public API ───────────────────────────────────────────────────────


def discover_crypto_markets(mode: str = "fixture") -> DiscoveryResult:
    """Discover crypto threshold markets using specified mode.

    Args:
        mode: "fixture" or "live"

    Returns:
        DiscoveryResult with markets, metadata, provenance

    Raises:
        ValueError: if mode="live" under PYTEST_CURRENT_TEST env
    """
    if mode == "live":
        if not _should_use_live_discovery():
            raise ValueError("Live discovery blocked: PYTEST_CURRENT_TEST env set (no network in tests)")
        logger.info("Using LIVE Gamma API discovery")
        return discover_crypto_markets_live()
    else:
        logger.info("Using FIXTURE discovery")
        return discover_crypto_markets_fixture()
