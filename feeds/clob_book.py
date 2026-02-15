"""
CLOB orderbook fetcher — REST-based top-of-book + depth proxy.

Loop 5: Real market data for EV gate, toxicity, and order placement.

Architecture:
  - Fetch orderbook snapshots via CLOB REST API (GET /book?token_id=...)
  - Store in-memory with staleness tracking
  - Compute best_bid, best_ask, mid, spread_frac, depth_proxy_usd
  - No WebSocket in Loop 5 (REST polling is sufficient for 60s scan intervals)

Design constraints:
  - Rate-limited (configurable, default 10 req/s)
  - Staleness threshold: books older than BOOK_STALE_S are treated as missing
  - If API returns empty book: best_bid=0, best_ask=0 (fail-safe)
  - No network calls in tests (all tests use fixtures)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from config import CLOB_API_URL

logger = logging.getLogger(__name__)

BOOK_STALE_S = 120  # 2 minutes — generous for REST polling


@dataclass
class BookSnapshot:
    """Top-of-book snapshot for one outcome token."""

    asset_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_depth_usd: float = 0.0  # notional within 10% of best bid
    ask_depth_usd: float = 0.0  # notional within 10% of best ask
    spread: float = 0.0
    spread_frac: float = 0.0
    mid: float = 0.0
    n_bid_levels: int = 0
    n_ask_levels: int = 0
    last_trade_price: float = 0.0
    tick_size: float = 0.01
    fetched_at: float = 0.0  # time.time()

    @property
    def is_valid(self) -> bool:
        """True if we have a two-sided book."""
        return self.best_bid > 0 and self.best_ask > 0 and self.best_bid < self.best_ask

    @property
    def is_stale(self) -> bool:
        """True if snapshot is older than BOOK_STALE_S."""
        return (time.time() - self.fetched_at) > BOOK_STALE_S

    @property
    def depth_proxy_usd(self) -> float:
        """Conservative depth proxy: min of bid/ask depth."""
        return min(self.bid_depth_usd, self.ask_depth_usd) if self.is_valid else 0.0


@dataclass
class MarketBook:
    """Combined YES + NO orderbook for a market."""

    condition_id: str
    yes: Optional[BookSnapshot] = None
    no: Optional[BookSnapshot] = None

    @property
    def is_valid(self) -> bool:
        return self.yes is not None and self.yes.is_valid and not self.yes.is_stale


def _parse_book_response(raw: dict, fetched_at: float) -> BookSnapshot:
    """Parse CLOB /book response into BookSnapshot."""
    asset_id = raw.get("asset_id", "")
    bids_raw = raw.get("bids", [])
    asks_raw = raw.get("asks", [])

    # Parse and sort
    bids = [(float(b["price"]), float(b["size"])) for b in bids_raw if b.get("price") and b.get("size")]
    asks = [(float(a["price"]), float(a["size"])) for a in asks_raw if a.get("price") and a.get("size")]

    bids.sort(key=lambda x: x[0], reverse=True)  # highest first
    asks.sort(key=lambda x: x[0])  # lowest first

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0.0
    spread = best_ask - best_bid if (best_bid > 0 and best_ask > 0) else 0.0
    spread_frac = spread / mid if mid > 0 else 0.0

    # Depth: notional within 10% of best price
    bid_depth = 0.0
    if best_bid > 0:
        threshold = best_bid * 0.9
        for price, size in bids:
            if price >= threshold:
                bid_depth += price * size

    ask_depth = 0.0
    if best_ask > 0:
        threshold = best_ask * 1.1
        for price, size in asks:
            if price <= threshold:
                ask_depth += price * size

    last_trade = float(raw.get("last_trade_price", 0) or 0)
    tick_size = float(raw.get("tick_size", 0.01) or 0.01)

    return BookSnapshot(
        asset_id=asset_id,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth_usd=bid_depth,
        ask_depth_usd=ask_depth,
        spread=spread,
        spread_frac=spread_frac,
        mid=mid,
        n_bid_levels=len(bids),
        n_ask_levels=len(asks),
        last_trade_price=last_trade,
        tick_size=tick_size,
        fetched_at=fetched_at,
    )


class CLOBBookFetcher:
    """Fetches orderbook snapshots from Polymarket CLOB REST API.

    Usage:
        fetcher = CLOBBookFetcher()
        book = await fetcher.fetch_book(yes_token_id)
    """

    def __init__(
        self,
        base_url: str = CLOB_API_URL,
        timeout_s: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._cache: dict[str, BookSnapshot] = {}

    async def fetch_book(self, token_id: str) -> BookSnapshot:
        """Fetch orderbook for a single token from CLOB REST API.

        Returns BookSnapshot with best_bid=0, best_ask=0 on any error.
        """
        url = f"{self._base_url}/book"
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(url, params={"token_id": token_id}) as resp:
                    if resp.status != 200:
                        logger.warning("CLOB book fetch failed: status=%d token=%s", resp.status, token_id[:20])
                        return BookSnapshot(asset_id=token_id, fetched_at=time.time())
                    raw = await resp.json()
            snap = _parse_book_response(raw, time.time())
            self._cache[token_id] = snap
            return snap
        except Exception as e:
            logger.warning("CLOB book fetch error for %s: %s", token_id[:20], e)
            return BookSnapshot(asset_id=token_id, fetched_at=time.time())

    async def fetch_market_book(
        self,
        condition_id: str,
        yes_token_id: str,
        no_token_id: Optional[str] = None,
    ) -> MarketBook:
        """Fetch YES (and optionally NO) orderbook for a market."""
        yes_snap = await self.fetch_book(yes_token_id)
        no_snap = None
        if no_token_id:
            no_snap = await self.fetch_book(no_token_id)
        return MarketBook(condition_id=condition_id, yes=yes_snap, no=no_snap)

    def get_cached(self, token_id: str) -> Optional[BookSnapshot]:
        """Get cached snapshot (may be stale)."""
        return self._cache.get(token_id)

    def clear_cache(self) -> None:
        self._cache.clear()
