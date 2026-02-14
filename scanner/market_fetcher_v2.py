"""
Enhanced Market Fetcher for Allocator-Grade System (GAP #6 FIX).

CRITICAL: Fetches ALL metadata needed for production:
- time_to_close (NOT hours_to_expiry)
- tick_size from market metadata
- feeRateBps from market metadata
- event_id + negRisk flags
- min_size (if available)
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp

from config import GAMMA_API_URL, MAX_REQUESTS_PER_MINUTE
from models.types import Event, Market, MarketState

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.calls: list[float] = []

    async def acquire(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= self.max_per_minute:
            sleep_time = 60 - (now - self.calls[0]) + 0.1
            await asyncio.sleep(sleep_time)
        self.calls.append(time.time())


_rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)


def _parse_json_field(value) -> list:
    """Parse a field that may be a JSON string or already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def compute_time_to_close(end_date_str: Optional[str]) -> Optional[float]:
    """
    Compute time_to_close in hours (GAP #6 FIX).

    CRITICAL: Returns time until trading ENDS (not resolution time).

    Args:
        end_date_str: ISO timestamp string

    Returns:
        Hours to close, or None if no end date
    """
    if not end_date_str:
        return None

    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours = (end - now).total_seconds() / 3600
        return hours if hours > 0 else 0.0
    except Exception as e:
        logger.warning(f"Failed to parse end_date: {end_date_str}: {e}")
        return None


def compute_activity_score(market: Market) -> float:
    """
    Compute activity likelihood score for market selection.

    Markets with higher activity are more likely to emit WS book updates,
    keeping books fresh and Active Set non-zero during burn-in.

    Components:
    - volume_24h (50%): Recent trading volume
    - liquidity (30%): Current liquidity depth
    - time_recency (20%): Closer to resolution = more activity

    Returns:
        Activity score (0-1), higher = more likely to have active book updates
    """
    from config import ACTIVITY_SCORE_WEIGHTS

    # Normalize volume (assume $10k is "high" volume)
    volume_score = min(1.0, market.volume_24h / 10000.0)

    # Normalize liquidity (assume $20k is "high" liquidity)
    liquidity_score = min(1.0, market.liquidity / 20000.0)

    # Time recency: markets closer to resolution tend to be more active
    # But not TOO close (we exclude close_window separately)
    if market.time_to_close is None:
        time_score = 0.0
    elif market.time_to_close < 24:
        time_score = 0.0  # Will be filtered by close_window check
    elif market.time_to_close < 168:  # < 1 week
        time_score = 0.8
    elif market.time_to_close < 720:  # < 1 month
        time_score = 0.5
    else:
        time_score = 0.3

    activity_score = (
        ACTIVITY_SCORE_WEIGHTS["volume_24h"] * volume_score
        + ACTIVITY_SCORE_WEIGHTS["liquidity"] * liquidity_score
        + ACTIVITY_SCORE_WEIGHTS["time_recency"] * time_score
    )

    return activity_score


async def fetch_markets_with_metadata(
    limit: int = 500,
    min_liquidity: float = 500.0,
    active_only: bool = True,
) -> List[Market]:
    """
    Fetch markets from Gamma API with complete metadata (GAP #6 FIX).

    CRITICAL: Extracts all fields needed for production validation:
    - time_to_close (computed from endDate)
    - tick_size (from market metadata, default 0.01)
    - feeRateBps (from market metadata, default 200)
    - event_id + negRisk flags
    - yes/no token IDs

    Args:
        limit: Maximum markets to fetch
        min_liquidity: Minimum liquidity filter
        active_only: Only fetch active markets

    Returns:
        List of Market instances with full metadata
    """
    markets = []  # type: ignore[var-annotated]
    offset = 0
    page_size = 100

    async with aiohttp.ClientSession() as session:
        while len(markets) < limit:
            await _rate_limiter.acquire()

            try:
                params = {
                    "limit": page_size,
                    "offset": offset,
                }
                if active_only:
                    params["active"] = "true"  # type: ignore[assignment]
                    params["closed"] = "false"  # type: ignore[assignment]

                async with session.get(
                    f"{GAMMA_API_URL}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Gamma API returned status {resp.status}")
                        break

                    data = await resp.json()

            except Exception as e:
                logger.error(f"Failed to fetch markets from Gamma API: {e}")
                break

            if not data or not isinstance(data, list):
                break

            for item in data:
                try:
                    market = _parse_market_with_metadata(item)
                    if market and market.liquidity >= min_liquidity:
                        markets.append(market)
                except Exception as e:
                    logger.warning(f"Failed to parse market: {e}")
                    continue

            logger.info(f"Fetched {len(markets)} markets (offset={offset})")

            offset += page_size
            if len(data) < page_size:
                break  # No more pages

    logger.info(f"Total markets fetched: {len(markets)}")
    return markets


def _parse_market_with_metadata(item: dict) -> Optional[Market]:
    """
    Parse market from Gamma API with full metadata (GAP #6 FIX).

    Args:
        item: Market dict from Gamma API

    Returns:
        Market instance with metadata, or None if invalid
    """
    # Basic fields
    condition_id = item.get("conditionId", "")
    question = item.get("question", "").strip()

    if not condition_id or not question:
        return None

    # Token IDs (may be JSON strings)
    token_ids = _parse_json_field(item.get("clobTokenIds", []))
    outcomes = _parse_json_field(item.get("outcomes", []))
    outcome_prices = _parse_json_field(item.get("outcomePrices", []))

    yes_token_id = None
    no_token_id = None
    yes_bid = 0.0
    yes_ask = 0.0
    no_bid = 0.0
    no_ask = 0.0

    # Parse YES/NO token IDs and prices
    for i, outcome in enumerate(outcomes):
        token_id = token_ids[i] if i < len(token_ids) else None
        price = float(outcome_prices[i]) if i < len(outcome_prices) else 0.0

        # Use price as both bid/ask (spread unknown from Gamma)
        # Real spread comes from CLOB order book
        if outcome.lower() == "yes":
            yes_token_id = token_id
            yes_bid = yes_ask = price
        elif outcome.lower() == "no":
            no_token_id = token_id
            no_bid = no_ask = price

    if not yes_token_id or not no_token_id:
        return None  # Need both tokens

    # Time model (GAP #6 FIX)
    end_date_str = item.get("endDate") or item.get("endDateIso")
    time_to_close = compute_time_to_close(end_date_str)

    # Market metadata (GAP #6 FIX)
    tick_size = item.get("tick_size", 0.01)  # Default 0.01 if not provided
    fee_rate_bps = item.get("feeRateBps", 200)  # Default 200 (2%)

    # Liquidity
    liquidity = float(item.get("liquidityNum") or item.get("liquidity") or 0)
    volume_24h = float(item.get("volumeNum") or item.get("volume") or 0)

    # Category inference
    tags = []  # type: ignore[var-annotated]
    events_data = item.get("events") or []
    for event_dict in events_data:
        if isinstance(event_dict, dict):
            tags.extend(event_dict.get("tags") or [])

    category = _infer_category(question, tags)

    # Event association (GAP #7 FIX)
    event = None
    if events_data and isinstance(events_data, list) and len(events_data) > 0:
        event_dict = events_data[0]  # Use first event
        if isinstance(event_dict, dict):
            event = Event(
                event_id=event_dict.get("id", ""),
                title=event_dict.get("title", ""),
                neg_risk=event_dict.get("negRisk", False),
                augmented_neg_risk=event_dict.get("augmentedNegRisk", False),
                metadata=event_dict,
            )

    # Build Market instance
    market = Market(
        condition_id=condition_id,
        question=question,
        description=item.get("description", ""),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        time_to_close=time_to_close,
        category=category,
        liquidity=liquidity,
        volume_24h=volume_24h,
        tick_size=tick_size,
        fee_rate_bps=fee_rate_bps,
        rules_text=item.get("rules", ""),
        resolution_source=item.get("resolutionSource", ""),
        state=MarketState.NORMAL,  # Will be updated by market_state module
        event=event,
        raw_metadata=item,
    )

    return market


def _infer_category(question: str, tags: list) -> str:
    """Infer market category from question and tags."""
    import re

    q = question.lower()
    tag_names = []
    for t in tags or []:
        if isinstance(t, dict):
            tag_names.append(t.get("label", "").lower())
        else:
            tag_names.append(str(t).lower())
    tag_str = " ".join(tag_names)

    def matches(keywords: list[str], text: str) -> bool:
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text):
                return True
        return False

    crypto_kw = [
        "btc",
        "bitcoin",
        "eth",
        "ethereum",
        "sol",
        "solana",
        "crypto",
        "xrp",
        "bnb",
        "doge",
        "usdt",
        "altcoin",
        "defi",
        "nft",
        "blockchain",
        "web3",
    ]
    sports_kw = [
        "nfl",
        "nba",
        "mlb",
        "nhl",
        "soccer",
        "football",
        "basketball",
        "championship",
        "super bowl",
        "playoffs",
        "mvp",
    ]

    if matches(crypto_kw, q) or matches(crypto_kw, tag_str):
        return "crypto"
    if matches(sports_kw, q) or matches(sports_kw, tag_str):
        return "sports"

    # Politics detection
    if any(
        kw in q
        for kw in [
            "election",
            "president",
            "senate",
            "congress",
            "democrat",
            "republican",
            "biden",
            "trump",
        ]
    ):
        return "politics"

    return "other"
