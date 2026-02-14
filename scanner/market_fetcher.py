"""
Fetches active markets and prices from Polymarket Gamma API.
No authentication required for read-only access.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from config import GAMMA_API_URL, MAX_REQUESTS_PER_MINUTE, MIN_HOURS_TO_EXPIRY, MIN_MARKET_LIQUIDITY


@dataclass
class Market:
    condition_id: str
    question: str
    end_date_iso: Optional[str]
    category: str
    yes_token_id: Optional[str]
    no_token_id: Optional[str]
    yes_price: float = 0.0  # outcome price from Gamma API (0-1)
    no_price: float = 0.0
    liquidity: float = 0.0
    volume: float = 0.0
    hours_to_expiry: float = float("inf")


class RateLimiter:
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


def _hours_to_expiry(end_date_str: Optional[str]) -> float:
    if not end_date_str:
        return float("inf")
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (end - now).total_seconds() / 3600
    except Exception:
        return float("inf")


def _infer_category(question: str, tags: list) -> str:
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
            # Use word boundaries for short tokens to avoid false substring matches
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text):
                return True
        return False

    # Crypto: use word-boundary matching to avoid "netherlands" matching "eth"
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
        "market cap",
    ]
    weather_kw = [
        "temperature",
        "hurricane",
        "tropical storm",
        "rain",
        "snow",
        "tornado",
        "flood",
        "drought",
        "weather",
        "celsius",
        "fahrenheit",
        "wildfire",
        "earthquake",
        "blizzard",
        "typhoon",
        "precipitation",
        "heat wave",
        "frost",
    ]
    sports_kw = [
        "nfl",
        "nba",
        "mlb",
        "nhl",
        "soccer",
        "football",
        "basketball",
        "baseball",
        "championship",
        "super bowl",
        "world cup",
        "playoffs",
        "mvp",
        "draft",
        "premier league",
        "champions league",
        "ufc",
        "nascar",
        "tennis",
        "golf",
        "formula 1",
        "f1",
    ]

    if matches(crypto_kw, q) or matches(crypto_kw, tag_str):
        return "crypto"
    if matches(weather_kw, q) or matches(weather_kw, tag_str):
        return "weather"
    if matches(sports_kw, q) or matches(sports_kw, tag_str):
        return "sports"
    return "other"


async def fetch_active_markets(limit: int = 500) -> list[Market]:
    """
    Fetch active markets from the Gamma API.
    Prices are included directly in the Gamma response via outcomePrices.
    """
    markets = []  # type: ignore[var-annotated]
    offset = 0
    page_size = 100

    async with aiohttp.ClientSession() as session:
        while len(markets) < limit:
            await _rate_limiter.acquire()
            try:
                async with session.get(
                    f"{GAMMA_API_URL}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": page_size,
                        "offset": offset,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
            except Exception:
                break

            if not data or not isinstance(data, list):
                break

            for item in data:
                try:
                    market = _parse_market(item)
                    if market:
                        markets.append(market)
                except Exception:
                    continue

            offset += page_size
            if len(data) < page_size:
                break

    return markets


def _parse_market(item: dict) -> Optional[Market]:
    """Parse a single market from the Gamma API response."""
    # Parse token IDs (stored as JSON string in API)
    token_ids = _parse_json_field(item.get("clobTokenIds", []))
    outcomes = _parse_json_field(item.get("outcomes", []))
    outcome_prices = _parse_json_field(item.get("outcomePrices", []))

    # Map outcomes to token IDs and prices
    yes_token_id = None
    no_token_id = None
    yes_price = 0.0
    no_price = 0.0

    for i, outcome in enumerate(outcomes):
        token_id = token_ids[i] if i < len(token_ids) else None
        price = float(outcome_prices[i]) if i < len(outcome_prices) else 0.0

        if outcome.lower() == "yes":
            yes_token_id = token_id
            yes_price = price
        elif outcome.lower() == "no":
            no_token_id = token_id
            no_price = price

    # Skip markets with no price data
    if yes_price <= 0 and no_price <= 0:
        return None

    end_date = item.get("endDate") or item.get("endDateIso")
    hours = _hours_to_expiry(end_date)
    if hours < MIN_HOURS_TO_EXPIRY:
        return None

    liquidity = float(item.get("liquidityNum") or item.get("liquidity") or 0)
    if liquidity < MIN_MARKET_LIQUIDITY:
        return None

    question = item.get("question", "").strip()
    if not question:
        return None

    # Get tags from events if available
    tags = []  # type: ignore[var-annotated]
    events = item.get("events") or []
    for event in events:
        if isinstance(event, dict):
            tags.extend(event.get("tags") or [])

    category = _infer_category(question, tags)

    return Market(
        condition_id=item.get("conditionId", ""),
        question=question,
        end_date_iso=end_date,
        category=category,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        yes_price=yes_price,
        no_price=no_price,
        liquidity=liquidity,
        volume=float(item.get("volumeNum") or item.get("volume") or 0),
        hours_to_expiry=hours,
    )


async def fetch_prices(markets: list[Market]) -> list[Market]:
    """
    Prices are already fetched from Gamma API in fetch_active_markets.
    This function is a no-op kept for API compatibility.
    """
    return markets
