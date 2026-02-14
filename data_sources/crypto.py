"""
Fetches crypto price data from CoinGecko (free, no API key needed).
Computes probability estimates for crypto price markets using
a lognormal distribution model.
"""

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Optional

import aiohttp

from config import COINGECKO_API_URL

# Map common token names/symbols to CoinGecko IDs
COINGECKO_ID_MAP = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "bnb": "binancecoin",
    "xrp": "ripple",
    "ripple": "ripple",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "ada": "cardano",
    "cardano": "cardano",
    "avax": "avalanche-2",
    "avalanche": "avalanche-2",
    "dot": "polkadot",
    "polkadot": "polkadot",
    "matic": "matic-network",
    "polygon": "matic-network",
    "link": "chainlink",
    "chainlink": "chainlink",
    "uni": "uniswap",
    "uniswap": "uniswap",
    "ltc": "litecoin",
    "litecoin": "litecoin",
    "atom": "cosmos",
    "cosmos": "cosmos",
    "near": "near",
    "sui": "sui",
    "ton": "the-open-network",
    "pepe": "pepe",
    "shib": "shiba-inu",
    "trump": "trump",  # TRUMP token
}

# Historical annualized volatility estimates (rough, used for BS model)
VOLATILITY_MAP = {
    "bitcoin": 0.65,
    "ethereum": 0.80,
    "solana": 1.10,
    "binancecoin": 0.75,
    "ripple": 0.95,
    "dogecoin": 1.20,
    "cardano": 0.95,
    "avalanche-2": 1.10,
    "matic-network": 1.20,
    "chainlink": 0.90,
    "uniswap": 0.95,
}
DEFAULT_VOLATILITY = 1.0


@dataclass
class CryptoContext:
    coin_id: str
    current_price: float
    price_change_24h: float  # percentage
    market_cap: float


async def fetch_prices(coin_ids: list[str]) -> dict[str, CryptoContext]:
    """Fetch current prices for a list of CoinGecko coin IDs."""
    if not coin_ids:
        return {}

    ids_param = ",".join(set(coin_ids))
    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ids_param,
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "price_change_percentage": "24h",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:  # type: ignore[arg-type]
                if resp.status != 200:
                    return {}
                data = await resp.json()
    except Exception:
        return {}

    result = {}
    for item in data:
        cid = item.get("id", "")
        result[cid] = CryptoContext(
            coin_id=cid,
            current_price=float(item.get("current_price") or 0),
            price_change_24h=float(item.get("price_change_percentage_24h") or 0),
            market_cap=float(item.get("market_cap") or 0),
        )
    return result


def _lognormal_prob_above(
    current_price: float, target_price: float, annualized_vol: float, days_to_expiry: float
) -> float:
    """
    Probability that price is above target_price at expiry, assuming lognormal.
    Uses risk-neutral drift = 0 (prediction market context).
    """
    if current_price <= 0 or target_price <= 0 or days_to_expiry <= 0:
        return 0.5

    t = days_to_expiry / 365
    sigma = annualized_vol
    log_return = math.log(target_price / current_price)
    # Sigma squared / 2 drift adjustment (risk neutral)
    d = (math.log(current_price / target_price) + (sigma**2 / 2) * t) / (sigma * math.sqrt(t))

    # Normal CDF approximation
    return _norm_cdf(d)


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""
    # Abramowitz & Stegun approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def parse_crypto_question(question: str) -> Optional[dict]:
    """
    Parse a crypto price market question to extract:
    - coin identifier
    - price target
    - direction (above/below)
    - date context

    Returns None if unparseable.

    Examples:
    - "Will BTC be above $100,000 by end of 2025?"
    - "Will ETH hit $5,000 before June 30?"
    - "Will Bitcoin close above $80k on March 1?"
    """
    q = question.lower()

    # Find coin
    coin_id = None
    for symbol, gid in COINGECKO_ID_MAP.items():
        if re.search(r"\b" + re.escape(symbol) + r"\b", q):
            coin_id = gid
            break

    if not coin_id:
        return None

    # Find direction
    above = bool(re.search(r"\babove\b|\bhigher than\b|\bexceed\b|\bhit\b|\breach\b|\bbreak\b", q))
    below = bool(re.search(r"\bbelow\b|\bunder\b|\bfall below\b|\bdrop below\b", q))

    if not above and not below:
        return None

    direction = "above" if above else "below"

    # Find price target
    # Match patterns like $1m, $100k, $100,000, $1.5b
    price_patterns = [
        (r"\$([0-9,]+(?:\.[0-9]+)?)\s*b(?:illion)?\b", 1e9),  # $1b, $1billion
        (r"\$([0-9,]+(?:\.[0-9]+)?)\s*m(?:illion)?\b", 1e6),  # $1m, $1million
        (r"\$([0-9,]+(?:\.[0-9]+)?)\s*k\b", 1e3),  # $80k
        (r"\$([0-9,]+(?:\.[0-9]+)?)\b", 1),  # $100,000
        (r"\b([0-9,]+(?:\.[0-9]+)?)\s*k\b", 1e3),  # 80k
        (r"\b([0-9]{4,}(?:,[0-9]{3})*(?:\.[0-9]+)?)\b", 1),  # 100,000
    ]

    target_price = None
    for pattern, multiplier in price_patterns:
        m = re.search(pattern, q)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw) * multiplier
                if val > 0:
                    target_price = val
                    break
            except ValueError:
                continue

    if not target_price:
        return None

    return {
        "coin_id": coin_id,
        "direction": direction,
        "target_price": target_price,
    }


def estimate_probability(question: str, days_to_expiry: float, price_data: dict[str, CryptoContext]) -> Optional[dict]:
    """
    Returns probability estimate and metadata for a crypto price market.
    Returns None if the question can't be parsed or data is missing.
    """
    parsed = parse_crypto_question(question)
    if not parsed:
        return None

    coin_id = parsed["coin_id"]
    ctx = price_data.get(coin_id)
    if not ctx or ctx.current_price <= 0:
        return None

    vol = VOLATILITY_MAP.get(coin_id, DEFAULT_VOLATILITY)
    prob_above = _lognormal_prob_above(ctx.current_price, parsed["target_price"], vol, days_to_expiry)

    if parsed["direction"] == "above":
        probability = prob_above
    else:
        probability = 1.0 - prob_above

    # Confidence: higher when current price is far from target (less model sensitivity)
    # Lower when near the boundary (small vol estimate errors matter a lot)
    distance_ratio = abs(math.log(ctx.current_price / parsed["target_price"])) / (
        vol * math.sqrt(max(days_to_expiry, 1) / 365)
    )
    confidence = min(0.85, 0.4 + distance_ratio * 0.15)

    return {
        "probability": round(probability, 4),
        "confidence": round(confidence, 4),
        "source": "crypto_lognormal",
        "details": {
            "coin": coin_id,
            "current_price": ctx.current_price,
            "target_price": parsed["target_price"],
            "direction": parsed["direction"],
            "days_to_expiry": days_to_expiry,
            "annualized_vol": vol,
        },
    }
