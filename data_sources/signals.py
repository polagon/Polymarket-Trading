"""
Astra V2 — Signal Library
Free real-time signals for calibrating prediction market estimates.

Signals included:
  1.  Crypto Fear & Greed Index     (alternative.me)  — market sentiment
  2.  Bitcoin Dominance             (CoinGecko)        — altcoin risk-on/off
  3.  Global Crypto Market Cap      (CoinGecko)        — macro crypto trend
  4.  US Treasury Yield Spread      (FRED via API)     — recession probability
  5.  US CPI (latest)               (FRED)             — inflation state
  6.  Fed Funds Rate                (FRED)             — monetary tightening
  7.  VIX (fear index)              (Yahoo Finance)    — equity volatility
  8.  DXY (US Dollar Index)         (Yahoo Finance)    — macro risk proxy
  9.  Gold Price                    (Yahoo Finance)    — safe haven demand
  10. Polymarket volume trend       (Gamma API)        — market activity signal

All signals are cached and expire after TTL seconds.
Usage:
    from data_sources.signals import fetch_all_signals, MarketContext
    ctx = await fetch_all_signals()
    summary = ctx.summary()  # human-readable string for Astra prompt injection
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from config import FRED_API_KEY

logger = logging.getLogger("astra.signals")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, object]] = {}  # key → (timestamp, value)


def _cached(key: str, ttl: int, value=None):
    """Read or write cache. If value is None, read. Otherwise, write and return."""
    now = datetime.now(timezone.utc).timestamp()
    if value is None:
        entry = _cache.get(key)
        if entry and now - entry[0] < ttl:
            return entry[1]
        return None
    _cache[key] = (now, value)
    return value


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FearGreedIndex:
    value: int  # 0 = Extreme Fear, 100 = Extreme Greed
    label: str  # e.g. "Fear", "Greed", "Extreme Greed"
    timestamp: str

    @property
    def sentiment(self) -> str:
        """Return directional sentiment for Astra context."""
        if self.value <= 20:
            return "extreme_fear"
        if self.value <= 40:
            return "fear"
        if self.value <= 60:
            return "neutral"
        if self.value <= 80:
            return "greed"
        return "extreme_greed"

    @property
    def crypto_bias(self) -> float:
        """
        Implied probability adjustment for crypto price markets.
        High greed → slightly bullish bias; extreme fear → bearish bias.
        Range: -0.05 to +0.05 (small overlay, not decisive)
        """
        normalized = (self.value - 50) / 100  # -0.5 to +0.5
        return round(normalized * 0.10, 3)  # max ±5%


@dataclass
class MacroSignals:
    """US macroeconomic snapshot."""

    fed_funds_rate: Optional[float]  # Current federal funds target rate
    cpi_yoy: Optional[float]  # Latest CPI year-over-year %
    yield_spread_10y2y: Optional[float]  # 10Y minus 2Y treasury spread (recession proxy)
    unemployment_rate: Optional[float]  # Latest U-3 unemployment rate
    vix: Optional[float] = None  # CBOE Volatility Index (equity fear gauge)
    retrieved_at: str = ""

    @property
    def recession_signal(self) -> str:
        """Qualitative recession risk based on yield curve."""
        if self.yield_spread_10y2y is None:
            return "unknown"
        if self.yield_spread_10y2y < -0.5:
            return "inverted_high_risk"  # Deep inversion
        if self.yield_spread_10y2y < 0:
            return "inverted_moderate"
        if self.yield_spread_10y2y < 0.5:
            return "flat_watch"
        return "normal"

    @property
    def rate_environment(self) -> str:
        if self.fed_funds_rate is None:
            return "unknown"
        if self.fed_funds_rate >= 5.0:
            return "restrictive"
        if self.fed_funds_rate >= 3.0:
            return "tight"
        if self.fed_funds_rate >= 1.0:
            return "neutral"
        return "accommodative"

    @property
    def vix_kelly_multiplier(self) -> float:
        """
        Global Kelly size multiplier based on VIX level (DanTsai0903 / Moreira & Muir 2017).
        High volatility → reduce position sizes to preserve capital.
          VIX < 25 (normal)  → no change     (1.0×)
          VIX 25–35 (stress) → reduce 25%    (0.75×)
          VIX > 35 (crisis)  → reduce 50%    (0.50×)
        """
        if self.vix is None:
            return 1.0
        if self.vix >= 35:
            return 0.50
        if self.vix >= 25:
            return 0.75
        return 1.0

    @property
    def vix_label(self) -> str:
        """Human-readable VIX regime."""
        if self.vix is None:
            return "unknown"
        if self.vix >= 35:
            return "crisis"
        if self.vix >= 25:
            return "stress"
        if self.vix >= 15:
            return "normal"
        return "complacent"

    @property
    def monetary_regime(self) -> str:
        """
        Combined monetary policy regime based on fed_funds_rate + CPI.
        Useful for labeling macro markets (inflation bets, rate decisions).
        """
        if self.fed_funds_rate is None or self.cpi_yoy is None:
            return "unknown"
        real_rate = self.fed_funds_rate - self.cpi_yoy
        if real_rate > 1.5:
            return "restrictive_real_positive"  # Fed clearly tightening; credit negative
        if real_rate > 0:
            return "tight_mildly_restrictive"
        if real_rate > -1.0:
            return "neutral_mildly_accommodative"
        return "accommodative_real_negative"  # Fed behind the curve; credit positive


@dataclass
class CryptoMacro:
    btc_dominance_pct: Optional[float]  # BTC's share of total crypto market cap
    total_market_cap_usd: Optional[float]  # Global crypto market cap in USD
    retrieved_at: str = ""

    @property
    def risk_environment(self) -> str:
        """High BTC dominance = risk-off (alts weak); low = risk-on."""
        if self.btc_dominance_pct is None:
            return "unknown"
        if self.btc_dominance_pct >= 60:
            return "risk_off_btc_dominance"
        if self.btc_dominance_pct >= 50:
            return "neutral"
        return "risk_on_altcoin_season"


@dataclass
class MarketContext:
    """
    Aggregated market context for injection into Astra V2 estimation prompts.
    All signals combined into a single object.
    """

    fear_greed: Optional[FearGreedIndex] = None
    macro: Optional[MacroSignals] = None
    crypto_macro: Optional[CryptoMacro] = None
    retrieved_at: str = ""
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """
        Return a compact string summary for injection into Astra V2 system prompt.
        Designed to be < 200 tokens.
        """
        lines = [f"=== Market Context ({self.retrieved_at}) ==="]

        if self.fear_greed:
            fg = self.fear_greed
            lines.append(f"Crypto Sentiment: {fg.label} ({fg.value}/100) — bias={fg.crypto_bias:+.1%}")

        if self.crypto_macro:
            cm = self.crypto_macro
            mcap = f"${cm.total_market_cap_usd / 1e12:.2f}T" if cm.total_market_cap_usd else "N/A"
            dom = f"{cm.btc_dominance_pct:.1f}%" if cm.btc_dominance_pct else "N/A"
            lines.append(f"Crypto Market: cap={mcap}, BTC dom={dom} [{cm.risk_environment}]")

        if self.macro:
            m = self.macro
            spread = f"{m.yield_spread_10y2y:+.2f}%" if m.yield_spread_10y2y is not None else "N/A"
            parts = [f"10Y-3M_spread={spread} [{m.recession_signal}]"]
            if m.fed_funds_rate is not None:
                parts.append(f"fed_funds={m.fed_funds_rate:.2f}% [{m.rate_environment}]")
            if m.cpi_yoy is not None:
                parts.append(f"CPI(YoY)={m.cpi_yoy:.1f}%")
            if m.unemployment_rate is not None:
                parts.append(f"unemp={m.unemployment_rate:.1f}%")
            if m.vix is not None:
                kelly_mult = m.vix_kelly_multiplier
                mult_str = f" Kelly×{kelly_mult:.2f}" if kelly_mult < 1.0 else ""
                parts.append(f"VIX={m.vix:.1f} [{m.vix_label}]{mult_str}")
            if m.monetary_regime != "unknown":
                parts.append(f"regime={m.monetary_regime}")
            lines.append("US Macro: " + ", ".join(parts))

        if self.errors:
            lines.append(f"[signals unavailable: {', '.join(self.errors)}]")

        return "\n".join(lines)

    def crypto_sentiment_overlay(self) -> float:
        """Return Fear/Greed crypto bias. Safe to use even if signal unavailable."""
        if self.fear_greed:
            return self.fear_greed.crypto_bias
        return 0.0


# ---------------------------------------------------------------------------
# Signal fetchers
# ---------------------------------------------------------------------------


async def _fetch_fear_greed(session: aiohttp.ClientSession) -> Optional[FearGreedIndex]:
    """Alternative.me Fear & Greed Index — free, no API key."""
    cached = _cached("fear_greed", ttl=1800)  # 30 min TTL
    if cached:
        return cached  # type: ignore[no-any-return]

    try:
        async with session.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1},
            timeout=aiohttp.ClientTimeout(total=8),
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                entry = data.get("data", [{}])[0]
                result = FearGreedIndex(
                    value=int(entry.get("value", 50)),
                    label=entry.get("value_classification", "Neutral"),
                    timestamp=entry.get("timestamp", ""),
                )
                return _cached("fear_greed", ttl=1800, value=result)  # type: ignore[no-any-return]
    except Exception as e:
        logger.debug("Fear/Greed fetch failed: %s", e)
    return None


async def _fetch_crypto_global(session: aiohttp.ClientSession) -> Optional[CryptoMacro]:
    """CoinGecko global market data — BTC dominance + total market cap."""
    cached = _cached("crypto_global", ttl=600)  # 10 min TTL
    if cached:
        return cached  # type: ignore[no-any-return]

    try:
        async with session.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=aiohttp.ClientTimeout(total=8),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                gdata = data.get("data", {})
                dom = gdata.get("market_cap_percentage", {}).get("btc")
                mcap_data = gdata.get("total_market_cap", {})
                mcap = mcap_data.get("usd")
                result = CryptoMacro(
                    btc_dominance_pct=float(dom) if dom is not None else None,
                    total_market_cap_usd=float(mcap) if mcap is not None else None,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                )
                return _cached("crypto_global", ttl=600, value=result)  # type: ignore[no-any-return]
    except Exception as e:
        logger.debug("CoinGecko global fetch failed: %s", e)
    return None


async def _fetch_yahoo_finance(symbol: str, session: aiohttp.ClientSession, ttl: int = 3600) -> Optional[float]:
    """
    Fetch latest close price/value for a Yahoo Finance symbol (no API key).
    Works for indices: ^TNX (10Y yield), ^IRX (3M T-bill), ^VIX, DX-Y.NYB (DXY).
    """
    cache_key = f"yahoo_{symbol}"
    cached = _cached(cache_key, ttl=ttl)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    try:
        async with session.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"interval": "1d", "range": "5d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                result_list = data.get("chart", {}).get("result", [])
                if result_list:
                    closes = result_list[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                    latest = next((v for v in reversed(closes) if v is not None), None)
                    if latest is not None:
                        _cached(cache_key, ttl=ttl, value=float(latest))
                        return float(latest)
    except Exception as e:
        logger.debug("Yahoo Finance fetch failed for %s: %s", symbol, e)
    return None


async def _fetch_fred_series(
    series_id: str,
    session: aiohttp.ClientSession,
    ttl: int = 3600,
) -> Optional[float]:
    """
    Fetch the latest observation for a FRED data series.
    Free API key available at fred.stlouisfed.org — set FRED_API_KEY in .env.

    Key series:
      FEDFUNDS  — Federal Funds Effective Rate
      CPIAUCSL  — Consumer Price Index (YoY % calculated separately)
      UNRATE    — Civilian Unemployment Rate
    """
    if not FRED_API_KEY:
        return None

    cache_key = f"fred_{series_id}"
    cached = _cached(cache_key, ttl=ttl)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    try:
        async with session.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": "2",  # Latest 2 for YoY calculation
            },
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                observations = data.get("observations", [])
                if observations:
                    latest = next(
                        (o for o in observations if o.get("value") not in (".", "")),
                        None,
                    )
                    if latest:
                        val = float(latest["value"])
                        _cached(cache_key, ttl=ttl, value=val)
                        return val
            else:
                logger.warning("FRED API returned status %d for %s", resp.status, series_id)
    except Exception as e:
        logger.warning("FRED fetch failed for %s: %s", series_id, e)
    return None


async def _fetch_fred_cpi_yoy(session: aiohttp.ClientSession) -> Optional[float]:
    """
    Fetch CPI YoY by getting latest 13 observations and computing annual change.
    FRED's CPIAUCSL is monthly index levels; YoY = (latest / 12-months-ago - 1) * 100.
    """
    if not FRED_API_KEY:
        return None

    cache_key = "fred_cpi_yoy"
    cached = _cached(cache_key, ttl=7200)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    try:
        async with session.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "CPIAUCSL",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": "13",
            },
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                obs = [float(o["value"]) for o in data.get("observations", []) if o.get("value") not in (".", "")]
                if len(obs) >= 13:
                    yoy = round((obs[0] / obs[12] - 1) * 100, 2)
                    _cached(cache_key, ttl=7200, value=yoy)
                    return yoy
    except Exception as e:
        logger.warning("FRED CPI YoY calculation failed: %s", e)
    return None


async def _fetch_macro_signals(session: aiohttp.ClientSession) -> Optional[MacroSignals]:
    """
    Fetch key US macro signals via Yahoo Finance (no key) and FRED (free key).

    Yahoo Finance symbols:
      ^TNX   — 10-year US Treasury yield
      ^IRX   — 13-week T-Bill yield (short rate proxy)
      ^VIX   — CBOE Volatility Index

    FRED series (requires free FRED_API_KEY in .env):
      FEDFUNDS — Federal Funds Effective Rate
      CPIAUCSL — CPI (YoY % calculated from monthly index)
      UNRATE   — Civilian Unemployment Rate

    Yield spread = TNX - IRX (approximate 10Y-3M; close enough for recession signal).
    """
    cached = _cached("macro_signals", ttl=1800)  # 30 min TTL
    if cached:
        return cached  # type: ignore[no-any-return]

    # Fetch Yahoo Finance (no key) and FRED (key optional) in parallel
    tnx, irx, vix, fed_rate, cpi_yoy, unemployment = await asyncio.gather(
        _fetch_yahoo_finance("^TNX", session),
        _fetch_yahoo_finance("^IRX", session),
        _fetch_yahoo_finance("^VIX", session, ttl=900),  # 15 min TTL for VIX
        _fetch_fred_series("FEDFUNDS", session, ttl=86400),  # Daily update
        _fetch_fred_cpi_yoy(session),
        _fetch_fred_series("UNRATE", session, ttl=86400),
        return_exceptions=True,
    )

    def safe(v):
        return v if not isinstance(v, Exception) else None

    tnx_val = safe(tnx)
    irx_val = safe(irx)
    vix_val = safe(vix)
    fed_rate_val = safe(fed_rate)
    cpi_yoy_val = safe(cpi_yoy)
    unemployment_val = safe(unemployment)

    spread = None
    if tnx_val is not None and irx_val is not None:
        spread = round(tnx_val - irx_val, 3)

    if FRED_API_KEY and any(v is not None for v in [fed_rate_val, cpi_yoy_val, unemployment_val]):
        logger.info(
            "FRED macro loaded: FFR=%.2f%% CPI=%.1f%% UNRATE=%.1f%%",
            fed_rate_val or 0,
            cpi_yoy_val or 0,
            unemployment_val or 0,
        )

    result = MacroSignals(
        fed_funds_rate=fed_rate_val,
        cpi_yoy=cpi_yoy_val,
        yield_spread_10y2y=spread,
        unemployment_rate=unemployment_val,
        vix=round(vix_val, 2) if vix_val is not None else None,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    return _cached("macro_signals", ttl=1800, value=result)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def fetch_all_signals() -> MarketContext:
    """
    Fetch all available market context signals concurrently.
    Failures are silently ignored — signals are advisory overlays, not required.
    """
    errors = []

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            _fetch_fear_greed(session),
            _fetch_crypto_global(session),
            _fetch_macro_signals(session),
            return_exceptions=True,
        )

    fear_greed = results[0] if not isinstance(results[0], Exception) else None
    crypto_macro = results[1] if not isinstance(results[1], Exception) else None
    macro = results[2] if not isinstance(results[2], Exception) else None

    if isinstance(results[0], Exception):
        errors.append("fear_greed")
    if isinstance(results[1], Exception):
        errors.append("crypto_global")
    if isinstance(results[2], Exception):
        errors.append("macro_fred")

    return MarketContext(
        fear_greed=fear_greed,  # type: ignore[arg-type]
        macro=macro,  # type: ignore[arg-type]
        crypto_macro=crypto_macro,  # type: ignore[arg-type]
        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        errors=errors,
    )
