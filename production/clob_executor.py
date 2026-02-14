"""
CLOB Executor — Production order placement for Polymarket.

⚠️  NOT ACTIVE — requires CLOB credentials (POLY_PRIVATE_KEY + Builder Program API key).
    Currently paper trading only. Enable when credentials are configured.

From research:
  - Panca2341/polymarket-trading-bot: Builder Program credentials for gasless execution
  - runesatsdev/polymarket-trading-bot: 10s order auto-cancel TTL, SOCKS5 proxy
  - metaggdev/Polymarket-Trading-Bot: SOCKS5 for US geo-restriction bypass
  - Rakshit2323/polymarket-trading-bot: Go CLOB client w/ auto-claim on resolution

Architecture (when live):
    Astra V2 → TradeSignal → ClobExecutor.submit() → Polymarket CLOB
                                    ↓
                             RiskManager.check()
                                    ↓
                         order_id → monitor → auto-cancel if unfilled in TTL

Builder Program:
  - Register at https://polymarket.com/build
  - Provides reduced/zero gas fees for order placement
  - Critical for arb strategies where gas costs eat the margin
  - Requires separate API key from regular Polymarket API

SOCKS5 Proxy:
  - Polymarket blocks US IPs for order placement (data reads are fine)
  - Required for live trading from US servers / standard cloud providers
  - Configure: POLY_SOCKS5_PROXY=socks5://user:pass@host:port in .env
  - aiohttp_socks library provides SOCKS5 connector support

Order TTL (auto-cancel):
  - All arb orders must expire after ORDER_TTL_SECONDS (default 30s) if unfilled
  - An unfilled arb order → naked directional bet (loses the guaranteed-profit property)
  - Implementation: pass expire_timestamp to CLOB post_order() call
  - Monitor order status; cancel if not filled within TTL

Auto-claim:
  - Resolved positions leave funds locked until claimed
  - Auto-claim via CLOB /redeem endpoint when position shows as resolved
  - Poll resolved positions every scan; batch-claim to reduce gas
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

# These will be needed when live:
# from py_clob_client.client import ClobClient
# from py_clob_client.clob_types import OrderArgs, OrderType
# import aiohttp_socks   # pip install aiohttp-socks


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (from .env when live)
# ─────────────────────────────────────────────────────────────────────────────

POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "")  # L1 wallet private key
POLY_API_KEY = os.getenv("POLY_API_KEY", "")  # Builder Program API key
POLY_API_SECRET = os.getenv("POLY_API_SECRET", "")
POLY_API_PASSPHRASE = os.getenv("POLY_API_PASSPHRASE", "")
POLY_SOCKS5_PROXY = os.getenv("POLY_SOCKS5_PROXY", "")  # e.g. socks5://user:pass@host:1080
CLOB_API_URL = "https://clob.polymarket.com"

ORDER_TTL_SECONDS = 30  # Auto-cancel arb orders after 30s if unfilled
MAX_POSITION_USD = 500  # Hard cap per order in live trading
MAX_DAILY_EXPOSURE_USD = 2000  # Daily total exposure limit


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    filled: bool = False
    filled_price: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def is_live_trading_enabled() -> bool:
    """Returns True only if all required credentials are configured."""
    return bool(POLY_PRIVATE_KEY and POLY_API_KEY and POLY_API_SECRET and POLY_API_PASSPHRASE)


def get_credentials_status() -> str:
    """Human-readable status of live trading credentials."""
    missing = []
    if not POLY_PRIVATE_KEY:
        missing.append("POLY_PRIVATE_KEY")
    if not POLY_API_KEY:
        missing.append("POLY_API_KEY (Builder Program)")
    if not POLY_API_SECRET:
        missing.append("POLY_API_SECRET")
    if not POLY_API_PASSPHRASE:
        missing.append("POLY_API_PASSPHRASE")

    if not missing:
        proxy_str = (
            f"  SOCKS5 proxy: {POLY_SOCKS5_PROXY}" if POLY_SOCKS5_PROXY else "  No SOCKS5 proxy (needed for US IPs)"
        )
        return f"✅ All CLOB credentials configured.\n{proxy_str}"
    return f"❌ Missing credentials: {', '.join(missing)}\n   Register at https://polymarket.com/build for Builder Program access."


async def submit_order(
    token_id: str,
    side: str,  # "BUY" or "SELL"
    price: float,  # limit price (0-1)
    size_usd: float,  # position size in USD
    is_arb: bool = False,  # arb orders get TTL auto-cancel
) -> OrderResult:
    """
    Submit a limit order to Polymarket CLOB.

    ⚠️  STUB — not active until credentials are configured.

    When live, this will:
    1. Check credentials are configured
    2. Connect via SOCKS5 proxy if configured (required for US IPs)
    3. Create order with Builder Program API key (gasless)
    4. For arb orders: set expire_timestamp = now + ORDER_TTL_SECONDS
    5. Return order_id for monitoring
    """
    if not is_live_trading_enabled():
        return OrderResult(
            success=False,
            error="Live trading not enabled — credentials not configured. "
            "Paper trading only. See production/clob_executor.py for setup.",
        )

    # ── LIVE IMPLEMENTATION (uncomment when credentials available) ───────────
    # connector = None
    # if POLY_SOCKS5_PROXY:
    #     from aiohttp_socks import ProxyConnector
    #     connector = ProxyConnector.from_url(POLY_SOCKS5_PROXY)
    #
    # client = ClobClient(
    #     host=CLOB_API_URL,
    #     key=POLY_PRIVATE_KEY,
    #     chain_id=137,   # Polygon mainnet
    #     creds=ApiCreds(
    #         api_key=POLY_API_KEY,
    #         api_secret=POLY_API_SECRET,
    #         api_passphrase=POLY_API_PASSPHRASE,
    #     ),
    # )
    #
    # expire_ts = None
    # if is_arb:
    #     expire_ts = int((datetime.now(timezone.utc) + timedelta(seconds=ORDER_TTL_SECONDS)).timestamp())
    #
    # order_args = OrderArgs(
    #     token_id=token_id,
    #     price=price,
    #     size=size_usd / price,   # convert USD to shares
    #     side=BUY if side == "BUY" else SELL,
    #     expiration=expire_ts,
    # )
    # signed_order = client.create_order(order_args)
    # resp = await client.post_order(signed_order, OrderType.GTC)
    # return OrderResult(success=True, order_id=resp["orderID"])

    return OrderResult(success=False, error="STUB — not yet implemented")


async def auto_claim_resolved(condition_ids: list[str]) -> dict[str, bool]:
    """
    Claim resolved positions from the CLOB.

    ⚠️  STUB — not active until credentials are configured.

    Resolved positions leave funds locked until claimed.
    Auto-claims prevent capital from sitting idle.
    Batch-claims to reduce gas costs.
    """
    if not is_live_trading_enabled():
        return {}
    # Implementation: call CLOB /redeem endpoint for each resolved condition_id
    return {}
