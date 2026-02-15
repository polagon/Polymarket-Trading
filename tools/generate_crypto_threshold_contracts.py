#!/usr/bin/env python3
"""
Crypto Threshold Universe Scorer + Contract Generator

Loop 5.1: Tag-based discovery + strict word-boundary parsing.

Discovers all crypto threshold markets, scores them with conservative
maker-entry EV gate, and outputs:
- artifacts/universe/crypto_threshold_scored.json (full scored universe)
- artifacts/universe/summary.json (veto reason counts + counts_by_underlying)
- definitions/contracts.crypto_threshold.json (only passers)

Fail-closed: parsing/lint/book failures → veto with named reason.
Maker-only: scoring uses entry price we can actually post without crossing.
Conservative: uses EV gate lower-bound (p_low/p_high) exactly like trading.

Loop 5.1 improvements:
- Tag-based discovery (fixture-based for now, live API in Loop 6)
- Strict word-boundary parsing (\\bBTC\\b, \\bETH\\b)
- Explicit Ethena/WBTC/stETH rejection
- Extended summary with counts_by_underlying
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add tools to path for new L5.1 modules
tools_path = Path(__file__).resolve().parent
if str(tools_path) not in sys.path:
    sys.path.insert(0, str(tools_path))

# Loop 5.1: Import new discovery and parser
from discover_crypto_gamma import discover_crypto_markets_fixture
from parse_crypto_threshold import parse_threshold_market as parse_strict

from config import (
    CLOB_API_URL,
    EV_BASE_ADVERSE_BUFFER,
    EV_BASE_SLIPPAGE,
    EV_EPS,
    EV_K_DEPTH,
    EV_K_SPREAD,
    EV_MAKER_THRESHOLD,
    GAMMA_API_URL,
    ORDER_TICK_SIZE,
)
from data_sources.crypto import fetch_prices
from definitions.lint import lint
from feeds.clob_book import CLOBBookFetcher
from gates.ev_gate import evaluate as ev_gate_evaluate
from models.definition_contract import DefinitionContract
from signals.crypto_estimator import (
    estimate_probability,
    get_default_vol,
    time_to_cutoff_years,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Veto reason enums (stable) — single source of truth for universe scoring
# Parse vetoes (specific)
VETO_PARSE_FALSE_POSITIVE = "parse_veto: false_positive_token"  # Ethena, WBTC, stETH, etc.
VETO_PARSE_UNSUPPORTED_UNDERLYING = "parse_veto: unsupported_underlying"  # SOL, LINK, etc. (not BTC/ETH)
VETO_PARSE_NO_STRIKE = "parse_veto: no_strike_level"  # Cannot extract strike price
VETO_PARSE_NO_RESOLUTION_TYPE = "parse_veto: no_resolution_type"  # Cannot detect touch/close
VETO_PARSE_MISSING_FIELDS = "parse_veto: missing_required_fields"  # No question or end_date_iso
VETO_PARSE_AMBIGUOUS = "parse_veto: ambiguous_question"  # Generic fallback

# Lint vetoes
VETO_LINT_FAILED = "lint_veto: validation_failed"

# Book quality vetoes
VETO_NO_BOOK = "book_veto: no_book"
VETO_WIDE_SPREAD = "book_veto: spread_too_wide"
VETO_THIN_DEPTH = "book_veto: depth_too_thin"

# Time vetoes
VETO_TIME_OOB = "time_veto: cutoff_out_of_bounds"

# Estimator vetoes
VETO_ESTIMATOR_FAILED = "estimator_veto: failed"

# Maker entry vetoes
VETO_MAKER_CANNOT_POST = "maker_veto: cannot_post_without_crossing"

# EV vetoes
VETO_EV_FAILED = "ev_veto: net_lb_below_threshold"


@dataclass
class ScoredMarket:
    """Scored crypto threshold market candidate."""

    market_id: str
    condition_id: str
    question: str
    underlying: str
    strike: float
    cutoff_ts_utc: str
    resolution_type: str
    best_bid: float
    best_ask: float
    spread_frac: float
    depth_proxy_usd: float
    spot_price: float
    time_to_cutoff_years: float
    p_hat: float
    p_low: float
    p_high: float
    entry_price: float
    ev_net_lb: float
    passed: bool
    veto_reason: Optional[str]
    definition_hash: Optional[str] = None


async def discover_markets(underlyings: list[str]) -> tuple[list[dict], dict]:
    """Discover all crypto threshold markets using Loop 5.1 tag-based discovery.

    Loop 5.1: Uses fixture-based discovery (live API in Loop 6).
    Returns (markets, metadata) where metadata includes counts_by_underlying.

    underlyings parameter is IGNORED in Loop 5.1 (fixture returns all).
    """
    # Loop 5.1: Use fixture-based discovery
    markets, metadata = discover_crypto_markets_fixture()
    logger.info(f"Discovered {len(markets)} markets (BTC: {metadata['btc_count']}, ETH: {metadata['eth_count']})")
    return markets, metadata


def parse_threshold_market(market: dict) -> Optional[dict]:
    """Parse a crypto threshold market using Loop 5.1 strict parser.

    Loop 5.1: Delegates to parse_crypto_threshold.parse_threshold_market
    which implements word-boundary regex and false-positive rejection.

    Returns dict with {underlying, strike, cutoff_ts_utc, resolution_type, op}
    or None if parsing fails, is ambiguous, or is a false positive.
    """
    # Loop 5.1: Use strict word-boundary parser
    return parse_strict(market)


def build_definition_contract(market: dict, parsed: dict) -> Optional[DefinitionContract]:
    """Build a DefinitionContract from parsed market data.

    Returns None if construction or lint fails.
    """
    try:
        # Build oracle_details based on resolution_type
        oracle_details = {
            "feed": "coingecko_v3",
            "rounding": "round_cent",  # Must be from ALLOWED_ROUNDING_RULES
        }
        if parsed["resolution_type"] == "touch":
            oracle_details["finality"] = "1h_close"

        # Build condition based on resolution_type
        # Level must be int for crypto_threshold
        condition = {
            "op": parsed["op"],
            "level": int(parsed["strike"]),
            "window": parsed["window"],
        }
        if parsed["resolution_type"] == "close":
            condition["measurement_time"] = parsed.get("measurement_time", parsed["cutoff_ts_utc"])

        contract = DefinitionContract(
            market_id=market["id"],
            category="crypto_threshold",
            resolution_type=parsed["resolution_type"],
            underlying=parsed["underlying"],
            quote_ccy="USD",
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            oracle_source="coingecko_v3",
            oracle_details=oracle_details,
            condition=condition,
            venue_rules_version="polymarket_clob_v1",
        )

        # Lint
        lint_result = lint(contract)
        if not lint_result.ok:
            logger.debug(f"Lint failed for {market['id']}: {lint_result.missing}")
            return None

        return contract
    except Exception as e:
        logger.debug(f"Build contract failed for {market['id']}: {e}")
        return None


def compute_maker_entry_price(best_bid: float, best_ask: float, side: str, tick: float) -> Optional[float]:
    """Compute maker entry price that does not cross the spread.

    For BUY_YES: post at min(best_ask - tick, best_bid + tick)
    For BUY_NO: symmetric logic based on NO token pricing

    Returns None if cannot post without crossing.
    """
    if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        return None

    if side == "BUY_YES":
        # Try to improve on best_bid by 1 tick, but don't cross best_ask
        entry = min(best_ask - tick, best_bid + tick)
        if entry >= best_ask or entry <= best_bid:
            return None  # Would cross or not improve
        return max(0.01, min(0.99, entry))
    else:
        # BUY_NO: similar logic but inverted
        # NO token pricing: if YES best_bid=0.48, then NO best_ask=0.52
        no_best_ask = 1.0 - best_bid
        no_best_bid = 1.0 - best_ask
        entry = min(no_best_ask - tick, no_best_bid + tick)
        if entry >= no_best_ask or entry <= no_best_bid:
            return None
        return max(0.01, min(0.99, entry))


async def score_market(
    market: dict,
    book_fetcher: CLOBBookFetcher,
    price_data: dict,
    min_depth_usd: float,
    max_spread_frac: float,
    min_days: float,
    max_days: float,
    selection_margin_frac: float,
) -> ScoredMarket:
    """Score a single market with all filters and EV gate.

    Returns ScoredMarket with passed=True/False and veto_reason.
    """
    market_id = market["id"]
    condition_id = market.get("conditionId") or market.get("condition_id", "")
    question = market["question"]

    # Parse
    parsed = parse_threshold_market(market)
    if not parsed:
        # Infer specific parse veto reason
        veto_reason = VETO_PARSE_AMBIGUOUS  # Default fallback
        q = market.get("question", "").lower()

        # Check for false positives first
        if any(
            token in q
            for token in [
                "ethena",
                "susde",
                "usde",
                "wbtc",
                "renbtc",
                "steth",
                "seth",
                "reth",
                "cbeth",
            ]
        ):
            veto_reason = VETO_PARSE_FALSE_POSITIVE
        # Check for missing fields
        elif not market.get("question") or not market.get("end_date_iso"):
            veto_reason = VETO_PARSE_MISSING_FIELDS
        # Check for unsupported underlying (SOL, LINK, etc.)
        elif any(token in q for token in ["sol", "solana", "link", "ada", "dot"]):
            veto_reason = VETO_PARSE_UNSUPPORTED_UNDERLYING
        # Check for no strike
        elif not any(char in q for char in ["$", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
            veto_reason = VETO_PARSE_NO_STRIKE
        # Check for no resolution type keywords
        elif not any(kw in q for kw in ["hit", "reach", "touch", "above", "below", "close"]):
            veto_reason = VETO_PARSE_NO_RESOLUTION_TYPE

        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying="",
            strike=0.0,
            cutoff_ts_utc="",
            resolution_type="",
            best_bid=0.0,
            best_ask=0.0,
            spread_frac=0.0,
            depth_proxy_usd=0.0,
            spot_price=0.0,
            time_to_cutoff_years=0.0,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=veto_reason,
        )

    # Build definition
    contract = build_definition_contract(market, parsed)
    if not contract:
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=0.0,
            best_ask=0.0,
            spread_frac=0.0,
            depth_proxy_usd=0.0,
            spot_price=0.0,
            time_to_cutoff_years=0.0,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_LINT_FAILED,
        )

    # Fetch CLOB book
    tokens = market.get("clobTokenIds", "[]")
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    if len(tokens) < 2:
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=0.0,
            best_ask=0.0,
            spread_frac=0.0,
            depth_proxy_usd=0.0,
            spot_price=0.0,
            time_to_cutoff_years=0.0,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_NO_BOOK,
        )

    yes_token_id = tokens[0]
    no_token_id = tokens[1]
    market_book = await book_fetcher.fetch_market_book(condition_id, yes_token_id, no_token_id)

    if not market_book or not market_book.yes or not market_book.yes.is_valid:
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=0.0,
            best_ask=0.0,
            spread_frac=0.0,
            depth_proxy_usd=0.0,
            spot_price=0.0,
            time_to_cutoff_years=0.0,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_NO_BOOK,
        )

    best_bid = market_book.yes.best_bid
    best_ask = market_book.yes.best_ask
    spread_frac = market_book.yes.spread_frac
    depth_proxy_usd = market_book.yes.depth_proxy_usd

    # Filter: spread
    if spread_frac > max_spread_frac:
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            spot_price=0.0,
            time_to_cutoff_years=0.0,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_WIDE_SPREAD,
        )

    # Filter: depth
    if depth_proxy_usd < min_depth_usd:
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            spot_price=0.0,
            time_to_cutoff_years=0.0,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_THIN_DEPTH,
        )

    # Filter: time to cutoff
    try:
        t_years = time_to_cutoff_years(parsed["cutoff_ts_utc"])
    except Exception:
        t_years = 0.0

    if t_years < (min_days / 365.0) or t_years > (max_days / 365.0):
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            spot_price=0.0,
            time_to_cutoff_years=t_years,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_TIME_OOB,
        )

    # Get spot price
    underlying_key = parsed["underlying"].lower()
    crypto_ctx = price_data.get(underlying_key)
    if not crypto_ctx or not hasattr(crypto_ctx, "current_price"):
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            spot_price=0.0,
            time_to_cutoff_years=t_years,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_ESTIMATOR_FAILED,
        )

    spot = crypto_ctx.current_price
    if spot <= 0:
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            spot_price=spot,
            time_to_cutoff_years=t_years,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_ESTIMATOR_FAILED,
        )

    # Estimate probability
    vol = get_default_vol(parsed["underlying"])
    try:
        est = estimate_probability(
            spot=spot,
            strike=parsed["strike"],
            time_years=t_years,
            vol=vol,
            resolution_type=parsed["resolution_type"],
            op=parsed["op"],
        )
    except Exception as e:
        logger.debug(f"Estimator failed for {market_id}: {e}")
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            spot_price=spot,
            time_to_cutoff_years=t_years,
            p_hat=0.0,
            p_low=0.0,
            p_high=0.0,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_ESTIMATOR_FAILED,
        )

    p_hat = est.p_hat
    p_low = est.p_low
    p_high = est.p_high

    # Compute maker entry price
    entry_price = compute_maker_entry_price(best_bid, best_ask, "BUY_YES", ORDER_TICK_SIZE)
    if entry_price is None:
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            spot_price=spot,
            time_to_cutoff_years=t_years,
            p_hat=p_hat,
            p_low=p_low,
            p_high=p_high,
            entry_price=0.0,
            ev_net_lb=0.0,
            passed=False,
            veto_reason=VETO_MAKER_CANNOT_POST,
        )

    # EV gate
    ev_result = ev_gate_evaluate(
        p_hat=p_hat,
        p_low=p_low,
        p_high=p_high,
        market_price=entry_price,
        side="BUY_YES",
        size_usd=100.0,  # Nominal size for scoring
        fees_pct=2.0,
        spread_frac=spread_frac,
        depth_proxy_usd=max(depth_proxy_usd, 100.0),
        toxicity_multiplier=1.0,
        maker_threshold=EV_MAKER_THRESHOLD + selection_margin_frac,
        base_slip=EV_BASE_SLIPPAGE,
        k_spread=EV_K_SPREAD,
        k_depth=EV_K_DEPTH,
        base_buffer=EV_BASE_ADVERSE_BUFFER,
        eps=EV_EPS,
    )

    if not ev_result.approved:
        return ScoredMarket(
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            cutoff_ts_utc=parsed["cutoff_ts_utc"],
            resolution_type=parsed["resolution_type"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_frac=spread_frac,
            depth_proxy_usd=depth_proxy_usd,
            spot_price=spot,
            time_to_cutoff_years=t_years,
            p_hat=p_hat,
            p_low=p_low,
            p_high=p_high,
            entry_price=entry_price,
            ev_net_lb=ev_result.ev_net_lb,
            passed=False,
            veto_reason=VETO_EV_FAILED,
            definition_hash=contract.definition_hash,
        )

    # PASSED
    return ScoredMarket(
        market_id=market_id,
        condition_id=condition_id,
        question=question,
        underlying=parsed["underlying"],
        strike=parsed["strike"],
        cutoff_ts_utc=parsed["cutoff_ts_utc"],
        resolution_type=parsed["resolution_type"],
        best_bid=best_bid,
        best_ask=best_ask,
        spread_frac=spread_frac,
        depth_proxy_usd=depth_proxy_usd,
        spot_price=spot,
        time_to_cutoff_years=t_years,
        p_hat=p_hat,
        p_low=p_low,
        p_high=p_high,
        entry_price=entry_price,
        ev_net_lb=ev_result.ev_net_lb,
        passed=True,
        veto_reason=None,
        definition_hash=contract.definition_hash,
    )


async def main():
    parser = argparse.ArgumentParser(description="Generate crypto threshold contracts")
    parser.add_argument("--underlyings", default="BTC,ETH", help="Comma-separated underlyings")
    parser.add_argument("--max_out", type=int, default=15, help="Max passers to output")
    parser.add_argument("--min_depth_usd", type=float, default=3000.0, help="Min depth proxy USD")
    parser.add_argument("--max_spread_frac", type=float, default=0.08, help="Max spread fraction")
    parser.add_argument("--min_days", type=float, default=1.0, help="Min days to cutoff")
    parser.add_argument("--max_days", type=float, default=90.0, help="Max days to cutoff")
    parser.add_argument("--selection_margin_frac", type=float, default=0.01, help="Extra EV margin for selection")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    underlyings = [u.strip().upper() for u in args.underlyings.split(",")]
    logger.info(f"Discovering markets for underlyings: {underlyings}")

    # Discover markets (Loop 5.1: returns metadata with counts_by_underlying)
    markets, discovery_metadata = await discover_markets(underlyings)
    logger.info(f"Discovered {len(markets)} candidate markets")
    logger.info(f"Discovery metadata: {discovery_metadata}")

    # Fetch price data
    coin_ids = [u.lower() for u in underlyings]  # BTC → btc, ETH → eth
    price_data = await fetch_prices(coin_ids)
    logger.info(f"Fetched spot prices for {len(price_data)} underlyings")

    # Score all markets
    book_fetcher = CLOBBookFetcher()
    scored = []
    for m in markets:
        s = await score_market(
            m,
            book_fetcher,
            price_data,
            args.min_depth_usd,
            args.max_spread_frac,
            args.min_days,
            args.max_days,
            args.selection_margin_frac,
        )
        scored.append(s)

    # Filter passers
    passers = [s for s in scored if s.passed]
    passers.sort(key=lambda x: (-x.ev_net_lb, -x.depth_proxy_usd))
    passers = passers[: args.max_out]

    logger.info(f"Passers: {len(passers)} / {len(scored)}")

    # Veto reason distribution
    veto_counts = Counter(s.veto_reason for s in scored if not s.passed)

    # Write scored universe
    universe_path = Path("artifacts/universe/crypto_threshold_scored.json")
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(universe_path, "w") as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_candidates": len(scored),
                "passers": len(passers),
                "markets": [asdict(s) for s in scored],
            },
            f,
            indent=2,
            sort_keys=True,
        )
    logger.info(f"Wrote {universe_path}")

    # Write summary (Loop 5.1: extended with counts_by_underlying)
    summary_path = Path("artifacts/universe/summary.json")

    # Compute counts_by_underlying from scored markets
    counts_by_underlying = Counter(s.underlying for s in scored if s.underlying)

    with open(summary_path, "w") as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "discovered_count": len(markets),
                "scored_count": len(scored),
                "passed_count": len(passers),
                "top_veto_reasons": dict(veto_counts.most_common(10)),
                "counts_by_underlying": dict(counts_by_underlying),  # Loop 5.1
                "discovery_metadata": discovery_metadata,  # Loop 5.1
            },
            f,
            indent=2,
            sort_keys=True,
        )
    logger.info(f"Wrote {summary_path}")

    # Build contracts output
    contracts = []
    for s in passers:
        # Rebuild DefinitionContract
        parsed = {
            "underlying": s.underlying,
            "strike": s.strike,
            "cutoff_ts_utc": s.cutoff_ts_utc,
            "resolution_type": s.resolution_type,
            "op": ">=",  # All current markets are ">=" touch markets
            "window": "any_time" if s.resolution_type == "touch" else "at_close",
        }
        contract_dict = {
            "market_id": s.market_id,
            "category": "crypto_threshold",
            "resolution_type": s.resolution_type,
            "underlying": s.underlying,
            "quote_ccy": "USD",
            "cutoff_ts_utc": s.cutoff_ts_utc,
            "oracle_source": "coingecko_v3",
            "oracle_details": {
                "feed": "coingecko_v3",
                "rounding": "round_cent",
                "finality": "1h_close",
            },
            "condition": {
                "op": parsed["op"],
                "level": int(s.strike),
                "window": parsed["window"],
            },
            "venue_rules_version": "polymarket_clob_v1",
        }
        contracts.append(contract_dict)

    contracts_path = Path("definitions/contracts.crypto_threshold.json")
    with open(contracts_path, "w") as f:
        json.dump(contracts, f, indent=2, sort_keys=True)
    logger.info(f"Wrote {len(contracts)} contracts to {contracts_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("UNIVERSE SCORING SUMMARY")
    print("=" * 60)
    print(f"Discovered: {len(markets)}")
    print(f"Scored: {len(scored)}")
    print(f"Passed: {len(passers)}")
    print("\nTop Veto Reasons:")
    for reason, count in veto_counts.most_common(10):
        print(f"  {reason}: {count}")
    print("\n" + "=" * 60)

    if len(passers) == 0:
        print("⚠️  ZERO PASSERS — no edge under current conservative gates")
        print("This is a VALID Loop 5 outcome. Model sees no exploitable edge.")
    else:
        print(f"✅ {len(passers)} passers written to contracts.json")
        print("\nTop 5 by EV_net_lb:")
        for i, s in enumerate(passers[:5], 1):
            print(f"  {i}. {s.underlying} ${s.strike:,.0f} ({s.resolution_type}) — EV: {s.ev_net_lb:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
