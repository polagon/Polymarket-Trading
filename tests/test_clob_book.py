"""Tests for feeds/clob_book.py — CLOB orderbook parsing (no network)."""

import time

import pytest

from feeds.clob_book import BookSnapshot, MarketBook, _parse_book_response

# ── Fixtures: recorded CLOB /book responses ────────────────────────────────


def _btc_book_fixture() -> dict:
    """Recorded BTC $1M touch market — YES token orderbook."""
    return {
        "market": "0xbb57ccf5853a85487bc3d83d04d669310d28c6c810758953b9d9b91d1aee89d2",
        "asset_id": "105267568073659068217311993901927962476298440625043565106676088842803600775810",
        "timestamp": "1771109855756",
        "bids": [
            {"price": "0.48", "size": "67686.29"},
            {"price": "0.46", "size": "16933.00"},
            {"price": "0.45", "size": "4510.00"},
            {"price": "0.44", "size": "2587.00"},
            {"price": "0.40", "size": "1013.00"},
            {"price": "0.01", "size": "1037309.36"},
        ],
        "asks": [
            {"price": "0.49", "size": "96261.32"},
            {"price": "0.50", "size": "5000.00"},
            {"price": "0.52", "size": "3000.00"},
            {"price": "0.55", "size": "2000.00"},
            {"price": "0.99", "size": "100.00"},
        ],
        "last_trade_price": "0.485",
        "tick_size": "0.01",
    }


def _thin_book_fixture() -> dict:
    """Thin book — low depth."""
    return {
        "market": "0xdeadbeef",
        "asset_id": "thin_token",
        "timestamp": "1771100000000",
        "bids": [{"price": "0.10", "size": "50"}],
        "asks": [{"price": "0.20", "size": "50"}],
        "last_trade_price": "0.15",
        "tick_size": "0.01",
    }


def _empty_book_fixture() -> dict:
    """Empty book — no bids or asks."""
    return {
        "market": "0x0000",
        "asset_id": "empty_token",
        "timestamp": "0",
        "bids": [],
        "asks": [],
    }


def _one_sided_fixture() -> dict:
    """Only bids, no asks."""
    return {
        "market": "0x1111",
        "asset_id": "one_sided",
        "timestamp": "0",
        "bids": [{"price": "0.50", "size": "1000"}],
        "asks": [],
    }


# ── Parse tests ────────────────────────────────────────────────────────────


class TestParseBookResponse:
    def test_btc_best_bid_ask(self) -> None:
        snap = _parse_book_response(_btc_book_fixture(), time.time())
        assert snap.best_bid == 0.48
        assert snap.best_ask == 0.49
        assert snap.is_valid

    def test_btc_mid_and_spread(self) -> None:
        snap = _parse_book_response(_btc_book_fixture(), time.time())
        assert abs(snap.mid - 0.485) < 1e-9
        assert abs(snap.spread - 0.01) < 1e-9
        assert abs(snap.spread_frac - 0.01 / 0.485) < 1e-6

    def test_btc_depth(self) -> None:
        snap = _parse_book_response(_btc_book_fixture(), time.time())
        # Bid depth: levels within 10% of 0.48 (>= 0.432)
        # 0.48 * 67686.29 + 0.46 * 16933 + 0.45 * 4510 + 0.44 * 2587 = ~43k
        assert snap.bid_depth_usd > 30000
        # Ask depth: levels within 10% of 0.49 (<= 0.539)
        # 0.49 * 96261.32 + 0.50 * 5000 + 0.52 * 3000 = ~51k
        assert snap.ask_depth_usd > 40000

    def test_btc_metadata(self) -> None:
        snap = _parse_book_response(_btc_book_fixture(), time.time())
        assert snap.n_bid_levels == 6
        assert snap.n_ask_levels == 5
        assert snap.last_trade_price == 0.485
        assert snap.tick_size == 0.01

    def test_empty_book(self) -> None:
        snap = _parse_book_response(_empty_book_fixture(), time.time())
        assert snap.best_bid == 0.0
        assert snap.best_ask == 0.0
        assert not snap.is_valid
        assert snap.depth_proxy_usd == 0.0

    def test_one_sided_book(self) -> None:
        snap = _parse_book_response(_one_sided_fixture(), time.time())
        assert snap.best_bid == 0.50
        assert snap.best_ask == 0.0
        assert not snap.is_valid  # Need both sides

    def test_thin_book(self) -> None:
        snap = _parse_book_response(_thin_book_fixture(), time.time())
        assert snap.best_bid == 0.10
        assert snap.best_ask == 0.20
        assert snap.is_valid
        # Wide spread
        assert snap.spread_frac > 0.5  # 0.10/0.15 = 66%


class TestBookSnapshot:
    def test_staleness(self) -> None:
        snap = BookSnapshot(asset_id="x", fetched_at=time.time() - 200)
        assert snap.is_stale  # 200s > 120s threshold

    def test_fresh(self) -> None:
        snap = BookSnapshot(asset_id="x", fetched_at=time.time())
        assert not snap.is_stale

    def test_depth_proxy_valid(self) -> None:
        snap = BookSnapshot(
            asset_id="x",
            best_bid=0.48,
            best_ask=0.49,
            bid_depth_usd=5000,
            ask_depth_usd=3000,
            fetched_at=time.time(),
        )
        assert snap.depth_proxy_usd == 3000  # min of bid/ask

    def test_depth_proxy_invalid(self) -> None:
        snap = BookSnapshot(asset_id="x")
        assert snap.depth_proxy_usd == 0.0


class TestMarketBook:
    def test_valid_market_book(self) -> None:
        yes = _parse_book_response(_btc_book_fixture(), time.time())
        mb = MarketBook(condition_id="0xabc", yes=yes)
        assert mb.is_valid

    def test_invalid_no_yes(self) -> None:
        mb = MarketBook(condition_id="0xabc")
        assert not mb.is_valid

    def test_invalid_stale(self) -> None:
        yes = _parse_book_response(_btc_book_fixture(), time.time() - 300)
        mb = MarketBook(condition_id="0xabc", yes=yes)
        assert not mb.is_valid  # stale
