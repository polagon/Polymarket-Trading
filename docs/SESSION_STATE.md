# Astra Session State — Development Notes

**Last updated:** 2026-02-15
**Status:** Loop 6.2 complete — Live crypto universe validation (zero tradable surface confirmed)
**Test count:** 602 passed, 2 skipped

---

## Loop 6.2: Live Universe Tradability Validation (COMPLETE)

**Goal:** Prove whether Polymarket crypto threshold markets are genuinely non-tradable or if we have an ID mapping bug.

### Discovery Findings (Live Mode - tag_id=744 "cryptocurrency")

**Universe Stats:**
- Total discovered: 100 markets
- BTC markets: 26
- ETH markets: 18
- Pages fetched: 2 (pagination complete, no exhaustion)

**Market Bucketing:**
- `threshold_like`: **7 markets (7%)** — Price-based threshold markets
- `directional_5m_like`: **0 markets (0%)** — No intraday directional markets exist
- `other_crypto_like`: **93 markets (93%)** — Event-driven (ETF approvals, FTX claims, regulation)

### Tradability Validation Results

**All 7 threshold markets failed with `book_veto: no_book`**

Validated via direct CLOB API testing:
```
Token ID: 79973951962333251705... (from market 253518)
CLOB Response: 404 "No orderbook exists for the requested token id"
```

**Root Cause Confirmed:**
- Token IDs are correctly extracted from Gamma API (`clobTokenIds` field)
- CLOB API correctly receives these token IDs
- Markets exist in Gamma but **have no CLOB orderbooks** (inactive/zero liquidity)
- This is NOT an ID mapping bug — it's real market state

### The 7 Threshold Markets (All Non-Tradable)

| Market | Underlying | Strike | Type | CLOB Status |
|--------|------------|--------|------|-------------|
| Will ETH hit $2,500 by Jan 31? | ETH | $2,500 | touch | ❌ No orderbook (404) |
| Will BTC hit $50,000 by Jan 31? | BTC | $50,000 | touch | ❌ No orderbook (404) |
| Will ETH hit $3,000 in January? | ETH | $3,000 | touch | ❌ No orderbook (404) |
| Will BTC hit $50,000 in February? | BTC | $50,000 | touch | ❌ No orderbook (404) |
| Will ETH hit $2,500 in February? | ETH | $2,500 | touch | ❌ No orderbook (404) |
| Will ETH hit $3,000 in February? | ETH | $3,000 | touch | ❌ No orderbook (404) |
| Will BTC hit $55,000 in February? | BTC | $55,000 | touch | ❌ No orderbook (404) |

### Scoring Results (--mode live)

**Passers: 0 / 100** ✅ **VALID conservative outcome**

**Top Veto Reasons:**
```
parse_veto: no_resolution_type: 54   (93% event markets without price levels)
parse_veto: no_strike_level: 28      (crypto markets without specific prices)
book_veto: no_book: 7                (threshold markets with ZERO liquidity)
parse_veto: ambiguous_question: 6
parse_veto: unsupported_underlying: 5
```

### System Validation

**✅ Bot Behavior is Correct:**
1. Discovery pipeline works (100 markets found, discovery_ok)
2. Parser correctly identifies 7 threshold markets
3. Fail-closed gates correctly veto all with `book_veto: no_book`
4. Zero passers is the **intended outcome** when no edge exists

**❌ Polymarket Crypto Surface Not Tradable:**
1. Zero liquidity on all threshold markets (no CLOB orderbooks)
2. 93% of crypto tag is event-driven (requires different estimators)
3. Zero intraday directional markets (planned Loop 6.2 doesn't exist)

### Implications

**Current State:**
- Bot is **production-ready** and **fail-closed correct**
- Polymarket crypto tag has **zero tradable surface** for threshold category
- Conservative gates working as designed (refuse to trade without edge)

**Roadmap Options:**

1. **Wait for Market Evolution** (Recommended)
   - Polymarket may add liquid crypto threshold markets later
   - Bot will automatically discover and trade them when available
   - Current state is stable and safe

2. **Expand Discovery Beyond "cryptocurrency" Tag**
   - Search all 100 Gamma tags for crypto price markets
   - May find threshold markets with liquidity elsewhere
   - Effort: 1-2 hours

3. **Build Event-Driven Category**
   - Add new category for ETF/regulatory markets (93% of surface)
   - Requires new estimator (not lognormal)
   - Effort: 5-10 hours

**Decision:** Documenting current state and pausing active development. Bot remains production-ready for when tradable crypto threshold markets emerge.

---

## Loop 6.1: Live Universe Reconciliation (COMPLETE)

**Goal:** Make it impossible for users to feel "the bot doesn't ingest all crypto markets" without auditable explanation.

### Key Features Implemented

1. **Pagination Exhaustion Detection**
   - Extended `DiscoveryResult` with `pagination_exhausted: bool` field
   - `GammaClient.fetch_markets_for_tag()` detects when `next_cursor` exists at `max_pages` boundary
   - Sets `REASON_DISCOVERY_PAGINATION_EXHAUSTED` when incomplete universe detected
   - Generator hard-stops unless: `discovery_reason == discovery_ok AND pagination_exhausted == False`

2. **Inventory-Only Mode**
   - CLI flag: `--inventory`
   - Deterministic market bucketing:
     - `threshold_like`: Markets that `parse_crypto_threshold` accepts
     - `directional_5m_like`: 5-minute up/down directional markets
     - `other_crypto_like`: BTC/ETH markets not matching above
   - Writes `artifacts/universe/inventory.json` with:
     - `counts_by_bucket`
     - `top_parse_veto_reasons`, `top_book_veto_reasons`
     - `excluded_samples` (max 25), `included_samples` (max 25)
   - Skips EV scoring and contract output

3. **Spot Price Provenance**
   - Writes `artifacts/universe/spot.json` immediately after `fetch_prices()`
   - Schema: `{schema_version, fetched_at, source, prices: [{underlying, spot_usd, price_change_24h, market_cap}]}`
   - Audit trail of prices used in scoring

4. **Fail-Closed Integration**
   - Generator refuses to proceed if:
     - `discovery_reason != REASON_DISCOVERY_OK`, OR
     - `pagination_exhausted == True`
   - Always writes `discovery.json` before exiting (even on failure)
   - Applies to both `--inventory` and normal generation modes

### Files Modified

| File | Changes |
|------|---------|
| `tools/discover_crypto_gamma.py` | Extended `DiscoveryResult`, modified `fetch_markets_for_tag()` return type to `tuple[list, bool]`, added pagination exhaustion detection |
| `tools/generate_crypto_threshold_contracts.py` | Added `--inventory` flag, `classify_market_bucket()` function, spot price provenance, fail-closed pagination check, inventory mode logic |
| `models/reasons.py` | Already had `REASON_DISCOVERY_PAGINATION_EXHAUSTED` (line 62) |

### Files Created

| File | Purpose |
|------|---------|
| `tests/test_inventory_mode.py` | 6 tests for bucketing logic and inventory.json schema |
| `tests/test_discovery_fail_closed_pagination.py` | 3 tests for fail-closed pagination behavior |
| `tests/test_spot_provenance_artifact.py` | 3 tests for spot.json artifact schema |

### Test Coverage

**New tests:** 12
**Total tests:** 602 passed, 2 skipped

#### Test Breakdown

**test_inventory_mode.py (6 tests):**
- `test_threshold_like_bucket_for_parseable_markets`: Parseable markets → threshold_like
- `test_directional_5m_like_bucket`: 5-minute directional → directional_5m_like
- `test_other_crypto_like_bucket`: Other BTC/ETH → other_crypto_like
- `test_false_positive_excluded_from_other_crypto`: Ethena/WBTC/stETH handling
- `test_inventory_mode_writes_artifact`: inventory.json schema validation
- `test_inventory_sample_capping`: Max 25 samples per category

**test_discovery_fail_closed_pagination.py (3 tests):**
- `test_generator_exits_on_pagination_exhausted`: SystemExit when `pagination_exhausted == True`
- `test_generator_proceeds_on_pagination_ok`: Normal flow when `pagination_exhausted == False`
- `test_discovery_artifact_includes_pagination_exhausted`: discovery.json includes new field

**test_spot_provenance_artifact.py (3 tests):**
- `test_spot_artifact_schema`: spot.json required fields
- `test_spot_artifact_handles_missing_prices`: Graceful handling of missing price data
- `test_spot_artifact_written_before_scoring`: Structural test for correct code ordering

### Artifact Schemas

#### `discovery.json`
```json
{
  "schema_version": "1.0",
  "discovery_mode": "fixture|live",
  "discovered_at": "2026-02-15T10:00:00Z",
  "discovery_reason": "discovery_ok",
  "pagination_exhausted": false,
  "tag_ids_used": ["crypto"],
  "pages_fetched": 3,
  "total_count": 223,
  "btc_count": 150,
  "eth_count": 73,
  "counts_by_underlying": {"BTC": 150, "ETH": 73}
}
```

#### `inventory.json`
```json
{
  "schema_version": "1.0",
  "generated_at": "2026-02-15T10:05:00Z",
  "discovery_mode": "fixture",
  "total_discovered": 223,
  "counts_by_bucket": {
    "threshold_like": 190,
    "directional_5m_like": 25,
    "other_crypto_like": 8
  },
  "top_parse_veto_reasons": {
    "parse_veto: false_positive_token": 5,
    "parse_veto: unsupported_underlying": 3
  },
  "top_book_veto_reasons": {},
  "excluded_samples": [
    {
      "market_id": "0x123",
      "question": "Will BTC be up in the next 5 minutes?",
      "bucket": "directional_5m_like",
      "end_date_iso": "2026-02-15T10:10:00Z"
    }
  ],
  "included_samples": [
    {
      "market_id": "0x456",
      "question": "Will BTC hit $100,000 by end of 2026?",
      "bucket": "threshold_like",
      "end_date_iso": "2026-12-31T23:59:59Z"
    }
  ]
}
```

#### `spot.json`
```json
{
  "schema_version": "1.0",
  "fetched_at": "2026-02-15T10:00:30Z",
  "source": "coingecko_v3",
  "prices": [
    {
      "underlying": "BTC",
      "spot_usd": 50000.0,
      "price_change_24h": 2.5,
      "market_cap": 1000000000000
    },
    {
      "underlying": "ETH",
      "spot_usd": 3000.0,
      "price_change_24h": 1.2,
      "market_cap": 400000000000
    }
  ]
}
```

---

## Loop 6.0: Tag-Based Live Discovery (COMPLETE)

**Status:** Merged to main in PR #3
**Tests:** 590 passing

### Key Features

1. **Tag-based discovery** via Gamma API `/tags` and `/markets`
2. **Mode switching**: `--mode fixture` (default) or `--mode live`
3. **Live discovery blocked under pytest** (`PYTEST_CURRENT_TEST` guard)
4. **Fail-closed discovery** with named reason enums:
   - `REASON_DISCOVERY_OK`
   - `REASON_DISCOVERY_TAG_NOT_FOUND`
   - `REASON_DISCOVERY_GAMMA_HTTP_ERROR`
   - `REASON_DISCOVERY_GAMMA_TIMEOUT`
   - `REASON_DISCOVERY_INVALID_RESPONSE`
   - `REASON_DISCOVERY_PAGINATION_EXHAUSTED` (Loop 6.1)

---

## Loop 5.1: Strict Parsing + Universe Scoring (COMPLETE)

**Status:** Merged to main
**Tests:** 53 new tests

### Key Features

1. **Word-boundary parsing**: `\bBTC\b`, `\bETH\b` regex
2. **False-positive rejection**: Ethena, WBTC, stETH explicit rejection
3. **Extended summary artifact**: `counts_by_underlying` in summary.json
4. **Fixture-based tests**: Zero network calls, deterministic

---

## Loop 5.0: Real CLOB Market Data + Crypto Estimator (COMPLETE)

**Status:** Merged to main
**Tests:** 53 new tests (clob_book + estimator + loader)

### Key Features

1. **CLOB orderbook fetcher**: REST-based, BookSnapshot + MarketBook
2. **Lognormal probability estimator**: Touch/close probability, vol buffers
3. **Definition contract loader**: Load contracts from JSON at startup
4. **Paper Run #3 verification**: Real book data + estimator working

---

## Development Philosophy

### Fail-Closed Design

- **Discovery**: Generator refuses to proceed on degraded/partial universes
- **Pagination**: Hard stop when `pagination_exhausted == True`
- **All gates**: Explicit veto reasons, never implicit failures

### Artifact Provenance

- **discovery.json**: Full discovery metadata (reason, pagination, tag IDs, pages fetched)
- **inventory.json**: Market bucketing + veto distributions + samples
- **spot.json**: Price provenance (source, timestamp, prices used)
- **All artifacts**: Schema versioning, sorted keys, 2-space indent

### Testing Strategy

- **Zero network in tests**: PYTEST_CURRENT_TEST guard prevents live API calls
- **Fixture-based**: Deterministic test data in discover_crypto_gamma.py
- **Async tests**: anyio backend for async test functions
- **Comprehensive coverage**: Parse logic, fail-closed behavior, artifact schemas

---

## Next Steps (Future Loops)

**Potential Loop 7 candidates:**
- Live API credential management (GAMMA_API_KEY)
- Rate limiting + retry budgets for Gamma API
- Historical pagination (fetch older markets beyond max_pages)
- Market metadata enrichment (tags, categories, volumes)
- Directional market support (5-minute up/down models)
