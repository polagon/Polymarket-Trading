# Astra V2 — Comprehensive Test Plan

## Phase 1: Kill Switch Test

```bash
# Terminal 1: Start scanner
./venv/bin/python3.12 main.py --once

# Terminal 2: Activate kill switch mid-scan
touch /tmp/astra_kill

# Expected: Clean exit with message "🛑 KILL SWITCH ACTIVATED"

# Cleanup
rm /tmp/astra_kill
```

---

## Phase 2: Arbitrage Scanner Test

**Test with mock market data:**
1. Look for markets where YES + NO ≠ $1.00
2. Verify `⚡ARB⚡` highlighting in report
3. Check score = 999.0+ for arbitrage opportunities

**Manual verification:**
- If arbitrage found: Calculate net_arbitrage = 1.0 - (yes_price + no_price) - 0.02
- Should be > 0.005 to trigger

---

## Phase 3: SQLite Logger Test

```bash
# Run a scan
./venv/bin/python3.12 main.py --once

# Verify database created
ls -lh memory/astra_trades.db

# Check tables populated
sqlite3 memory/astra_trades.db "SELECT COUNT(*) FROM market_snapshots"
sqlite3 memory/astra_trades.db "SELECT COUNT(*) FROM estimates"
sqlite3 memory/astra_trades.db "SELECT * FROM market_snapshots LIMIT 5"
```

**Expected:**
- `market_snapshots`: ~300 rows (one per market scanned)
- `estimates`: ~300 rows (one per market)
- `trades`: Empty initially (populates during paper trading)

---

## Phase 4: Sharpe/Sortino Test

```bash
# Run paper trader until ≥10 positions resolve
./venv/bin/python3.12 paper_trader.py --fast

# Wait for 10+ resolved positions
# Then check scan header output for:
#   "Risk-adjusted: Sharpe: X.XX  Sortino: X.XX"
```

**Expected:**
- Sharpe/Sortino appear after 10 resolved positions
- Green if Sharpe > 1.0, yellow if > 0.5, red otherwise

---

## Phase 5: Chain-of-Thought Prompts Test

**Manual verification in Claude API responses:**
1. Run scan with `--once`
2. Check logs for Claude responses containing:
   - "STEP 1: State the base rate"
   - "STEP 2: Identify the 3 strongest"
   - "STEP 3: Weight evidence by tier"

**Note:** This is internal to Claude's reasoning, not visible in final output.

---

## Phase 6: Few-Shot Examples Test

```bash
# Resolve 5+ predictions first
# Then run new scan and check if learning agent injects examples

# Look for in logs:
# "━━━ RECENT SUCCESSFUL PREDICTIONS ━━━"
```

**Expected:**
- After 5 resolved correct predictions (confidence ≥ 0.70)
- Examples injected into PRO/CON/Synthesizer prompts
- Should see improvement in similar market categories

---

## Phase 7: VIX Regime Test

```bash
# Run scan
./venv/bin/python3.12 main.py --once

# Check if VIX context injected into prompts
# Look for in logs:
# "━━━ MARKET REGIME (VIX) ━━━"
# "VIX: XX.X — Regime: NORMAL/STRESS/CRISIS"
```

**Test scenarios:**
- VIX < 16: "NORMAL VOLATILITY: Standard confidence thresholds apply"
- VIX 25-30: "ELEVATED VOLATILITY: Exercise caution"
- VIX > 30: "HIGH VOLATILITY: Reduce confidence on tail-risk markets"

---

## Phase 8: Multi-Timeframe Price History Test

```bash
# Run scanner multiple times (hourly)
./venv/bin/python3.12 main.py --once
# Wait 1 hour
./venv/bin/python3.12 main.py --once
# Wait 1 hour
./venv/bin/python3.12 main.py --once

# Check price snapshots stored
cat memory/price_snapshots.json | jq 'keys'

# Verify momentum calculations
python3.12 -c "
from data_sources.price_history import get_momentum_summary
summary = get_momentum_summary('bitcoin')
print(summary)
"
```

**Expected:**
- `price_snapshots.json` contains entries for each market scanned
- After 24 hours: momentum_24hr populated
- Rolling 168-hour window (old data pruned)

---

## Phase 9: Verification Loop Test

**Trigger conditions:**
1. Confidence < 0.70
2. Edge > 0.10
3. Not already marked no_trade

**Look for in logs:**
```
Verification passed for <market>: conf 0.65→0.75, p̂ 0.550→0.580
```
or
```
Verification failed for <market>: vetoing trade (conf=0.68, edge=+2.3%)
```

**Manual trigger:**
- Find market with conf=0.65, edge=0.12
- Verification should fetch momentum + VIX data
- Secondary Claude pass with 256 tokens

---

## Integrated End-to-End Test

```bash
# 1. Clean slate
rm /tmp/astra_kill
rm memory/astra_trades.db
rm memory/price_snapshots.json
rm memory/paper_positions.json

# 2. Run paper trader for 24 hours
./venv/bin/python3.12 paper_trader.py --fast

# 3. Verify all systems working:
#    - Kill switch responsive
#    - Arbitrage opportunities flagged
#    - SQLite populated
#    - Sharpe/Sortino calculated after 10 resolutions
#    - Chain-of-thought in Claude reasoning
#    - Few-shot examples injected after 5 correct predictions
#    - VIX context in every scan
#    - Price history growing
#    - Verification loop triggered on low-confidence trades

# 4. Check database stats
sqlite3 memory/astra_trades.db "SELECT COUNT(*) FROM market_snapshots"
sqlite3 memory/astra_trades.db "SELECT COUNT(*) FROM estimates"
sqlite3 memory/astra_trades.db "SELECT * FROM trades"

# 5. Review performance metrics
#    - Expected Sharpe > 1.5 after 50+ trades
#    - Expected hit rate > 60%
#    - Expected max drawdown < -15%
```

---

## Known Test Limitations

1. **Chain-of-thought**: Internal to Claude, not directly verifiable in output
2. **Few-shot**: Requires 5 resolved predictions first
3. **Sharpe/Sortino**: Requires 10 resolved positions
4. **Momentum**: Requires 24+ hours of price history
5. **Verification loop**: Only triggers on specific conditions (conf < 0.70, edge > 0.10)

---

## Success Criteria

✅ **Kill switch**: Clean exit within 1 second
✅ **Arbitrage**: Opportunities flagged with score 999.0+
✅ **SQLite**: All 3 tables populated
✅ **Sharpe/Sortino**: Displayed after 10 resolutions
✅ **VIX context**: Injected into every Astra estimation
✅ **Price history**: Snapshots stored every scan
✅ **Verification**: Triggered on low-confidence + high-edge signals
✅ **Overall performance**: Sharpe > 1.5, hit rate > 60%, drawdown < -15%

---

## Debugging Tools

```bash
# View SQLite data
sqlite3 memory/astra_trades.db

# Check price snapshots
cat memory/price_snapshots.json | jq '.'

# Monitor logs for verification
tail -f logs/astra.log | grep "Verification"

# Check learning agent predictions
cat memory/predictions.json | jq '.[] | select(.resolved == true)'
```
