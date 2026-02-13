#!/bin/bash
# Verification script for paper burn-in production hardening fixes
# Run after 4+ hours of burn-in to validate WS reconnection and reseed stability

set -e

LOG_DIR="/Users/pads/Claude/Polymarket/logs"
LOG_FILE="$LOG_DIR/paper_burnin.log"

echo "=========================================="
echo "Paper Burn-In Fix Verification"
echo "=========================================="
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ ERROR: Log file not found: $LOG_FILE"
    exit 1
fi

echo "Analyzing log file: $LOG_FILE"
echo "Log size: $(du -h "$LOG_FILE" | cut -f1)"
echo ""

# 1. Check for reconnection attempts
echo "1️⃣ Checking WS reconnection behavior..."
reconnect_count=$(grep -c "RECONNECT attempt" "$LOG_FILE" || true)
reconnect_success_count=$(grep -c "RECONNECT successful" "$LOG_FILE" || true)
echo "   Reconnect attempts: $reconnect_count"
echo "   Reconnect successes: $reconnect_success_count"

if [ "$reconnect_count" -gt 0 ]; then
    echo "   ✅ WS reconnection logic triggered"
    echo "   Last 5 reconnect attempts:"
    grep "RECONNECT attempt" "$LOG_FILE" | tail -n 5 | sed 's/^/      /'
else
    echo "   ℹ️  No reconnections needed (connection stable)"
fi
echo ""

# 2. Check for reseed storms (should NOT see hundreds of consecutive attempts)
echo "2️⃣ Checking for reseed storms..."
reseed_trigger_count=$(grep -c "RESEED TRIGGER" "$LOG_FILE" || true)
reseed_complete_count=$(grep -c "Reseed complete" "$LOG_FILE" || true)
echo "   Reseed triggers: $reseed_trigger_count"
echo "   Reseed completions: $reseed_complete_count"

if [ "$reseed_trigger_count" -gt 100 ]; then
    echo "   ⚠️  WARNING: High reseed trigger count (possible storm)"
    echo "   Check for consecutive failures:"
    grep "RESEED TRIGGER" "$LOG_FILE" | tail -n 10 | sed 's/^/      /'
elif [ "$reseed_trigger_count" -gt 0 ]; then
    echo "   ✅ Reseed triggers within normal range"
else
    echo "   ℹ️  No reseeds triggered (Active set stable)"
fi
echo ""

# 3. Check for skip-fetch after reseed
echo "3️⃣ Checking skip-fetch optimization..."
skip_fetch_count=$(grep -c "SKIP_FETCH_AFTER_RESEED" "$LOG_FILE" || true)
echo "   Skip-fetch executions: $skip_fetch_count"

if [ "$skip_fetch_count" -gt 0 ]; then
    echo "   ✅ Skip-fetch working"
    echo "   Sample:"
    grep "SKIP_FETCH_AFTER_RESEED" "$LOG_FILE" | head -n 3 | sed 's/^/      /'
else
    echo "   ⚠️  Skip-fetch not observed (check reseed completion)"
fi
echo ""

# 4. Check for degraded mode entries
echo "4️⃣ Checking degraded mode behavior..."
ws_degraded_count=$(grep -c "ENTER_DEGRADED_MODE" "$LOG_FILE" || true)
reseed_breaker_count=$(grep -c "RESEED circuit breaker" "$LOG_FILE" || true)
echo "   WS degraded mode entries: $ws_degraded_count"
echo "   Reseed circuit breaker triggers: $reseed_breaker_count"

if [ "$ws_degraded_count" -gt 0 ]; then
    echo "   ⚠️  WS entered degraded mode (10+ reconnect failures)"
    grep "ENTER_DEGRADED_MODE" "$LOG_FILE" | sed 's/^/      /'
fi

if [ "$reseed_breaker_count" -gt 0 ]; then
    echo "   ⚠️  Reseed circuit breaker triggered (5+ consecutive failures)"
    grep "RESEED circuit breaker" "$LOG_FILE" | sed 's/^/      /'
fi

if [ "$ws_degraded_count" -eq 0 ] && [ "$reseed_breaker_count" -eq 0 ]; then
    echo "   ✅ No degraded mode entries (healthy operation)"
fi
echo ""

# 5. Check ConnectionClosedError spam
echo "5️⃣ Checking for ConnectionClosedError spam..."
conn_closed_count=$(grep -c "ConnectionClosedError" "$LOG_FILE" || true)
echo "   ConnectionClosedError occurrences: $conn_closed_count"

if [ "$conn_closed_count" -gt 50 ]; then
    echo "   ❌ FAILURE: Excessive ConnectionClosedError spam (should be rare)"
    echo "   This indicates reseed storm or missing WS reconnection"
elif [ "$conn_closed_count" -gt 0 ]; then
    echo "   ✅ Bounded ConnectionClosedError (expected during actual failures)"
else
    echo "   ✅ No ConnectionClosedError (clean run)"
fi
echo ""

# 6. Check ping/pong health
echo "6️⃣ Checking ping/pong watchdog..."
ping_success_count=$(grep -c "Ping successful" "$LOG_FILE" || true)
ping_failed_count=$(grep -c "Ping failed" "$LOG_FILE" || true)
echo "   Ping successes: $ping_success_count"
echo "   Ping failures: $ping_failed_count"

if [ "$ping_success_count" -gt 0 ]; then
    echo "   ✅ Ping watchdog active"
fi

if [ "$ping_failed_count" -gt 0 ]; then
    echo "   ℹ️  Ping failures detected (triggered reconnection):"
    grep "Ping failed" "$LOG_FILE" | tail -n 3 | sed 's/^/      /'
fi
echo ""

# 7. Check for rate limit initialization
echo "7️⃣ Checking rate limit initialization..."
# Look for immediate reseed on startup (would indicate bug)
first_reseed_line=$(grep -n "RESEED TRIGGER" "$LOG_FILE" | head -n 1 | cut -d: -f1)
if [ -n "$first_reseed_line" ]; then
    first_cycle_line=$(grep -n "--- Cycle 1 ---" "$LOG_FILE" | head -n 1 | cut -d: -f1)
    if [ -n "$first_cycle_line" ] && [ "$first_reseed_line" -lt "$((first_cycle_line + 100))" ]; then
        echo "   ⚠️  WARNING: Reseed triggered very early (possible rate limit bug)"
    else
        echo "   ✅ No immediate reseed on startup (rate limit working)"
    fi
else
    echo "   ℹ️  No reseeds in log"
fi
echo ""

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="

issues=0

if [ "$conn_closed_count" -gt 50 ]; then
    echo "❌ FAIL: ConnectionClosedError spam detected"
    ((issues++))
fi

if [ "$reseed_trigger_count" -gt 100 ] && [ "$reseed_complete_count" -lt 5 ]; then
    echo "❌ FAIL: Reseed storm detected (many triggers, few completions)"
    ((issues++))
fi

if [ "$reseed_complete_count" -gt 0 ] && [ "$skip_fetch_count" -eq 0 ]; then
    echo "⚠️  WARN: Skip-fetch not working (reseeds completed but skip not observed)"
    ((issues++))
fi

if [ $issues -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED"
    echo ""
    echo "Key metrics:"
    echo "  - WS reconnections: $reconnect_count (successes: $reconnect_success_count)"
    echo "  - Reseed cycles: $reseed_complete_count"
    echo "  - Skip-fetch optimizations: $skip_fetch_count"
    echo "  - ConnectionClosedError: $conn_closed_count (bounded)"
    echo ""
    echo "System is production-ready for 24h burn-in ✅"
else
    echo "❌ $issues issue(s) found. Review logs for details."
    exit 1
fi
