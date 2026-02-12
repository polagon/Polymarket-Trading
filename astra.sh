#!/bin/bash
# Astra V2 — daemon management + live portfolio viewer
# Usage:
#   ./astra.sh start      — load & start daemon (runs on every login)
#   ./astra.sh stop       — unload & stop daemon
#   ./astra.sh restart    — stop + start
#   ./astra.sh status     — show if running + recent log lines
#   ./astra.sh portfolio  — show current paper portfolio
#   ./astra.sh log        — tail live logs
#   ./astra.sh once       — run a single scan (no loop) for testing

PLIST="$HOME/Library/LaunchAgents/com.astra.paper-trader.plist"
LABEL="com.astra.paper-trader"
LOG="$HOME/Library/Logs/astra-paper-trader.log"
ERRLOG="$HOME/Library/Logs/astra-paper-trader-error.log"
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/venv/bin/python3.12"

case "$1" in
  start)
    echo "Starting Astra daemon..."
    launchctl load -w "$PLIST"
    sleep 1
    launchctl list | grep "$LABEL" && echo "✅ Astra daemon started" || echo "❌ Failed to start"
    ;;

  stop)
    echo "Stopping Astra daemon..."
    launchctl unload "$PLIST"
    echo "✅ Astra daemon stopped"
    ;;

  restart)
    echo "Restarting Astra daemon..."
    launchctl unload "$PLIST" 2>/dev/null
    sleep 2
    launchctl load -w "$PLIST"
    sleep 1
    launchctl list | grep "$LABEL" && echo "✅ Astra daemon restarted" || echo "❌ Failed to restart"
    ;;

  status)
    echo "=== Astra Daemon Status ==="
    if launchctl list | grep -q "$LABEL"; then
      PID=$(launchctl list | grep "$LABEL" | awk '{print $1}')
      echo "✅ RUNNING (PID: $PID)"
    else
      echo "❌ NOT RUNNING"
    fi
    echo ""
    echo "=== Last 20 log lines ==="
    [ -f "$LOG" ] && tail -20 "$LOG" || echo "(no log yet)"
    echo ""
    [ -f "$ERRLOG" ] && [ -s "$ERRLOG" ] && echo "=== Recent errors ===" && tail -10 "$ERRLOG"
    ;;

  portfolio)
    echo "=== Astra V2 Paper Portfolio ==="
    cd "$DIR"
    "$PYTHON" -c "
import json
from pathlib import Path
from datetime import datetime, timezone

pos_file = Path('memory/paper_positions.json')
if not pos_file.exists():
    print('No portfolio file found.')
    exit()

positions = json.loads(pos_file.read_text())
from config import BANKROLL

open_pos = [p for p in positions if not p['resolved']]
resolved = [p for p in positions if p['resolved'] and p.get('pnl') is not None]

total_invested = sum(p['position_size'] for p in open_pos)
total_pnl = sum(p['pnl'] for p in resolved)
wins = [p for p in resolved if p['pnl'] > 0]
cash = BANKROLL - sum(p['position_size'] for p in positions if not p['resolved']) + total_pnl
portfolio_value = cash + total_invested

print(f'Bankroll:  \${BANKROLL:.2f} starting')
print(f'Cash:      \${cash:.2f}')
print(f'Invested:  \${total_invested:.2f}')
print(f'Value:     \${portfolio_value:.2f}  ({(portfolio_value/BANKROLL-1)*100:+.1f}%)')
print(f'P&L:       \${total_pnl:+.2f}  ({len(wins)}W / {len(resolved)-len(wins)}L)')
print()

if open_pos:
    print(f'--- OPEN POSITIONS ({len(open_pos)}) ---')
    for p in open_pos:
        print(f'  [{p[\"direction\"]}] {p[\"question\"][:55]}')
        print(f'    Entry: {p[\"entry_price\"]:.3f}  Size: \${p[\"position_size\"]:.2f}  Our P: {p[\"our_probability\"]:.1%}')
else:
    print('No open positions.')
print()

if resolved:
    print(f'--- RESOLVED ({len(resolved)}) ---')
    for p in resolved:
        sign = '+' if p['pnl'] > 0 else ''
        outcome = 'YES' if p['outcome'] else 'NO'
        print(f'  [{\"✅\" if p[\"pnl\"] > 0 else \"❌\"}] {p[\"question\"][:55]}')
        print(f'    Resolved: {outcome}  P&L: \${sign}{p[\"pnl\"]:.2f}')
"
    ;;

  log)
    echo "Tailing Astra logs (Ctrl+C to stop)..."
    tail -f "$LOG"
    ;;

  once)
    echo "Running single scan..."
    cd "$DIR"
    "$PYTHON" paper_trader.py --fast &
    PID=$!
    sleep 5
    echo "(scan running as PID $PID, check memory/paper_positions.json for results)"
    ;;

  watchdog)
    echo "=== Astra Watchdog Status ==="
    echo ""
    echo "--- Cron job ---"
    if crontab -l 2>/dev/null | grep -q "watchdog.sh"; then
      crontab -l | grep "watchdog.sh"
      echo "✅ Cron watchdog installed"
    else
      echo "❌ Cron watchdog NOT installed"
      echo "   Run: (crontab -l 2>/dev/null; echo '*/15 * * * * $DIR/watchdog.sh >> ~/Library/Logs/astra-watchdog.log 2>&1') | crontab -"
    fi
    echo ""
    echo "--- Watchdog log (last 10 lines) ---"
    WLOG="$HOME/Library/Logs/astra-watchdog.log"
    [ -f "$WLOG" ] && tail -10 "$WLOG" || echo "(no watchdog activity yet)"
    echo ""
    echo "--- launchd daemon ---"
    if launchctl list | grep -q "$LABEL"; then
      PID=$(launchctl list | grep "$LABEL" | awk '{print $1}')
      echo "✅ RUNNING (PID: $PID)"
    else
      echo "❌ NOT RUNNING"
    fi
    ;;

  *)
    echo "Usage: ./astra.sh [start|stop|restart|status|portfolio|log|once|watchdog]"
    echo ""
    echo "  start      Load & start the background daemon (persists across reboots)"
    echo "  stop       Stop the daemon"
    echo "  restart    Restart the daemon"
    echo "  status     Show daemon status + recent logs"
    echo "  portfolio  Show current paper portfolio P&L"
    echo "  log        Tail live log output"
    echo "  once       Run a single scan for testing"
    echo "  watchdog   Show cron watchdog status + recent watchdog activity"
    ;;
esac
