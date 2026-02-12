#!/bin/bash
# Astra Watchdog — runs via cron every 15 minutes
# Restarts the launchd daemon if it has stopped unexpectedly.
#
# Cron entry (add via: crontab -e):
#   */15 * * * * /Users/pads/Claude/Polymarket/watchdog.sh >> /Users/pads/Library/Logs/astra-watchdog.log 2>&1

LABEL="com.astra.paper-trader"
PLIST="$HOME/Library/LaunchAgents/com.astra.paper-trader.plist"
LOG="$HOME/Library/Logs/astra-watchdog.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Check if launchd job is running
if launchctl list | grep -q "$LABEL"; then
    # Daemon is running — all good, silent exit
    exit 0
fi

# Daemon is not running — attempt restart
echo "[$TIMESTAMP] ⚠️  Astra daemon NOT running — attempting restart..."

# Try launchctl load
if launchctl load -w "$PLIST" 2>/dev/null; then
    sleep 2
    if launchctl list | grep -q "$LABEL"; then
        echo "[$TIMESTAMP] ✅ Astra daemon restarted successfully"
    else
        echo "[$TIMESTAMP] ❌ Restart attempt failed — check plist and logs"
    fi
else
    echo "[$TIMESTAMP] ❌ launchctl load failed — plist may be missing or invalid"
fi
