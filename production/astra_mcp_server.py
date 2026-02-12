"""
Astra V2 MCP Server — Claude-callable tools for the Polymarket trading system.

Inspired by invest-composer/composer-trade-mcp (the most architecturally
forward-thinking repo in our research). That MCP server exposed 30+ trading
tools to Claude/Cursor, enabling conversational interaction with a strategy
database. We build the same pattern for Polymarket.

This MCP server wraps Astra's core functions as Claude-callable tools:
  - get_portfolio()         → paper portfolio stats + open positions
  - scan_opportunities()   → run a full Astra scan + return top opportunities
  - evaluate_market()      → adversarial PRO/CON estimate on one specific market
  - get_whale_signals()    → current volume spike signals
  - get_arb_signals()      → YES+NO arbitrage opportunities
  - get_calendar_events()  → upcoming FOMC/CPI/NFP in next 48h
  - get_system_status()    → health check: API, credentials, VIX, heartbeat

Usage (run as MCP server):
    ./venv/bin/python3.12 production/astra_mcp_server.py

Then add to Claude Code's MCP config (.claude/mcp.json or via /mcp command):
    {
      "astra": {
        "command": "/Users/pads/Claude/Polymarket/venv/bin/python3.12",
        "args": ["/Users/pads/Claude/Polymarket/production/astra_mcp_server.py"]
      }
    }

Once connected, you can ask Claude Code:
  "What are Astra's top opportunities right now?"
  "Evaluate this Polymarket market: Will Bitcoin hit $150k by June?"
  "Show me the current paper portfolio"
  "Any whale signals or arbitrage right now?"
"""
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path so we can import Astra modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Astra imports
from config import BANKROLL, ANTHROPIC_API_KEY
from scanner.market_fetcher import fetch_active_markets
from scanner.longshot_screener import scan_for_arbitrage, screen_longshot_markets
from data_sources.signals import fetch_all_signals
from data_sources.whale_tracker import track_volume_and_detect_whales
from data_sources.economic_calendar import check_markets_for_events, get_todays_events
from production.clob_executor import get_credentials_status, is_live_trading_enabled


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server setup
# ─────────────────────────────────────────────────────────────────────────────

app = Server("astra-polymarket")


# ─────────────────────────────────────────────────────────────────────────────
# Tool: get_portfolio
# ─────────────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_portfolio",
            description=(
                "Get current Astra paper trading portfolio: open positions, "
                "resolved trades, P&L, win rate, and bankroll status."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="scan_opportunities",
            description=(
                "Run a full Astra market scan and return the top opportunities. "
                "Fetches up to 200 markets, runs longshot screener and arb scanner. "
                "Takes 10-30 seconds. Returns ranked list of opportunities with edge, "
                "EV, and Kelly sizing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max markets to scan (default 200, max 400)",
                        "default": 200,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="evaluate_market",
            description=(
                "Run Astra's full adversarial PRO/CON pipeline on a specific market "
                "question. Returns p_hat, confidence, edge, Kelly size, PRO summary, "
                "CON summary, and evidence tier analysis. Takes 5-15 seconds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The Polymarket market question to evaluate",
                    },
                    "market_price": {
                        "type": "number",
                        "description": "Current YES price on Polymarket (0-1)",
                    },
                },
                "required": ["question", "market_price"],
            },
        ),
        Tool(
            name="get_whale_signals",
            description=(
                "Fetch current whale/large-order signals from volume spike detection. "
                "Returns markets with 3×+ volume spikes vs recent baseline. "
                "These indicate informed money movement (Tier B evidence)."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_arb_signals",
            description=(
                "Scan for YES+NO arbitrage opportunities on Polymarket. "
                "Returns markets where combined YES+NO price < $0.97 (guaranteed profit). "
                "Only includes markets resolving within 7 days."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_calendar_events",
            description=(
                "Get upcoming high-impact economic events (FOMC, CPI, NFP) "
                "in the next 48 hours, and check which Polymarket markets are "
                "resolving near these events."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "window_hours": {
                        "type": "number",
                        "description": "Look-ahead window in hours (default 48)",
                        "default": 48,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_system_status",
            description=(
                "Get Astra system health status: API connectivity, credentials, "
                "VIX level, circuit breaker state, last successful fetch timestamp."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    # ── get_portfolio ─────────────────────────────────────────────────────────
    if name == "get_portfolio":
        positions_file = Path("memory/paper_positions.json")
        if not positions_file.exists():
            return [TextContent(type="text", text="No paper positions found. Run paper_trader.py first.")]

        try:
            positions = json.loads(positions_file.read_text())
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading positions: {e}")]

        open_pos = [p for p in positions if not p.get("resolved")]
        resolved = [p for p in positions if p.get("resolved") and p.get("pnl") is not None]
        total_pnl = sum(p["pnl"] for p in resolved)
        wins = [p for p in resolved if p["pnl"] > 0]
        win_rate = len(wins) / len(resolved) if resolved else 0.0

        lines = [
            f"# Astra Paper Portfolio — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"",
            f"**Bankroll:** ${BANKROLL:.2f} starting",
            f"**Open positions:** {len(open_pos)}",
            f"**Resolved trades:** {len(resolved)}",
            f"**Win rate:** {win_rate:.0%}",
            f"**Total P&L:** ${total_pnl:+.2f}",
            f"",
            f"## Open Positions",
        ]
        for p in open_pos[-10:]:   # last 10
            lines.append(
                f"- {p['direction']} ${p['position_size']:.2f} @ {p['entry_price']:.3f} | "
                f"{p['question'][:60]}"
            )
        if len(open_pos) > 10:
            lines.append(f"  ... and {len(open_pos) - 10} more")

        lines.extend(["", "## Recent Resolved"])
        for p in resolved[-5:]:
            pnl_str = f"${p['pnl']:+.2f}"
            outcome = "YES ✓" if p.get("outcome") else "NO ✗"
            lines.append(f"- {outcome} {pnl_str} | {p['question'][:55]}")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── scan_opportunities ────────────────────────────────────────────────────
    elif name == "scan_opportunities":
        limit = min(int(arguments.get("limit", 200)), 400)
        try:
            markets = await fetch_active_markets(limit=limit)
            if not markets:
                return [TextContent(type="text", text="No markets returned from Gamma API.")]

            longshot_signals = screen_longshot_markets(markets)
            arb_signals = scan_for_arbitrage(markets)
            whale_signals = track_volume_and_detect_whales(markets)
            calendar_events = check_markets_for_events(markets, 48)

            lines = [
                f"# Astra Scan Results — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"Scanned **{len(markets)}** markets",
                f"",
                f"## Arbitrage Signals ({len(arb_signals)} found)",
            ]
            for arb in arb_signals[:5]:
                lines.append(
                    f"- **{arb.guaranteed_profit_pct:.1%}** guaranteed profit | "
                    f"YES={arb.yes_price:.3f} + NO={arb.no_price:.3f} = {arb.combined_price:.3f} | "
                    f"{arb.market.question[:60]}"
                )

            lines.extend([f"", f"## Longshot BUY NO Signals ({len(longshot_signals)} found)"])
            for ls in longshot_signals[:5]:
                lines.append(
                    f"- YES@{ls.yes_price:.1%} → BUY NO | EV={ls.ev_after_costs:+.3f} | "
                    f"{ls.market.question[:60]}"
                )

            lines.extend([f"", f"## Whale Signals ({len(whale_signals)} volume spikes)"])
            for sig in list(whale_signals.values())[:3]:
                lines.append(
                    f"- {sig.spike_ratio:.1f}× volume spike | ${sig.current_volume:,.0f} | "
                    f"{sig.question[:60]}"
                )

            lines.extend([f"", f"## Calendar Proximity ({len(calendar_events)} markets near events)"])
            for cid, events in list(calendar_events.items())[:3]:
                market = next((m for m in markets if m.condition_id == cid), None)
                if market and events:
                    ev, hrs = events[0]
                    lines.append(f"- {ev.name} in {abs(hrs):.0f}h | {market.question[:55]}")

            return [TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            return [TextContent(type="text", text=f"Scan error: {e}")]

    # ── evaluate_market ───────────────────────────────────────────────────────
    elif name == "evaluate_market":
        question = arguments.get("question", "")
        market_price = float(arguments.get("market_price", 0.5))

        if not question:
            return [TextContent(type="text", text="Error: question is required.")]
        if not ANTHROPIC_API_KEY:
            return [TextContent(type="text", text="Error: ANTHROPIC_API_KEY not configured.")]

        try:
            from scanner.probability_estimator import _astra_batch_adversarial
            from scanner.market_fetcher import Market

            # Create a synthetic Market object for evaluation
            synthetic_market = Market(
                condition_id="mcp_eval",
                question=question,
                end_date_iso=None,
                category="other",
                yes_token_id=None,
                no_token_id=None,
                yes_price=market_price,
                no_price=1.0 - market_price,
                liquidity=10000.0,
                volume=1000.0,
                hours_to_expiry=float("inf"),
            )

            results = await _astra_batch_adversarial([(synthetic_market, None)], "")
            if not results:
                return [TextContent(type="text", text="No estimate produced.")]

            est = results[0]
            lines = [
                f"# Astra Adversarial Evaluation",
                f"**Question:** {question}",
                f"**Market price:** {market_price:.3f}",
                f"",
                f"## Result",
                f"- **p_hat:** {est.probability:.3f}",
                f"- **p_neutral:** {est.p_neutral:.3f} (ignores market price)",
                f"- **p_aware:** {est.p_aware:.3f} (Bayesian with market prior)",
                f"- **Confidence:** {est.confidence:.0%}",
                f"- **Edge:** {est.edge:+.3f}",
                f"- **EV after costs:** {est.ev_after_costs:+.4f}",
                f"- **Robustness:** {est.robustness_score}/5",
                f"- **Kelly:** {est.kelly_position_pct:.2%} of bankroll",
                f"- **Truth state:** {est.truth_state}",
                f"- **Trade?** {'✅ YES' if not est.no_trade else f'❌ NO — {est.no_trade_reason}'}",
                f"",
                f"## Adversarial Analysis",
                f"**PRO (YES case):** {est.pro_summary}",
                f"**CON (NO case):** {est.con_summary}",
                f"",
                f"## Synthesis",
                f"{est.reasoning}",
                f"",
                f"*Evidence tier: {est.dominant_evidence_tier} | "
                f"Correlation collapses: {est.correlation_collapses} | "
                f"Mode: {'adversarial' if est.adversarial_mode else 'single-pass'}*",
            ]
            return [TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            return [TextContent(type="text", text=f"Evaluation error: {e}")]

    # ── get_whale_signals ─────────────────────────────────────────────────────
    elif name == "get_whale_signals":
        try:
            markets = await fetch_active_markets(limit=200)
            signals = track_volume_and_detect_whales(markets)
            if not signals:
                return [TextContent(type="text", text="No whale signals detected in current scan.")]
            lines = [f"# Whale Signals — {len(signals)} volume spikes detected\n"]
            for sig in sorted(signals.values(), key=lambda s: s.spike_ratio, reverse=True):
                lines.append(
                    f"- **{sig.spike_ratio:.1f}× spike** | "
                    f"${sig.current_volume:,.0f} vs ${sig.baseline_volume:,.0f} baseline | "
                    f"{sig.question[:70]}"
                )
            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── get_arb_signals ───────────────────────────────────────────────────────
    elif name == "get_arb_signals":
        try:
            markets = await fetch_active_markets(limit=400)
            arb_signals = scan_for_arbitrage(markets)
            if not arb_signals:
                return [TextContent(type="text", text="No arbitrage opportunities found currently.")]
            lines = [f"# Arbitrage Signals — {len(arb_signals)} found\n"]
            for arb in arb_signals:
                hours = arb.market.hours_to_expiry
                lines.append(
                    f"- **{arb.guaranteed_profit_pct:.2%}** profit | "
                    f"YES={arb.yes_price:.3f} + NO={arb.no_price:.3f} | "
                    f"Resolves in {hours:.0f}h | Min ${arb.min_position_usd:.0f} | "
                    f"{arb.market.question[:60]}"
                )
            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── get_calendar_events ───────────────────────────────────────────────────
    elif name == "get_calendar_events":
        window = float(arguments.get("window_hours", 48))
        events = get_todays_events(window_hours=window)
        try:
            markets = await fetch_active_markets(limit=200)
            market_events = check_markets_for_events(markets, window)
        except Exception:
            markets = []
            market_events = {}

        lines = [f"# Economic Calendar — Next {window:.0f}h\n"]
        if events:
            lines.append(f"## Upcoming Events ({len(events)})")
            for ev in events:
                lines.append(f"- **{ev.name}** ({ev.event_type}) — {ev.date_utc[:16]} UTC")
                lines.append(f"  {ev.description}")
        else:
            lines.append(f"No high-impact events in next {window:.0f}h.")

        if market_events:
            lines.extend([f"", f"## Markets Near Events ({len(market_events)})"])
            market_by_id = {m.condition_id: m for m in markets}
            for cid, evts in list(market_events.items())[:10]:
                market = market_by_id.get(cid)
                if market and evts:
                    ev, hrs = evts[0]
                    lines.append(f"- {ev.name} in {abs(hrs):.0f}h | {market.question[:65]}")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── get_system_status ─────────────────────────────────────────────────────
    elif name == "get_system_status":
        try:
            ctx = await fetch_all_signals()
            vix_str = (
                f"{ctx.macro.vix:.1f} [{ctx.macro.vix_label}] "
                f"Kelly×{ctx.macro.vix_kelly_multiplier:.2f}"
                if ctx.macro and ctx.macro.vix is not None else "unavailable"
            )
            spread_str = (
                f"{ctx.macro.yield_spread_10y2y:+.2f}% [{ctx.macro.recession_signal}]"
                if ctx.macro and ctx.macro.yield_spread_10y2y is not None else "unavailable"
            )
            fg_str = (
                f"{ctx.fear_greed.label} ({ctx.fear_greed.value}/100)"
                if ctx.fear_greed else "unavailable"
            )
        except Exception:
            vix_str = spread_str = fg_str = "unavailable (signals fetch failed)"

        positions_file = Path("memory/paper_positions.json")
        has_positions = positions_file.exists()

        lines = [
            f"# Astra System Status — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"",
            f"## API & Credentials",
            f"- Anthropic API: {'✅ configured' if ANTHROPIC_API_KEY else '❌ not set'}",
            f"- {get_credentials_status()}",
            f"- Paper positions file: {'✅ exists' if has_positions else '⚠️  not found (run paper_trader.py)'}",
            f"",
            f"## Market Signals",
            f"- VIX: {vix_str}",
            f"- Yield spread: {spread_str}",
            f"- Crypto Fear/Greed: {fg_str}",
            f"",
            f"## Configuration",
            f"- Bankroll: ${BANKROLL:.2f}",
            f"- Circuit breaker: 5% daily loss limit",
            f"- Order TTL (arb): 30s (when CLOB live)",
        ]
        return [TextContent(type="text", text="\n".join(lines))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
