"""
Polymarket Trading Scanner — Main Entry Point

Phase 1: Scan-only mode (no trading execution)
Phase 2: Add trading credentials to .env to enable execution

Run: python main.py
     python main.py --once    (single scan, no loop)
     python main.py --stats   (show learning stats only)
"""
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import SCAN_INTERVAL_SECONDS, BANKROLL, ANTHROPIC_API_KEY, ODDS_API_KEY
from scanner.market_fetcher import fetch_active_markets, fetch_prices
from scanner.probability_estimator import estimate_markets
from scanner.mispricing_detector import find_opportunities
from scanner.learning_agent import LearningAgent, Prediction
from scanner.trade_logger import init_db, log_market_snapshot, log_estimate
from data_sources import crypto as crypto_source
from data_sources.weather import parse_weather_question, fetch_forecast
from data_sources.sports import get_sports_estimates
from data_sources.signals import fetch_all_signals
from data_sources.price_history import store_price_snapshot
import report


async def run_scan(learning_agent: LearningAgent, scan_number: int):
    """Run a single scan cycle."""
    # KILL SWITCH: Check for emergency halt file
    KILL_SWITCH_PATH = Path("/tmp/astra_kill")
    if KILL_SWITCH_PATH.exists():
        report.console.print("[red bold]🛑 KILL SWITCH ACTIVATED[/red bold]")
        report.console.print(f"[red]Found /tmp/astra_kill — halting immediately.[/red]")
        report.console.print(f"[yellow]Remove file to resume: rm /tmp/astra_kill[/yellow]")
        sys.exit(0)

    t_start = time.time()

    # 1. Fetch markets
    report.console.print(f"[dim]Fetching markets...[/dim]", end="\r")
    markets = await fetch_active_markets(limit=300)

    if not markets:
        report.console.print("[red]Failed to fetch markets. Check internet connection.[/red]")
        return

    # 2. Get current prices
    report.console.print(f"[dim]Fetching prices for {len(markets)} markets...[/dim]", end="\r")
    markets = await fetch_prices(markets)

    # 3. Categorize
    crypto_markets = [m for m in markets if m.category == "crypto"]
    weather_markets = [m for m in markets if m.category == "weather"]
    sports_markets = [m for m in markets if m.category == "sports"]
    other_markets = [m for m in markets if m.category not in ("crypto", "weather", "sports")]

    # 4. Fetch external data
    report.console.print("[dim]Fetching external data...[/dim]", end="\r")

    # Crypto prices
    coin_ids_needed = set()
    for m in crypto_markets:
        parsed = crypto_source.parse_crypto_question(m.question)
        if parsed:
            coin_ids_needed.add(parsed["coin_id"])

    price_data = {}
    if coin_ids_needed:
        price_data = await crypto_source.fetch_prices(list(coin_ids_needed))

    # Weather forecasts
    locations_needed = set()
    for m in weather_markets:
        parsed = parse_weather_question(m.question)
        if parsed and parsed.get("location"):
            locations_needed.add(parsed["location"])

    forecasts = {}
    if locations_needed:
        forecast_tasks = {loc: fetch_forecast(loc) for loc in locations_needed}
        forecast_results = await asyncio.gather(*forecast_tasks.values(), return_exceptions=True)
        for loc, result in zip(forecast_tasks.keys(), forecast_results):
            if isinstance(result, Exception) or result is None:
                continue
            forecasts[loc] = result

    # Sports odds (via The Odds API)
    sports_estimates = {}
    if ODDS_API_KEY and sports_markets:
        try:
            sports_estimates = await get_sports_estimates(
                [m.question for m in sports_markets]
            )
        except Exception:
            pass

    # Market signals (Fear/Greed, macro)
    market_context = None
    try:
        market_context = await fetch_all_signals()
        signals_summary = market_context.summary()
    except Exception:
        signals_summary = ""

    # 5. Estimate probabilities
    report.console.print("[dim]Estimating probabilities...[/dim]", end="\r")
    learning_context = learning_agent.get_strategy_context()
    if signals_summary:
        learning_context = signals_summary + "\n\n" + learning_context

    estimates = await estimate_markets(
        markets=markets,
        price_data=price_data,
        forecasts=forecasts,
        learning_context=learning_context,
        sports_estimates=sports_estimates,
        macro_signals=market_context,
        learning_agent=learning_agent,
    )

    # 6. Store price snapshots for multi-timeframe momentum (Phase 8)
    for market in markets:
        store_price_snapshot(market.condition_id, market.yes_price)

    # 7. Log market snapshots and estimates to SQLite
    for market in markets:
        log_market_snapshot(market)

    # Create estimate map for logging
    estimate_map = {e.market_condition_id: e for e in estimates}
    for market in markets:
        if market.condition_id in estimate_map:
            log_estimate(estimate_map[market.condition_id], market.condition_id, market)

    # 8. Detect mispricings
    opportunities = find_opportunities(markets, estimates)

    # 9. Record predictions for Astra learning
    for opp in opportunities:
        pred = Prediction(
            market_condition_id=opp.market.condition_id,
            question=opp.market.question,
            category=opp.market.category,
            our_probability=opp.our_estimate,
            probability_low=opp.estimate.probability_low,
            probability_high=opp.estimate.probability_high,
            market_price=opp.market_price,
            direction=opp.direction,
            source=opp.estimate.source,
            truth_state=opp.estimate.truth_state,
            reasoning=opp.estimate.reasoning,
            key_unknowns=opp.estimate.key_evidence_needed,
            no_trade=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        learning_agent.record_prediction(pred)

    elapsed = time.time() - t_start

    # 10. Count estimation sources
    n_algo = sum(1 for e in estimates if e.source in ("crypto_lognormal", "noaa_forecast"))
    n_astra = sum(1 for e in estimates if e.source == "astra_v2")

    # 11. Print report
    stats = learning_agent.get_stats()
    report.print_scan_header(scan_number, len(markets), stats)
    report.print_no_credentials_warning()
    report.print_opportunities(opportunities)
    report.print_summary(
        n_markets=len(markets),
        n_crypto=len(crypto_markets),
        n_weather=len(weather_markets),
        n_other=len(other_markets) + len(sports_markets),
        n_algo=n_algo,
        n_astra=n_astra,
        elapsed=elapsed,
    )

    # 12. Periodically trigger learning cycle
    if stats.get("resolved", 0) > 0 and stats["resolved"] % 10 == 0:
        report.console.print("[dim]Running learning cycle...[/dim]")
        await learning_agent.evolve()
        report.console.print("[dim]Strategy updated.[/dim]")


async def show_stats(learning_agent: LearningAgent):
    """Display learning stats and exit."""
    stats = learning_agent.get_stats()
    calibration = learning_agent.calibration_buckets()

    from rich.table import Table
    from rich import box

    report.console.print("\n[bold]Learning Agent Stats[/bold]\n")
    report.console.print(f"Total predictions tracked: [white]{stats['total_predictions']}[/white]")
    report.console.print(f"Resolved: [white]{stats.get('resolved', 0)}[/white]")
    report.console.print(f"Unresolved: [white]{stats.get('unresolved', 0)}[/white]")

    if stats.get("accuracy") is not None:
        report.console.print(f"Overall accuracy: [white]{stats['accuracy']:.1%}[/white]")

    report.console.print(f"Total P&L: [{'green' if stats['total_pnl'] >= 0 else 'red'}]${stats['total_pnl']:+.2f}[/]")

    if stats.get("by_category"):
        report.console.print("\n[bold]By Category:[/bold]")
        for cat, data in stats["by_category"].items():
            report.console.print(f"  {cat}: accuracy={data['accuracy']:.1%} n={data['n']} pnl=${data['pnl']:+.2f}")

    if calibration:
        table = Table(title="Calibration", box=box.SIMPLE)
        table.add_column("Bucket")
        table.add_column("Predicted Avg", justify="right")
        table.add_column("Actual Rate", justify="right")
        table.add_column("Bias", justify="right")
        table.add_column("N", justify="right")
        for c in calibration:
            bias_color = "green" if abs(c.bias) < 0.05 else "yellow" if abs(c.bias) < 0.10 else "red"
            table.add_row(
                c.bucket,
                f"{c.predicted_avg:.3f}",
                f"{c.actual_rate:.3f}",
                f"[{bias_color}]{c.bias:+.3f}[/{bias_color}]",
                str(c.n),
            )
        report.console.print(table)

    strategy = learning_agent.get_strategy_context()
    if strategy:
        from rich.panel import Panel
        report.console.print(Panel(strategy[:1000], title="[dim]Current Strategy Context[/dim]"))


async def main():
    args = sys.argv[1:]

    # Initialize memory directory and database
    import os
    os.makedirs("memory", exist_ok=True)
    init_db()  # Initialize SQLite trade logger

    learning_agent = LearningAgent()

    if "--stats" in args:
        await show_stats(learning_agent)
        return

    once = "--once" in args

    report.console.print()
    report.console.print("[bold cyan]Polymarket Scanner[/bold cyan] starting up...")
    if not ANTHROPIC_API_KEY:
        report.console.print("[yellow]Warning: ANTHROPIC_API_KEY not set. Claude estimation disabled.[/yellow]")
    report.console.print(f"Bankroll: [green]${BANKROLL:.2f}[/green]")
    report.console.print(f"Scan interval: [white]{SCAN_INTERVAL_SECONDS}s[/white]")
    report.console.print()

    scan_number = 1
    while True:
        try:
            await run_scan(learning_agent, scan_number)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            report.console.print(f"[red]Scan error: {e}[/red]")

        scan_number += 1

        if once:
            break

        report.console.print(f"[dim]Next scan in {SCAN_INTERVAL_SECONDS}s. Ctrl+C to stop.[/dim]\n")

        try:
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        report.console.print("\n[dim]Stopped.[/dim]")
