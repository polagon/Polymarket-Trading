"""
Astra V2 — Terminal Reporting
Displays opportunities, Astra's reasoning, and portfolio stats.
"""
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from scanner.mispricing_detector import Opportunity
from config import BANKROLL

console = Console()

# Truth state display colors
TRUTH_COLORS = {
    "Verified": "bold green",
    "Supported": "green",
    "Assumed": "yellow",
    "Speculative": "dim red",
}


def print_scan_header(scan_number: int, markets_scanned: int, stats: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"[bold cyan]Astra V2[/bold cyan] — Scan #{scan_number}",
        f"[dim]{now}[/dim]  |  [white]{markets_scanned}[/white] markets  |  Bankroll: [green]${BANKROLL:.2f}[/green]",
    ]
    if stats.get("resolved", 0) > 0:
        acc = stats.get("accuracy")
        brier = stats.get("brier_score_avg")
        pnl = stats.get("total_pnl", 0)
        oc = stats.get("overconfidence_index", 0)
        sharpe = stats.get("sharpe_ratio", 0)
        sortino = stats.get("sortino_ratio", 0)

        pnl_color = "green" if pnl >= 0 else "red"
        oc_color = "yellow" if abs(oc) > 0.05 else "green"
        sharpe_color = "green" if sharpe > 1.0 else "yellow" if sharpe > 0.5 else "red"

        lines.append(
            f"Resolved: [white]{stats['resolved']}[/white]  "
            f"Acc: [white]{acc:.1%}[/white]  "
            f"Brier: [white]{brier:.3f}[/white]  "
            f"OC index: [{oc_color}]{oc:+.3f}[/{oc_color}]  "
            f"P&L: [{pnl_color}]${pnl:+.2f}[/{pnl_color}]"
        )

        # Add Sharpe/Sortino if ≥10 resolved positions
        if stats.get("resolved", 0) >= 10:
            lines.append(
                f"[dim]Risk-adjusted:[/dim] "
                f"Sharpe: [{sharpe_color}]{sharpe:.2f}[/{sharpe_color}]  "
                f"Sortino: [{sharpe_color}]{sortino:.2f}[/{sharpe_color}]"
            )
    console.print(Panel("\n".join(lines), border_style="dim cyan"))


def print_opportunities(opportunities: list[Opportunity]):
    if not opportunities:
        console.print("[dim]Astra: No trades meet minimum bar this scan. (EV > 0, robustness ≥ 3, confidence ≥ 60%)[/dim]\n")
        return

    table = Table(
        title=f"[bold green]Astra V2 — {len(opportunities)} Trade Proposal{'s' if len(opportunities) != 1 else ''}[/bold green]",
        box=box.ROUNDED,
        show_lines=True,
        border_style="green",
    )

    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Market", style="white", min_width=25, no_wrap=False)
    table.add_column("Mkt", justify="right", width=5)
    table.add_column("Astra", justify="right", width=5)
    table.add_column("Edge", justify="right", width=7)
    table.add_column("EV", justify="right", width=6)
    table.add_column("Rob", justify="right", width=4)
    table.add_column("Action", width=8)
    table.add_column("Size%", justify="right", width=5)
    table.add_column("T", width=2)  # Truth state initial

    for i, opp in enumerate(opportunities[:20], 1):
        est = opp.estimate
        edge_pct = opp.edge * 100

        # Edge coloring
        if abs(edge_pct) >= 20:
            edge_str = f"[bold green]{edge_pct:+.1f}%[/bold green]"
        elif abs(edge_pct) >= 12:
            edge_str = f"[green]{edge_pct:+.1f}%[/green]"
        else:
            edge_str = f"[yellow]{edge_pct:+.1f}%[/yellow]"

        # EV coloring
        ev = opp.ev_after_costs
        ev_color = "green" if ev > 0.05 else "yellow"
        ev_str = f"[{ev_color}]{ev:+.3f}[/{ev_color}]"

        # Robustness coloring
        rob = opp.robustness_score
        rob_color = "green" if rob >= 4 else "yellow" if rob >= 3 else "red"
        rob_str = f"[{rob_color}]{rob}/5[/{rob_color}]"

        # Direction (with arbitrage highlighting)
        if opp.direction == "BUY BOTH":
            dir_str = "[bold magenta on white]⚡ARB⚡[/bold magenta on white]"  # Risk-free arbitrage
        elif opp.direction == "BUY YES":
            dir_str = "[bold green]YES↑[/bold green]"
        else:
            dir_str = "[bold red]NO↓[/bold red]"

        # Kelly size
        kelly = opp.kelly_pct * 100
        size_str = f"[bold]{kelly:.1f}%[/bold]"

        # Truth state initial
        ts = est.truth_state[0] if est.truth_state else "A"
        ts_color = TRUTH_COLORS.get(est.truth_state, "white")
        ts_str = f"[{ts_color}]{ts}[/{ts_color}]"

        # Question truncation
        q = opp.market.question
        q = q[:52] + "..." if len(q) > 55 else q

        table.add_row(
            str(i), q,
            f"{opp.market_price:.3f}",
            f"{opp.our_estimate:.3f}",
            edge_str, ev_str, rob_str,
            dir_str, size_str, ts_str,
        )

    console.print(table)

    # Print Astra's reasoning for top 3 opportunities
    for i, opp in enumerate(opportunities[:3], 1):
        est = opp.estimate
        if est.reasoning:
            ts_color = TRUTH_COLORS.get(est.truth_state, "white")
            # Show adversarial mode indicator
            mode_tag = ""
            if getattr(est, "adversarial_mode", False):
                tier = getattr(est, "dominant_evidence_tier", "?")
                collapses = getattr(est, "correlation_collapses", 0)
                mode_tag = (
                    f" [bold magenta]⚔ADV[/bold magenta]"
                    f"[dim] tier:{tier}"
                    + (f" collapse:{collapses}" if collapses else "")
                    + "[/dim]"
                )
            console.print(
                f"  [dim]#{i}[/dim] [{ts_color}][{est.truth_state}][/{ts_color}]"
                f"{mode_tag} {est.reasoning}"
            )
            # Show PRO/CON summaries for adversarial estimates
            if getattr(est, "adversarial_mode", False):
                if getattr(est, "pro_summary", ""):
                    console.print(f"       [green]✓ PRO:[/green] [dim]{est.pro_summary}[/dim]")
                if getattr(est, "con_summary", ""):
                    console.print(f"       [red]✗ CON:[/red] [dim]{est.con_summary}[/dim]")
                p_neutral = getattr(est, "p_neutral", 0.0)
                p_aware = getattr(est, "p_aware", 0.0)
                if p_neutral:
                    edge_sig = abs(p_neutral - opp.market_price)
                    console.print(
                        f"       [dim]pNeutral={p_neutral:.3f} pAware={p_aware:.3f} "
                        f"edge_signal={edge_sig:.3f}[/dim]"
                    )
            if est.key_evidence_needed:
                console.print(f"       [dim]↳ Needs: {est.key_evidence_needed}[/dim]")

    if opportunities:
        console.print()


def print_no_credentials_warning():
    console.print(Panel(
        "[yellow]LIVE-REVIEW MODE[/yellow] — Trade proposals shown for human approval.\n"
        "No orders will execute until Polymarket CLOB credentials are configured.\n\n"
        "To enable execution:\n"
        "  1. Create a Polygon wallet with USDC\n"
        "  2. Generate Polymarket CLOB API key\n"
        "  3. Add to .env: POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE",
        title="[dim]Astra V2: Scan + Propose Only[/dim]",
        border_style="dim yellow",
    ))
    console.print()


def print_summary(n_markets: int, n_crypto: int, n_weather: int, n_other: int,
                  n_algo: int, n_astra: int, elapsed: float):
    console.print(
        f"[dim]Scanned [white]{n_markets}[/white] markets "
        f"([cyan]{n_crypto}[/cyan] crypto, [blue]{n_weather}[/blue] weather, "
        f"[dim]{n_other}[/dim] other) in [white]{elapsed:.1f}s[/white] — "
        f"[white]{n_algo}[/white] algo + [white]{n_astra}[/white] Astra estimates[/dim]"
    )
    console.print()
