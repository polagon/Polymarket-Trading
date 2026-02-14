"""
Economic Calendar Signal — pre-tag markets near high-impact events.

Markets resolving within ±48h of a scheduled FOMC/CPI/NFP/Jobs event get a
flag in Astra's context. These windows have historically higher information
content and edge density because:
  - Informed actors price in event outcomes before they happen
  - Uncertainty collapses rapidly after the event → prices move sharply
  - Many prediction markets have resolution criteria tied to post-event data

From research:
  - rahulpatil0001/forex_newstrading_bot: timestamp-trigger trading around
    FOMC/CPI/NFP events (MT5 pending order strategy)
  - 1kx bottlenecks paper: "AI is the catalyst" for event-driven markets
  - LiveTradeBench: "causal reasoning > surface correlation" — White House
    visit → edge, Zelenskyy statement → no edge. Event proximity matters.

Schedule source:
  - Federal Reserve publishes FOMC schedule in January each year (fomc-calendar)
  - BLS publishes CPI/NFP schedule in January each year
  - These are deterministic — no API needed

Update cadence: Update this file in January each year with the new schedule.
Last updated: 2026-02-11
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class EconomicEvent:
    name: str  # e.g. "FOMC Rate Decision"
    event_type: str  # "FOMC" | "CPI" | "NFP" | "JOBS" | "GDP"
    date_utc: str  # ISO format "YYYY-MM-DDTHH:MM:00+00:00"
    impact: str  # "HIGH" | "MEDIUM"
    description: str  # Brief description for Astra context


# ─────────────────────────────────────────────────────────────────────────────
# 2025–2026 High-Impact Economic Calendar
# All times in UTC. FOMC decisions at 19:00 UTC (2pm ET).
# CPI at 13:30 UTC (8:30am ET). NFP at 13:30 UTC first Friday of month.
# ─────────────────────────────────────────────────────────────────────────────

ECONOMIC_CALENDAR: list[EconomicEvent] = [
    # ── FOMC Meetings (Federal Open Market Committee rate decisions) ─────────
    EconomicEvent(
        "FOMC Rate Decision",
        "FOMC",
        "2025-01-29T19:00:00+00:00",
        "HIGH",
        "Fed funds rate decision. Markets highly sensitive to rate surprises.",
    ),
    EconomicEvent(
        "FOMC Rate Decision",
        "FOMC",
        "2025-03-19T19:00:00+00:00",
        "HIGH",
        "Fed funds rate decision + updated dot plot projections.",
    ),
    EconomicEvent("FOMC Rate Decision", "FOMC", "2025-05-07T19:00:00+00:00", "HIGH", "Fed funds rate decision."),
    EconomicEvent(
        "FOMC Rate Decision",
        "FOMC",
        "2025-06-18T19:00:00+00:00",
        "HIGH",
        "Fed funds rate decision + updated economic projections.",
    ),
    EconomicEvent("FOMC Rate Decision", "FOMC", "2025-07-30T19:00:00+00:00", "HIGH", "Fed funds rate decision."),
    EconomicEvent(
        "FOMC Rate Decision", "FOMC", "2025-09-17T19:00:00+00:00", "HIGH", "Fed funds rate decision + updated dot plot."
    ),
    EconomicEvent("FOMC Rate Decision", "FOMC", "2025-10-29T19:00:00+00:00", "HIGH", "Fed funds rate decision."),
    EconomicEvent(
        "FOMC Rate Decision",
        "FOMC",
        "2025-12-10T19:00:00+00:00",
        "HIGH",
        "Fed funds rate decision + updated projections.",
    ),
    EconomicEvent("FOMC Rate Decision", "FOMC", "2026-01-28T19:00:00+00:00", "HIGH", "Fed funds rate decision."),
    EconomicEvent(
        "FOMC Rate Decision",
        "FOMC",
        "2026-03-18T19:00:00+00:00",
        "HIGH",
        "Fed funds rate decision + updated dot plot projections.",
    ),
    EconomicEvent("FOMC Rate Decision", "FOMC", "2026-04-29T19:00:00+00:00", "HIGH", "Fed funds rate decision."),
    EconomicEvent(
        "FOMC Rate Decision",
        "FOMC",
        "2026-06-17T19:00:00+00:00",
        "HIGH",
        "Fed funds rate decision + updated economic projections.",
    ),
    EconomicEvent("FOMC Rate Decision", "FOMC", "2026-07-29T19:00:00+00:00", "HIGH", "Fed funds rate decision."),
    EconomicEvent(
        "FOMC Rate Decision", "FOMC", "2026-09-16T19:00:00+00:00", "HIGH", "Fed funds rate decision + updated dot plot."
    ),
    EconomicEvent("FOMC Rate Decision", "FOMC", "2026-11-04T19:00:00+00:00", "HIGH", "Fed funds rate decision."),
    EconomicEvent(
        "FOMC Rate Decision",
        "FOMC",
        "2026-12-16T19:00:00+00:00",
        "HIGH",
        "Fed funds rate decision + updated projections.",
    ),
    # ── CPI (Consumer Price Index — monthly inflation) ───────────────────────
    EconomicEvent(
        "US CPI Release",
        "CPI",
        "2025-01-15T13:30:00+00:00",
        "HIGH",
        "December 2024 CPI. Key inflation metric watched by Fed.",
    ),
    EconomicEvent("US CPI Release", "CPI", "2025-02-12T13:30:00+00:00", "HIGH", "January 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-03-12T13:30:00+00:00", "HIGH", "February 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-04-10T13:30:00+00:00", "HIGH", "March 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-05-13T13:30:00+00:00", "HIGH", "April 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-06-11T13:30:00+00:00", "HIGH", "May 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-07-15T13:30:00+00:00", "HIGH", "June 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-08-12T13:30:00+00:00", "HIGH", "July 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-09-10T13:30:00+00:00", "HIGH", "August 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-10-15T13:30:00+00:00", "HIGH", "September 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-11-13T13:30:00+00:00", "HIGH", "October 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2025-12-10T13:30:00+00:00", "HIGH", "November 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-01-14T13:30:00+00:00", "HIGH", "December 2025 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-02-11T13:30:00+00:00", "HIGH", "January 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-03-11T13:30:00+00:00", "HIGH", "February 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-04-10T13:30:00+00:00", "HIGH", "March 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-05-13T13:30:00+00:00", "HIGH", "April 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-06-10T13:30:00+00:00", "HIGH", "May 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-07-14T13:30:00+00:00", "HIGH", "June 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-08-12T13:30:00+00:00", "HIGH", "July 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-09-09T13:30:00+00:00", "HIGH", "August 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-10-14T13:30:00+00:00", "HIGH", "September 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-11-12T13:30:00+00:00", "HIGH", "October 2026 CPI."),
    EconomicEvent("US CPI Release", "CPI", "2026-12-09T13:30:00+00:00", "HIGH", "November 2026 CPI."),
    # ── NFP (Non-Farm Payrolls — first Friday of each month) ─────────────────
    EconomicEvent(
        "US Non-Farm Payrolls",
        "NFP",
        "2025-01-10T13:30:00+00:00",
        "HIGH",
        "December 2024 jobs report. Labour market health indicator.",
    ),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-02-07T13:30:00+00:00", "HIGH", "January 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-03-07T13:30:00+00:00", "HIGH", "February 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-04-04T13:30:00+00:00", "HIGH", "March 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-05-02T13:30:00+00:00", "HIGH", "April 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-06-06T13:30:00+00:00", "HIGH", "May 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-07-03T13:30:00+00:00", "HIGH", "June 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-08-01T13:30:00+00:00", "HIGH", "July 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-09-05T13:30:00+00:00", "HIGH", "August 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-10-03T13:30:00+00:00", "HIGH", "September 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-11-07T13:30:00+00:00", "HIGH", "October 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2025-12-05T13:30:00+00:00", "HIGH", "November 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-01-09T13:30:00+00:00", "HIGH", "December 2025 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-02-06T13:30:00+00:00", "HIGH", "January 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-03-06T13:30:00+00:00", "HIGH", "February 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-04-03T13:30:00+00:00", "HIGH", "March 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-05-01T13:30:00+00:00", "HIGH", "April 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-06-05T13:30:00+00:00", "HIGH", "May 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-07-02T13:30:00+00:00", "HIGH", "June 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-08-07T13:30:00+00:00", "HIGH", "July 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-09-04T13:30:00+00:00", "HIGH", "August 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-10-02T13:30:00+00:00", "HIGH", "September 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-11-06T13:30:00+00:00", "HIGH", "October 2026 jobs report."),
    EconomicEvent("US Non-Farm Payrolls", "NFP", "2026-12-04T13:30:00+00:00", "HIGH", "November 2026 jobs report."),
]


def get_nearby_events(
    target_date: str,
    window_hours: float = 48.0,
) -> list[tuple[EconomicEvent, float]]:
    """
    Return events within ±window_hours of target_date.

    Args:
        target_date: ISO format datetime string (market end_date_iso)
        window_hours: Search window in hours (default 48h)

    Returns:
        List of (event, hours_delta) tuples sorted by |hours_delta|.
        hours_delta is positive if event is AFTER the target date.
    """
    try:
        target = datetime.fromisoformat(target_date.replace("Z", "+00:00"))
    except Exception:
        return []

    results = []
    window = timedelta(hours=window_hours)

    for event in ECONOMIC_CALENDAR:
        try:
            event_dt = datetime.fromisoformat(event.date_utc)
        except Exception:
            continue

        delta = event_dt - target
        if abs(delta) <= window:
            hours_delta = delta.total_seconds() / 3600
            results.append((event, round(hours_delta, 1)))

    return sorted(results, key=lambda x: abs(x[1]))


def check_markets_for_events(
    markets: list,
    window_hours: float = 48.0,
) -> dict[str, list[tuple[EconomicEvent, float]]]:
    """
    Check a list of Market objects for nearby economic events.

    Returns dict: {condition_id: [(event, hours_delta), ...]}
    Only markets with at least one nearby event are included.
    """
    result = {}
    for market in markets:
        end_date = getattr(market, "end_date_iso", None)
        if not end_date:
            continue
        nearby = get_nearby_events(end_date, window_hours)
        if nearby:
            result[market.condition_id] = nearby
    return result


def format_calendar_context(
    event_map: dict[str, list[tuple[EconomicEvent, float]]],
    markets: list,
) -> str:
    """
    Format economic calendar flags for injection into Astra V2 learning context.

    Returns a compact string labelled as context for the Astra prompt.
    Only surfaces events where the market is resolving near a high-impact date.
    """
    if not event_map:
        return ""

    market_by_id = {m.condition_id: m for m in markets}
    lines = ["=== ECONOMIC CALENDAR FLAGS ==="]
    lines.append(
        "Markets resolving near high-impact economic events "
        "(FOMC/CPI/NFP). These windows have elevated information content."
    )

    for cid, events in event_map.items():
        market = market_by_id.get(cid)
        if not market:
            continue
        for event, hours_delta in events[:2]:  # top 2 events per market
            direction = "after" if hours_delta > 0 else "before"
            abs_h = abs(hours_delta)
            time_str = (
                f"{abs_h:.0f}h {direction} market resolves"
                if abs_h >= 1
                else f"{abs_h * 60:.0f}min {direction} market resolves"
            )
            lines.append(
                f"  • {market.question[:60]}\n"
                f"    ↳ {event.name} ({event.event_type}) in {time_str}\n"
                f"    ↳ {event.description}"
            )

    return "\n".join(lines)


def get_todays_events(window_hours: float = 24.0) -> list[EconomicEvent]:
    """Return events happening within the next window_hours (for startup display)."""
    now = datetime.now(timezone.utc)
    upcoming = []
    for event in ECONOMIC_CALENDAR:
        try:
            event_dt = datetime.fromisoformat(event.date_utc)
        except Exception:
            continue
        delta = (event_dt - now).total_seconds() / 3600
        if 0 <= delta <= window_hours:
            upcoming.append(event)
    return sorted(upcoming, key=lambda e: e.date_utc)
