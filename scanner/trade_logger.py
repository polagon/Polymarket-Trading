"""
Astra V2 — SQLite Trade Logger

Persistent database for market snapshots, estimates, and trade history.
Enables backtesting, drawdown tracking, and performance analysis.

Tables:
  - market_snapshots: Price/liquidity snapshots every scan
  - estimates: Astra probability estimates with confidence scores
  - trades: Paper/live trade entries and exits with P&L tracking

Usage:
    from scanner.trade_logger import init_db, log_market_snapshot, log_estimate

    init_db()  # Call once at startup
    log_market_snapshot(market)
    log_estimate(estimate, market.condition_id)
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from scanner.market_fetcher import Market
from scanner.probability_estimator import Estimate


DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "astra_trades.db"


def init_db():
    """
    Initialize SQLite database with tables for market snapshots, estimates, and trades.
    Safe to call multiple times (CREATE TABLE IF NOT EXISTS).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for concurrent reads

    # Market snapshots: price and liquidity over time
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            condition_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            yes_price REAL NOT NULL,
            liquidity REAL,
            volume_24h REAL,
            category TEXT,
            question TEXT,
            PRIMARY KEY (condition_id, timestamp)
        )
    """)

    # Probability estimates: Astra's predictions over time
    conn.execute("""
        CREATE TABLE IF NOT EXISTS estimates (
            condition_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            probability REAL NOT NULL,
            probability_low REAL,
            probability_high REAL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            robustness_score INTEGER,
            edge REAL,
            ev_after_costs REAL,
            kelly_position_pct REAL,
            truth_state TEXT,
            no_trade BOOLEAN,
            no_trade_reason TEXT,
            PRIMARY KEY (condition_id, timestamp)
        )
    """)

    # Trade execution log: paper and live trades
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            condition_id TEXT NOT NULL,
            question TEXT,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            size_usd REAL NOT NULL,
            profit_loss REAL,
            profit_loss_pct REAL,
            holding_hours REAL,
            category TEXT,
            source TEXT,
            resolution_outcome BOOLEAN,
            brier_score REAL
        )
    """)

    # Index for fast time-series queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_estimates_time ON estimates(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_time)")

    conn.commit()
    conn.close()


def log_market_snapshot(market: Market):
    """Store current market price and liquidity snapshot."""
    conn = sqlite3.connect(DB_PATH)
    timestamp = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT OR REPLACE INTO market_snapshots
        (condition_id, timestamp, yes_price, liquidity, volume_24h, category, question)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        market.condition_id,
        timestamp,
        market.yes_price,
        market.liquidity,
        getattr(market, 'volume_24h', None),  # Optional field
        market.category,
        market.question,
    ))

    conn.commit()
    conn.close()


def log_estimate(estimate: Estimate, condition_id: str, market: Optional[Market] = None):
    """Store Astra probability estimate."""
    conn = sqlite3.connect(DB_PATH)
    timestamp = datetime.now(timezone.utc).isoformat()

    # Calculate edge if market provided
    edge = None
    if market:
        edge = estimate.probability - market.yes_price

    conn.execute("""
        INSERT OR REPLACE INTO estimates
        (condition_id, timestamp, probability, probability_low, probability_high,
         confidence, source, robustness_score, edge, ev_after_costs, kelly_position_pct,
         truth_state, no_trade, no_trade_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        condition_id,
        timestamp,
        estimate.probability,
        estimate.probability_low,
        estimate.probability_high,
        estimate.confidence,
        estimate.source,
        estimate.robustness_score,
        edge,
        estimate.ev_after_costs,
        estimate.kelly_position_pct,
        estimate.truth_state,
        estimate.no_trade,
        estimate.no_trade_reason if estimate.no_trade else None,
    ))

    conn.commit()
    conn.close()


def log_trade_entry(
    trade_id: str,
    condition_id: str,
    question: str,
    direction: str,
    entry_price: float,
    size_usd: float,
    category: str,
    source: str,
):
    """Record trade entry (paper or live)."""
    conn = sqlite3.connect(DB_PATH)
    timestamp = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO trades
        (trade_id, condition_id, question, entry_time, direction, entry_price,
         size_usd, category, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_id,
        condition_id,
        question,
        timestamp,
        direction,
        entry_price,
        size_usd,
        category,
        source,
    ))

    conn.commit()
    conn.close()


def log_trade_exit(
    trade_id: str,
    exit_price: float,
    profit_loss: float,
    resolution_outcome: Optional[bool] = None,
    brier_score: Optional[float] = None,
):
    """Update trade with exit price and P&L."""
    conn = sqlite3.connect(DB_PATH)
    timestamp = datetime.now(timezone.utc).isoformat()

    # Calculate holding period
    cursor = conn.execute("SELECT entry_time, entry_price FROM trades WHERE trade_id = ?", (trade_id,))
    row = cursor.fetchone()
    if row:
        entry_time_str, entry_price = row
        entry_time = datetime.fromisoformat(entry_time_str)
        holding_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
        profit_loss_pct = profit_loss / entry_price if entry_price > 0 else 0.0

        conn.execute("""
            UPDATE trades
            SET exit_time = ?, exit_price = ?, profit_loss = ?, profit_loss_pct = ?,
                holding_hours = ?, resolution_outcome = ?, brier_score = ?
            WHERE trade_id = ?
        """, (
            timestamp,
            exit_price,
            profit_loss,
            profit_loss_pct,
            holding_hours,
            resolution_outcome,
            brier_score,
            trade_id,
        ))

    conn.commit()
    conn.close()


def get_current_drawdown() -> float:
    """
    Calculate current drawdown from peak equity.
    Returns: (current_equity - peak_equity) / peak_equity (negative value)
    """
    conn = sqlite3.connect(DB_PATH)

    # Get all resolved trades ordered by exit time
    cursor = conn.execute("""
        SELECT profit_loss
        FROM trades
        WHERE exit_time IS NOT NULL
        ORDER BY exit_time ASC
    """)

    pnl_series = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not pnl_series:
        return 0.0

    # Calculate cumulative equity and track max drawdown
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0

    for pnl in pnl_series:
        equity += pnl
        if equity > peak_equity:
            peak_equity = equity

        if peak_equity > 0:
            drawdown = (equity - peak_equity) / peak_equity
            max_drawdown = min(max_drawdown, drawdown)

    return max_drawdown


def get_trade_stats() -> dict:
    """Get aggregate trade statistics from database."""
    conn = sqlite3.connect(DB_PATH)

    # Total trades and win rate
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl,
            AVG(holding_hours) as avg_holding_hours
        FROM trades
        WHERE exit_time IS NOT NULL
    """)

    row = cursor.fetchone()
    total, wins, total_pnl, avg_pnl, avg_holding_hours = row if row else (0, 0, 0.0, 0.0, 0.0)

    win_rate = (wins / total) if total > 0 else 0.0

    # Category breakdown
    cursor = conn.execute("""
        SELECT category, COUNT(*), SUM(profit_loss), AVG(profit_loss)
        FROM trades
        WHERE exit_time IS NOT NULL
        GROUP BY category
    """)

    category_stats = {row[0]: {"count": row[1], "total_pnl": row[2], "avg_pnl": row[3]}
                      for row in cursor.fetchall()}

    conn.close()

    return {
        "total_trades": total or 0,
        "win_rate": win_rate,
        "total_pnl": total_pnl or 0.0,
        "avg_pnl_per_trade": avg_pnl or 0.0,
        "avg_holding_hours": avg_holding_hours or 0.0,
        "max_drawdown": get_current_drawdown(),
        "by_category": category_stats,
    }
