"""
Astra V2 — SQLite Trade Logger

Persistent database for market snapshots, estimates, and trade history.
Enables backtesting, drawdown tracking, and performance analysis.

Tables:
  - market_snapshots: Price/liquidity snapshots every scan
  - estimates: Astra probability estimates with confidence scores
  - trades: Paper/live trade entries and exits with P&L tracking
  - config_snapshots: Configuration state at each run
  - run_manifests: Run-level metadata (git sha, versions, etc.)

Usage:
    from scanner.trade_logger import init_db, log_market_snapshot, log_estimate

    init_db()  # Call once at startup
    log_market_snapshot(market)
    log_estimate(estimate, market.condition_id)
"""

import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scanner.market_fetcher import Market
from scanner.probability_estimator import Estimate

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "astra_trades.db"

CURRENT_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    """Check whether *column_name* already exists on *table_name* (idempotent DDL)."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1] for row in cursor.fetchall()}
    return column_name in columns


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """
    Idempotent migration from schema v0/v1 → v2.

    Adds provenance columns to estimates, trades, market_snapshots and
    creates config_snapshots + run_manifests tables.
    """
    log.info("Running migration v1 → v2 …")

    # -- estimates provenance columns --
    for col in (
        "run_id TEXT",
        "config_snapshot_id TEXT",
        "config_hash TEXT",
        "prompt_bundle_hash TEXT",
        "estimator_version TEXT",
    ):
        col_name = col.split()[0]
        if not _column_exists(conn, "estimates", col_name):
            conn.execute(f"ALTER TABLE estimates ADD COLUMN {col}")

    # -- trades provenance columns --
    for col in ("run_id TEXT", "config_snapshot_id TEXT", "config_hash TEXT"):
        col_name = col.split()[0]
        if not _column_exists(conn, "trades", col_name):
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col}")

    # -- market_snapshots provenance columns --
    if not _column_exists(conn, "market_snapshots", "run_id"):
        conn.execute("ALTER TABLE market_snapshots ADD COLUMN run_id TEXT")

    # -- config_snapshots table --
    conn.execute("""
        CREATE TABLE IF NOT EXISTS config_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT,
            timestamp TEXT,
            config_hash TEXT,
            config_json TEXT
        )
    """)

    # -- run_manifests table --
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_manifests (
            run_id TEXT PRIMARY KEY,
            git_sha TEXT,
            started_at TEXT,
            config_hash TEXT,
            prompt_bundle_hash TEXT,
            python_version TEXT,
            manifest_json TEXT
        )
    """)

    conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
    conn.commit()
    log.info("Migration v1 → v2 complete.")


# ---------------------------------------------------------------------------
# Backup / restore
# ---------------------------------------------------------------------------


def backup_db(db_path: Optional[Path] = None) -> None:
    """
    Rotate backups: bak1 → bak2 → bak3, then copy current DB → bak1.

    Uses shutil.copy2 to preserve metadata.
    """
    db = Path(db_path) if db_path else DB_PATH
    if not db.exists():
        return

    bak1 = db.with_suffix(".db.bak1")
    bak2 = db.with_suffix(".db.bak2")
    bak3 = db.with_suffix(".db.bak3")

    # Rotate: bak2 → bak3, bak1 → bak2
    if bak2.exists():
        shutil.copy2(bak2, bak3)
    if bak1.exists():
        shutil.copy2(bak1, bak2)

    shutil.copy2(db, bak1)
    log.info("Backup rotated: %s → bak1", db.name)


def restore_from_backup(db_path: Optional[Path] = None) -> bool:
    """
    Try to restore from bak1, bak2, bak3 in order.

    Each candidate is validated with ``PRAGMA integrity_check`` before use.
    Returns True if a valid backup was restored, False otherwise.
    """
    db = Path(db_path) if db_path else DB_PATH
    for suffix in (".db.bak1", ".db.bak2", ".db.bak3"):
        bak = db.with_suffix(suffix)
        if not bak.exists():
            continue
        try:
            test_conn = sqlite3.connect(bak)
            result = test_conn.execute("PRAGMA integrity_check").fetchone()
            test_conn.close()
            if result and result[0] == "ok":
                shutil.copy2(bak, db)
                log.info("Restored DB from %s", bak.name)
                return True
            else:
                log.warning("Backup %s failed integrity check, skipping.", bak.name)
        except Exception as exc:
            log.warning("Backup %s unusable: %s", bak.name, exc)
    return False


# ---------------------------------------------------------------------------
# init_db — full rewrite with WAL, FK, integrity, migration
# ---------------------------------------------------------------------------


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Initialize SQLite database with tables for market snapshots, estimates,
    trades, config_snapshots, and run_manifests.

    * Sets WAL journal mode, foreign keys, and busy timeout.
    * Runs integrity check (attempts restore from backup on failure).
    * Applies schema migrations when needed.
    * For a fresh DB (user_version == 0) creates all tables with provenance
      columns included from the start.

    Returns the open connection so callers can reuse it if desired.
    """
    db = Path(db_path) if db_path else DB_PATH
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db)

    # -- PRAGMAs --
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    # -- Integrity check --
    integrity = conn.execute("PRAGMA integrity_check").fetchone()
    if not integrity or integrity[0] != "ok":
        log.error("DB integrity check failed: %s — attempting restore …", integrity)
        conn.close()
        restored = restore_from_backup(db)
        if not restored:
            raise RuntimeError(f"Database {db} is corrupt and no valid backup found.")
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")

    # -- Schema version check --
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]

    if user_version > CURRENT_SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(
            f"DB schema (v{user_version}) is newer than code (v{CURRENT_SCHEMA_VERSION}). Update code first."
        )

    if user_version == 0:
        # Fresh database — create all tables with full schema including
        # provenance columns so no migration is needed later.
        _create_all_tables_fresh(conn)
    elif user_version < CURRENT_SCHEMA_VERSION:
        # Existing database that needs migration
        backup_db(db)
        _migrate_v1_to_v2(conn)

    return conn


def _create_all_tables_fresh(conn: sqlite3.Connection) -> None:
    """Create every table from scratch with the full v2 schema."""

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
            run_id TEXT,
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
            run_id TEXT,
            config_snapshot_id TEXT,
            config_hash TEXT,
            prompt_bundle_hash TEXT,
            estimator_version TEXT,
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
            brier_score REAL,
            run_id TEXT,
            config_snapshot_id TEXT,
            config_hash TEXT
        )
    """)

    # Configuration snapshots
    conn.execute("""
        CREATE TABLE IF NOT EXISTS config_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT,
            timestamp TEXT,
            config_hash TEXT,
            config_json TEXT
        )
    """)

    # Run manifests
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_manifests (
            run_id TEXT PRIMARY KEY,
            git_sha TEXT,
            started_at TEXT,
            config_hash TEXT,
            prompt_bundle_hash TEXT,
            python_version TEXT,
            manifest_json TEXT
        )
    """)

    # Indexes for fast time-series queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_estimates_time ON estimates(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_time)")

    conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
    conn.commit()
    log.info("Fresh DB created at schema v%d.", CURRENT_SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# Logging helpers (unchanged API)
# ---------------------------------------------------------------------------


def log_market_snapshot(market: Market):
    """Store current market price and liquidity snapshot."""
    conn = sqlite3.connect(DB_PATH)
    timestamp = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT OR REPLACE INTO market_snapshots
        (condition_id, timestamp, yes_price, liquidity, volume_24h, category, question)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            market.condition_id,
            timestamp,
            market.yes_price,
            market.liquidity,
            getattr(market, "volume_24h", None),  # Optional field
            market.category,
            market.question,
        ),
    )

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

    conn.execute(
        """
        INSERT OR REPLACE INTO estimates
        (condition_id, timestamp, probability, probability_low, probability_high,
         confidence, source, robustness_score, edge, ev_after_costs, kelly_position_pct,
         truth_state, no_trade, no_trade_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
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
        ),
    )

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

    conn.execute(
        """
        INSERT INTO trades
        (trade_id, condition_id, question, entry_time, direction, entry_price,
         size_usd, category, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            trade_id,
            condition_id,
            question,
            timestamp,
            direction,
            entry_price,
            size_usd,
            category,
            source,
        ),
    )

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

        conn.execute(
            """
            UPDATE trades
            SET exit_time = ?, exit_price = ?, profit_loss = ?, profit_loss_pct = ?,
                holding_hours = ?, resolution_outcome = ?, brier_score = ?
            WHERE trade_id = ?
        """,
            (
                timestamp,
                exit_price,
                profit_loss,
                profit_loss_pct,
                holding_hours,
                resolution_outcome,
                brier_score,
                trade_id,
            ),
        )

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

    category_stats = {row[0]: {"count": row[1], "total_pnl": row[2], "avg_pnl": row[3]} for row in cursor.fetchall()}

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
