"""
Tests for Astra trade logger — schema versioning, migrations, backup/restore.
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.trade_logger import (
    CURRENT_SCHEMA_VERSION,
    _column_exists,
    backup_db,
    init_db,
    restore_from_backup,
)


class TestSchemaVersioning:
    """Activity 4: DB schema versioning and migrations."""

    def test_fresh_db_gets_current_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path=db_path)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == CURRENT_SCHEMA_VERSION
            conn.close()

    def test_v1_db_migrated_to_v2(self):
        """Create a v1-style DB (no provenance columns), run init_db, verify migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            # Create v1 schema manually
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE estimates (
                condition_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                probability REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY (condition_id, timestamp)
            )""")
            conn.execute("""CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                condition_id TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size_usd REAL NOT NULL
            )""")
            conn.execute("""CREATE TABLE market_snapshots (
                condition_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                yes_price REAL NOT NULL,
                PRIMARY KEY (condition_id, timestamp)
            )""")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
            conn.close()

            # Now init_db should migrate to v2
            conn = init_db(db_path=db_path)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == CURRENT_SCHEMA_VERSION

            # Check new columns exist
            assert _column_exists(conn, "estimates", "run_id")
            assert _column_exists(conn, "estimates", "config_hash")
            assert _column_exists(conn, "estimates", "prompt_bundle_hash")
            assert _column_exists(conn, "trades", "run_id")
            assert _column_exists(conn, "trades", "config_hash")
            assert _column_exists(conn, "market_snapshots", "run_id")

            # Check new tables exist
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            assert "config_snapshots" in tables
            assert "run_manifests" in tables
            conn.close()

    def test_migration_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn1 = init_db(db_path=db_path)
            conn1.close()
            # Run again — should not error
            conn2 = init_db(db_path=db_path)
            version = conn2.execute("PRAGMA user_version").fetchone()[0]
            assert version == CURRENT_SCHEMA_VERSION
            conn2.close()

    def test_future_version_halts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA user_version = 99")
            conn.commit()
            conn.close()
            with pytest.raises(RuntimeError, match="newer than code"):
                init_db(db_path=db_path)

    def test_integrity_check_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path=db_path)
            # Should pass without error
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            assert result == "ok"
            conn.close()

    def test_integrity_check_fails_halts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            # Create a corrupt DB
            db_path.write_bytes(b"this is not a valid sqlite database at all" * 10)
            # init_db should fail (no backup to restore from, but it's a corrupt file)
            with pytest.raises((RuntimeError, sqlite3.DatabaseError)):
                init_db(db_path=db_path)


class TestBackupRestore:
    """Activity 5: DB backup rotation and recovery."""

    def test_backup_creates_bak1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path=db_path)
            conn.close()
            backup_db(db_path=db_path)
            assert (Path(tmpdir) / "test.db.bak1").exists()

    def test_backup_rotates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path=db_path)
            conn.execute("INSERT INTO market_snapshots (condition_id, timestamp, yes_price) VALUES ('c1', 't1', 0.5)")
            conn.commit()
            conn.close()

            # First backup
            backup_db(db_path=db_path)
            assert (Path(tmpdir) / "test.db.bak1").exists()

            # Second backup — bak1 should rotate to bak2
            backup_db(db_path=db_path)
            assert (Path(tmpdir) / "test.db.bak1").exists()
            assert (Path(tmpdir) / "test.db.bak2").exists()

    def test_restore_from_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path=db_path)
            conn.execute("INSERT INTO market_snapshots (condition_id, timestamp, yes_price) VALUES ('c1', 't1', 0.5)")
            conn.commit()
            conn.close()

            # Create backup
            backup_db(db_path=db_path)

            # Corrupt the primary
            db_path.write_bytes(b"corrupted" * 100)

            # Restore should succeed
            result = restore_from_backup(db_path=db_path)
            assert result is True

            # Verify data survived
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT yes_price FROM market_snapshots WHERE condition_id='c1'").fetchone()
            assert row is not None
            assert row[0] == 0.5
            conn.close()
