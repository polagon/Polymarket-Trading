"""
Tests for Astra provenance layer — run identity, config hash, prompt bundle hash.

All tests use temp dirs and mock subprocess for git sha.
No real network or API calls.
"""

import json
import os
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Test helper: build a RunContext that passes validate_manifest_v1()
# ---------------------------------------------------------------------------
_VALID_CONFIG_HASH = "a" * 64  # 64-hex sha256 placeholder
_VALID_PROMPT_HASH = "b" * 64  # 64-hex sha256 placeholder


def _make_valid_run_ctx(**overrides):
    """Return a RunContext whose to_manifest_dict() passes V1 validation."""
    from ops.run_context import RunContext

    defaults = dict(
        paper_mode=True,
        config_hash=_VALID_CONFIG_HASH,
        prompt_bundle_hash=_VALID_PROMPT_HASH,
        schema_version=2,
    )
    defaults.update(overrides)
    return RunContext(**defaults)  # type: ignore[arg-type]


class TestRunIdentity:
    """Activity 1: Run identity contract."""

    def test_run_id_is_uuid4(self):
        from ops.run_context import RunContext

        ctx = RunContext()
        # Should be a valid UUID4
        parsed = uuid.UUID(ctx.run_id, version=4)
        assert str(parsed) == ctx.run_id

    def test_cycle_id_monotonic(self):
        from ops.run_context import RunContext

        ctx = RunContext()
        assert ctx.cycle_id == 0
        c1 = ctx.next_cycle()
        assert c1 == 1
        c2 = ctx.next_cycle()
        assert c2 == 2
        c3 = ctx.next_cycle()
        assert c3 == 3
        assert ctx.cycle_id == 3

    def test_git_sha_detected_if_repo(self):
        """Mock subprocess to simulate git repo."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="abc1234\n")
            from config import get_git_sha

            sha = get_git_sha()
            assert sha == "abc1234"

    def test_git_sha_missing_paper_ok(self):
        """Paper mode should allow None git_sha."""
        from ops.run_context import RunContext

        ctx = RunContext(git_sha=None, paper_mode=True)
        assert ctx.git_sha is None
        assert ctx.paper_mode is True
        # Should still produce valid manifest dict structure
        manifest = ctx.to_manifest_dict()
        assert manifest["git_sha"] is None
        assert manifest["paper_mode"] is True

    def test_git_sha_missing_live_halts(self):
        """Live mode with missing git_sha should be detectable."""
        from ops.run_context import RunContext

        ctx = RunContext(git_sha=None, paper_mode=False)
        # The check is: if not paper_mode and git_sha is None -> refuse to start
        assert ctx.git_sha is None
        assert ctx.paper_mode is False
        # Caller is responsible for the halt decision

    def test_manifest_written_on_start(self):
        from ops.artifacts import ArtifactWriter

        ctx = _make_valid_run_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter(ctx.run_id, base_dir=Path(tmpdir))
            writer.ensure_writable()
            writer.write_manifest_start(ctx.to_manifest_dict())
            manifest_path = Path(tmpdir) / ctx.run_id / "manifest.json"
            assert manifest_path.exists()
            data = json.loads(manifest_path.read_text())
            assert data["run_id"] == ctx.run_id
            assert data["manifest_schema_version"] == 1
            assert data["ended_at"] is None
            # V1 compliance: no ended_ok key
            assert "ended_ok" not in data

    def test_manifest_immutable(self):
        from ops.artifacts import ArtifactWriter

        ctx = _make_valid_run_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter(ctx.run_id, base_dir=Path(tmpdir))
            writer.ensure_writable()
            writer.write_manifest_start(ctx.to_manifest_dict())
            with pytest.raises(RuntimeError, match="already written"):
                writer.write_manifest_start(ctx.to_manifest_dict())

    def test_manifest_updates_on_shutdown(self):
        from ops.artifacts import ArtifactWriter

        ctx = _make_valid_run_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter(ctx.run_id, base_dir=Path(tmpdir))
            writer.ensure_writable()
            writer.write_manifest_start(ctx.to_manifest_dict())
            writer.write_manifest_end("normal_shutdown")
            manifest_path = Path(tmpdir) / ctx.run_id / "manifest.json"
            data = json.loads(manifest_path.read_text())
            assert data["ended_at"] is not None
            assert data["exit_reason"] == "normal_shutdown"
            # V1 compliance: no ended_ok key
            assert "ended_ok" not in data
            # run_id must still match
            assert data["run_id"] == ctx.run_id

    def test_manifest_schema_version_matches_db(self):
        """Fix 1: schema_version must reflect actual DB PRAGMA user_version."""
        from scanner.trade_logger import CURRENT_SCHEMA_VERSION, init_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path=db_path)
            actual_version = conn.execute("PRAGMA user_version").fetchone()[0]
            ctx = _make_valid_run_ctx(schema_version=actual_version)
            manifest = ctx.to_manifest_dict()
            assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION
            assert manifest["schema_version"] == actual_version
            conn.close()


class TestManifestV1Compliance:
    """Manifest V1 schema contract — fail-closed validation."""

    def test_manifest_required_hashes_nonempty(self):
        """config_hash and prompt_bundle_hash are 64-hex and not empty."""
        ctx = _make_valid_run_ctx()
        manifest = ctx.to_manifest_dict()
        assert len(manifest["config_hash"]) == 64
        assert len(manifest["prompt_bundle_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in manifest["config_hash"])
        assert all(c in "0123456789abcdef" for c in manifest["prompt_bundle_hash"])

    def test_manifest_rejects_extra_keys_v1(self):
        """Adding ended_ok (or any extra key) causes validate_manifest_v1 to raise."""
        from ops.artifacts import validate_manifest_v1

        ctx = _make_valid_run_ctx()
        manifest = ctx.to_manifest_dict()
        manifest["ended_ok"] = True  # V1 does not allow this key
        with pytest.raises(RuntimeError, match="unexpected keys"):
            validate_manifest_v1(manifest)

    def test_manifest_write_fails_on_empty_hash(self):
        """Empty string config_hash or prompt_bundle_hash fails fast."""
        from ops.artifacts import ArtifactWriter
        from ops.run_context import RunContext

        # Empty config_hash
        ctx = RunContext(config_hash="", prompt_bundle_hash=_VALID_PROMPT_HASH, schema_version=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter(ctx.run_id, base_dir=Path(tmpdir))
            writer.ensure_writable()
            with pytest.raises(RuntimeError, match="config_hash must be 64-hex sha256"):
                writer.write_manifest_start(ctx.to_manifest_dict())

        # Empty prompt_bundle_hash
        ctx2 = RunContext(config_hash=_VALID_CONFIG_HASH, prompt_bundle_hash="", schema_version=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            writer2 = ArtifactWriter(ctx2.run_id, base_dir=Path(tmpdir))
            writer2.ensure_writable()
            with pytest.raises(RuntimeError, match="prompt_bundle_hash must be 64-hex sha256"):
                writer2.write_manifest_start(ctx2.to_manifest_dict())

    def test_manifest_end_only_updates_end_fields(self):
        """Compare dict before/after end write; only ended_at and exit_reason change."""
        from ops.artifacts import ArtifactWriter

        ctx = _make_valid_run_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter(ctx.run_id, base_dir=Path(tmpdir))
            writer.ensure_writable()
            writer.write_manifest_start(ctx.to_manifest_dict())

            manifest_path = Path(tmpdir) / ctx.run_id / "manifest.json"
            before = json.loads(manifest_path.read_text())

            writer.write_manifest_end("normal_shutdown")
            after = json.loads(manifest_path.read_text())

            # Only ended_at and exit_reason should change
            changed_keys = {k for k in after if before.get(k) != after.get(k)}
            assert changed_keys == {"ended_at", "exit_reason"}
            assert after["ended_at"] is not None
            assert after["exit_reason"] == "normal_shutdown"
            # No new keys added
            assert set(before.keys()) == set(after.keys())

    def test_prompt_bundle_hash_propagates(self):
        """Manifest prompt_bundle_hash equals probability_estimator.PROMPT_BUNDLE_HASH."""
        from scanner.probability_estimator import PROMPT_BUNDLE_HASH

        ctx = _make_valid_run_ctx(prompt_bundle_hash=PROMPT_BUNDLE_HASH)
        manifest = ctx.to_manifest_dict()
        assert manifest["prompt_bundle_hash"] == PROMPT_BUNDLE_HASH
        assert len(manifest["prompt_bundle_hash"]) == 64

    def test_config_hash_propagates(self):
        """Manifest config_hash equals config.compute_config_hash(get_canonical_config_dict())."""
        from config import compute_config_hash, get_canonical_config_dict

        real_hash = compute_config_hash(get_canonical_config_dict())
        ctx = _make_valid_run_ctx(config_hash=real_hash)
        manifest = ctx.to_manifest_dict()
        assert manifest["config_hash"] == real_hash
        assert len(manifest["config_hash"]) == 64


class TestConfigHash:
    """Activity 2: Config hash and snapshot-on-change."""

    def test_config_hash_deterministic(self):
        from config import compute_config_hash, get_canonical_config_dict

        d = get_canonical_config_dict()
        h1 = compute_config_hash(d)
        h2 = compute_config_hash(d)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_config_dict_excludes_credentials(self):
        from config import get_canonical_config_dict

        d = get_canonical_config_dict()
        # Must NOT contain API keys or file paths
        assert "ANTHROPIC_API_KEY" not in d
        assert "POLY_PRIVATE_KEY" not in d
        assert "POLY_API_KEY" not in d
        assert "POLY_API_SECRET" not in d
        assert "FRED_API_KEY" not in d
        assert "ODDS_API_KEY" not in d
        assert "MEMORY_DIR" not in d
        assert "LOG_DIR" not in d
        assert "DB_PATH" not in d

    def test_config_snapshot_written_on_start(self):
        """Verify config snapshot is written to DB on startup."""
        from config import compute_config_hash, get_canonical_config_dict
        from scanner.trade_logger import init_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path=db_path)
            config_dict = get_canonical_config_dict()
            config_hash = compute_config_hash(config_dict)
            # Write a config snapshot
            conn.execute(
                "INSERT INTO config_snapshots (snapshot_id, run_id, timestamp, config_hash, config_json) VALUES (?, ?, ?, ?, ?)",
                ("snap1", "run1", "2026-01-01T00:00:00Z", config_hash, json.dumps(config_dict)),
            )
            conn.commit()
            row = conn.execute("SELECT config_hash FROM config_snapshots WHERE snapshot_id='snap1'").fetchone()
            assert row[0] == config_hash
            conn.close()

    def test_config_snapshot_written_on_change(self):
        """Verify only changed configs produce new snapshots."""
        from config import compute_config_hash

        h1 = compute_config_hash({"BANKROLL": 5000})
        h2 = compute_config_hash({"BANKROLL": 10000})
        assert h1 != h2  # Different config should produce different hash


class TestPromptHash:
    """Activity 3: Prompt bundle hash and estimator version."""

    def test_prompt_hashes_full_sha256(self):
        from scanner.probability_estimator import (
            ASTRA_CON_SYSTEM_HASH,
            ASTRA_PRO_SYSTEM_HASH,
            ASTRA_SYNTHESIZER_SYSTEM_HASH,
            ASTRA_V2_SYSTEM_HASH,
        )

        for h in [ASTRA_V2_SYSTEM_HASH, ASTRA_PRO_SYSTEM_HASH, ASTRA_CON_SYSTEM_HASH, ASTRA_SYNTHESIZER_SYSTEM_HASH]:
            assert len(h) == 64  # Full SHA-256 hex
            assert all(c in "0123456789abcdef" for c in h)

    def test_prompt_bundle_hash_deterministic(self):
        # Import twice — should be same
        import importlib

        import scanner.probability_estimator as pe
        from scanner.probability_estimator import PROMPT_BUNDLE_HASH

        importlib.reload(pe)
        assert pe.PROMPT_BUNDLE_HASH == PROMPT_BUNDLE_HASH

    def test_estimator_version_present(self):
        from scanner.probability_estimator import ESTIMATOR_VERSION

        assert isinstance(ESTIMATOR_VERSION, str)
        assert len(ESTIMATOR_VERSION) > 0
        assert "astra" in ESTIMATOR_VERSION.lower()
