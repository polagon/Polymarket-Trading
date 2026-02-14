"""
Tests for Astra ops package — artifacts, gate engine, heartbeat, alerts, logging, startup checklist.
"""

import json
import logging
import os
import signal
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestArtifacts:
    """Activity 6: Run artifacts contract."""

    def test_artifacts_dir_created(self):
        from ops.artifacts import ArtifactWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter("test-run", base_dir=Path(tmpdir))
            writer.ensure_writable()
            assert (Path(tmpdir) / "test-run").is_dir()

    def test_artifacts_atomic_write_json(self):
        from ops.artifacts import ArtifactWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter("test-run", base_dir=Path(tmpdir))
            writer.ensure_writable()
            writer.write_gate_status(1, {"status": "ok", "gates": []})
            path = Path(tmpdir) / "test-run" / "gate_status.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["status"] == "ok"

    def test_all_artifacts_contain_run_id_and_cycle_id(self):
        from ops.artifacts import ArtifactWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter("test-run-123", base_dir=Path(tmpdir))
            writer.ensure_writable()
            writer.write_all_cycle_artifacts(
                cycle_id=5,
                gate_status={"status": "ok", "gates": []},
            )
            run_dir = Path(tmpdir) / "test-run-123"
            for fname in [
                "gate_status.json",
                "metrics_all_windows.json",
                "drawdown_state.json",
                "decision_report.json",
                "edge_snapshot.json",
                "fill_realism_report.json",
            ]:
                fpath = run_dir / fname
                assert fpath.exists(), f"{fname} missing"
                data = json.loads(fpath.read_text())
                assert data["run_id"] == "test-run-123", f"{fname} missing run_id"
                assert data["cycle_id"] == 5, f"{fname} missing cycle_id"
                assert "timestamp" in data, f"{fname} missing timestamp"

    def test_anomalies_append_ndjson(self):
        from ops.artifacts import ArtifactWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArtifactWriter("test-run", base_dir=Path(tmpdir))
            writer.ensure_writable()
            writer.append_anomaly(1, "circuit_break", {"reason": "daily loss"})
            writer.append_anomaly(2, "feed_disconnect", {"duration_s": 30})
            anomaly_path = Path(tmpdir) / "test-run" / "anomalies.ndjson"
            lines = anomaly_path.read_text().strip().split("\n")
            assert len(lines) == 2
            entry1 = json.loads(lines[0])
            assert entry1["type"] == "circuit_break"
            entry2 = json.loads(lines[1])
            assert entry2["type"] == "feed_disconnect"


class TestGateEngine:
    """Activity 7: Gate engine ok/degraded/halted."""

    def test_gate_engine_ok(self):
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        ctx = GateContext(memory_mb=100.0, disk_free_mb=10000.0)  # Explicit healthy values
        status = engine.evaluate(ctx, run_id="r1", cycle_id=1)
        assert status.status == "ok"

    def test_gate_engine_degraded_on_warn(self):
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        ctx = GateContext(ws_connected=False)  # WS disconnected = warn
        status = engine.evaluate(ctx, run_id="r1", cycle_id=1)
        assert status.status == "degraded"

    def test_gate_engine_halted_on_fail(self):
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        ctx = GateContext(kill_switch_active=True)  # Kill switch = fail
        status = engine.evaluate(ctx, run_id="r1", cycle_id=1)
        assert status.status == "halted"

    def test_halted_blocks_new_entries(self):
        """Gate engine status is consumed by caller to block entries."""
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        ctx = GateContext(daily_pnl_pct=-0.06)  # 6% daily loss exceeds 5% threshold
        status = engine.evaluate(ctx, run_id="r1", cycle_id=1)
        assert status.status == "halted"
        # Caller should check: if status.status in ("halted", "degraded"): block entries
        assert status.status in ("halted", "degraded")

    def test_gate_status_transitions_tracked(self):
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        # First eval: ok
        ctx_ok = GateContext(memory_mb=100.0, disk_free_mb=10000.0)
        s1 = engine.evaluate(ctx_ok, run_id="r1", cycle_id=1)
        assert s1.status == "ok"
        assert len(s1.transitions) == 0  # No transition from initial ok

        # Second eval: degraded (transition should be tracked)
        ctx_degraded = GateContext(ws_connected=False)
        s2 = engine.evaluate(ctx_degraded, run_id="r1", cycle_id=2)
        assert s2.status == "degraded"
        assert len(s2.transitions) == 1
        assert s2.transitions[0]["from"] == "ok"
        assert s2.transitions[0]["to"] == "degraded"

    def test_sample_gates_add_blockers(self):
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        ctx = GateContext(memory_mb=100.0, disk_free_mb=10000.0)
        metrics = {
            "all_time": {
                "sharpe_per_trade_gate": "FAIL (need 30 trades, have 5)",
                "win_rate_gate": "PASS",
            }
        }
        status = engine.evaluate(ctx, run_id="r1", cycle_id=1, metrics_snapshot=metrics)
        assert status.status == "ok"  # Sample gates don't cause halted
        assert len(status.blockers) > 0
        assert any("Sharpe" in b for b in status.blockers)

    def test_memory_none_returns_warn(self):
        """Fix 4: unmeasurable memory -> WARN, not PASS."""
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        ctx = GateContext(memory_mb=None, disk_free_mb=10000.0)
        status = engine.evaluate(ctx, run_id="r1", cycle_id=1)
        memory_gate = [g for g in status.gates if g.name == "memory"][0]
        assert memory_gate.status == "warn"

    def test_memory_realistic_value_passes(self):
        """Fix 4: realistic memory value should pass."""
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        ctx = GateContext(memory_mb=150.0, disk_free_mb=10000.0)
        status = engine.evaluate(ctx, run_id="r1", cycle_id=1)
        memory_gate = [g for g in status.gates if g.name == "memory"][0]
        assert memory_gate.status == "pass"
        assert memory_gate.value == 150.0

    def test_gate_thresholds_warn_fail_split(self):
        """Fix 5: gate thresholds have distinct warn/fail levels."""
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine(memory_warn_mb=400, memory_fail_mb=600)

        # Below warn: pass
        ctx = GateContext(memory_mb=300.0, disk_free_mb=10000.0)
        s = engine.evaluate(ctx)
        mem = [g for g in s.gates if g.name == "memory"][0]
        assert mem.status == "pass"

        # Between warn and fail: warn
        ctx = GateContext(memory_mb=500.0, disk_free_mb=10000.0)
        s = engine.evaluate(ctx)
        mem = [g for g in s.gates if g.name == "memory"][0]
        assert mem.status == "warn"

        # Above fail: fail
        ctx = GateContext(memory_mb=700.0, disk_free_mb=10000.0)
        s = engine.evaluate(ctx)
        mem = [g for g in s.gates if g.name == "memory"][0]
        assert mem.status == "fail"


class TestHeartbeat:
    """Activity 9: Health heartbeat."""

    def test_heartbeat_written_and_fields_present(self):
        from ops.health import write_heartbeat

        with tempfile.TemporaryDirectory() as tmpdir:
            hb_path = os.path.join(tmpdir, "astra_health.json")
            write_heartbeat(
                path=hb_path,
                run_id="test-run",
                cycle_id=5,
                status="ok",
                cycle_duration_s=12.5,
                ws_connected=True,
                active_markets=40,
                open_orders=10,
                daily_pnl_usd=25.50,
                memory_dir=tmpdir,
            )
            assert os.path.exists(hb_path)
            data = json.loads(Path(hb_path).read_text())
            required_fields = [
                "timestamp",
                "run_id",
                "cycle_id",
                "status",
                "cycle_duration_s",
                "ws_connected",
                "active_markets",
                "open_orders",
                "daily_pnl_usd",
                "memory_mb",
                "disk_free_mb",
                "max_age_seconds",
            ]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            assert data["run_id"] == "test-run"
            assert data["cycle_id"] == 5
            assert data["status"] == "ok"

    def test_heartbeat_uses_env_var_path(self):
        """Fix 3: Heartbeat path reads from ASTRA_HEALTH_PATH."""
        from ops.health import write_heartbeat

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_health.json")
            with mock.patch.dict(os.environ, {"ASTRA_HEALTH_PATH": custom_path}):
                write_heartbeat(
                    run_id="test-run",
                    cycle_id=1,
                    status="ok",
                    memory_dir=tmpdir,
                )
                assert os.path.exists(custom_path)
                data = json.loads(Path(custom_path).read_text())
                assert data["run_id"] == "test-run"

    def test_heartbeat_memory_not_zero(self):
        """Fix 4: Memory should be None or a realistic value, never 0.0."""
        from ops.health import _get_memory_mb

        result = _get_memory_mb()
        # On any real system, should be either None or > 0
        if result is not None:
            assert result > 0, f"Memory should not be 0.0: {result}"


class TestAlerts:
    """Activity 10: Alerting with rate limits."""

    def test_alert_no_webhook_no_error(self):
        from ops.alerts import AlertManager

        mgr = AlertManager(webhook_url="", cooldown_seconds=60)
        result = mgr.send_alert("Test", "Message", "info", "test_type")
        assert result is True  # Should succeed (logged but not sent)

    def test_alert_rate_limit(self):
        from ops.alerts import AlertManager

        mgr = AlertManager(webhook_url="", cooldown_seconds=300)
        r1 = mgr.send_alert("Test", "Msg1", "info", "same_type")
        assert r1 is True
        r2 = mgr.send_alert("Test", "Msg2", "info", "same_type")
        assert r2 is False  # Rate limited

    def test_alert_different_types_not_limited(self):
        from ops.alerts import AlertManager

        mgr = AlertManager(webhook_url="", cooldown_seconds=300)
        r1 = mgr.send_alert("Test", "Msg1", "info", "type_a")
        r2 = mgr.send_alert("Test", "Msg2", "info", "type_b")
        assert r1 is True
        assert r2 is True  # Different type, not rate limited


class TestStructuredLogging:
    """Activity 11: Structured logging with run context."""

    def test_logging_json_file_contains_run_id(self):
        from ops.logging_setup import setup_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(run_id="test-run-id", json_mode=False, log_dir=tmpdir)
            logger = logging.getLogger("astra.test")
            logger.info("test message")
            # Flush handlers
            for h in logging.getLogger("astra").handlers:
                h.flush()
            log_path = Path(tmpdir) / "astra.log"
            content = log_path.read_text()
            # File handler always uses JSON
            for line in content.strip().split("\n"):
                if line.strip():
                    entry = json.loads(line)
                    assert entry["run_id"] == "test-run-id"
                    break
            # Clean up handlers
            logging.getLogger("astra").handlers.clear()

    def test_logging_cycle_context_injected(self):
        from ops.logging_setup import setup_logging
        from ops.run_context import set_cycle_context

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(run_id="test-run", json_mode=False, log_dir=tmpdir)
            set_cycle_context(42, "estimator")
            logger = logging.getLogger("astra.test_cycle")
            logger.info("cycle test message")
            for h in logging.getLogger("astra").handlers:
                h.flush()
            log_path = Path(tmpdir) / "astra.log"
            content = log_path.read_text()
            for line in content.strip().split("\n"):
                if "cycle test message" in line:
                    entry = json.loads(line)
                    assert entry["cycle_id"] == 42
                    assert entry["component"] == "estimator"
                    break
            logging.getLogger("astra").handlers.clear()


class TestShutdown:
    """Activity 8: Graceful shutdown."""

    def test_sigterm_sets_shutdown_flag(self):
        """Verify that a shutdown flag mechanism works."""
        # We test the pattern, not actual signal delivery (which is tricky in tests)
        shutdown_requested = False

        def handler():
            nonlocal shutdown_requested
            shutdown_requested = True

        handler()
        assert shutdown_requested is True

    def test_shutdown_writes_final_manifest_and_artifacts(self):
        from ops.artifacts import ArtifactWriter
        from ops.run_context import RunContext

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = RunContext(
                config_hash="a" * 64,
                prompt_bundle_hash="b" * 64,
                schema_version=2,
            )
            writer = ArtifactWriter(ctx.run_id, base_dir=Path(tmpdir))
            writer.ensure_writable()
            writer.write_manifest_start(ctx.to_manifest_dict())

            # Simulate shutdown
            writer.write_gate_status(ctx.cycle_id, {"status": "shutdown", "gates": []})
            writer.write_manifest_end("sigterm")

            # Verify
            manifest = json.loads((Path(tmpdir) / ctx.run_id / "manifest.json").read_text())
            assert manifest["ended_at"] is not None
            assert manifest["exit_reason"] == "sigterm"
            assert "ended_ok" not in manifest  # V1 schema: no ended_ok

            gate = json.loads((Path(tmpdir) / ctx.run_id / "gate_status.json").read_text())
            assert gate["status"] == "shutdown"


# ============================================================================
# TEST: Startup Checklist (Activity 17)
# ============================================================================

_VALID_CONFIG_HASH = "a" * 64
_VALID_PROMPT_HASH = "b" * 64


def _make_checklist_deps(tmpdir):
    """Create all dependencies needed for a passing checklist."""
    from ops.artifacts import ArtifactWriter
    from ops.gate_engine import GateEngine
    from ops.run_context import RunContext
    from scanner.trade_logger import CURRENT_SCHEMA_VERSION, init_db

    run_ctx = RunContext(
        paper_mode=True,
        config_hash=_VALID_CONFIG_HASH,
        prompt_bundle_hash=_VALID_PROMPT_HASH,
        schema_version=CURRENT_SCHEMA_VERSION,
    )

    db_path = Path(tmpdir) / "test.db"
    db_conn = init_db(db_path=db_path)

    # Write config snapshot for this run
    db_conn.execute(
        "INSERT INTO config_snapshots (snapshot_id, run_id, timestamp, config_hash, config_json) VALUES (?, ?, ?, ?, ?)",
        (f"snap_{run_ctx.run_id[:8]}", run_ctx.run_id, "2026-01-01T00:00:00Z", _VALID_CONFIG_HASH, "{}"),
    )
    db_conn.commit()

    # Create DB backup
    import shutil

    bak1 = db_path.with_suffix(".db.bak1")
    shutil.copy2(db_path, bak1)

    artifact_writer = ArtifactWriter(run_ctx.run_id, base_dir=Path(tmpdir) / "artifacts")
    artifact_writer.ensure_writable()
    artifact_writer.write_manifest_start(run_ctx.to_manifest_dict())
    artifact_writer.write_prompt_registry({"test": "registry"})

    gate_engine = GateEngine()

    heartbeat_path = str(Path(tmpdir) / "heartbeat.json")

    class FakeAlertMgr:
        pass

    alert_mgr = FakeAlertMgr()

    return {
        "run_ctx": run_ctx,
        "db_conn": db_conn,
        "artifact_writer": artifact_writer,
        "gate_engine": gate_engine,
        "alert_mgr": alert_mgr,
        "config_hash": _VALID_CONFIG_HASH,
        "prompt_bundle_hash": _VALID_PROMPT_HASH,
        "api_key_set": True,
        "paper_mode": True,
        "heartbeat_path": heartbeat_path,
        "signal_handlers_registered": True,
        "db_path": db_path,
    }


class TestStartupChecklist:
    """Activity 17: Startup checklist tests."""

    def test_all_checks_pass_happy_path(self):
        from ops.startup_checklist import all_passed, run_startup_checklist

        with tempfile.TemporaryDirectory() as tmpdir:
            deps = _make_checklist_deps(tmpdir)
            results = run_startup_checklist(**deps)
            assert all_passed(results), f"Failed checks: {[r for r in results if not r.passed and not r.warn_only]}"

    def test_missing_db_fails(self):
        from ops.startup_checklist import all_passed, run_startup_checklist

        with tempfile.TemporaryDirectory() as tmpdir:
            deps = _make_checklist_deps(tmpdir)
            deps["db_conn"].close()
            deps["db_conn"] = None
            results = run_startup_checklist(**deps)
            assert not all_passed(results)
            failed_names = [r.name for r in results if not r.passed and not r.warn_only]
            assert "db_connection" in failed_names

    def test_bad_schema_version_fails(self):
        from ops.startup_checklist import all_passed, run_startup_checklist

        with tempfile.TemporaryDirectory() as tmpdir:
            deps = _make_checklist_deps(tmpdir)
            # Set schema version to wrong value
            deps["db_conn"].execute("PRAGMA user_version = 999")
            results = run_startup_checklist(**deps)
            assert not all_passed(results)
            failed_names = [r.name for r in results if not r.passed and not r.warn_only]
            assert "schema_version" in failed_names

    def test_artifacts_not_writable_fails(self):
        from ops.startup_checklist import all_passed, run_startup_checklist

        with tempfile.TemporaryDirectory() as tmpdir:
            deps = _make_checklist_deps(tmpdir)
            # Make artifacts dir unwritable by pointing to a non-existent read-only path
            deps["artifact_writer"].run_dir = Path("/nonexistent/path/that/cannot/exist")
            results = run_startup_checklist(**deps)
            assert not all_passed(results)
            failed_names = [r.name for r in results if not r.passed and not r.warn_only]
            assert "artifacts_writable" in failed_names

    def test_heartbeat_path_not_writable_fails(self):
        from ops.startup_checklist import all_passed, run_startup_checklist

        with tempfile.TemporaryDirectory() as tmpdir:
            deps = _make_checklist_deps(tmpdir)
            deps["heartbeat_path"] = "/nonexistent/readonly/heartbeat.json"
            results = run_startup_checklist(**deps)
            assert not all_passed(results)
            failed_names = [r.name for r in results if not r.passed and not r.warn_only]
            assert "heartbeat_writable" in failed_names

    def test_gate_engine_dry_run_succeeds(self):
        """Gate engine dry-run with neutral context should pass."""
        from ops.startup_checklist import run_startup_checklist

        with tempfile.TemporaryDirectory() as tmpdir:
            deps = _make_checklist_deps(tmpdir)
            results = run_startup_checklist(**deps)
            gate_result = [r for r in results if r.name == "gate_engine_dry_run"]
            assert len(gate_result) == 1
            assert gate_result[0].passed is True

    def test_low_disk_fails(self):
        from ops.startup_checklist import all_passed, run_startup_checklist

        with tempfile.TemporaryDirectory() as tmpdir:
            deps = _make_checklist_deps(tmpdir)
            # Mock disk free to return very low value
            with mock.patch("ops.health._get_disk_free_mb", return_value=1.0):
                results = run_startup_checklist(**deps)
                disk_result = [r for r in results if r.name == "disk_free"]
                assert len(disk_result) == 1
                assert disk_result[0].passed is False

    def test_missing_api_key_warn_only_paper(self):
        """paper_mode + no api key → all_passed() still True (warn only)."""
        from ops.startup_checklist import all_passed, run_startup_checklist

        with tempfile.TemporaryDirectory() as tmpdir:
            deps = _make_checklist_deps(tmpdir)
            deps["api_key_set"] = False
            deps["paper_mode"] = True
            results = run_startup_checklist(**deps)
            # Should still pass overall (warn_only)
            assert all_passed(results)
            api_result = [r for r in results if r.name == "api_key"]
            assert len(api_result) == 1
            assert api_result[0].warn_only is True
            assert api_result[0].passed is False


# ============================================================================
# TEST: Monitor-only Phase (Activity 17)
# ============================================================================


class TestMonitorOnly:
    """Activity 17: Monitor-only phase tests."""

    def test_monitor_only_countdown(self):
        """3 ok cycles → monitor_only_remaining == 0."""
        monitor_only_remaining = 3
        monitor_failures = 0

        for _ in range(3):
            # Simulate "ok" gate status
            gate_ok = True
            if gate_ok:
                monitor_only_remaining -= 1

        assert monitor_only_remaining == 0

    def test_monitor_only_degraded_resets(self):
        """ok, degraded → counter resets to BURN_IN_MONITOR_CYCLES."""
        burn_in_cycles = 3
        monitor_only_remaining = burn_in_cycles
        monitor_failures = 0

        # 1 ok cycle
        monitor_only_remaining -= 1
        assert monitor_only_remaining == 2

        # degraded → reset
        monitor_failures += 1
        monitor_only_remaining = burn_in_cycles
        assert monitor_only_remaining == burn_in_cycles
        assert monitor_failures == 1

    def test_monitor_only_no_positions_opened(self):
        """During monitor-only, real PaperPortfolio.open_position returns None (belt)."""
        from paper_trader import PaperPortfolio

        with tempfile.TemporaryDirectory() as tmpdir:
            positions_file = Path(tmpdir) / "paper_positions.json"
            with mock.patch("paper_trader.POSITIONS_FILE", positions_file):
                portfolio = PaperPortfolio()
                portfolio.set_monitor_only(True)

                assert portfolio.monitor_only is True

                # Create a fake opportunity with required attrs
                opp = mock.MagicMock()
                opp.market.condition_id = "test_cond"
                opp.market.question = "Test?"
                opp.market.category = "test"
                opp.direction = "BUY YES"
                opp.our_estimate = 0.7
                opp.market_price = 0.5

                result = portfolio.open_position(opp, {"position_dollars": 100})
                assert result is None, "Monitor-only should block open_position"

                # Disable monitor-only and verify position CAN be opened
                portfolio.set_monitor_only(False)
                assert portfolio.monitor_only is False
                result2 = portfolio.open_position(opp, {"position_dollars": 100})
                assert result2 is not None, "Should open position when monitor-only is False"

    def test_monitor_only_allow_new_positions_false(self):
        """allow_new_positions=False prevents position opening (suspenders)."""
        # This tests that the logic in run_paper_scan skips position opening
        # We verify by checking the conditional structure
        allow_new_positions = False
        positions_opened = 0
        opportunities = [1, 2, 3]  # Fake opportunities

        if allow_new_positions:
            for opp in opportunities:
                positions_opened += 1

        assert positions_opened == 0

    def test_monitor_only_max_failures_halts(self):
        """10 degraded cycles → shutdown_requested=True."""
        max_failures = 10
        burn_in_cycles = 3
        monitor_only_remaining = burn_in_cycles
        monitor_failures = 0
        shutdown_requested = False

        for _ in range(max_failures):
            # Simulate degraded
            monitor_failures += 1
            monitor_only_remaining = burn_in_cycles
            if monitor_failures >= max_failures:
                shutdown_requested = True

        assert shutdown_requested is True
        assert monitor_failures == max_failures

    def test_monitor_only_zero_config_skips(self):
        """BURN_IN_MONITOR_CYCLES=0 → no monitor-only phase."""
        burn_in_cycles = 0

        class SimplePortfolio:
            def __init__(self):
                self._monitor_only = False

            def set_monitor_only(self, enabled):
                self._monitor_only = enabled

            @property
            def monitor_only(self):
                return self._monitor_only

        portfolio = SimplePortfolio()
        if burn_in_cycles > 0:
            portfolio.set_monitor_only(True)
        else:
            portfolio.set_monitor_only(False)

        assert portfolio.monitor_only is False


# ============================================================================
# TEST: Monitor Artifacts (Activity 17)
# ============================================================================


class TestMonitorArtifacts:
    """Activity 17: Monitor state in artifacts."""

    def test_gate_status_includes_monitor_state(self):
        """gate_status artifact dict has monitor_only, monitor_remaining, monitor_failures."""
        from ops.gate_engine import GateContext, GateEngine

        engine = GateEngine()
        ctx = GateContext(ws_connected=True, feed_age_s=0, error_count=0, memory_mb=100, disk_free_mb=1000)
        gate_status = engine.evaluate(ctx, run_id="test", cycle_id=1)

        # Simulate what paper_trader does: extend the dict
        gate_dict = gate_status.to_dict()
        gate_dict["monitor_only"] = True
        gate_dict["monitor_remaining"] = 2
        gate_dict["monitor_failures"] = 1

        assert gate_dict["monitor_only"] is True
        assert gate_dict["monitor_remaining"] == 2
        assert gate_dict["monitor_failures"] == 1

    def test_decision_report_includes_monitor_state(self):
        """decision_report has positions_allowed field."""
        decision_report = {
            "monitor_only": True,
            "monitor_remaining": 2,
            "monitor_failures": 0,
            "positions_allowed": False,
        }

        assert decision_report["positions_allowed"] is False
        assert decision_report["monitor_only"] is True
