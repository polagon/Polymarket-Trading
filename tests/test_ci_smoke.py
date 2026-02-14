"""
CI smoke tests — verify CI infrastructure and package structure.

Loop 4: CI gate validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Project root (2 levels up from tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCIWorkflow:
    """Verify CI workflow exists and references required tools."""

    def test_ci_workflow_exists(self) -> None:
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists(), f"CI workflow not found at {ci_path}"

    def test_ci_references_pytest(self) -> None:
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        assert "pytest" in content, "CI workflow must reference pytest"

    def test_ci_references_precommit(self) -> None:
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        assert "pre-commit" in content, "CI workflow must reference pre-commit"


class TestPackageStructure:
    """Verify all new packages have __init__.py files."""

    @pytest.mark.parametrize(
        "package",
        [
            "definitions",
            "gates",
            "signals",
            "telemetry",
            "strategies",
        ],
    )
    def test_package_init_exists(self, package: str) -> None:
        init_path = PROJECT_ROOT / package / "__init__.py"
        assert init_path.exists(), f"Missing __init__.py for {package}"


class TestKeyModulesExist:
    """Verify key Loop 4 modules exist."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "models/definition_contract.py",
            "models/reasons.py",
            "definitions/registry.py",
            "definitions/lint.py",
            "gates/definition_gate.py",
            "gates/ev_gate.py",
            "signals/flow_toxicity.py",
            "risk/risk_engine.py",
            "execution/order_manager.py",
            "telemetry/trade_telemetry.py",
            "strategies/crypto_threshold.py",
        ],
    )
    def test_module_exists(self, module_path: str) -> None:
        full_path = PROJECT_ROOT / module_path
        assert full_path.exists(), f"Missing module: {module_path}"
