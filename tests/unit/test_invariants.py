"""Codebase-level invariants from the Phase 1 acceptance criteria."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "engine"


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


BOUNDARY_DIRS = {"broker", "marketdata", "service"}


def _float_uses(path: Path) -> list[str]:
    out: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            out.append(f"{path.name}:{node.lineno} float literal")
        if isinstance(node, ast.Name) and node.id == "float":
            out.append(f"{path.name}:{node.lineno} float")
    return out


def test_money_path_has_no_floats() -> None:
    """Money and quantities are Decimal only. The sole mentions are the validator that rejects floats and
    the clock's time arithmetic."""
    allowed = {"types.py", "clock.py"}
    offenders = [
        o
        for p in _py_files()
        if p.parent.name not in BOUNDARY_DIRS and p.name not in allowed
        for o in _float_uses(p)
    ]
    assert offenders == []


def test_adapters_never_turn_floats_into_money() -> None:
    """Boundary code may type timing callbacks as float, but must not call float() or use float literals."""
    offenders = []
    for p in _py_files():
        if p.parent.name in BOUNDARY_DIRS and p.name != "http.py":  # http.py: backoff/rate timing
            tree = ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    offenders.append(f"{p.name}:{node.lineno} float literal")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
                    offenders.append(f"{p.name}:{node.lineno} float()")
    assert offenders == []


def test_risk_engine_is_pure_no_io_clock_or_network_imports() -> None:
    forbidden = re.compile(
        r"^\s*(import|from)\s+(requests|httpx|urllib|socket|psycopg|subprocess|random|time|datetime\.datetime\.now)\b",
        re.M,
    )
    for path in (SRC / "risk").glob("*.py"):
        assert not forbidden.search(path.read_text()), path.name
        assert "now()" not in path.read_text().replace("system.now", ""), f"{path.name} reads the clock"


SECRET_ALLOWED_DIRS = {"broker", "marketdata", "service"}


def test_credentials_are_confined_to_the_adapter_market_data_and_service_layers() -> None:
    """Risk, orders, ledger, events and the control plane must never see or handle a credential."""
    pattern = re.compile(r"os\.environ|getenv|api[_-]?key|secret|password|token|SecretStr", re.I)
    for path in _py_files():
        if path.parent.name in SECRET_ALLOWED_DIRS:
            continue
        assert not pattern.search(path.read_text()), f"{path.relative_to(SRC)} touches credentials"


def test_only_the_service_layer_reads_the_environment() -> None:
    for path in _py_files():
        if path.parent.name != "service":
            assert "os.environ" not in path.read_text() and "getenv" not in path.read_text(), path.name


def test_only_the_order_manager_talks_to_the_broker() -> None:
    callers = {
        p.name
        for p in _py_files()
        if re.search(r"\.(submit|cancel)\(", p.read_text()) and p.parent.name != "broker"
    }
    assert callers == {"manager.py"}
