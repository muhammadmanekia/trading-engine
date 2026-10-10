"""Codebase-level invariants from the Phase 1 acceptance criteria."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "engine"


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_no_float_type_or_literal_anywhere_in_the_engine() -> None:
    """Money and quantities are Decimal only. The sole mention of `float` is the validator that rejects it."""
    offenders: list[str] = []
    for path in _py_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                offenders.append(f"{path.name}:{node.lineno} float literal")
            if isinstance(node, ast.Name) and node.id == "float":
                offenders.append(f"{path.name}:{node.lineno} float name")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
                offenders.append(f"{path.name}:{node.lineno} float()")
    allowed = {"types.py", "clock.py"}  # types: the guard that rejects floats; clock: time, not money
    assert [o for o in offenders if o.split(":")[0] not in allowed] == []


def test_risk_engine_is_pure_no_io_clock_or_network_imports() -> None:
    forbidden = re.compile(
        r"^\s*(import|from)\s+(requests|httpx|urllib|socket|psycopg|subprocess|random|time|datetime\.datetime\.now)\b",
        re.M,
    )
    for path in (SRC / "risk").glob("*.py"):
        assert not forbidden.search(path.read_text()), path.name
        assert "now()" not in path.read_text().replace("system.now", ""), f"{path.name} reads the clock"


def test_no_broker_credentials_or_env_access_in_phase_1() -> None:
    pattern = re.compile(r"os\.environ|getenv|api[_-]?key|secret|password|token", re.I)
    for path in _py_files():
        assert not pattern.search(path.read_text()), path.name


def test_only_the_order_manager_talks_to_the_broker() -> None:
    callers = {
        p.name
        for p in _py_files()
        if re.search(r"\.(submit|cancel)\(", p.read_text()) and p.parent.name != "broker"
    }
    assert callers == {"manager.py"}
