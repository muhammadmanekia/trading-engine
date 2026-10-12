"""Codebase-level invariants from the Phase 1 acceptance criteria."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "engine"


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


BOUNDARY_DIRS = {"broker", "marketdata", "service"}
# Model plumbing and news filtering use floats only for timeouts, backoff and text-similarity thresholds, never for
# money; test_llm_and_news_never_convert_to_float below guards that.
NON_MONEY_FLOAT_DIRS = {"llm", "news"}


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
    # schema.py: turns TOML floats into exact decimals; llm_schema.py: JSON-schema bounds and float->str on input
    allowed = {"types.py", "clock.py", "schema.py", "llm_schema.py"}
    offenders = [
        o
        for p in _py_files()
        if p.parent.name not in BOUNDARY_DIRS | NON_MONEY_FLOAT_DIRS and p.name not in allowed
        for o in _float_uses(p)
    ]
    assert offenders == []


def test_adapters_never_turn_floats_into_money() -> None:
    """Boundary code may type timing callbacks as float, but must not call float() or use float literals."""
    offenders = []
    for p in _py_files():
        if p.parent.name in BOUNDARY_DIRS and p.name not in {
            "http.py",
            "alerts.py",
            "config.py",
            "executor.py",
        }:  # backoff / retry-after timing
            tree = ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    offenders.append(f"{p.name}:{node.lineno} float literal")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
                    offenders.append(f"{p.name}:{node.lineno} float()")
    assert offenders == []


def test_llm_and_news_never_convert_to_float() -> None:
    """No float() calls on values in model/news code, except parsing a Retry-After header."""
    offenders = []
    for p in _py_files():
        if p.parent.name in NON_MONEY_FLOAT_DIRS and p.name != "provider.py":
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
                    offenders.append(f"{p.name}:{node.lineno}")
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
LLM_KEY_MODULES = {"anthropic.py", "xai.py"}  # the only model-side code that handles an API key


def test_credentials_are_confined_to_the_adapter_market_data_and_service_layers() -> None:
    """Risk, orders, ledger, events and the control plane must never see or handle a credential."""
    pattern = re.compile(r"os\.environ|getenv|api[_-]?key|secret|password|token|SecretStr", re.I)
    for path in _py_files():
        if path.parent.name in SECRET_ALLOWED_DIRS or path.name in LLM_KEY_MODULES:
            continue
        text = path.read_text()
        if path.parent.name in {"llm", "news"} or path.name.startswith("llm_"):
            # these legitimately talk about token COUNTS; filters.py holds detection patterns, not secrets
            pattern_here = re.compile(r"os\.environ|getenv|api[_-]?key|secret|password|SecretStr", re.I)
            if path.name == "filters.py":
                continue
            assert not pattern_here.search(text), f"{path.relative_to(SRC)} touches credentials"
            continue
        assert not pattern.search(text), f"{path.relative_to(SRC)} touches credentials"


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
