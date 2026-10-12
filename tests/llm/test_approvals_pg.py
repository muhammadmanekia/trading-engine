"""Proposal storage in Postgres, the audit/recorder tables, the [llm] config, and the proposals CLI."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import psycopg

from engine.clock import SystemClock
from engine.llm.recorder import PostgresLLMRecorder
from engine.service.cli import main
from engine.service.config import Secrets, load_config
from engine.strategy.approvals import PostgresProposalStore, Proposal
from tests.helpers import D, make_intent

TOML = """
[llm]
daily_usd = 3
monthly_usd = 40.5

[llm.providers.claude]
kind = "anthropic"
model = "claude-sonnet-5-5"
knowledge_cutoff = 2026-01-31
input_per_mtok = 3.0
output_per_mtok = 15.0

[llm.providers.grok]
kind = "xai"
model = "grok-4"

[[accounts]]
id = "alpaca-paper"
broker = "alpaca"
llm_strategies = ["strategies/examples/news-swing-stocks.toml"]
"""


def proposal(now: datetime, minutes: int = 30, pid: str = "p1") -> Proposal:
    intent = make_intent(
        intent_id=pid, strategy_id="news", created_at=now, expires_at=now + timedelta(minutes=minutes)
    )
    return Proposal(
        pid, "acct", "news", intent, "Beat estimates.", ("n1", "n2"), now, now + timedelta(minutes=minutes)
    )


def run(pg_dsn: str, tmp_path: Path, *argv: str) -> int:
    cfg = tmp_path / "c.toml"
    cfg.write_text(TOML)
    return main([argv[0], "--config", str(cfg), *argv[1:]], {"ENGINE_DATABASE_URL": pg_dsn})


def test_proposals_roundtrip_expire_and_decide_once(pg_dsn: str) -> None:
    store, now = PostgresProposalStore(pg_dsn), SystemClock().now()
    assert store.add(proposal(now)) and not store.add(proposal(now))  # duplicate id ignored
    got = store.get("p1")
    assert (
        got is not None
        and got.intent.quantity == D(10)
        and got.cited == ("n1", "n2")
        and got.status == "pending"
    )
    assert [p.proposal_id for p in store.pending(now)] == ["p1"]
    assert store.pending(now + timedelta(hours=1)) == []  # expired proposals are not offered
    assert store.count_since(now - timedelta(minutes=1), "news") == 1
    assert store.decide("p1", "approved", now) and not store.decide("p1", "rejected", now)
    assert store.get("p1").status == "approved"  # type: ignore[union-attr]
    store.close()


def test_config_parses_llm_section_with_exact_decimals_and_dates(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(TOML)
    llm = load_config(cfg).llm
    assert llm is not None and llm.daily_usd == 3 and llm.monthly_usd == D("40.5")
    c = llm.providers["claude"]
    assert c.input_per_mtok == D(3) and c.knowledge_cutoff is not None and c.knowledge_cutoff.year == 2026
    assert llm.providers["grok"].input_per_mtok is None  # unpriced -> the budget uses a pessimistic default


def test_secrets_pick_up_model_keys_and_redact_them() -> None:
    s = Secrets.from_env(
        {"ENGINE_DATABASE_URL": "postgresql://x", "ANTHROPIC_API_KEY": "sk-ant-1", "XAI_API_KEY": "xai-2"}
    )
    assert s.anthropic_key is not None and s.xai_key is not None
    assert "sk-ant-1" in s.values() and "xai-2" in s.values() and "sk-ant-1" not in repr(s)


def test_cli_lists_approves_and_rejects_proposals(pg_dsn: str, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    now = SystemClock().now()
    store = PostgresProposalStore(pg_dsn)
    store.add(proposal(now, pid="a1"))
    store.add(proposal(now, pid="a2"))
    assert run(pg_dsn, tmp_path, "proposals") == 0
    out = capsys.readouterr().out
    assert "a1" in out and "a2" in out and "Beat estimates." in out and "n1, n2" in out

    assert run(pg_dsn, tmp_path, "approve", "--id", "a1") == 0
    with psycopg.connect(pg_dsn) as c:  # the approved intent was queued in the inbox
        assert c.execute("SELECT count(*) FROM intent_inbox").fetchone()[0] == 1  # type: ignore[index]
    assert run(pg_dsn, tmp_path, "approve", "--id", "a1") == 1  # not twice
    assert "already approved" in capsys.readouterr().err
    assert run(pg_dsn, tmp_path, "reject", "--id", "a2") == 0
    assert run(pg_dsn, tmp_path, "approve", "--id", "a2") == 1
    assert run(pg_dsn, tmp_path, "approve", "--id", "nope") == 1
    assert run(pg_dsn, tmp_path, "proposals") == 0 and "no proposals waiting" in capsys.readouterr().out
    store.close()


def test_the_llm_report_cli_needs_prices_and_says_so(pg_dsn: str, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert run(pg_dsn, tmp_path, "llm-report") == 2
    assert "Alpaca credentials" in capsys.readouterr().err
    assert PostgresLLMRecorder(pg_dsn).calls_since(SystemClock().now() - timedelta(days=1)) == []
