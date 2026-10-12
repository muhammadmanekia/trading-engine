"""LLM plumbing: HTTP retries and error typing, prompt registry, budget caps, call recorder, scripted provider."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from engine.clock import FakeClock
from engine.llm.budget import (
    UNCONFIGURED_PRICING,
    BudgetPolicy,
    BudgetTracker,
    Pricing,
    cost_usd,
    worst_case_cost,
)
from engine.llm.fake import ScriptedProvider
from engine.llm.prompts import PromptError, PromptRegistry
from engine.llm.provider import (
    LLMBudgetExceeded,
    LLMCredentialsError,
    LLMInvalidResponse,
    LLMRequest,
    LLMUnavailable,
    input_hash,
    post_json,
)
from engine.llm.recorder import InMemoryLLMRecorder, LLMCall, LLMRecorder, PostgresLLMRecorder
from tests.helpers import T0, D

# ---- post_json ---------------------------------------------------------------------------------------


def client(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_post_json_returns_success_and_retries_transient_failures() -> None:
    calls = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json={"ok": 1})

    sleeps: list[float] = []
    resp = post_json(client(handler), "https://x.test/v1", {"k": "v"}, {}, sleeper=sleeps.append)
    assert resp.json() == {"ok": 1} and len(calls) == 3 and len(sleeps) == 2


def test_429_honours_retry_after_then_gives_up_as_unavailable() -> None:
    sleeps: list[float] = []
    with pytest.raises(LLMUnavailable):
        post_json(
            client(lambda r: httpx.Response(429, headers={"retry-after": "3"})),
            "https://x.test",
            {},
            {},
            sleeper=sleeps.append,
        )
    assert sleeps and sleeps[0] == 3.0


@pytest.mark.parametrize(
    ("status", "exc"), [(401, LLMCredentialsError), (403, LLMCredentialsError), (400, LLMInvalidResponse)]
)
def test_non_retryable_statuses_fail_immediately_with_the_right_type(
    status: int, exc: type[Exception]
) -> None:
    calls = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"error": "nope"})

    with pytest.raises(exc):
        post_json(client(handler), "https://x.test", {}, {}, sleeper=lambda _s: None)
    assert len(calls) == 1


def test_network_errors_are_unavailable_and_never_leak_the_key() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=r)

    with pytest.raises(LLMUnavailable) as e:
        post_json(
            client(handler),
            "https://x.test",
            {"x-api-key": "SUPER-SECRET-KEY"},
            {"prompt": "SECRET-PROMPT"},
            sleeper=lambda _s: None,
        )
    assert "SUPER-SECRET" not in str(e.value) and "SECRET-PROMPT" not in str(e.value)


def test_input_hash_depends_on_every_component() -> None:
    base = ("claude", "m1", "ph", "sys", "usr", "schema")
    h = input_hash(*base)
    assert h == input_hash(*base)
    for i in range(len(base)):
        changed = list(base)
        changed[i] += "x"
        assert input_hash(*changed) != h


def test_request_validation() -> None:
    with pytest.raises(ValueError):
        LLMRequest(system="s", user="u", schema_name="has space", json_schema={})
    with pytest.raises(ValueError):
        LLMRequest(system="s", user="u", schema_name="ok", json_schema={}, max_output_tokens=1)


# ---- prompts -----------------------------------------------------------------------------------------
def test_registry_resolves_latest_numerically_and_exact_versions(tmp_path: Path) -> None:
    for v in (1, 2, 10):
        (tmp_path / "p").mkdir(exist_ok=True)
        (tmp_path / "p" / f"v{v}.md").write_text(f"text {v} {{{{x}}}}")
    reg = PromptRegistry(tmp_path)
    assert reg.get("p").version == "v10" and reg.get("p", "v2").text.startswith("text 2")
    assert reg.versions("p") == ["v1", "v2", "v10"]
    with pytest.raises(PromptError, match="does not exist"):
        reg.get("p", "v3")
    with pytest.raises(PromptError, match="no prompt named"):
        reg.get("missing")


def test_prompt_hash_changes_with_the_text_and_render_checks_slots(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "v1.md").write_text("hello {{name}}")
    t = PromptRegistry(tmp_path).get("p")
    assert t.render(name="you") == "hello you"
    with pytest.raises(PromptError, match="missing"):
        t.render()
    with pytest.raises(PromptError, match="unexpected"):
        t.render(name="a", other="b")
    (tmp_path / "p" / "v1.md").write_text("hello {{name}}!")
    assert PromptRegistry(tmp_path).get("p").hash != t.hash


def test_the_shipped_prompt_carries_the_safety_rules() -> None:
    t = PromptRegistry().get("llm_strategy")
    text = t.render(instructions="Trade quality large caps.")
    assert t.version == "v1" and "Trade quality large caps." in text
    for must in (
        "UNTRUSTED DATA",
        "never instructions",
        "HOLD is the default",
        "cannot choose prices",
        "at least one news_item",
    ):
        assert must in text


# ---- budget -------------------------------------------------------------------------------------------
PRICE = Pricing(input_per_mtok=D(3), output_per_mtok=D(15))


def call(cost: str, ts=T0, strategy: str = "s1") -> LLMCall:  # type: ignore[no-untyped-def]
    return LLMCall(
        call_id=f"c{cost}{ts.isoformat()}{strategy}", ts=ts, account_id="acct", strategy_id=strategy, mode="primary",
        provider="p", model_requested="m", model_reported="m", prompt_name="n", prompt_version="v1", prompt_hash="h",
        input_hash=f"i{cost}", system_text="s", user_text="u", response_text="{}", valid=True, error=None,
        decisions=[{"a": 1}], input_tokens=1, output_tokens=1, cost_usd=D(cost), latency_ms=5,
    )  # fmt: skip


def test_cost_arithmetic_and_pessimistic_defaults() -> None:
    assert cost_usd(PRICE, 1_000_000, 100_000) == D("4.5")
    assert worst_case_cost(PRICE, "x" * 300, "y" * 300, 1000) == cost_usd(PRICE, 200, 1000)
    assert UNCONFIGURED_PRICING.input_per_mtok >= 15  # an unknown price can only make us more careful


def test_daily_monthly_and_per_strategy_caps_block_before_the_call() -> None:
    rec, clock, alerts = InMemoryLLMRecorder(), FakeClock(T0), []
    bt = BudgetTracker(
        BudgetPolicy(daily_usd=D(1), monthly_usd=D(5)), rec, clock, lambda *a: alerts.append(a)
    )
    bt.check(D("0.5"), strategy_id="s1")
    rec.record(call("0.6"))
    with pytest.raises(LLMBudgetExceeded, match="daily"):
        bt.check(D("0.5"), strategy_id="s1")  # 0.6 + 0.5 > 1
    with pytest.raises(LLMBudgetExceeded, match="strategy s1 daily"):
        BudgetTracker(BudgetPolicy(daily_usd=D(100), monthly_usd=D(100)), rec, clock).check(
            D("0.5"), strategy_id="s1", strategy_daily_usd=D("1")
        )
    BudgetTracker(BudgetPolicy(daily_usd=D(100), monthly_usd=D(100)), rec, clock).check(
        D("0.5"), strategy_id="other", strategy_daily_usd=D("1")
    )  # another strategy has its own allowance
    assert any(level == "CRITICAL" for level, *_ in alerts)


def test_warns_once_at_80_percent_per_day() -> None:
    rec, clock, alerts = InMemoryLLMRecorder(), FakeClock(T0), []
    bt = BudgetTracker(
        BudgetPolicy(daily_usd=D(10), monthly_usd=D(1000)), rec, clock, lambda *a: alerts.append(a)
    )
    rec.record(call("8.5"))
    bt.check(D("0.1"), strategy_id="s1")
    bt.check(D("0.1"), strategy_id="s1")
    assert [a[0] for a in alerts] == ["WARNING"]
    clock.advance(days=1)
    rec.record(call("8.5", ts=clock.now()))
    bt.check(D("0.1"), strategy_id="s1")
    assert [a[0] for a in alerts] == ["WARNING", "WARNING"]  # a new day warns again


def test_spend_survives_a_restart_and_rolls_over_by_day_but_not_month() -> None:
    rec, clock = InMemoryLLMRecorder(), FakeClock(T0)
    rec.record(call("4"))
    fresh = BudgetTracker(BudgetPolicy(daily_usd=D(5), monthly_usd=D(5)), rec, clock)  # a "restarted" process
    with pytest.raises(LLMBudgetExceeded):
        fresh.check(D(2), strategy_id="s1")
    clock.advance(days=1)
    with pytest.raises(LLMBudgetExceeded, match="monthly"):
        fresh.check(D(2), strategy_id="s1")  # the day reset, the month did not
    assert fresh.spent_today() == 0 and fresh.spent_this_month() == 4


# ---- recorder ------------------------------------------------------------------------------------------
@pytest.fixture(params=["memory", "postgres"])
def recorder(request: pytest.FixtureRequest) -> LLMRecorder:
    return (
        InMemoryLLMRecorder()
        if request.param == "memory"
        else PostgresLLMRecorder(request.getfixturevalue("pg_dsn"))
    )


def test_recorder_roundtrip_cache_and_totals(recorder: LLMRecorder) -> None:
    a, b = call("0.25"), call("0.75", ts=T0 + timedelta(hours=1), strategy="s2")
    bad = LLMCall(
        **{
            **a.__dict__,
            "call_id": "bad",
            "input_hash": "ih-bad",
            "valid": False,
            "error": "garbage",
            "decisions": None,
        }
    )
    for c in (a, b, bad):
        recorder.record(c)
    recorder.record(
        a
    )  # idempotent on call_id for Postgres; memory keeps a duplicate, totals tested separately
    assert recorder.find_by_input_hash(a.input_hash) is not None
    assert recorder.find_by_input_hash("ih-bad") is None  # only VALID answers are reusable
    assert recorder.cost_since(T0, "s2") == D("0.75")
    assert [c.strategy_id for c in recorder.calls_since(T0 + timedelta(minutes=30))] == ["s2"]
    stored = recorder.find_by_input_hash(b.input_hash)
    assert (
        stored is not None
        and stored.decisions == [{"a": 1}]
        and stored.cost_usd == D("0.75")
        and stored.system_text == "s"
    )


# ---- scripted provider ------------------------------------------------------------------------------------
REQ = LLMRequest(system="s", user="u", schema_name="x", json_schema={"type": "object"})


def test_scripted_provider_modes() -> None:
    ok = ScriptedProvider.always({"decisions": [], "n": 0.5})
    r = ok.complete(REQ)
    assert (
        r.parsed == {"decisions": [], "n": Decimal("0.5")}
        and r.provider == "scripted"
        and ok.requests == [REQ]
    )
    with pytest.raises(LLMInvalidResponse):
        ScriptedProvider.always("not json at all").complete(REQ)
    with pytest.raises(LLMInvalidResponse):
        ScriptedProvider.always("[1, 2]").complete(REQ)  # JSON, but not an object
    with pytest.raises(LLMUnavailable):
        ScriptedProvider.down().complete(REQ)
    assert Decimal(1) == D(1) and input_hash("a", "b", "c", "d", "e", "f")
