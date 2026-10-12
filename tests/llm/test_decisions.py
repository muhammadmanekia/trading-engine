"""Decision schema and validator: the security boundary. Every rule individually, then a property test."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.strategy.llm_schema import (
    DECISIONS_JSON_SCHEMA,
    LLMDecision,
    LLMSizing,
    load_any_strategy,
    parse_decisions,
    parse_llm_strategy,
    quantity_for,
    tier_fraction,
    validate_decisions,
)
from engine.strategy.schema import StrategyDef, StrategyError
from tests.helpers import D

ROOT = Path(__file__).resolve().parents[2]
BASE: dict[str, Any] = {
    "type": "llm",
    "id": "news-test",
    "instructions": "Trade only on genuinely new company-specific news; prefer to hold.",
    "universe": ["AAPL", "MSFT"],
    "models": {"primary": "claude"},
    "sizing": {"max_notional": 1000},
}
STRAT = parse_llm_strategy(BASE)


def dec(**kw: Any) -> LLMDecision:
    base = dict(symbol="AAPL", action="enter", confidence="0.8", stop_loss_pct="0.03", thesis="Beat estimates.",
                cited_ids=("n1",))  # fmt: skip
    return LLMDecision.model_validate({**base, **kw})


def check(
    *decisions: LLMDecision,
    held: set[str] = frozenset(),
    entering: set[str] = frozenset(),  # type: ignore[assignment]
    exiting: set[str] = frozenset(),
    ids: set[str] = frozenset({"n1", "n2"}),
    entries_today: int = 0,
    strat=STRAT,
):  # type: ignore[no-untyped-def,assignment]
    return validate_decisions(
        decisions, strat, supplied_ids=set(ids),
        position_qty=lambda s: D(10) if s in held else D(0),
        pending_entry=lambda s: s in entering, pending_exit=lambda s: s in exiting,
        entries_today=entries_today, news_triggered=True,
    )  # fmt: skip


def reasons(c) -> list[str]:  # type: ignore[no-untyped-def]
    return [r.reason for r in c.rejected]


# ---- parsing -----------------------------------------------------------------------------------------------
def reply(*decisions: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps({"decisions": list(decisions)}), parse_float=Decimal)


GOOD = {"symbol": "AAPL", "action": "enter", "confidence": 0.8, "horizon": "days", "stop_loss_pct": 0.03,
        "thesis": "Beat.", "invalidation": "", "cited_ids": ["n1"]}  # fmt: skip


def test_a_well_formed_reply_parses_to_exact_decimals() -> None:
    (d,) = parse_decisions(reply(GOOD))
    assert d.confidence == Decimal("0.8") and d.stop_loss_pct == Decimal("0.03") and d.cited_ids == ("n1",)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {"decisions": "buy"},
        {"decisions": [], "extra": 1},  # unexpected top-level key
        {"other": []},
    ],
)
def test_malformed_replies_are_rejected(bad: Any) -> None:
    with pytest.raises(ValueError):
        parse_decisions(bad)


@pytest.mark.parametrize(
    "mutation",
    [
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"action": "BUY"},  # not one of enter/exit/hold
        {"action": "short"},
        {"symbol": ""},
        {"stop_loss_pct": 1.2},
        {"horizon": "years"},
        {"thesis": "x" * 500},
        {"price": 123.4},  # fields the model must not be able to set
        {"quantity": 1000},
        {"order_type": "market"},
        {"cited_ids": ["a"] * 11},
    ],
)
def test_a_single_bad_field_invalidates_the_whole_reply(mutation: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_decisions(reply({**GOOD, **mutation}))


def test_the_wire_schema_matches_the_model() -> None:
    item = DECISIONS_JSON_SCHEMA["properties"]["decisions"]["items"]
    assert set(item["required"]) == set(item["properties"]) and item["additionalProperties"] is False
    assert set(item["properties"]) == set(
        LLMDecision.model_fields
    )  # nothing the model can send that we ignore
    assert "price" not in item["properties"] and "quantity" not in item["properties"]


# ---- strategy definition ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "change",
    [
        {"instructions": "too short"},
        {"universe": ["AAPL", "BTC-USD"]},
        {"universe": ["AAPL", "AAPL"]},
        {"universe": []},
        {"models": {"primary": "claude", "shadow": ["claude"]}},
        {"models": {"primary": "claude", "consensus": True}},  # nobody to agree with
        {"models": {"primary": "claude", "mode": "yolo"}},
        {"limits": {"min_stop_pct": 0.1, "max_stop_pct": 0.05}},
        {"limits": {"min_confidence": 0}},
        {"sizing": {"max_notional": 100, "tiers": []}},
        {"sizing": {"max_notional": 100, "tiers": [[0.9, 1], [0.8, 0.5]]}},  # not ascending
        {"sizing": {"max_notional": 100, "tiers": [[0.7, 1.5]]}},
        {"id": "Bad Id"},
        {"surprise": 1},
    ],
)
def test_invalid_llm_strategies_are_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(StrategyError):
        parse_llm_strategy({**BASE, **change})


def test_defaults_are_conservative() -> None:
    assert STRAT.models.mode == "propose_only" and not STRAT.review_on_bar_close
    assert STRAT.limits.min_confidence >= D("0.65") and STRAT.limits.max_entries_per_day <= 3


def test_strategy_files_load_by_type_and_hash_is_stable() -> None:
    llm = load_any_strategy(ROOT / "strategies/examples/news-swing-stocks.toml")
    rules = load_any_strategy(ROOT / "strategies/examples/ema-cross-btc.toml")
    assert llm.type == "llm" and isinstance(rules, StrategyDef)  # type: ignore[union-attr]
    assert llm.hash == load_any_strategy(ROOT / "strategies/examples/news-swing-stocks.toml").hash  # type: ignore[union-attr]
    assert llm.models.mode == "propose_only" and llm.models.shadow == ("grok",)  # type: ignore[union-attr]


# ---- validator rules -----------------------------------------------------------------------------------------------
def test_a_good_entry_is_accepted_with_cleaned_text() -> None:
    c = check(dec(thesis="Beat <b>estimates</b>​."))
    assert len(c.accepted) == 1 and c.rejected == []
    assert "<" not in c.accepted[0].thesis and "​" not in c.accepted[0].thesis


def test_hold_is_never_an_error_and_never_acts() -> None:
    c = check(dec(action="hold", confidence="0.1", stop_loss_pct=None, cited_ids=()))
    assert c.accepted == [] and len(c.holds) == 1 and c.rejected == []


@pytest.mark.parametrize(
    ("decision", "kwargs", "why"),
    [
        (dec(symbol="TSLA"), {}, "not in the strategy universe"),
        (dec(confidence="0.64"), {}, "below the minimum"),
        (dec(stop_loss_pct=None), {}, "needs a stop_loss_pct"),
        (dec(stop_loss_pct="0.005"), {}, "outside"),
        (dec(stop_loss_pct="0.3"), {}, "outside"),
        (dec(cited_ids=("ghost",)), {}, "fabricated"),
        (dec(cited_ids=()), {}, "at least 1"),
        (dec(), {"held": {"AAPL"}}, "already holding"),
        (dec(), {"entering": {"AAPL"}}, "already holding or entering"),
        (dec(), {"entries_today": 3}, "daily entry limit"),
        (dec(action="exit"), {}, "nothing is held"),
        (dec(action="exit", cited_ids=("ghost",)), {"held": {"AAPL"}}, "not supplied"),
        (dec(action="exit"), {"held": {"AAPL"}, "exiting": {"AAPL"}}, "already in flight"),
    ],
)
def test_each_rule_rejects_with_its_reason(decision: LLMDecision, kwargs: dict[str, Any], why: str) -> None:
    c = check(decision, **kwargs)
    assert c.accepted == [] and any(why in r for r in reasons(c)), reasons(c)


def test_exit_of_a_held_position_is_accepted() -> None:
    assert (
        len(
            check(
                dec(action="exit", confidence="0.2", stop_loss_pct=None, cited_ids=()), held={"AAPL"}
            ).accepted
        )
        == 1
    )


def test_duplicate_symbols_drop_both() -> None:
    c = check(dec(), dec(confidence="0.9"))
    assert c.accepted == [] and reasons(c) == ["duplicate decisions for one symbol"] * 2


def test_the_daily_entry_cap_counts_entries_inside_one_reply() -> None:
    capped = parse_llm_strategy({**BASE, "limits": {"max_entries_per_day": 1}})
    c = check(dec(symbol="AAPL"), dec(symbol="MSFT"), strat=capped)
    assert [d.symbol for d in c.accepted] == ["AAPL"] and any("daily entry limit" in r for r in reasons(c))


# ---- sizing -----------------------------------------------------------------------------------------------------
SIZING = LLMSizing(max_notional=D(1000))


def test_tiers_map_confidence_to_a_fraction() -> None:
    assert [tier_fraction(D(c), SIZING) for c in ("0.5", "0.65", "0.79", "0.8", "0.95")] == [
        D(0),
        D("0.25"),
        D("0.25"),
        D("0.5"),
        D(1),
    ]


def test_quantity_never_exceeds_the_cap_and_floors_to_the_lot() -> None:
    assert quantity_for(dec(confidence="0.95"), SIZING, D(100), D(1)) == 10  # $1000 / $100
    assert quantity_for(dec(confidence="0.95"), SIZING, D(300), D(1)) == 3  # floored: 3.33 -> 3
    assert quantity_for(dec(confidence="0.7"), SIZING, D(100), D(1)) == 2  # $250 / $100 = 2.5 -> 2
    assert quantity_for(dec(confidence="0.1"), SIZING, D(100), D(1)) == 0  # below every tier: no trade
    assert quantity_for(dec(confidence="0.95"), SIZING, D(0), D(1)) == 0


@settings(max_examples=200, deadline=None)
@given(
    conf=st.decimals(min_value=D(0), max_value=D(1), places=3),
    price=st.decimals(min_value=D("0.01"), max_value=D(100000), places=2),
    step=st.sampled_from([D(1), D("0.001"), D("0.00001")]),
)
def test_property_sizing_is_bounded_and_monotonic(conf, price, step) -> None:  # type: ignore[no-untyped-def]
    q = quantity_for(dec(confidence=str(conf)), SIZING, price, step)
    assert 0 <= q * price <= SIZING.max_notional
    higher = quantity_for(dec(confidence=str(min(D(1), conf + D("0.1")))), SIZING, price, step)
    assert higher >= q


# ---- property: nothing the validator accepts can break the strategy's limits ------------------------------------
symbols = st.sampled_from(["AAPL", "MSFT", "TSLA", "", "BTC-USD"])
decisions = st.builds(
    lambda s, a, c, sl, ids: LLMDecision.model_validate(
        {
            "symbol": s or "X",
            "action": a,
            "confidence": str(c),
            "stop_loss_pct": None if sl is None else str(sl),
            "cited_ids": ids,
            "thesis": "t",
        }  # fmt: skip
    ),
    symbols,
    st.sampled_from(["enter", "exit", "hold"]),
    st.decimals(min_value=D(0), max_value=D(1), places=2),
    st.one_of(st.none(), st.decimals(min_value=D("0.001"), max_value=D("0.9"), places=3)),
    st.lists(st.sampled_from(["n1", "n2", "ghost", "evil"]), max_size=4).map(tuple),
)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(decisions, max_size=6),
    st.sets(st.sampled_from(["AAPL", "MSFT"])),
    st.integers(min_value=0, max_value=5),
)
def test_property_every_accepted_decision_satisfies_every_limit(ds, held, entries_today) -> None:  # type: ignore[no-untyped-def]
    c = check(*ds, held=held, entries_today=entries_today)
    lim = STRAT.limits
    n_entries = 0
    for d in c.accepted:
        assert d.symbol in STRAT.universe
        if d.action == "enter":
            n_entries += 1
            assert d.symbol not in held and d.confidence >= lim.min_confidence
            assert d.stop_loss_pct is not None and lim.min_stop_pct <= d.stop_loss_pct <= lim.max_stop_pct
            assert d.cited_ids and set(d.cited_ids) <= {"n1", "n2"}
        else:
            assert d.action == "exit" and d.symbol in held
    assert entries_today + n_entries <= max(entries_today, lim.max_entries_per_day)
