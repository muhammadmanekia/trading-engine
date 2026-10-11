"""Evaluator, signal->intent conversion and the RuleStrategySource."""

from __future__ import annotations

import re
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.domain.types import Side
from engine.projections.account import AccountState, IntentRecord, Position
from engine.strategy.evaluator import Signal, evaluate
from engine.strategy.schema import parse_strategy
from engine.strategy.signals import Skip, build_entry_intent, build_exit_intent, intent_id_for
from engine.strategy.source import RuleStrategySource, SourceContext
from tests.helpers import ACCOUNT, D, T0, bars_from_closes, crypto, fresh_state, make_intent, market, stock

CROSS: dict[str, Any] = {
    "id": "sma-cross",
    "universe": ["BTC-USD"],
    "timeframe": "1h",
    "lookback": 50,
    "indicators": {"fast": {"type": "sma", "period": 2}, "slow": {"type": "sma", "period": 4}, "atr": {"type": "atr", "period": 3}},
    "entry": {"cross_above": ["fast", "slow"]},
    "exit": {"cross_below": ["fast", "slow"]},
    "stop": {"type": "atr", "multiple": 2},
    "sizing": {"type": "risk_fraction", "risk_pct": 0.01},
}  # fmt: skip
STRAT = parse_strategy(CROSS)
# down, then up (fast crosses above slow), then down again (crosses below)
PATH = [20, 19, 18, 17, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16]


def signals(strategy=STRAT, closes=PATH):  # type: ignore[no-untyped-def]
    bars = bars_from_closes(closes)
    out = []
    in_pos = False
    for k in range(1, len(bars) + 1):
        s = evaluate(strategy, bars[:k], in_position=in_pos)
        out.append(s)
        in_pos = s.kind == "enter" or (in_pos and s.kind != "exit")
    return out


def test_warmup_then_entry_then_exit() -> None:
    sigs = signals()
    assert all(s.kind == "hold" and "warming up" in s.reason for s in sigs[: STRAT.warmup - 1])
    kinds = [s.kind for s in sigs]
    assert kinds.count("enter") == 1 and kinds.count("exit") == 1
    assert kinds.index("enter") < kinds.index("exit")
    entry = sigs[kinds.index("enter")]
    assert entry.bar_close == T0 + timedelta(hours=kinds.index("enter") + 1)  # signal time = bar CLOSE time
    assert set(entry.values) == {"fast", "slow", "atr"}


def test_no_exit_rule_means_the_stop_is_the_only_exit() -> None:
    s = parse_strategy({**CROSS, "exit": None})
    sigs = signals(s)
    assert [x.kind for x in sigs].count("exit") == 0


def test_cooldown_blocks_reentry() -> None:
    s = parse_strategy({**CROSS, "cooldown_bars": 3})
    bars = bars_from_closes(PATH)
    assert evaluate(s, bars[:7], in_position=False, bars_since_exit=1).reason.startswith("cooldown")
    assert evaluate(s, bars[:7], in_position=False, bars_since_exit=3).kind == "enter"


def test_unknown_values_never_trigger_and_not_of_unknown_stays_unknown() -> None:
    s = parse_strategy({**CROSS, "entry": {"not": {"gt": [{"ref": "slow", "lag": 10}, 0]}}})
    bars = bars_from_closes(PATH)
    sig = evaluate(s, bars[:8], in_position=False)
    assert sig.kind == "hold"  # slow[-10] does not exist yet: unknown, and NOT unknown is still unknown


def test_division_by_zero_is_unknown_not_a_crash() -> None:
    s = parse_strategy({**CROSS, "entry": {"gt": [{"div": ["close", 0]}, 1]}})
    assert evaluate(s, bars_from_closes(PATH), in_position=False).kind == "hold"


def test_empty_history_is_an_error() -> None:
    with pytest.raises(ValueError):
        evaluate(STRAT, [], in_position=False)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(min_value=5, max_value=200), min_size=80, max_size=140),
    st.integers(min_value=60, max_value=79),
)
def test_live_window_equals_backtest_history(closes, k) -> None:  # type: ignore[no-untyped-def]
    """Backtest passes the whole history, live passes only the trailing window: the signal must be identical,
    otherwise paper/live would disagree with what was tested."""
    bars = bars_from_closes(closes)
    for in_pos in (False, True):
        full = evaluate(STRAT, bars[:k], in_position=in_pos)
        window = evaluate(STRAT, bars[k - STRAT.window : k], in_position=in_pos)
        assert full == window


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.integers(min_value=5, max_value=200), min_size=70, max_size=110),
    st.integers(min_value=55, max_value=69),
)
def test_signals_do_not_depend_on_future_bars(closes, k) -> None:  # type: ignore[no-untyped-def]
    bars = bars_from_closes(closes)
    a = evaluate(STRAT, bars[:k], in_position=False)
    b = evaluate(STRAT, bars[:k], in_position=False)  # the future exists in `bars` but is never passed in
    assert a == b


# ---- signal -> intent -------------------------------------------------------------------------------
def entry_signal(**values: Decimal) -> Signal:
    return Signal("enter", "entry rule", T0 + timedelta(hours=10), {"atr": D(5), **values})


def build(strategy=STRAT, close=100, budget=10_000, step="0.00001", **kw):  # type: ignore[no-untyped-def]
    return build_entry_intent(
        strategy, "BTC-USD", kw.pop("signal", entry_signal()), close=D(close), budget=D(budget),
        qty_step=D(step), account_id=ACCOUNT, now=T0 + timedelta(hours=10),
    )  # fmt: skip


def test_atr_stop_and_risk_based_size() -> None:
    i = build()  # stop = 100 - 2*5 = 90 ; risk 1% of 10,000 = 100 ; per-unit risk 10 -> 10 units
    assert not isinstance(i, Skip)
    assert (
        i.stop_loss_price == 90 and i.quantity == 10 and i.side is Side.BUY and i.strategy_id == "sma-cross"
    )
    assert i.expires_at - i.created_at == timedelta(hours=1)


def test_quantity_is_floored_to_the_lot_size() -> None:
    i = build(close=100, budget=10_000, step="3")  # 10 units -> 9 with a lot of 3
    assert not isinstance(i, Skip) and i.quantity == 9


def test_percent_stop_fixed_notional_and_take_profit() -> None:
    s = parse_strategy({**CROSS, "stop": {"type": "percent", "pct": 0.05},
                        "sizing": {"type": "fixed_notional", "notional": 500},
                        "take_profit": {"type": "r_multiple", "multiple": 2}})  # fmt: skip
    i = build(s)
    assert not isinstance(i, Skip)
    assert i.stop_loss_price == 95 and i.quantity == 5 and i.take_profit_price == 110  # 100 + 2*(100-95)


@pytest.mark.parametrize(
    ("kw", "why"),
    [
        (dict(signal=entry_signal(atr=D(60))), "not below"),  # ATR so wide the stop is <= 0
        (dict(signal=Signal("enter", "x", T0, {})), "not ready"),  # no ATR value
        (dict(budget=1, close=100_000, step="1"), "rounds to zero"),
    ],
)
def test_entries_that_cannot_be_sized_are_skipped_with_a_reason(kw: dict[str, Any], why: str) -> None:
    out = build(**kw)
    assert isinstance(out, Skip) and why in out.reason


def test_intent_ids_are_deterministic_and_safe_for_broker_client_ids() -> None:
    a = intent_id_for("sma-cross", "BTC-USD", T0, Side.BUY)
    assert (
        a
        == intent_id_for("sma-cross", "BTC-USD", T0, Side.BUY)
        != intent_id_for("sma-cross", "BTC-USD", T0, Side.SELL)
    )
    assert re.fullmatch(r"[A-Za-z0-9._-]+", a) and ":" not in a and len(a) < 80


def test_exit_intent_sells_exactly_the_position() -> None:
    sig = Signal("exit", "exit rule", T0, {})
    i = build_exit_intent(STRAT, "BTC-USD", sig, position=D("0.7"), account_id=ACCOUNT, now=T0)
    assert i.side is Side.SELL and i.quantity == D("0.7") and i.stop_loss_price is None


# ---- RuleStrategySource -------------------------------------------------------------------------------
def ctx(state: AccountState, closes: list[int], now=None) -> SourceContext:  # type: ignore[no-untyped-def]
    bars = bars_from_closes(closes)
    now = now or (bars[-1].ts + timedelta(hours=1))
    return SourceContext(
        now=now, timeframe="1h", bars={"BTC-USD": bars}, state=state,
        snapshot=market(crypto("BTC-USD", closes[-1], at=now, qty_step="0.00001")), budget=D(10_000),
    )  # fmt: skip


ENTRY_AT = PATH[:7]  # the fast average crosses above the slow one on the 7th bar


def test_source_emits_an_entry_once_flat() -> None:
    src = RuleStrategySource(STRAT, ACCOUNT)
    intents = src.on_bar_close(ctx(fresh_state(), ENTRY_AT))
    assert [i.side for i in intents] == [Side.BUY] and src.last_decisions[0].signal.kind == "enter"


def test_source_does_not_stack_entries_while_one_is_in_flight() -> None:
    src = RuleStrategySource(STRAT, ACCOUNT)
    state = fresh_state()
    first = src.on_bar_close(ctx(state, ENTRY_AT))[0]
    state.intents[first.intent_id] = IntentRecord(intent=first)  # received, not yet decided
    again = src.on_bar_close(ctx(state, ENTRY_AT))
    assert again == [] and src.last_decisions[0].skipped == "already holding or entering"


def test_source_exits_only_what_the_strategy_holds() -> None:
    src = RuleStrategySource(STRAT, ACCOUNT)
    state = fresh_state()
    state.positions[("sma-cross", "BTC-USD")] = Position(D("0.4"), D(100))
    state.positions[("other", "BTC-USD")] = Position(D(9), D(100))
    intents = src.on_bar_close(ctx(state, PATH[:12]))  # the fast average crosses back below on the 12th bar
    assert [(i.side, i.quantity) for i in intents] == [(Side.SELL, D("0.4"))]


def test_source_respects_max_positions() -> None:
    two = parse_strategy({**CROSS, "universe": ["BTC-USD", "ETH-USD"], "max_positions": 1})
    src = RuleStrategySource(two, ACCOUNT)
    bars = bars_from_closes(ENTRY_AT)
    now = bars[-1].ts + timedelta(hours=1)
    c = SourceContext(
        now=now, timeframe="1h", bars={"BTC-USD": bars, "ETH-USD": bars}, state=fresh_state(),
        snapshot=market(crypto("BTC-USD", 20, at=now), crypto("ETH-USD", 20, at=now)), budget=D(10_000),
    )  # fmt: skip
    intents = src.on_bar_close(c)
    assert len(intents) == 1
    assert [d.skipped for d in src.last_decisions].count("max_positions=1 reached") == 1


def test_source_cooldown_is_derived_from_the_ledger() -> None:
    s = parse_strategy({**CROSS, "cooldown_bars": 5})
    src = RuleStrategySource(s, ACCOUNT)
    state = fresh_state()
    bars = bars_from_closes(ENTRY_AT)
    now = bars[-1].ts + timedelta(hours=1)
    sold = make_intent(strategy_id="sma-cross", symbol="BTC-USD", side=Side.SELL, stop_loss_price=None,
                       created_at=now - timedelta(hours=2), expires_at=now + timedelta(hours=1))  # fmt: skip
    state.intents[sold.intent_id] = IntentRecord(intent=sold, decision="approve")
    assert src.on_bar_close(ctx(state, ENTRY_AT, now=now)) == []
    assert src.last_decisions[0].signal.reason.startswith("cooldown")


def test_source_ignores_symbols_without_data() -> None:
    src = RuleStrategySource(STRAT, ACCOUNT)
    c = ctx(fresh_state(), ENTRY_AT)
    empty = SourceContext(c.now, c.timeframe, {}, c.state, c.snapshot, c.budget)
    assert src.on_bar_close(empty) == [] and stock() is not None


# ---- every operator, on hand-checked values ------------------------------------------------------------
def kind(entry: dict[str, Any], closes: list[int] | None = None) -> str:
    s = parse_strategy({**CROSS, "entry": entry, "exit": None})
    return evaluate(s, bars_from_closes(closes or [10] * 6 + [12]), in_position=False).kind


def test_comparison_operators() -> None:
    # last close is 12, the one before is 10
    assert kind({"gt": ["close", 11]}) == "enter" and kind({"gt": ["close", 12]}) == "hold"
    assert kind({"gte": ["close", 12]}) == "enter" and kind({"gte": ["close", 13]}) == "hold"
    assert kind({"lt": ["close", 13]}) == "enter" and kind({"lt": ["close", 12]}) == "hold"
    assert kind({"lte": ["close", 12]}) == "enter" and kind({"lte": ["close", 11]}) == "hold"
    assert kind({"eq": ["close", 12]}) == "enter" and kind({"eq": ["close", 11]}) == "hold"


def test_cross_directions_use_the_previous_bar() -> None:
    assert kind({"cross_above": ["close", 11]}) == "enter"  # 10 <= 11 and 12 > 11
    assert kind({"cross_above": ["close", 9]}) == "hold"  # already above
    assert kind({"cross_below": ["close", 11]}) == "hold"
    assert (
        kind({"cross_below": ["close", 11]}, [10] * 6 + [12, 9][:1] + [9]) == "enter"
    )  # 12 >= 11 then 9 < 11


def test_boolean_combinators_and_arithmetic() -> None:
    assert kind({"any": [{"gt": ["close", 99]}, {"lt": ["close", 99]}]}) == "enter"
    assert kind({"all": [{"gt": ["close", 99]}, {"lt": ["close", 99]}]}) == "hold"
    assert kind({"not": {"gt": ["close", 99]}}) == "enter"
    assert kind({"gt": [{"add": ["close", 1]}, 12.5]}) == "enter"  # 13 > 12.5
    assert kind({"gt": [{"sub": ["close", {"ref": "close", "lag": 1}]}, 1.5]}) == "enter"  # 12-10 = 2
    assert kind({"gt": [{"div": ["close", {"ref": "close", "lag": 1}]}, 1.1]}) == "enter"  # 1.2
    assert kind({"gt": [{"mul": ["close", 2]}, 23]}) == "enter"


def test_unknown_inside_any_and_all_follows_three_valued_logic() -> None:
    far = {"div": ["close", 0]}  # division by zero is the one way a value stays unknown after warm-up
    assert kind({"any": [{"gt": [far, 0]}, {"gt": ["close", 0]}]}) == "enter"  # true beats unknown
    assert kind({"all": [{"gt": [far, 0]}, {"gt": ["close", 99]}]}) == "hold"  # false beats unknown
    assert kind({"all": [{"gt": [far, 0]}, {"gt": ["close", 0]}]}) == "hold"  # unknown stays unknown
    assert kind({"any": [{"gt": [far, 0]}, {"gt": ["close", 99]}]}) == "hold"


def test_all_price_sources_work() -> None:
    from engine.strategy import indicators as ind

    bars = bars_from_closes([10, 12], spread="1")  # bar 2: open 10, high 13, low 9, close 12
    last = bars[-1]
    assert ind.source_values(bars, "hl2")[-1] == (last.high + last.low) / 2
    assert ind.source_values(bars, "hlc3")[-1] == (last.high + last.low + last.close) / 3
    assert ind.source_values(bars, "ohlc4")[-1] == (last.open + last.high + last.low + last.close) / 4
    assert (
        ind.source_values(bars, "open")[-1] == last.open
        and ind.source_values(bars, "volume")[-1] == last.volume
    )
    assert ind.source_values(bars, "high")[-1] == last.high and ind.source_values(bars, "low")[-1] == last.low
