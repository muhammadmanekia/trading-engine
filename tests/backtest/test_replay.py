"""Replay: re-simulating a real run's intents should reproduce its fills; differences are the tracking error."""

from __future__ import annotations

from datetime import timedelta

from engine.backtest.engine import BacktestConfig
from engine.backtest.replay import (
    ReplaySource,
    actual_fills,
    intents_from_events,
    replay_account,
    starting_cash,
)
from engine.risk.config import disposable_money_preset
from tests.backtest.test_parity import FREE, live, wavy_bars
from tests.helpers import D, T0, make_intent

CFG = BacktestConfig(
    venue=FREE, starting_cash=D(1_000_000), risk=disposable_money_preset(), account_id="replay"
)


def test_replaying_a_real_run_on_the_same_bars_reproduces_every_fill() -> None:
    bars = wavy_bars()
    store, rt, fake = live(bars)
    events = store.read_all()
    cfg = BacktestConfig(
        venue=FREE,
        starting_cash=starting_cash(events, "acct"),
        risk=disposable_money_preset(),
        account_id="acct",
    )
    report = replay_account(events, "acct", {"AAPL": bars}, cfg, "1h")
    assert len(report.comparisons) >= 6 and report.mismatches == []
    assert (
        report.mean_abs_error_bps == 0 and report.max_abs_error_bps == 0
    )  # a perfect model of a perfect world
    assert report.warnings == []
    real = actual_fills(events, "acct")
    assert all(
        c.actual_qty == real[f"ord-{c.intent_id}"][0] == c.simulated_qty
        for c in report.comparisons
        if c.actual_qty
    )


def test_a_pessimistic_model_shows_up_as_price_error_not_as_a_mismatch() -> None:
    bars = wavy_bars()
    store, _, _ = live(bars)
    events = store.read_all()
    costly = BacktestConfig(
        venue=FREE.with_costs(spread_bps=D(100)),  # the model charges 50 bps per side that reality did not
        starting_cash=starting_cash(events, "acct"), risk=disposable_money_preset(), account_id="acct",
    )  # fmt: skip
    report = replay_account(events, "acct", {"AAPL": bars}, costly, "1h")
    assert report.mismatches == [] and report.mean_abs_error_bps == D(50)
    buys = [c for c in report.comparisons if c.side == "BUY" and c.price_error_bps is not None]
    sells = [c for c in report.comparisons if c.side == "SELL" and c.price_error_bps is not None]
    assert all(c.price_error_bps > 0 for c in buys) and all(c.price_error_bps < 0 for c in sells)  # type: ignore[operator]


def test_a_model_that_refuses_what_reality_did_is_reported_as_a_mismatch() -> None:
    bars = wavy_bars()
    store, _, _ = live(bars)
    events = store.read_all()
    strict = BacktestConfig(
        venue=FREE, starting_cash=starting_cash(events, "acct"), account_id="acct",
        risk=disposable_money_preset().model_copy(update={"max_position_pct": D("0.0001")}),
    )  # fmt: skip
    report = replay_account(events, "acct", {"AAPL": bars}, strict, "1h")
    assert report.mismatches and any("only one of reality and the simulation" in w for w in report.warnings)


def test_no_intents_is_a_message_not_a_crash() -> None:
    from engine.eventlog.memory import InMemoryEventStore

    r = replay_account([], "acct", {"AAPL": wavy_bars(10)}, CFG, "1h")
    assert r.comparisons == [] and r.warnings and isinstance(InMemoryEventStore().read_all(), list)


def test_the_replay_source_emits_each_intent_at_the_bar_close_just_before_it() -> None:
    i1 = make_intent(
        intent_id="a", created_at=T0 + timedelta(hours=3, seconds=30), expires_at=T0 + timedelta(hours=9)
    )
    src = ReplaySource([i1], "1h")
    from tests.helpers import fresh_state, market, stock
    from engine.strategy.source import SourceContext

    def ctx(hours: int) -> SourceContext:
        return SourceContext(T0 + timedelta(hours=hours), "1h", {}, fresh_state(), market(stock()), D(1))

    assert src.on_bar_close(ctx(2)) == [] and src.on_bar_close(ctx(4)) == []
    (out,) = src.on_bar_close(ctx(3))
    assert (
        out.intent_id == "a"
        and out.created_at == T0 + timedelta(hours=3)
        and out.expires_at == T0 + timedelta(hours=4)
    )
    assert len(intents_from_events([], "acct")) == 0
