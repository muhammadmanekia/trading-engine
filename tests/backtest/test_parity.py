"""THE guarantee of Phase 3: the strategy that was backtested is the strategy that trades.

The same TOML strategy runs (a) through the BacktestEngine and (b) through the whole live stack (executor,
StrategyRunner, risk engine, order manager, broker adapter) fed the same bars one at a time. The trades must be
identical: same intent ids, quantities, prices, holding periods and exit reasons.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import pytest
from pydantic import SecretStr

from engine.backtest.engine import BacktestConfig, BacktestEngine, build_trades
from engine.backtest.venue import ALPACA_STOCKS
from engine.clock import FakeClock
from engine.eventlog.base import EventStore, decode
from engine.eventlog.memory import InMemoryEventStore
from engine.marketdata.alpaca import AlpacaBars
from engine.marketdata.history import Bar
from engine.risk.config import disposable_money_preset
from engine.service.alerts import RecordingAlertSink
from engine.service.executor import Executor
from engine.service.inbox import InMemoryInbox
from engine.service.leader import InProcessLeaderLock
from engine.service.runtime import attach_rules
from engine.strategy.schema import parse_strategy
from engine.strategy.source import RuleStrategySource
from tests.fakes.alpaca_server import BASE
from tests.helpers import D, T0, alpaca_runtime

FREE = ALPACA_STOCKS.with_costs(spread_bps=D(0), slippage_bps=D(0), fee_rate=D(0), adv_usd=D(10**9))
SPEC: dict[str, Any] = {
    "id": "wave",
    "universe": ["AAPL"],
    "timeframe": "1h",
    "lookback": 60,
    "indicators": {"fast": {"type": "sma", "period": 3}, "slow": {"type": "sma", "period": 9}},
    "entry": {"cross_above": ["fast", "slow"]},
    "exit": {"cross_below": ["fast", "slow"]},
    "stop": {"type": "percent", "pct": 0.5},  # far away: this test is about signals and fills, not stops
    "sizing": {"type": "fixed_quantity", "quantity": 100},
}


@pytest.fixture(autouse=True)
def _release_locks() -> object:
    yield
    InProcessLeaderLock._held.clear()


def wavy_bars(n: int = 150) -> list[Bar]:
    """A wave with opening gaps (-1, 0, +1) so the next open is NOT the previous close."""
    closes = [100 + round(12 * math.sin(i / 7)) for i in range(n)]
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        o = prev + (i % 3) - 1
        out.append(
            Bar(
                ts=T0 + timedelta(hours=i),
                open=D(o),
                high=D(max(o, c) + 1),
                low=D(min(o, c) - 1),
                close=D(c),
                volume=D(10_000),
            )
        )
        prev = c
    return out


def backtest(bars: list[Bar]):  # type: ignore[no-untyped-def]
    cfg = BacktestConfig(
        venue=FREE, starting_cash=D(1_000_000), risk=disposable_money_preset(), account_id="acct"
    )
    strategy = parse_strategy(SPEC)
    return BacktestEngine([RuleStrategySource(strategy, "acct")], {"AAPL": bars}, cfg).run()


def live(bars: list[Bar], store: EventStore | None = None, stop_at: int | None = None):  # type: ignore[no-untyped-def]
    """Drive the real service bar by bar: only bars that have closed are visible, the market sits at the next open."""
    store = store or InMemoryEventStore()
    clock = FakeClock(T0)
    rt, fake, clock = alpaca_runtime(store, clock)
    fake.mock.fee_rate = D(0)
    history = AlpacaBars(
        SecretStr("k"), SecretStr("s"), clock, base_url=BASE, client=fake.client, sleeper=lambda _s: None
    )
    attach_rules(rt, [parse_strategy(SPEC)], history, clock, min_interval=timedelta(0))
    ex = Executor(store, [rt], InMemoryInbox(), InProcessLeaderLock("parity"), RecordingAlertSink(), clock)
    assert ex.start()
    for k in range((stop_at or len(bars)) - 1):
        clock.set(bars[k].ts + timedelta(hours=1, seconds=30))  # bar k has just closed
        fake.history["AAPL"] = bars[: k + 1]
        fake.mock.prices["AAPL"] = bars[k + 1].open  # orders placed now execute at the next bar's open
        assert ex.tick()
    ex.stop()  # release the leader lock so the next scenario can start its own service
    return store, rt, fake


def fingerprint(trades):  # type: ignore[no-untyped-def]
    return [(t.quantity, t.entry_price, t.exit_price, t.pnl, t.bars_held, t.exit_reason) for t in trades]


def intent_ids(events) -> list[str]:  # type: ignore[no-untyped-def]
    return [decode(e).intent.intent_id for e in events if e.type == "IntentReceived"]  # type: ignore[attr-defined]


def test_live_trades_are_exactly_the_backtested_trades() -> None:
    bars = wavy_bars()
    bt = backtest(bars)
    store, rt, fake = live(bars)
    live_events = store.read_all()
    live_trades = build_trades(live_events, "1h")

    assert len(bt.trades) >= 3, "the scenario must actually trade or the comparison proves nothing"
    assert fingerprint(live_trades) == fingerprint(bt.trades)
    assert intent_ids(live_events) == intent_ids(bt.events)  # even the intent ids are identical
    held_live = rt.core.projector.refresh().position_qty("wave", "AAPL")
    assert held_live == sum((p for p in bt.open_positions.values()), D(0))
    assert rt.core.projector.refresh().anomalies == []


def test_the_run_actually_exercises_entries_exits_and_the_risk_engine() -> None:
    bt = backtest(wavy_bars())
    assert bt.signals["enter"] >= 3 and bt.signals["exit"] >= 3 and bt.risk_decisions["APPROVED"] >= 6
    assert all(t.exit_reason == "rule" for t in bt.trades)


def test_each_closed_bar_is_evaluated_exactly_once() -> None:
    bars = wavy_bars()
    store, rt, fake = live(bars, stop_at=60)
    ids = intent_ids(store.read_all())
    assert len(ids) == len(set(ids))
    n_before = len(ids)
    for _ in range(3):  # ticking again with no new bar must not re-submit anything
        rt.runner.tick()  # type: ignore[union-attr]
    assert len(intent_ids(store.read_all())) == n_before


def test_a_restart_in_the_middle_changes_nothing() -> None:
    bars = wavy_bars()
    store, rt, fake = live(bars, stop_at=70)
    cut = len(store.read_all())
    # a brand-new process over the same log and the same broker; it re-reads history and re-evaluates the last bar
    clock = rt.core.clock
    rt2, _, _ = alpaca_runtime(store, clock, fake=fake)  # type: ignore[arg-type]
    history = AlpacaBars(
        SecretStr("k"), SecretStr("s"), clock, base_url=BASE, client=fake.client, sleeper=lambda _s: None
    )
    attach_rules(rt2, [parse_strategy(SPEC)], history, clock, min_interval=timedelta(0))
    ex2 = Executor(store, [rt2], InMemoryInbox(), InProcessLeaderLock("parity"), RecordingAlertSink(), clock)
    InProcessLeaderLock._held.clear()
    assert ex2.start()
    ex2.tick()
    ids = intent_ids(store.read_all())
    assert len(ids) == len(set(ids)) and len(store.read_all()) >= cut
    assert fake.mock.get_positions().get("AAPL", D(0)) == rt2.core.projector.refresh().position_qty(
        "wave", "AAPL"
    )


def test_stocks_are_not_evaluated_while_the_market_is_closed_and_catch_up_at_the_open() -> None:
    bars = wavy_bars()
    store = InMemoryEventStore()
    clock = FakeClock(T0)
    rt, fake, clock = alpaca_runtime(store, clock)
    history = AlpacaBars(
        SecretStr("k"), SecretStr("s"), clock, base_url=BASE, client=fake.client, sleeper=lambda _s: None
    )
    attach_rules(
        rt,
        [parse_strategy({**SPEC, "entry": {"gt": ["close", 0]}})],
        history,
        clock,
        min_interval=timedelta(0),
    )
    ex = Executor(store, [rt], InMemoryInbox(), InProcessLeaderLock("closed"), RecordingAlertSink(), clock)
    ex.start()
    fake.history["AAPL"] = bars[:30]
    clock.set(bars[29].ts + timedelta(hours=1, seconds=30))
    fake.mock.prices["AAPL"] = bars[30].open
    fake.market_open = False
    ex.tick()
    assert intent_ids(store.read_all()) == []  # closed: nothing proposed
    fake.market_open = True
    clock.advance(seconds=60)
    ex.tick()
    assert len(intent_ids(store.read_all())) == 1  # the open: the same last closed bar is acted on, once
    assert intent_ids(store.read_all())[0].startswith("wave.AAPL.")


def test_a_data_outage_is_retried_not_fatal() -> None:
    bars = wavy_bars()
    store = InMemoryEventStore()
    clock = FakeClock(T0)
    rt, fake, clock = alpaca_runtime(store, clock)
    history = AlpacaBars(
        SecretStr("k"), SecretStr("s"), clock, base_url=BASE, client=fake.client, sleeper=lambda _s: None
    )
    attach_rules(
        rt,
        [parse_strategy({**SPEC, "entry": {"gt": ["close", 0]}})],
        history,
        clock,
        min_interval=timedelta(0),
    )
    ex = Executor(store, [rt], InMemoryInbox(), InProcessLeaderLock("outage"), RecordingAlertSink(), clock)
    ex.start()
    fake.history["AAPL"] = bars[:30]
    clock.set(bars[29].ts + timedelta(hours=1, seconds=30))
    fake.mock.prices["AAPL"] = bars[30].open
    fake.outage = True
    assert ex.tick() and not rt.fatal
    fake.outage = False
    ex.tick()
    assert len(intent_ids(store.read_all())) == 1


def test_a_strategy_the_venue_cannot_hold_is_refused_at_attach_time() -> None:
    from engine.strategy.schema import StrategyError

    store = InMemoryEventStore()
    rt, fake, clock = alpaca_runtime(store)
    crypto = parse_strategy(
        {**SPEC, "universe": ["BTC-USD"], "take_profit": {"type": "r_multiple", "multiple": 2}}
    )
    with pytest.raises(StrategyError):  # a stock account cannot run a crypto strategy
        attach_rules(rt, [crypto], None, clock)  # type: ignore[arg-type]
