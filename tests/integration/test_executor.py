"""Execution service: end to end from the inbox to a filled, protected, reconciled position."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from engine.clock import FakeClock
from engine.eventlog.base import EventStore
from engine.orders.state import OrderStatus as S
from engine.service.alerts import Level, RecordingAlertSink
from engine.service.executor import Executor, Intervals
from engine.service.inbox import InMemoryInbox
from engine.service.leader import InProcessLeaderLock
from engine.service.runtime import AccountRuntime
from tests.fakes.alpaca_server import FakeAlpaca
from tests.helpers import D, alpaca_runtime, intent_payload, paper_crypto_runtime


def make(
    store: EventStore, *, key: str = "key", lock_name: str = "exec-test", **kw: Any
) -> tuple[
    Executor, AccountRuntime, FakeAlpaca, FakeClock, InMemoryInbox, RecordingAlertSink, InProcessLeaderLock
]:
    rt, fake, clock = alpaca_runtime(store, key=key)
    inbox, alerts, lock = InMemoryInbox(), RecordingAlertSink(), InProcessLeaderLock(lock_name)
    ex = Executor(store, [rt], inbox, lock, alerts, clock, **kw)  # type: ignore[arg-type]
    return ex, rt, fake, clock, inbox, alerts, lock


@pytest.fixture(autouse=True)
def _release_locks() -> object:
    yield
    InProcessLeaderLock._held.clear()


def test_intent_flows_from_inbox_to_a_protected_filled_position(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store)
    assert ex.start()
    inbox.put(intent_payload(clock))
    assert ex.tick()
    st = rt.core.projector.refresh()
    entry = next(o for o in st.orders.values() if o.role == "entry")
    assert entry.status is S.FILLED and st.position_qty("s1", "AAPL") == 10
    stop = st.orders[f"{entry.order_id}:stop"]
    assert stop.status is S.OPEN  # native bracket leg is in the ledger and live at the broker
    assert fake.mock.positions["AAPL"] == 10
    titles = [a.title for a in alerts.alerts]
    assert "Execution service started" in titles and any(t.startswith("Fill: BUY 10 AAPL") for t in titles)
    assert ex.health()["ok"] is True


def test_the_stop_fires_through_the_service_and_the_reconciler_stays_quiet(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(
        store, intervals=Intervals(sync=timedelta(seconds=60), reconcile=timedelta(0))
    )
    ex.start()
    inbox.put(intent_payload(clock))
    ex.tick()
    fake.set_price("AAPL", D(85))
    ex.tick()
    ex.tick()
    st = rt.core.projector.refresh()
    assert st.position_qty("s1", "AAPL") == 0 == fake.mock.positions["AAPL"]
    assert st.realized_total < 0 and not st.kill_switch
    assert [e for e in store.read_all() if e.type == "DriftDetected"] == []


def test_rejected_intents_are_decided_once_and_never_reach_the_broker(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store)
    ex.start()
    inbox.put(intent_payload(clock, stop_loss_price=None))
    ex.tick()
    ex.tick()
    decisions = [e.payload for e in store.read_all() if e.type == "RiskDecisionMade"]
    assert len(decisions) == 1 and decisions[0]["code"] == "NO_STOP_LOSS"
    assert fake.mock.placed_orders == []


def test_bad_inbox_rows_raise_a_warning_but_never_stop_the_service(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store)
    ex.start()
    bad = intent_payload(clock)
    bad["quantity"] = 1.5
    inbox.put(bad)
    inbox.put(intent_payload(clock, account_id="other-account"))
    inbox.put(intent_payload(clock))
    ex.tick()
    assert sum(1 for a in alerts.alerts if a.title == "Bad intent in inbox") == 2
    assert fake.mock.positions["AAPL"] == 10  # the good one still traded


def test_only_one_service_can_lead(store: EventStore) -> None:
    ex1, *_ = make(store, lock_name="same")
    rt2, _, clock2 = alpaca_runtime(store)
    ex2 = Executor(store, [rt2], InMemoryInbox(), InProcessLeaderLock("same"), RecordingAlertSink(), clock2)
    assert ex1.start() and not ex2.start()


def test_losing_the_lock_stops_the_service_immediately(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, lock = make(store)
    ex.start()
    lock.lose_on_next_check = True
    inbox.put(intent_payload(clock))
    assert ex.tick() is False
    assert fake.mock.placed_orders == [] and not ex.health()["leader"]  # no trading after losing leadership
    assert any(a.level is Level.CRITICAL and "lost leadership" in a.title for a in alerts.alerts)


def test_bad_credentials_halt_the_account_and_are_never_retried(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store, key="")  # the fake answers 401
    ex.start()
    assert rt.fatal and rt.core.projector.refresh().kill_switch
    n = len(fake.requests)
    inbox.put(intent_payload(clock))
    ex.tick()
    ex.tick()
    assert len(fake.requests) == n  # a fatal account makes no further broker calls
    assert any(a.level is Level.CRITICAL and "trading halted" in a.title for a in alerts.alerts)
    assert ex.health()["ok"] is False


def test_an_outage_never_crashes_the_loop_and_recovers(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store)
    ex.start()
    fake.outage = True
    for _ in range(4):
        assert ex.tick()
        clock.advance(seconds=2)
    warnings = [a for a in alerts.alerts if "unreachable" in a.title]
    assert len(warnings) == 1  # once per outage, not once per tick
    assert not rt.fatal and not rt.core.projector.refresh().kill_switch
    fake.outage = False
    inbox.put(intent_payload(clock))
    ex.tick()
    assert fake.mock.positions["AAPL"] == 10 and rt.consecutive_errors == 0


def test_intent_during_a_data_outage_fails_closed(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store)
    ex.start()
    fake.outage = True
    inbox.put(intent_payload(clock))
    ex.tick()
    codes = [e.payload["code"] for e in store.read_all() if e.type == "RiskDecisionMade"]
    assert codes == ["PROVIDER_ERROR"] and fake.mock.placed_orders == []


def test_runaway_unexpected_errors_halt_the_account(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store, max_consecutive_errors=3)
    ex.start()

    def boom() -> None:
        raise RuntimeError("bug")

    rt.core.process_pending = boom  # type: ignore[method-assign]
    for _ in range(3):
        ex.tick()
    assert rt.fatal and rt.core.projector.refresh().kill_switch
    assert any("consecutive unexpected errors" in a.body for a in alerts.alerts)


def test_restart_resumes_without_duplicating_orders(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, lock = make(store)
    ex.start()
    inbox.put(intent_payload(clock))
    ex.tick()
    ex.stop()
    rt2, _, _ = alpaca_runtime(store, clock, fake=fake)  # a new process, same log, same broker
    ex2 = Executor(store, [rt2], inbox, InProcessLeaderLock("exec-test"), RecordingAlertSink(), clock)
    assert ex2.start()
    ex2.tick()
    assert len(fake.mock.placed_orders) == 1 and fake.mock.positions["AAPL"] == 10
    assert rt2.core.projector.refresh().anomalies == []


def test_fills_alerts_can_be_turned_off(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store, alert_on_fills=False)
    ex.start()
    inbox.put(intent_payload(clock))
    ex.tick()
    assert not any(a.title.startswith("Fill:") for a in alerts.alerts)


def test_health_reflects_staleness(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make(store)
    assert ex.health()["ok"] is False  # not leader yet
    ex.start()
    ex.tick()
    assert ex.health()["ok"] is True
    clock.advance(minutes=5)
    assert ex.health()["ok"] is False and ex.health()["accounts"]["acct"]["stale"] is True


# ---- crypto paper trading: simulator + live (fake) Robinhood quotes --------------------------------
def crypto_intent(clock: FakeClock, **kw: object) -> dict[str, object]:
    base = dict(symbol="BTC-USD", asset_class="CRYPTO", quantity=D("0.5"), stop_loss_price=90)
    return intent_payload(clock, **{**base, **kw})


def make_crypto(store: EventStore):  # type: ignore[no-untyped-def]
    rt, fake, clock, paper_state = paper_crypto_runtime(store)
    inbox, alerts = InMemoryInbox(), RecordingAlertSink()
    ex = Executor(store, [rt], inbox, InProcessLeaderLock("crypto"), alerts, clock)
    return ex, rt, fake, clock, inbox, alerts, paper_state


def test_crypto_paper_trade_end_to_end_with_a_separate_stop(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make_crypto(store)
    assert ex.start()
    assert rt.core.projector.refresh().cash == 10_000  # paper starting cash, funded once
    inbox.put(crypto_intent(clock))
    ex.tick()
    st = rt.core.projector.refresh()
    entry = next(o for o in st.orders.values() if o.role == "entry")
    assert entry.status is S.FILLED and st.position_qty("s1", "BTC-USD") == D("0.5")
    stop = st.orders[f"{entry.order_id}:stop:0"]
    assert stop.managed and stop.status is S.OPEN and stop.stop_price == 90
    assert fake.mock.placed_orders == []  # paper mode never touches the real exchange, only reads quotes
    fake.set_price("BTC-USD", D(85))  # the live market falls through the stop
    clock.advance(seconds=6)
    ex.tick()
    ex.tick()
    st = rt.core.projector.refresh()
    assert st.position_qty("s1", "BTC-USD") == 0 and st.realized_total < 0 and st.anomalies == []
    assert not st.kill_switch and [e for e in store.read_all() if e.type == "DriftDetected"] == []


def test_crypto_outside_the_allowlist_never_trades(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make_crypto(store)
    ex.start()
    inbox.put(crypto_intent(clock, symbol="PEPE-USD"))
    ex.tick()
    codes = [e.payload["code"] for e in store.read_all() if e.type == "RiskDecisionMade"]
    assert codes == ["PROVIDER_ERROR"] and rt.paper is not None and rt.paper.placed_orders == []


def test_crypto_take_profit_is_refused_because_robinhood_cannot_hold_both_exits(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make_crypto(store)
    ex.start()
    inbox.put(crypto_intent(clock, take_profit_price=120))
    ex.tick()
    st = rt.core.projector.refresh()
    assert [o.reject_code for o in st.orders.values()] == ["TP_NOT_SUPPORTED"]


def test_crypto_paper_state_survives_a_full_restart(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, paper_state = make_crypto(store)
    ex.start()
    inbox.put(crypto_intent(clock))
    ex.tick()
    ex.stop()
    rt2, _, _, _ = paper_crypto_runtime(store, clock, fake=fake, paper_state=paper_state)  # brand-new process
    ex2 = Executor(store, [rt2], inbox, InProcessLeaderLock("crypto"), RecordingAlertSink(), clock)
    assert ex2.start()
    ex2.tick()
    st = rt2.core.projector.refresh()
    assert st.position_qty("s1", "BTC-USD") == D("0.5") == rt2.paper.positions["BTC-USD"]  # type: ignore[union-attr]
    assert st.anomalies == [] and len(rt2.paper.placed_orders) == 2  # type: ignore[union-attr]  # entry + stop leg
    assert rt2.reconciler.run().ok


def test_paper_enforces_the_same_order_rules_as_the_live_adapter(store: EventStore) -> None:
    ex, rt, fake, clock, inbox, alerts, _ = make_crypto(store)
    ex.start()
    inbox.put(crypto_intent(clock, quantity=D("0.000015")))  # not a multiple of the 0.00001 increment
    ex.tick()
    st = rt.core.projector.refresh()
    assert [o.reject_code for o in st.orders.values()] in (["BAD_INCREMENT"], [])  # risk may floor it first


# ---- quote recording (feeds honest backtest costs) -----------------------------------------------
def test_the_executor_records_quotes_at_most_once_per_interval(store: EventStore) -> None:
    from engine.marketdata.recorder import InMemoryQuoteRecorder

    recorder = InMemoryQuoteRecorder(interval=timedelta(seconds=60))
    ex, rt, fake, clock, inbox, alerts, _ = make(store, recorder=recorder)
    ex.start()
    for _ in range(5):
        ex.tick()
        clock.advance(seconds=10)  # 50s in total: still inside one recording interval
    assert len(recorder.rows) == 1 and recorder.rows[0][2] == "AAPL"
    clock.advance(seconds=30)
    ex.tick()
    assert len(recorder.rows) == 2


def test_a_broken_recorder_never_disturbs_trading(store: EventStore) -> None:
    class Broken:
        def due(self, account_id: str, now: object) -> bool:
            return True

        def store(self, account_id: str, snapshot: object, now: object) -> int:
            raise RuntimeError("disk full")

    ex, rt, fake, clock, inbox, alerts, _ = make(store, recorder=Broken())
    ex.start()
    inbox.put(intent_payload(clock))
    assert ex.tick() and not rt.fatal
    assert fake.mock.positions["AAPL"] == 10  # the trade still happened


def test_recorded_spreads_become_a_measured_backtest_cost(pg_dsn: str) -> None:
    from datetime import datetime, timezone

    from engine.domain.market import MarketSnapshot
    from engine.marketdata.recorder import PostgresQuoteRecorder, recorded_spread_bps
    from tests.helpers import crypto

    rec = PostgresQuoteRecorder(pg_dsn, interval=timedelta(0))
    t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
    for i, bps in enumerate([180, 190, 200, 210, 220] * 8):  # 40 samples, median 200
        snap = MarketSnapshot(
            as_of=t0, instruments={"BTC-USD": crypto("BTC-USD", 60000, est_slippage_bps=D(bps))}
        )
        rec.store("acct", snap, t0 + timedelta(minutes=i))
    assert recorded_spread_bps(pg_dsn, "BTC-USD") == D(200)
    assert recorded_spread_bps(pg_dsn, "ETH-USD") is None  # nothing recorded
    assert (
        recorded_spread_bps(pg_dsn, "BTC-USD", min_samples=500) is None
    )  # too little data to claim a number
    rec.close()
