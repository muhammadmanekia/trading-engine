"""Crash matrix: kill the 'process' at every step boundary, restart from the log, check invariants.

Invariants after recovery, for every crash point and broker fault:
  * the broker holds exactly one order for the intent (never zero-then-two, never duplicates)
  * the ledger agrees with the broker (position, cash, order status)
  * the hash chain verifies and there are no anomalies
  * a redelivered intent is a no-op
"""

from __future__ import annotations

import pytest

from engine.broker.mock import MockBroker, SubmitFault
from engine.clock import FakeClock
from engine.core import TradingCore
from engine.eventlog.base import EventStore
from engine.orders.state import OrderStatus as S
from engine.projections.account import replay
from tests.helpers import ACCOUNT, Crasher, D, SimulatedCrash, build_core, intent_now

POINTS = [
    "after_received",
    "after_decision",
    "after_created",
    "after_attempt",
    "after_broker_submit",
    "after_ack",
]
FAULTS: list[SubmitFault | None] = [None, "unavailable_before", "unavailable_after"]


def restart(store: EventStore, broker: MockBroker, clock: FakeClock, hook=None) -> TradingCore:  # type: ignore[no-untyped-def]
    """A brand-new process: nothing in memory except what the log and the broker hold."""
    core, _, _ = build_core(store, clock, broker, fund=False, hook=hook)
    return core


def converge(core: TradingCore, rounds: int = 3) -> None:
    """Recovery plus the periodic syncs the execution service would run. A still-queued broker fault may
    consume the first retry; the system must converge, never diverge."""
    core.recover()
    for _ in range(rounds - 1):
        core.manager.sync()


def assert_consistent(core: TradingCore, broker: MockBroker, store: EventStore, intent_id: str) -> None:
    st = core.projector.refresh()
    oid = f"ord-{intent_id}"
    assert broker.placed_orders == [oid], broker.placed_orders
    assert st.orders[oid].status is S.FILLED
    assert st.position_qty("s1", "AAPL") == broker.positions["AAPL"] == 10
    assert st.cash == D(10_000) - 1000 - D("1.000")
    assert st.anomalies == []
    store.verify_chain()
    # replaying the whole log from scratch gives identical economics
    fresh = replay(ACCOUNT, store.read_all(account_id=ACCOUNT))
    assert (fresh.cash, dict(fresh.positions)) == (st.cash, dict(st.positions))
    # only one OrderCreated for the entry, ever
    created = [e for e in store.read_all() if e.type == "OrderCreated" and e.payload["order_id"] == oid]
    assert len(created) == 1
    # redelivery of the same intent does nothing
    again = core.submit_intent(core.state.intents[intent_id].intent)
    assert again.duplicate and len(broker.placed_orders) == 1


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("point", POINTS)
def test_single_crash_then_recover(store: EventStore, point: str, fault: SubmitFault | None) -> None:
    crasher = Crasher(point)
    core, broker, clock = build_core(store, hook=crasher)
    if fault:
        broker.fail_next_submit(fault)
    intent = intent_now(clock)
    try:
        core.submit_intent(intent)
    except SimulatedCrash:
        pass
    # the crash must actually have happened where we said (otherwise the test proves nothing)
    # 'after_ack' is only reachable when the broker call returned normally
    if not crasher.fired:
        assert point in ("after_broker_submit", "after_ack") and fault in (
            "unavailable_before",
            "unavailable_after",
        )

    survivor = restart(store, broker, clock)
    converge(survivor)
    assert_consistent(survivor, broker, store, intent.intent_id)


@pytest.mark.parametrize("first", POINTS)
@pytest.mark.parametrize("second", POINTS)
def test_crash_during_recovery_then_recover_again(store: EventStore, first: str, second: str) -> None:
    core, broker, clock = build_core(store, hook=Crasher(first))
    intent = intent_now(clock)
    try:
        core.submit_intent(intent)
    except SimulatedCrash:
        pass
    mid = restart(store, broker, clock, hook=Crasher(second))
    try:
        mid.recover()
    except SimulatedCrash:
        pass
    final = restart(store, broker, clock)
    converge(final)
    assert_consistent(final, broker, store, intent.intent_id)


def test_recover_is_idempotent(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    intent = intent_now(clock)
    core.submit_intent(intent)
    for _ in range(3):
        restart(store, broker, clock).recover()
    final = restart(store, broker, clock)
    final.recover()
    assert_consistent(final, broker, store, intent.intent_id)


def test_crash_between_stop_fill_and_ledger_is_repaired(store: EventStore) -> None:
    """The stop fills at the broker while the engine is down; recovery books it exactly once."""
    core, broker, clock = build_core(store)
    intent = intent_now(clock)
    core.submit_intent(intent)
    broker.set_price("AAPL", D(85))  # stop fires while we are 'offline'
    survivor = restart(store, broker, clock)
    survivor.recover()
    survivor.recover()
    st = survivor.projector.refresh()
    assert st.position_qty("s1", "AAPL") == 0 == broker.positions["AAPL"]
    assert st.realized_total < 0 and st.anomalies == []
    store.verify_chain()


def test_approved_but_never_ordered_intent_expires_instead_of_trading_late(store: EventStore) -> None:
    core, broker, clock = build_core(store, hook=Crasher("after_decision"))
    intent = intent_now(clock)
    try:
        core.submit_intent(intent)
    except SimulatedCrash:
        pass
    clock.advance(hours=3)  # down for a long time; the opportunity has passed
    survivor = restart(store, broker, clock)
    survivor.recover()
    assert broker.placed_orders == []
    assert survivor.projector.refresh().intents[intent.intent_id].abandoned
    store.verify_chain()


def test_undecided_intent_is_reevaluated_on_recovery(store: EventStore) -> None:
    core, broker, clock = build_core(store, hook=Crasher("after_received"))
    intent = intent_now(clock)
    try:
        core.submit_intent(intent)
    except SimulatedCrash:
        pass
    assert core.projector.refresh().intents[intent.intent_id].decision is None
    survivor = restart(store, broker, clock)
    survivor.recover()
    assert_consistent(survivor, broker, store, intent.intent_id)
