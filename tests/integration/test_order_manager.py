"""Order manager behaviour against the mock broker, on both event stores."""

from __future__ import annotations

from engine.broker.base import BrokerFill
from engine.broker.mock import MockBroker
from engine.clock import FakeClock
from engine.core import TradingCore
from engine.eventlog.base import EventStore
from engine.orders.state import OrderStatus as S
from tests.helpers import D, build_core, intent_now


def place(core: TradingCore, clock: FakeClock, **kw: object) -> str:
    res = core.submit_intent(intent_now(clock, **kw))
    assert res.decision is not None and res.decision.allowed, res.decision
    assert res.order_id
    return res.order_id


def test_entry_fills_and_stop_leg_enters_the_ledger(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    oid = place(core, clock)
    st = core.projector.refresh()
    assert st.orders[oid].status is S.FILLED
    assert st.position_qty("s1", "AAPL") == 10
    assert st.cash == D(10_000) - 1000 - D("1.000")
    assert st.orders[f"{oid}:stop"].status is S.OPEN
    assert st.anomalies == []


def test_stop_loss_triggers_and_books_the_loss(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    oid = place(core, clock)
    broker.set_price("AAPL", D(85))
    core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[f"{oid}:stop"].status is S.FILLED
    assert st.position_qty("s1", "AAPL") == 0
    assert st.realized_total < 0 and st.consecutive_losses["s1"] == 1
    assert st.anomalies == []


def test_take_profit_fills_and_cancels_the_stop(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    oid = place(core, clock, take_profit_price=120)
    broker.set_price("AAPL", D(125))
    core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[f"{oid}:tp"].status is S.FILLED
    assert st.orders[f"{oid}:stop"].status is S.CANCELLED
    assert st.realized_total > 0 and st.position_qty("s1", "AAPL") == 0


def test_partial_fills_complete_over_several_syncs(store: EventStore) -> None:
    broker = MockBroker(FakeClock(), fill_fraction=D("0.4"))
    core, broker, clock = build_core(store, FakeClock(), broker)
    broker.clock = clock
    oid = place(core, clock)
    st = core.projector.refresh()
    assert st.orders[oid].status is S.PARTIAL and st.position_qty("s1", "AAPL") == 4
    assert f"{oid}:stop" not in st.orders  # legs appear once the entry size is final
    broker.step()
    broker.step()
    core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[oid].status is S.FILLED and st.position_qty("s1", "AAPL") == 10
    assert st.orders[f"{oid}:stop"].quantity == 10


def test_cancel_after_partial_fill_keeps_and_protects_the_filled_part(store: EventStore) -> None:
    broker = MockBroker(FakeClock(), fill_fraction=D("0.4"))
    core, broker, clock = build_core(store, FakeClock(), broker)
    oid = place(core, clock)
    assert core.manager.cancel(oid)
    st = core.projector.refresh()
    assert st.orders[oid].status is S.CANCELLED
    assert st.position_qty("s1", "AAPL") == 4
    assert st.orders[f"{oid}:stop"].quantity == 4 and st.orders[f"{oid}:stop"].status is S.OPEN


def test_resting_limit_order_can_be_cancelled(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    oid = place(core, clock, limit_price=95)
    st = core.projector.refresh()
    assert st.orders[oid].status is S.OPEN and st.position_qty("s1", "AAPL") == 0
    assert core.manager.cancel(oid)
    assert core.projector.refresh().orders[oid].status is S.CANCELLED
    assert not core.manager.cancel(oid)  # already terminal


def test_broker_rejection_is_recorded_and_nothing_is_held(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    broker.fail_next_submit("reject")
    oid = place(core, clock)
    st = core.projector.refresh()
    assert st.orders[oid].status is S.REJECTED and st.orders[oid].reject_code == "MOCK_REJECT"
    assert st.position_qty("s1", "AAPL") == 0 and st.cash == 10_000


def test_ambiguous_timeout_after_placement_never_double_places(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    broker.fail_next_submit("unavailable_after")
    oid = place(core, clock)
    assert core.projector.refresh().orders[oid].status is S.SUBMITTING
    core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[oid].status is S.FILLED
    assert broker.placed_orders == [oid] and broker.submit_calls == 1


def test_timeout_before_placement_is_resubmitted_once(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    broker.fail_next_submit("unavailable_before")
    oid = place(core, clock)
    assert broker.placed_orders == []
    core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[oid].status is S.FILLED and st.orders[oid].attempts == 2
    assert broker.placed_orders == [oid]


def test_outage_changes_nothing_and_recovers(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    broker.fail_next_submit("unavailable_before")
    oid = place(core, clock)
    broker.outage = True
    core.manager.sync()
    core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[oid].status is S.SUBMITTING and st.orders[oid].attempts == 1 and st.anomalies == []
    broker.outage = False
    core.manager.sync()
    assert core.projector.refresh().orders[oid].status is S.FILLED


def test_gives_up_only_after_the_broker_definitively_has_no_order(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    broker.fail_next_submit("unavailable_before", times=10)
    oid = place(core, clock)
    for _ in range(6):
        core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[oid].status is S.REJECTED and st.orders[oid].reject_code == "SUBMIT_FAILED"
    assert broker.placed_orders == [] and st.orders[oid].attempts == 5


def test_duplicate_fill_delivery_is_idempotent(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    broker.duplicate_fills = True
    place(core, clock)
    for _ in range(3):
        core.manager.sync()
    st = core.projector.refresh()
    assert st.position_qty("s1", "AAPL") == 10 and st.cash == D(10_000) - 1000 - D("1.000")
    assert st.anomalies == []


def test_overfill_is_flagged_and_not_booked(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    oid = place(core, clock)
    broker._orders[oid].fills.append(
        BrokerFill(fill_id="rogue", quantity=D(5), price=D(100), fee=D(0), ts=clock.now())
    )
    core.manager.sync()
    st = core.projector.refresh()
    assert st.position_qty("s1", "AAPL") == 10
    assert any("OVERFILL" in a for a in st.anomalies)


def test_unprotectable_entry_is_rejected_when_broker_has_no_brackets(store: EventStore) -> None:
    broker = MockBroker(FakeClock(), supports_protective_orders=False)
    core, broker, clock = build_core(store, FakeClock(), broker)
    oid = place(core, clock)
    st = core.projector.refresh()
    assert st.orders[oid].status is S.REJECTED and st.orders[oid].reject_code == "NO_PROTECTION"
    assert broker.placed_orders == [] and broker.submit_calls == 0


def test_placing_the_same_intent_twice_creates_one_order(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    i = intent_now(clock)
    core.submit_intent(i)
    core.manager.place(i, D(10))
    core.manager.place(i, D(10))
    assert len([o for o in core.projector.refresh().orders.values() if o.role == "entry"]) == 1
    assert len(broker.placed_orders) == 1 and core.state.position_qty("s1", "AAPL") == 10


def test_manual_exit_cancels_protective_legs(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    oid = place(core, clock, take_profit_price=120)
    broker.prices["AAPL"] = D(105)
    ex = place(core, clock, side="SELL", stop_loss_price=None)
    st = core.projector.refresh()
    assert st.orders[ex].status is S.FILLED and st.position_qty("s1", "AAPL") == 0
    assert st.orders[f"{oid}:stop"].status is S.CANCELLED
    assert st.orders[f"{oid}:tp"].status is S.CANCELLED
    broker.set_price("AAPL", D(80))  # a stale stop must not fire against a closed position
    core.manager.sync()
    assert core.projector.refresh().position_qty("s1", "AAPL") == 0
