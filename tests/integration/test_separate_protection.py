"""SEPARATE_ORDERS protection (Robinhood-style: no brackets) and the protection invariant."""

from __future__ import annotations

import pytest

from engine.broker.base import OrderRequest, OrderType, ProtectionMode
from engine.broker.mock import MockBroker
from engine.clock import FakeClock
from engine.core import TradingCore
from engine.eventlog.base import EventStore
from engine.orders.state import OrderStatus as S
from tests.helpers import Crasher, D, SimulatedCrash, build_core, intent_now


def separate(clock: FakeClock, **kw: object) -> MockBroker:
    return MockBroker(clock, protection=ProtectionMode.SEPARATE_ORDERS, **kw)  # type: ignore[arg-type]


def setup(store: EventStore, **kw: object) -> tuple[TradingCore, MockBroker, FakeClock]:
    clock = FakeClock()
    return build_core(store, clock, separate(clock, **kw))


def legs(core: TradingCore, entry_id: str) -> list[str]:
    return sorted(
        o.order_id for o in core.projector.refresh().orders.values() if o.parent_order_id == entry_id
    )


def test_entry_fill_places_a_standalone_stop_for_the_filled_quantity(store: EventStore) -> None:
    core, broker, clock = setup(store)
    oid = core.submit_intent(intent_now(clock)).order_id
    assert oid
    st = core.projector.refresh()
    leg = st.orders[f"{oid}:stop:0"]
    assert leg.managed and leg.status is S.OPEN and leg.quantity == 10 and leg.stop_price == 90
    assert broker.get_order(leg.order_id) is not None
    assert broker.placed_orders == [oid, leg.order_id]  # two independent orders, no native children


def test_stop_triggers_and_books_the_loss(store: EventStore) -> None:
    core, broker, clock = setup(store)
    oid = core.submit_intent(intent_now(clock)).order_id
    broker.set_price("AAPL", D(85))
    core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[f"{oid}:stop:0"].status is S.FILLED
    assert st.position_qty("s1", "AAPL") == 0 == broker.positions["AAPL"]
    assert st.realized_total < 0 and st.consecutive_losses["s1"] == 1 and st.anomalies == []


def test_take_profit_is_rejected_because_two_sells_cannot_share_the_same_shares(store: EventStore) -> None:
    core, broker, clock = setup(store)
    res = core.submit_intent(intent_now(clock, take_profit_price=120))
    st = core.projector.refresh()
    assert res.order_id and st.orders[res.order_id].reject_code == "TP_NOT_SUPPORTED"
    assert broker.placed_orders == []


def test_partial_fills_are_protected_incrementally(store: EventStore) -> None:
    core, broker, clock = setup(store, fill_fraction=D("0.4"))
    oid = core.submit_intent(intent_now(clock)).order_id
    assert oid
    st = core.projector.refresh()
    assert st.position_qty("s1", "AAPL") == 4 and st.orders[f"{oid}:stop:0"].quantity == 4
    broker.step()
    core.manager.sync()
    broker.step()
    core.manager.sync()
    st = core.projector.refresh()
    stops = [o for o in st.orders.values() if o.role == "stop"]
    assert sum(o.quantity for o in stops) == st.position_qty("s1", "AAPL") == 10
    assert all(o.status is S.OPEN for o in stops) and len(stops) == 3
    core.manager.sync()  # idempotent: no extra legs
    assert len([o for o in core.projector.refresh().orders.values() if o.role == "stop"]) == 3


def test_rejected_stop_flattens_the_position_and_trips_the_strategy(store: EventStore) -> None:
    core, broker, clock = setup(store)
    broker.reject_rules.append(lambda r: "BAD_STOP" if r.kind is OrderType.STOP else None)
    oid = core.submit_intent(intent_now(clock)).order_id
    core.manager.sync()
    st = core.projector.refresh()
    assert st.orders[f"{oid}:stop:0"].status is S.REJECTED
    assert st.orders[f"{oid}:stop:0:flatten"].status is S.FILLED
    assert st.position_qty("s1", "AAPL") == 0 == broker.positions["AAPL"]
    assert "s1" in st.halted_strategies
    assert any("PROTECTION_FAILED" in a for a in st.anomalies)
    assert len([o for o in st.orders.values() if o.role == "stop"]) == 1  # no retry storm
    blocked = core.submit_intent(intent_now(clock))
    assert blocked.decision is not None and blocked.decision.code == "HALTED_STRATEGY"


def test_failed_flatten_is_flagged_loudly(store: EventStore) -> None:
    core, broker, clock = setup(store)
    broker.reject_rules.append(lambda r: "NOPE" if r.side.value == "SELL" else None)
    core.submit_intent(intent_now(clock))
    core.manager.sync()
    st = core.projector.refresh()
    assert any("FLATTEN_FAILED" in a for a in st.anomalies)


def test_exit_that_fails_leaves_the_position_protected_again(store: EventStore) -> None:
    core, broker, clock = setup(store)
    oid = core.submit_intent(intent_now(clock)).order_id
    broker.fail_next_submit("reject")
    ex = core.submit_intent(intent_now(clock, side="SELL", stop_loss_price=None)).order_id
    st = core.projector.refresh()
    assert ex and st.orders[ex].status is S.REJECTED
    assert st.orders[f"{oid}:stop:0"].status is S.CANCELLED  # released to free the shares
    assert st.orders[f"{oid}:stop:1"].status is S.OPEN and st.orders[f"{oid}:stop:1"].quantity == 10
    broker.set_price("AAPL", D(80))
    core.manager.sync()
    assert core.projector.refresh().position_qty("s1", "AAPL") == 0  # the replacement stop works


def test_successful_exit_frees_the_stop_and_does_not_recreate_it(store: EventStore) -> None:
    core, broker, clock = setup(store)
    oid = core.submit_intent(intent_now(clock)).order_id
    broker.prices["AAPL"] = D(105)
    ex = core.submit_intent(intent_now(clock, side="SELL", stop_loss_price=None)).order_id
    core.manager.sync()
    st = core.projector.refresh()
    assert ex and st.orders[ex].status is S.FILLED
    assert st.orders[f"{oid}:stop:0"].status is S.CANCELLED
    assert [o for o in st.orders.values() if o.role == "stop" and not o.is_terminal] == []
    assert st.position_qty("s1", "AAPL") == 0


@pytest.mark.parametrize("point", ["after_fills", "after_leg_created", "after_ack"])
def test_crash_around_protection_is_repaired_exactly_once(store: EventStore, point: str) -> None:
    clock = FakeClock()
    broker = separate(clock)
    core, _, _ = build_core(store, clock, broker, hook=Crasher(point))
    i = intent_now(clock)
    try:
        core.submit_intent(i)
    except SimulatedCrash:
        pass
    survivor, _, _ = build_core(store, clock, broker, fund=False)
    survivor.recover()
    survivor.recover()
    st = survivor.projector.refresh()
    stops = [o for o in st.orders.values() if o.role == "stop"]
    assert len(stops) == 1 and stops[0].status is S.OPEN and stops[0].quantity == 10
    assert sum(1 for cid in broker.placed_orders if cid.endswith(":stop:0")) == 1
    assert st.anomalies == []
    store.verify_chain()


def test_cumulative_fill_reporting_matches_explicit_fills(store: EventStore) -> None:
    """Alpaca-style brokers give only cumulative qty and average price; the ledger must still be exact."""
    clock = FakeClock()
    broker = MockBroker(clock, fill_fraction=D("0.4"), cumulative_only=True)
    core, broker, clock = build_core(store, clock, broker)
    oid = core.submit_intent(intent_now(clock)).order_id
    assert oid
    broker.set_price("AAPL", D(110))  # later slices fill at a different price
    broker.step()
    broker.set_price("AAPL", D(120))
    broker.step()
    core.manager.sync()
    st = core.projector.refresh()
    ref = MockBroker(clock, fill_fraction=D("0.4"))
    assert st.orders[oid].status is S.FILLED and st.orders[oid].filled_quantity == 10
    assert st.position_qty("s1", "AAPL") == broker.positions["AAPL"] == 10
    expected_notional = sum(f.quantity * f.price for f in broker._orders[oid].fills)
    assert abs(st.orders[oid].filled_notional - expected_notional) < D("0.0000001")
    assert st.anomalies == [] and ref is not None
    core.manager.sync()  # re-polling the same cumulative state adds nothing
    assert core.projector.refresh().orders[oid].filled_quantity == 10


def test_order_request_kind_defaults() -> None:
    base = dict(client_order_id="c", symbol="X", asset_class="STOCK", side="BUY", quantity=1)
    assert OrderRequest.model_validate(base).kind is OrderType.MARKET
    assert OrderRequest.model_validate({**base, "limit_price": 5}).kind is OrderType.LIMIT
    assert OrderRequest.model_validate({**base, "order_type": "STOP", "stop_price": 3}).kind is OrderType.STOP
