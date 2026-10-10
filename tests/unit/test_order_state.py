from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.domain.events import (
    OrderAcked,
    OrderCancelled,
    OrderCreated,
    OrderExpired,
    OrderFilled,
    OrderRejected,
    OrderSubmitAttempted,
)
from engine.domain.types import AssetClass, Side
from engine.eventlog.base import make_event
from engine.eventlog.memory import InMemoryEventStore
from engine.orders.state import TERMINAL, IllegalTransition, OrderStatus, transition
from engine.projections.account import replay
from tests.helpers import ACCOUNT, T0, D

S = OrderStatus
ALL_EVENTS = [
    "OrderSubmitAttempted",
    "OrderAcked",
    "OrderFilled",
    "OrderRejected",
    "OrderCancelled",
    "OrderExpired",
]


def test_happy_path() -> None:
    s = transition(S.CREATED, "OrderSubmitAttempted")
    assert s is S.SUBMITTING
    s = transition(s, "OrderAcked")
    assert s is S.OPEN
    s = transition(s, "OrderFilled", fully_filled=False)
    assert s is S.PARTIAL
    s = transition(s, "OrderFilled", fully_filled=True)
    assert s is S.FILLED


def test_retry_keeps_submitting() -> None:
    assert transition(S.SUBMITTING, "OrderSubmitAttempted") is S.SUBMITTING


def test_fill_can_arrive_before_ack() -> None:
    assert transition(S.SUBMITTING, "OrderFilled", fully_filled=True) is S.FILLED


@pytest.mark.parametrize("status", sorted(TERMINAL - {S.CANCELLED, S.EXPIRED}))
def test_filled_and_rejected_are_final(status: OrderStatus) -> None:
    for e in ALL_EVENTS:
        with pytest.raises(IllegalTransition):
            transition(status, e)


@pytest.mark.parametrize("status", [S.CANCELLED, S.EXPIRED])
def test_late_fill_after_cancel_is_a_legit_race(status: OrderStatus) -> None:
    assert transition(status, "OrderFilled", fully_filled=True) is status
    with pytest.raises(IllegalTransition):
        transition(status, "OrderCancelled")


def test_cannot_ack_after_ack_or_resubmit_open_order() -> None:
    with pytest.raises(IllegalTransition):
        transition(S.OPEN, "OrderAcked")
    with pytest.raises(IllegalTransition):
        transition(S.OPEN, "OrderSubmitAttempted")


def test_unknown_event_is_illegal() -> None:
    with pytest.raises(IllegalTransition):
        transition(S.OPEN, "Bogus")


@settings(max_examples=150, deadline=None)
@given(st.lists(st.sampled_from(ALL_EVENTS), max_size=15))
def test_projection_survives_any_event_order(seq: list[str]) -> None:
    """Replay is total: out-of-order or duplicated lifecycle events become anomalies, never exceptions."""
    store = InMemoryEventStore()
    base = {"account_id": ACCOUNT, "order_id": "o1"}

    def put(model) -> None:  # type: ignore[no-untyped-def]
        store.append([make_event(model, stream_id="order:o1", ts=T0)])

    put(
        OrderCreated(
            **base,
            intent_id="i",
            strategy_id="s1",
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            side=Side.BUY,
            quantity=D(10),
            role="entry",
        )
    )
    for n, name in enumerate(seq):
        match name:
            case "OrderSubmitAttempted":
                put(OrderSubmitAttempted(**base, attempt=n + 1))
            case "OrderAcked":
                put(OrderAcked(**base, broker_order_id="b"))
            case "OrderFilled":
                put(
                    OrderFilled(
                        **base,
                        strategy_id="s1",
                        symbol="AAPL",
                        side=Side.BUY,
                        quantity=D(3),
                        price=D(100),
                        fee=D(0),
                        fill_id=f"f{n}",
                    )
                )
            case "OrderRejected":
                put(OrderRejected(**base, code="X"))
            case "OrderCancelled":
                put(OrderCancelled(**base))
            case "OrderExpired":
                put(OrderExpired(**base))
    state = replay(ACCOUNT, store.read_all())
    assert state.orders["o1"].status in set(OrderStatus)
    assert state.orders["o1"].filled_quantity >= 0
