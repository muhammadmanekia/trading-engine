"""MockBroker behaviour. Phase 2 real adapters must satisfy the same semantics (see broker/base.py)."""

from __future__ import annotations

import pytest

from engine.broker.base import BrokerRejected, BrokerStatus, BrokerUnavailable, OrderRequest
from engine.broker.mock import MockBroker
from engine.clock import FakeClock
from engine.domain.types import AssetClass, Side
from tests.helpers import D


def req(cid: str = "c1", side: Side = Side.BUY, qty: int = 10, **kw: object) -> OrderRequest:
    base: dict[str, object] = {
        "client_order_id": cid,
        "symbol": "AAPL",
        "asset_class": AssetClass.STOCK,
        "side": side,
        "quantity": D(qty),
    }
    base.update(kw)
    return OrderRequest.model_validate(base)


def broker(**kw: object) -> MockBroker:
    b = MockBroker(FakeClock(), **kw)  # type: ignore[arg-type]
    b.prices["AAPL"] = D(100)
    return b


def test_market_buy_fills_with_fee() -> None:
    b = broker()
    o = b.submit(req())
    assert o.status is BrokerStatus.FILLED
    assert o.fills[0].price == 100 and o.fills[0].fee == D("1.000")
    assert b.positions["AAPL"] == 10


def test_submit_is_idempotent_on_client_order_id() -> None:
    b = broker()
    first = b.submit(req())
    again = b.submit(req())
    assert first.broker_order_id == again.broker_order_id
    assert len(b.placed_orders) == 1 and b.positions["AAPL"] == 10


def test_cannot_sell_what_is_not_held() -> None:
    with pytest.raises(BrokerRejected, match="INSUFFICIENT_POSITION"):
        broker().submit(req(side=Side.SELL))


def test_unknown_symbol_rejected() -> None:
    with pytest.raises(BrokerRejected, match="NO_PRICE"):
        broker().submit(req().model_copy(update={"symbol": "NOPE"}))


def test_get_order_unknown_is_none() -> None:
    assert broker().get_order("nope") is None


def test_limit_buy_rests_until_price_falls() -> None:
    b = broker()
    o = b.submit(req(limit_price=95))
    assert o.status is BrokerStatus.ACCEPTED and not o.fills
    b.set_price("AAPL", D(94))
    got = b.get_order("c1")
    assert got is not None and got.status is BrokerStatus.FILLED and got.fills[0].price == 94


def test_partial_fills_progress_each_step() -> None:
    b = broker(fill_fraction=D("0.4"))
    o = b.submit(req())
    assert o.status is BrokerStatus.PARTIALLY_FILLED and o.fills[0].quantity == 4
    b.step()
    b.step()
    got = b.get_order("c1")
    assert got is not None and got.status is BrokerStatus.FILLED
    assert sum(f.quantity for f in got.fills) == 10
    assert len({f.fill_id for f in got.fills}) == len(got.fills)


def test_bracket_legs_activate_after_entry_fills_and_are_oco() -> None:
    b = broker()
    o = b.submit(req(stop_loss_price=90, take_profit_price=120))
    assert {c.client_order_id for c in o.children} == {"c1:stop", "c1:tp"}
    b.set_price("AAPL", D(121))
    got = b.get_order("c1")
    assert got is not None
    legs = {c.client_order_id: c for c in got.children}
    assert legs["c1:tp"].status is BrokerStatus.FILLED
    assert legs["c1:stop"].status is BrokerStatus.CANCELLED
    assert b.positions["AAPL"] == 0


def test_stop_triggers_on_price_drop() -> None:
    b = broker()
    b.submit(req(stop_loss_price=90))
    b.set_price("AAPL", D(89))
    child = b.get_order("c1:stop")
    assert child is not None and child.status is BrokerStatus.FILLED and child.fills[0].price == 89


def test_gap_through_stop_on_activation_fires_immediately() -> None:
    b = broker(fill_fraction=D("0.5"))
    b.submit(req(stop_loss_price=90))  # half filled: legs inactive
    b.prices["AAPL"] = D(80)  # gap down without a step
    b.step()  # second half fills at 80, legs activate and fire at once
    child = b.get_order("c1:stop")
    assert child is not None and child.status is BrokerStatus.FILLED
    assert b.positions["AAPL"] == 0


def test_cancel_after_partial_protects_only_the_filled_quantity() -> None:
    b = broker(fill_fraction=D("0.5"))
    b.submit(req(stop_loss_price=90))
    b.cancel("c1")
    stop = b.get_order("c1:stop")
    assert stop is not None and stop.quantity == 5
    b.set_price("AAPL", D(85))
    assert b.positions["AAPL"] == 0


def test_cancel_with_no_fills_cancels_legs() -> None:
    b = broker()
    b.submit(req(limit_price=90, stop_loss_price=80))
    b.cancel("c1")
    stop = b.get_order("c1:stop")
    assert stop is not None and stop.status is BrokerStatus.CANCELLED


def test_outage_raises_unavailable_everywhere() -> None:
    b = broker()
    b.outage = True
    assert not b.healthy()
    with pytest.raises(BrokerUnavailable):
        b.submit(req())
    with pytest.raises(BrokerUnavailable):
        b.get_order("c1")
    with pytest.raises(BrokerUnavailable):
        b.cancel("c1")
    b.outage = False
    assert b.get_order("c1") is None


def test_injected_reject_places_nothing() -> None:
    b = broker()
    b.fail_next_submit("reject")
    with pytest.raises(BrokerRejected):
        b.submit(req())
    assert b.placed_orders == []
    assert b.submit(req()).status is BrokerStatus.FILLED  # fault is one-shot


def test_unavailable_before_places_nothing_unavailable_after_places_one() -> None:
    b = broker()
    b.fail_next_submit("unavailable_before")
    with pytest.raises(BrokerUnavailable):
        b.submit(req("a"))
    assert b.get_order("a") is None
    b.fail_next_submit("unavailable_after")
    with pytest.raises(BrokerUnavailable):
        b.submit(req("b"))
    got = b.get_order("b")
    assert got is not None and got.status is BrokerStatus.FILLED


def test_duplicate_fill_delivery_has_stable_fill_ids() -> None:
    b = broker()
    b.duplicate_fills = True
    o = b.submit(req())
    assert len(o.fills) == 2 and o.fills[0].fill_id == o.fills[1].fill_id
