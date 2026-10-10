"""Broker contract suite. EVERY broker implementation (mock, paper, and the real adapters against their fake
servers) must pass these tests unchanged. The behaviours here are what OrderManager relies on."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from engine.broker.base import (
    Broker,
    BrokerOrder,
    BrokerRejected,
    BrokerStatus,
    BrokerUnavailable,
    OrderRequest,
    OrderType,
    ProtectionMode,
)
from engine.broker.mock import MockBroker, SubmitFault
from engine.broker.paper import InMemoryPaperState, PaperBroker
from engine.clock import FakeClock
from engine.domain.types import AssetClass, Side
from pydantic import SecretStr

from engine.broker.alpaca import AlpacaBroker
from engine.domain.types import Mode
from tests.fakes.alpaca_server import BASE, FakeAlpaca
from tests.fakes.robinhood_server import API_KEY, BASE as RH_BASE, FakeRobinhood, new_keypair
from engine.broker.robinhood_crypto import PairInfo, RobinhoodApi, RobinhoodCryptoBroker
from tests.helpers import D

PX = D(100)


@dataclass
class Harness:
    """Uniform test controls over very different brokers."""

    name: str
    broker: Broker
    symbol: str
    asset_class: AssetClass
    set_price: Callable[[Decimal], None]
    fail_next_submit: Callable[[SubmitFault], None]
    set_outage: Callable[[bool], None]
    cleanup: Callable[[], None] = lambda: None
    fractional: bool = False

    def req(self, cid: str = "c1", side: Side = Side.BUY, qty: int = 10, **kw: Any) -> OrderRequest:
        return OrderRequest.model_validate(
            {
                "client_order_id": cid,
                "symbol": self.symbol,
                "asset_class": self.asset_class,
                "side": side,
                "quantity": D(qty),
                **kw,
            }
        )


def _mock(name: str, **kw: Any) -> Harness:
    b = MockBroker(FakeClock(), starting_cash=D(100_000), **kw)
    b.prices["AAPL"] = PX
    return Harness(
        name, b, "AAPL", AssetClass.STOCK,
        set_price=lambda p: b.set_price("AAPL", p),
        fail_next_submit=b.fail_next_submit,
        set_outage=lambda v: setattr(b, "outage", v),
    )  # fmt: skip


def _paper() -> Harness:
    b = PaperBroker(FakeClock(), "acct", InMemoryPaperState(), starting_cash=D(100_000))
    b.prices["BTC-USD"] = PX
    return Harness(
        "paper", b, "BTC-USD", AssetClass.CRYPTO,
        set_price=lambda p: b.tick({"BTC-USD": p}),
        fail_next_submit=b.fail_next_submit,
        set_outage=lambda v: setattr(b, "outage", v),
        fractional=True,
    )  # fmt: skip


def _alpaca() -> Harness:
    fake = FakeAlpaca(FakeClock())
    broker = AlpacaBroker(
        SecretStr("key"),
        SecretStr("secret"),
        Mode.PAPER,
        base_url=BASE,
        client=fake.client,
        sleeper=lambda _s: None,
    )
    return Harness(
        "alpaca", broker, "AAPL", AssetClass.STOCK,
        set_price=lambda p: fake.set_price("AAPL", p),
        fail_next_submit=fake.fail_next_submit,
        set_outage=lambda v: setattr(fake, "outage", v),
    )  # fmt: skip


def _robinhood() -> Harness:
    clock = FakeClock()
    key, seed_b64 = new_keypair()
    fake = FakeRobinhood(clock, key)
    api = RobinhoodApi(
        SecretStr(API_KEY), SecretStr(seed_b64), base_url=RH_BASE, client=fake.client,
        time_source=lambda: clock.now().timestamp(), sleeper=lambda _s: None,
    )  # fmt: skip
    pairs = {"BTC-USD": PairInfo(asset_increment=D("0.00000001"), price_increment=D("0.01"))}
    broker = RobinhoodCryptoBroker(api, pairs, now=lambda: clock.now().timestamp())
    return Harness(
        "robinhood-fake", broker, "BTC-USD", AssetClass.CRYPTO,
        set_price=lambda p: fake.set_price("BTC-USD", p),
        fail_next_submit=fake.fail_next_submit,
        set_outage=lambda v: setattr(fake, "outage", v),
        fractional=True,
    )  # fmt: skip


FACTORIES: dict[str, Callable[[], Harness]] = {
    "robinhood-fake": _robinhood,
    "alpaca-fake": _alpaca,
    "mock-native": lambda: _mock("mock-native"),
    "mock-separate": lambda: _mock("mock-separate", protection=ProtectionMode.SEPARATE_ORDERS),
    "mock-cumulative": lambda: _mock("mock-cumulative", cumulative_only=True),
    "paper": _paper,
}


@pytest.fixture(params=sorted(FACTORIES))
def h(request: pytest.FixtureRequest) -> Iterator[Harness]:
    harness = FACTORIES[request.param]()
    try:
        yield harness
    finally:
        harness.cleanup()


def filled_qty(o: BrokerOrder) -> Decimal:
    if o.fills:
        return sum((f.quantity for f in o.fills), Decimal(0))
    return o.filled_quantity or Decimal(0)


# ---- submit / get ------------------------------------------------------------------------------
def test_market_buy_fills_completely(h: Harness) -> None:
    o = h.broker.submit(h.req())
    assert o.status is BrokerStatus.FILLED and filled_qty(o) == 10
    got = h.broker.get_order("c1")
    assert got is not None and got.status is BrokerStatus.FILLED and filled_qty(got) == 10


def test_get_order_unknown_is_none(h: Harness) -> None:
    assert h.broker.get_order("never-submitted") is None


def test_submit_is_idempotent_on_client_order_id(h: Harness) -> None:
    first = h.broker.submit(h.req())
    again = h.broker.submit(h.req())
    assert first.broker_order_id == again.broker_order_id
    assert h.broker.get_positions() == {h.symbol: D(10)}


def test_fill_ids_are_stable_across_polls(h: Harness) -> None:
    h.broker.submit(h.req())
    a, b = h.broker.get_order("c1"), h.broker.get_order("c1")
    assert a is not None and b is not None
    assert [f.fill_id for f in a.fills] == [f.fill_id for f in b.fills]
    assert len({f.fill_id for f in a.fills}) == len(a.fills)


def test_cannot_sell_what_is_not_held(h: Harness) -> None:
    with pytest.raises(BrokerRejected):
        h.broker.submit(h.req(side=Side.SELL))
    assert h.broker.get_order("c1") is None


def test_injected_reject_places_nothing(h: Harness) -> None:
    h.fail_next_submit("reject")
    with pytest.raises(BrokerRejected):
        h.broker.submit(h.req())
    assert h.broker.get_order("c1") is None
    assert h.broker.submit(h.req()).status is BrokerStatus.FILLED  # the fault was one-shot


def test_timeout_before_placement_leaves_no_order(h: Harness) -> None:
    h.fail_next_submit("unavailable_before")
    with pytest.raises(BrokerUnavailable):
        h.broker.submit(h.req())
    assert h.broker.get_order("c1") is None
    assert h.broker.submit(h.req()).status is BrokerStatus.FILLED


def test_timeout_after_placement_is_discoverable(h: Harness) -> None:
    h.fail_next_submit("unavailable_after")
    with pytest.raises(BrokerUnavailable):
        h.broker.submit(h.req())
    got = h.broker.get_order("c1")
    assert got is not None and got.status is BrokerStatus.FILLED
    assert h.broker.submit(h.req()).broker_order_id == got.broker_order_id  # retry never duplicates


# ---- account state -----------------------------------------------------------------------------
def test_positions_and_cash_follow_fills(h: Harness) -> None:
    before = h.broker.get_cash()
    h.broker.submit(h.req())
    assert h.broker.get_positions() == {h.symbol: D(10)}
    assert before - h.broker.get_cash() >= D(1000)  # notional plus any fees
    h.broker.submit(h.req("c2", Side.SELL, 10))
    assert h.broker.get_positions() == {}


def test_list_open_orders_shows_working_orders_only(h: Harness) -> None:
    h.broker.submit(h.req("done"))
    h.broker.submit(h.req("resting", limit_price=90))
    open_ids = {o.client_order_id for o in h.broker.list_open_orders()}
    assert "resting" in open_ids and "done" not in open_ids


def test_cancel_resting_order(h: Harness) -> None:
    h.broker.submit(h.req("resting", limit_price=90))
    h.broker.cancel("resting")
    got = h.broker.get_order("resting")
    assert got is not None and got.status is BrokerStatus.CANCELLED
    h.broker.cancel("resting")  # already terminal: no-op
    h.broker.cancel("unknown")  # unknown: no-op


def test_limit_order_fills_when_price_crosses(h: Harness) -> None:
    o = h.broker.submit(h.req("lim", limit_price=95))
    assert o.status is BrokerStatus.ACCEPTED
    h.set_price(D(94))
    got = h.broker.get_order("lim")
    assert got is not None and got.status is BrokerStatus.FILLED


# ---- protection --------------------------------------------------------------------------------
def test_standalone_stop_sell_triggers_on_price_drop(h: Harness) -> None:
    h.broker.submit(h.req("entry"))
    h.broker.submit(h.req("stop", Side.SELL, 10, order_type=OrderType.STOP, stop_price=90))
    resting = h.broker.get_order("stop")
    assert resting is not None and resting.status is BrokerStatus.ACCEPTED
    h.set_price(D(85))
    got = h.broker.get_order("stop")
    assert got is not None and got.status is BrokerStatus.FILLED
    assert h.broker.get_positions() == {}


def test_two_sells_cannot_reserve_the_same_shares(h: Harness) -> None:
    h.broker.submit(h.req("entry"))
    h.broker.submit(h.req("stop", Side.SELL, 10, order_type=OrderType.STOP, stop_price=90))
    with pytest.raises(BrokerRejected):
        h.broker.submit(h.req("exit", Side.SELL, 10))
    h.broker.cancel("stop")
    assert h.broker.submit(h.req("exit", Side.SELL, 10)).status is BrokerStatus.FILLED


def test_native_bracket_children_protect_the_entry(h: Harness) -> None:
    if h.broker.protection is not ProtectionMode.NATIVE_BRACKET:
        pytest.skip("broker has no native brackets")
    o = h.broker.submit(h.req(stop_loss_price=90, take_profit_price=120))
    assert {c.client_order_id for c in o.children} == {"c1:stop", "c1:tp"}
    h.set_price(D(85))
    got = h.broker.get_order("c1")
    assert got is not None
    legs = {c.client_order_id: c for c in got.children}
    assert legs["c1:stop"].status is BrokerStatus.FILLED
    assert legs["c1:tp"].status is BrokerStatus.CANCELLED
    assert h.broker.get_positions() == {}


# ---- failure behaviour -------------------------------------------------------------------------
def test_outage_raises_unavailable_for_every_call(h: Harness) -> None:
    h.broker.submit(h.req())
    h.set_outage(True)
    for call in (
        lambda: h.broker.submit(h.req("c9")),
        lambda: h.broker.get_order("c1"),
        lambda: h.broker.cancel("c1"),
        h.broker.get_positions,
        h.broker.get_cash,
        h.broker.list_open_orders,
    ):
        with pytest.raises(BrokerUnavailable):
            call()
    assert not h.broker.healthy()  # health is learned from failed calls
    h.set_outage(False)
    assert h.broker.get_order("c1") is not None
    assert h.broker.healthy()
