"""Alpaca adapter specifics beyond the shared contract."""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest
from pydantic import SecretStr

from engine.broker.alpaca import AlpacaBroker
from engine.broker.base import (
    BrokerRejected,
    BrokerStatus,
    BrokerUnavailable,
    CredentialsError,
    OrderRequest,
    OrderType,
)
from engine.clock import FakeClock
from engine.domain.types import AssetClass, Mode, Side
from tests.fakes.alpaca_server import BASE, FakeAlpaca
from tests.helpers import D


def build(key: str = "key") -> tuple[AlpacaBroker, FakeAlpaca]:
    fake = FakeAlpaca(FakeClock())
    broker = AlpacaBroker(
        SecretStr(key),
        SecretStr("secret"),
        Mode.PAPER,
        base_url=BASE,
        client=fake.client,
        sleeper=lambda _s: None,
    )
    return broker, fake


def req(cid: str = "c1", side: Side = Side.BUY, qty: str = "10", **kw: object) -> OrderRequest:
    return OrderRequest.model_validate(
        {
            "client_order_id": cid,
            "symbol": "AAPL",
            "asset_class": AssetClass.STOCK,
            "side": side,
            "quantity": D(qty),
            **kw,
        }
    )


def test_stop_only_entry_uses_an_oto_order_because_brackets_need_both_legs() -> None:
    broker, fake = build()
    broker.submit(req(stop_loss_price=90))
    body = fake.posted[0]
    assert body["order_class"] == "oto" and body["stop_loss"] == {"stop_price": "90.00"}
    assert "take_profit" not in body


def test_both_legs_make_a_bracket() -> None:
    broker, fake = build()
    broker.submit(req(stop_loss_price=90, take_profit_price=120))
    body = fake.posted[0]
    assert body["order_class"] == "bracket"
    assert body["take_profit"] == {"limit_price": "120.00"} and body["stop_loss"] == {"stop_price": "90.00"}


def test_plain_entry_is_a_simple_gtc_market_order_with_our_client_id() -> None:
    broker, fake = build()
    broker.submit(req())
    assert fake.posted[0] == {
        "symbol": "AAPL",
        "qty": "10",
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "client_order_id": "c1",
    }


def test_prices_are_quantised_to_alpaca_tick_rules() -> None:
    broker, fake = build()
    broker.submit(req(limit_price=D("99.999"), stop_loss_price=D("89.994")))
    assert fake.posted[0]["limit_price"] == "100.00" and fake.posted[0]["stop_loss"]["stop_price"] == "89.99"
    broker.submit(req("c2", limit_price=D("0.12345"), stop_loss_price=D("0.1")))
    assert fake.posted[1]["limit_price"] == "0.1235"  # sub-dollar prices keep 4 decimals


@pytest.mark.parametrize(
    ("kw", "code"),
    [
        ({"qty": "1.5"}, "FRACTIONAL_NOT_SUPPORTED"),
        ({"symbol": "BTC-USD"}, "UNSUPPORTED_SYMBOL"),
        ({"cid": "x" * 129}, "CLIENT_ID_TOO_LONG"),
    ],
)
def test_orders_alpaca_cannot_safely_take_are_rejected_locally(kw: dict[str, str], code: str) -> None:
    broker, fake = build()
    r = req(**{k: v for k, v in kw.items() if k in ("cid", "qty")})
    if "symbol" in kw:
        r = r.model_copy(update={"symbol": kw["symbol"]})
    with pytest.raises(BrokerRejected) as exc:
        broker.submit(r)
    assert exc.value.code == code and fake.requests == []  # nothing was even sent


def test_insufficient_buying_power_is_a_rejection_not_an_auth_failure() -> None:
    broker, fake = build()
    fake.mock.cash = D(50)
    fake.mock.enforce_cash = True
    with pytest.raises(BrokerRejected) as exc:
        broker.submit(req())
    assert exc.value.code == "INSUFFICIENT_FUNDS"


def test_bad_keys_are_fatal() -> None:
    broker, _ = build(key="")
    with pytest.raises(CredentialsError):
        broker.get_cash()


def test_secrets_are_not_in_repr_or_error_text() -> None:
    broker, fake = build(key="SUPER-SECRET-KEY-ID")
    assert "SUPER-SECRET" not in repr(broker.__dict__) and "SUPER-SECRET" not in repr(
        SecretStr("SUPER-SECRET")
    )
    fake.outage = True
    with pytest.raises(BrokerUnavailable) as exc:
        broker.get_cash()
    assert "SUPER-SECRET" not in str(exc.value)


def test_native_legs_get_deterministic_ids_and_can_be_cancelled_individually() -> None:
    broker, fake = build()
    broker.submit(req(stop_loss_price=90, take_profit_price=120))
    stop = broker.get_order("c1:stop")
    tp = broker.get_order("c1:tp")
    assert stop is not None and tp is not None and stop.client_order_id == "c1:stop"
    broker.cancel("c1:tp")
    after = broker.get_order("c1")
    assert after is not None
    legs = {c.client_order_id: c.status for c in after.children}
    assert legs["c1:tp"] is BrokerStatus.CANCELLED


def test_cumulative_fill_data_is_exposed_for_the_ledger() -> None:
    broker, fake = build()
    fake.mock.fill_fraction = D("0.5")
    o = broker.submit(req())
    assert o.status is BrokerStatus.PARTIALLY_FILLED and o.filled_quantity == 5 and o.avg_fill_price == 100
    assert o.fills == ()


@pytest.mark.parametrize(
    ("alpaca", "expected"),
    [
        ("new", BrokerStatus.ACCEPTED),
        ("held", BrokerStatus.ACCEPTED),
        ("pending_cancel", BrokerStatus.ACCEPTED),
        ("partially_filled", BrokerStatus.PARTIALLY_FILLED),
        ("filled", BrokerStatus.FILLED),
        ("canceled", BrokerStatus.CANCELLED),
        ("expired", BrokerStatus.EXPIRED),
        ("rejected", BrokerStatus.REJECTED),
        ("suspended", BrokerStatus.REJECTED),
        ("some_future_status", BrokerStatus.ACCEPTED),  # unknown stays 'working' so reconciliation decides
    ],
)
def test_status_mapping(alpaca: str, expected: BrokerStatus) -> None:
    broker, _ = build()
    raw = {"id": "i", "client_order_id": "c", "symbol": "AAPL", "side": "buy", "qty": "1", "status": alpaca}
    assert broker._to_order(raw).status is expected


def test_order_working_with_fills_is_reported_partial_even_if_status_lags() -> None:
    broker, _ = build()
    raw = {
        "id": "i", "client_order_id": "c", "symbol": "AAPL", "side": "buy", "qty": "10",
        "status": "new", "filled_qty": "3", "filled_avg_price": "100",
    }  # fmt: skip
    assert broker._to_order(raw).status is BrokerStatus.PARTIALLY_FILLED


def test_duplicate_client_id_with_missing_order_is_surfaced_not_swallowed() -> None:
    broker, fake = build()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(422, json={"message": "client_order_id must be unique"})
        return httpx.Response(404, json={"message": "not found"})

    broker2 = AlpacaBroker(
        SecretStr("k"), SecretStr("s"), Mode.PAPER, base_url=BASE,
        client=httpx.Client(transport=httpx.MockTransport(handler)), sleeper=lambda _s: None,
    )  # fmt: skip
    with pytest.raises(BrokerRejected) as exc:
        broker2.submit(req())
    assert exc.value.code == "DUPLICATE_CLIENT_ORDER_ID" and broker is not None


def test_positions_cash_and_clock_endpoints() -> None:
    broker, fake = build()
    broker.submit(req())
    assert broker.get_positions() == {"AAPL": D(10)}
    assert broker.get_cash() < Decimal(100_000)
    assert broker.market_clock()["is_open"] is True
    fake.market_open = False
    assert broker.market_clock()["is_open"] is False
    assert OrderType.STOP.value == "STOP"
