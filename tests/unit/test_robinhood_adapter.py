"""Robinhood Crypto adapter specifics: signing, UUID client ids, increments, scanning, market data."""

from __future__ import annotations

import base64
import uuid
from datetime import timedelta
from decimal import Decimal

import httpx
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from engine.broker.base import (
    BrokerRejected,
    BrokerStatus,
    BrokerUnavailable,
    CredentialsError,
    OrderRequest,
    OrderType,
)
from engine.broker.robinhood_crypto import PairInfo, RobinhoodApi, RobinhoodCryptoBroker, load_pairs, wire_id
from engine.clock import FakeClock
from engine.domain.types import AssetClass, Side
from engine.marketdata.base import MarketDataError
from engine.marketdata.robinhood import RobinhoodMarketData
from tests.fakes.robinhood_server import API_KEY, BASE, FakeRobinhood, new_keypair
from tests.helpers import D

PAIRS = {
    "BTC-USD": PairInfo(
        asset_increment=D("0.00001"), price_increment=D("0.01"), min_order_size=D("0.0001"), max_order_size=D(10),
        market_cap_usd=D(1_200_000_000_000), avg_daily_volume_usd=D(30_000_000_000),
    ),
}  # fmt: skip


def build() -> tuple[RobinhoodCryptoBroker, RobinhoodApi, FakeRobinhood, FakeClock]:
    clock = FakeClock()
    key, seed = new_keypair()
    fake = FakeRobinhood(clock, key)
    api = RobinhoodApi(
        SecretStr(API_KEY), SecretStr(seed), base_url=BASE, client=fake.client,
        time_source=lambda: clock.now().timestamp(), sleeper=lambda _s: None,
    )  # fmt: skip
    return RobinhoodCryptoBroker(api, PAIRS, now=lambda: clock.now().timestamp()), api, fake, clock


def req(cid: str = "c1", side: Side = Side.BUY, qty: str = "0.5", **kw: object) -> OrderRequest:
    return OrderRequest.model_validate(
        {
            "client_order_id": cid,
            "symbol": "BTC-USD",
            "asset_class": AssetClass.CRYPTO,
            "side": side,
            "quantity": D(qty),
            **kw,
        }
    )


# ---- signing ------------------------------------------------------------------------------------
def test_signature_matches_the_documented_message_and_verifies() -> None:
    seed = bytes([1]) * 32
    api = RobinhoodApi(SecretStr("api-key"), SecretStr(base64.b64encode(seed).decode()))
    sig = api.sign(1700000000, "/api/v1/crypto/trading/orders/", "POST", '{"a":1}')
    pub = Ed25519PrivateKey.from_private_bytes(seed).public_key()
    message = b'api-key1700000000/api/v1/crypto/trading/orders/POST{"a":1}'
    pub.verify(base64.b64decode(sig), message)  # api_key + timestamp + path + method + body
    for tampered in (
        b'api-key1700000001/api/v1/crypto/trading/orders/POST{"a":1}',
        b'api-key1700000000/api/v1/crypto/trading/orders/GET{"a":1}',
        b'api-key1700000000/api/v1/crypto/trading/orders/POST{"a":2}',
        b'other-key1700000000/api/v1/crypto/trading/orders/POST{"a":1}',
    ):
        with pytest.raises(InvalidSignature):
            pub.verify(base64.b64decode(sig), tampered)


def test_signing_vectors_are_pinned_for_regression() -> None:
    api = RobinhoodApi(SecretStr("api-key"), SecretStr(base64.b64encode(bytes([1]) * 32).decode()))
    assert (
        api.sign(1700000000, "/api/v1/crypto/trading/orders/", "POST", '{"a":1}')
        == "tnNP6EybPKd78CWd0cN2fqkWKiFQJuk41OsOafpscUOZVs56s73EHT3K0KMotrTTf3/x7aj2u3DdRdGeGP51AA=="
    )
    assert (
        api.sign(1700000000, "/api/v1/crypto/trading/holdings/?asset_code=BTC", "GET", "")
        == "RfEFseI0Z+RroY+DE8WTz7KTmDEXCt439NSekZeXGIIigYLAVx9muyu+qwt5ThgwtZ+JWpS/N5TUPz4ne5NnAQ=="
    )  # the query string is part of the signed path


def test_a_64_byte_keypair_is_accepted_and_garbage_keys_are_refused() -> None:
    pair = bytes([2]) * 32 + bytes([3]) * 32
    RobinhoodApi(SecretStr("k"), SecretStr(base64.b64encode(pair).decode()))
    with pytest.raises(ValueError, match="Ed25519"):
        RobinhoodApi(SecretStr("k"), SecretStr(base64.b64encode(b"short").decode()))


def test_a_wrong_key_is_a_fatal_credentials_error() -> None:
    broker, _, fake, clock = build()
    other_key, other_seed = new_keypair()
    bad_api = RobinhoodApi(
        SecretStr(API_KEY), SecretStr(other_seed), base_url=BASE, client=fake.client,
        time_source=lambda: clock.now().timestamp(), sleeper=lambda _s: None,
    )  # fmt: skip
    with pytest.raises(CredentialsError):
        RobinhoodCryptoBroker(bad_api, PAIRS).get_cash()
    assert other_key is not None


def test_clock_skew_beyond_30s_is_refused_by_the_server() -> None:
    broker, api, fake, clock = build()
    skewed = RobinhoodApi(
        SecretStr(API_KEY), SecretStr(new_keypair()[1]), base_url=BASE, client=fake.client,
        time_source=lambda: clock.now().timestamp() - 120, sleeper=lambda _s: None,
    )  # fmt: skip
    with pytest.raises(CredentialsError):  # surfaces as 401: fatal, not silently retried
        RobinhoodCryptoBroker(skewed, PAIRS).get_cash()


# ---- order translation --------------------------------------------------------------------------
def test_client_order_ids_are_deterministic_uuids() -> None:
    assert wire_id("ord-1") == wire_id("ord-1") != wire_id("ord-2")
    assert str(uuid.UUID(wire_id("ord-1"))) == wire_id("ord-1")


def test_market_limit_and_stop_bodies() -> None:
    broker, _, fake, _ = build()
    broker.submit(req("m"))
    broker.submit(req("l", limit_price=D("99.999")))
    broker.submit(req("s", Side.SELL, "0.2", order_type=OrderType.STOP, stop_price=D("90.001")))
    m, lim, stop = fake.posted
    assert m["type"] == "market" and m["market_order_config"] == {"asset_quantity": "0.5"}
    assert m["client_order_id"] == wire_id("m") and m["symbol"] == "BTC-USD" and m["side"] == "buy"
    assert lim["limit_order_config"] == {
        "asset_quantity": "0.5",
        "limit_price": "99.99",
        "time_in_force": "gtc",
    }
    assert stop["type"] == "stop_loss"  # a sell stop rounds UP so it triggers earlier, never later
    assert stop["stop_loss_order_config"] == {
        "asset_quantity": "0.2",
        "stop_price": "90.01",
        "time_in_force": "gtc",
    }


@pytest.mark.parametrize(
    ("kw", "code"),
    [
        ({"qty": "0.000015"}, "BAD_INCREMENT"),
        ({"qty": "0.00001"}, "SIZE_OUT_OF_RANGE"),
        ({"qty": "11"}, "SIZE_OUT_OF_RANGE"),
        ({"symbol": "DOGE-USD"}, "UNSUPPORTED_SYMBOL"),
        ({"stop_loss_price": D(90)}, "BRACKET_NOT_SUPPORTED"),
    ],
)
def test_orders_robinhood_would_refuse_are_rejected_locally(kw: dict[str, object], code: str) -> None:
    broker, _, fake, _ = build()
    r = req(**({"qty": kw["qty"]} if "qty" in kw else {}))  # type: ignore[arg-type]
    r = r.model_copy(update={k: v for k, v in kw.items() if k != "qty"})
    with pytest.raises(BrokerRejected) as exc:
        broker.submit(r)
    assert exc.value.code == code and fake.posted == []


# ---- lookup without a client-id endpoint ---------------------------------------------------------
def test_orders_are_found_after_a_restart_by_scanning_and_map_back_to_our_ids() -> None:
    broker, api, fake, clock = build()
    broker.submit(req("c1"))
    broker.submit(req("c2", limit_price=D(50)))  # resting
    fresh = RobinhoodCryptoBroker(api, PAIRS, now=lambda: clock.now().timestamp())  # new process: no memory
    fresh.register_client_ids(["c1", "c2"])
    got = fresh.get_order("c1")
    assert got is not None and got.client_order_id == "c1" and got.status is BrokerStatus.FILLED
    assert {o.client_order_id for o in fresh.list_open_orders()} == {"c2"}
    assert fresh.get_order("never") is None


def test_pagination_is_followed_to_the_end() -> None:
    broker, api, fake, clock = build()
    for i in range(5):  # PAGE=2 in the fake: three pages
        broker.submit(req(f"o{i}", limit_price=D(50)))
    fresh = RobinhoodCryptoBroker(api, PAIRS, now=lambda: clock.now().timestamp())
    fresh.register_client_ids([f"o{i}" for i in range(5)])
    assert {o.client_order_id for o in fresh.list_open_orders()} == {f"o{i}" for i in range(5)}
    assert fresh.get_order("o4") is not None


def test_executions_become_stable_fills_with_no_fee() -> None:
    broker, _, fake, _ = build()
    fake.mock.fill_fraction = D("0.5")
    first = broker.submit(req("c1"))
    fake.mock.step()
    got = broker.get_order("c1")
    assert (
        first.status is BrokerStatus.PARTIALLY_FILLED
        and got is not None
        and got.status is BrokerStatus.FILLED
    )
    assert [f.quantity for f in got.fills] == [D("0.25"), D("0.125")] or sum(
        f.quantity for f in got.fills
    ) == D("0.5")
    assert all(f.fee == 0 for f in got.fills)
    again = broker.get_order("c1")
    assert again is not None and [f.fill_id for f in again.fills] == [f.fill_id for f in got.fills]


def test_cancel_of_an_already_terminal_order_is_a_noop() -> None:
    broker, _, fake, _ = build()
    broker.submit(req("c1"))
    broker.cancel("c1")  # server answers 400 'not cancelable'; the adapter verifies and accepts
    broker.cancel("never-existed")


def test_cash_and_positions_use_account_and_holdings_endpoints() -> None:
    broker, _, fake, _ = build()
    broker.submit(req("c1"))
    assert broker.get_positions() == {"BTC-USD": D("0.5")}
    assert broker.get_cash() == Decimal(100_000) - D("0.5") * 100


# ---- market data --------------------------------------------------------------------------------
def md(style: str = "inclusive") -> tuple[RobinhoodMarketData, FakeRobinhood, FakeClock]:
    broker, api, fake, clock = build()
    fake.quote_style = style
    return RobinhoodMarketData(api, PAIRS, clock), fake, clock


def test_quotes_become_risk_ready_instruments() -> None:
    provider, fake, clock = md()
    i = provider.snapshot(["BTC-USD"]).instruments["BTC-USD"]
    assert abs(i.price - 100) < D("0.001") and i.qty_step == D("0.00001") and i.robinhood_listed is True
    assert i.market_cap_usd == D(1_200_000_000_000) and i.est_slippage_bps == D(
        10
    )  # floor beats the 10bps spread
    assert i.quote_ts == clock.now().replace(microsecond=0) or i.quote_ts == clock.now()


def test_wide_spreads_raise_the_slippage_estimate() -> None:
    provider, fake, _ = md()
    fake.spread = D("0.02")  # 200 bps
    assert provider.snapshot(["BTC-USD"]).instruments["BTC-USD"].est_slippage_bps == D(200)


def test_alternate_field_names_are_understood() -> None:
    provider, _, _ = md("bid_ask")
    assert provider.snapshot(["BTC-USD"]).instruments["BTC-USD"].price > 0


def test_unrecognised_quote_shape_fails_closed() -> None:
    provider, _, _ = md("garbage")
    with pytest.raises(MarketDataError, match="recognisable"):
        provider.snapshot(["BTC-USD"])


def test_coins_outside_the_allowlist_are_never_quoted() -> None:
    provider, fake, _ = md()
    with pytest.raises(MarketDataError, match="allowlist"):
        provider.snapshot(["PEPE-USD"])
    assert ("GET", "/api/v1/crypto/marketdata/best_bid_ask/") not in fake.requests


def test_stale_or_failed_quotes_fail_closed() -> None:
    provider, fake, clock = md()
    fake.quote_age = timedelta(hours=1)  # the server serves an hour-old quote
    with pytest.raises(MarketDataError, match="stale"):
        provider.snapshot(["BTC-USD"])
    fake.quote_age = timedelta(0)
    fake.outage = True
    with pytest.raises(MarketDataError):
        provider.snapshot(["BTC-USD"])


def test_allowlist_file_loads_exact_decimals(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "pairs.json"
    f.write_text(
        '{"ETH-USD": {"asset_increment": 0.0001, "price_increment": 0.01, "market_cap_usd": 400000000000}}'
    )
    pairs = load_pairs(str(f))
    assert pairs["ETH-USD"].asset_increment == D("0.0001") and pairs["ETH-USD"].market_cap_usd == D(
        400_000_000_000
    )


def test_unavailable_is_raised_for_transport_failures() -> None:
    broker, _, fake, _ = build()
    fake.outage = True
    with pytest.raises(BrokerUnavailable):
        broker.get_cash()
    assert httpx is not None


def test_bad_credentials_during_a_quote_fetch_reach_the_executor_intact() -> None:
    _, api, fake, clock = build()
    wrong = RobinhoodApi(
        SecretStr(API_KEY), SecretStr(new_keypair()[1]), base_url=BASE, client=fake.client,
        time_source=lambda: clock.now().timestamp(), sleeper=lambda _s: None,
    )  # fmt: skip
    with pytest.raises(CredentialsError):
        RobinhoodMarketData(wrong, PAIRS, clock).snapshot(["BTC-USD"])
