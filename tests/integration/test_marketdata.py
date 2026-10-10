"""Market data providers: parsing, caching, and failing closed."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import SecretStr

from engine.broker.alpaca import AlpacaBroker
from engine.clock import FakeClock
from engine.domain.types import Mode
from engine.marketdata.alpaca import AlpacaMarketData
from engine.marketdata.base import MarketDataError, parse_rfc3339
from tests.fakes.alpaca_server import BASE, FakeAlpaca
from tests.helpers import D


def build(**kw: object) -> tuple[AlpacaMarketData, FakeAlpaca, FakeClock]:
    clock = FakeClock()
    fake = FakeAlpaca(clock)
    broker = AlpacaBroker(
        SecretStr("k"), SecretStr("s"), Mode.PAPER, base_url=BASE, client=fake.client, sleeper=lambda _s: None
    )
    md = AlpacaMarketData(
        SecretStr("k"),
        SecretStr("s"),
        broker,
        clock,
        base_url=BASE,
        client=fake.client,
        sleeper=lambda _s: None,
        **kw,  # type: ignore[arg-type]
    )
    return md, fake, clock


def test_snapshot_builds_risk_ready_instruments() -> None:
    md, fake, clock = build()
    snap = md.snapshot(["AAPL"])
    i = snap.instruments["AAPL"]
    assert i.price == D(100) and i.qty_step == 1 and i.tradable and i.market_open
    assert i.avg_daily_volume_usd == D(50_000_000)
    assert i.quote_ts == clock.now().replace(microsecond=123456)  # nanosecond timestamps are parsed


def test_mid_price_from_bid_ask() -> None:
    md, fake, _ = build()
    fake.set_price("AAPL", D(200))
    assert md.snapshot(["AAPL"]).instruments["AAPL"].price == D(200)  # symmetric spread around the price


def test_falls_back_to_last_trade_when_there_is_no_quote() -> None:
    md, fake, _ = build()
    fake.no_quotes.add("AAPL")
    assert md.snapshot(["AAPL"]).instruments["AAPL"].price == D(100)


def test_market_closed_and_untradable_are_reported() -> None:
    md, fake, _ = build()
    fake.market_open = False
    fake.untradable.add("AAPL")
    i = md.snapshot(["AAPL"]).instruments["AAPL"]
    assert not i.market_open and not i.tradable


def test_volume_scale_corrects_for_the_iex_feed() -> None:
    md, fake, _ = build(volume_scale=D(40))
    assert md.snapshot(["AAPL"]).instruments["AAPL"].avg_daily_volume_usd == D(2_000_000_000)


def test_slow_data_is_cached_and_refreshes_after_the_ttl() -> None:
    md, fake, clock = build()
    md.snapshot(["AAPL"])
    n = sum(1 for _m, p in fake.requests if p == "/v2/stocks/bars")
    md.snapshot(["AAPL"])
    assert sum(1 for _m, p in fake.requests if p == "/v2/stocks/bars") == n  # cached
    clock.advance(hours=7)
    md.snapshot(["AAPL"])
    assert sum(1 for _m, p in fake.requests if p == "/v2/stocks/bars") == n + 1


def test_failures_raise_market_data_error_never_a_guess() -> None:
    md, fake, _ = build()
    fake.outage = True
    with pytest.raises(MarketDataError):
        md.snapshot(["AAPL"])
    fake.outage = False
    with pytest.raises(MarketDataError):
        md.snapshot(["UNKNOWN"])  # no quote, no asset


def test_empty_request_is_fine() -> None:
    md, _, _ = build()
    assert md.snapshot([]).instruments == {}


@pytest.mark.parametrize(
    ("raw", "micro"),
    [
        ("2026-01-05T15:00:00.123456789Z", 123456),
        ("2026-01-05T15:00:00Z", 0),
        ("2026-01-05T15:00:00.5+00:00", 500000),
        ("2026-01-05T10:00:00-05:00", 0),
    ],
)
def test_rfc3339_variants(raw: str, micro: int) -> None:
    dt = parse_rfc3339(raw)
    assert dt.utcoffset() == timedelta(0) and dt.microsecond == micro and dt.hour == 15
    assert isinstance(D(1), Decimal)


def test_naive_timestamps_are_refused() -> None:
    with pytest.raises(MarketDataError):
        parse_rfc3339("2026-01-05T15:00:00")
