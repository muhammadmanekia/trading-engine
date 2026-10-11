"""Bars: validation, CSV loading, exactness."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from engine.marketdata.history import (
    Bar,
    CsvBars,
    DataQualityError,
    InMemoryBars,
    close_time,
    load_csv,
    timeframe_delta,
    to_csv,
    validate_bars,
)
from tests.helpers import D, T0, bars_from_closes


def test_ohlc_must_be_consistent() -> None:
    Bar(ts=T0, open=D(10), high=D(12), low=D(9), close=D(11))
    for bad in (
        dict(open=10, high=9, low=8, close=9),  # high below open
        dict(open=10, high=12, low=11, close=11),  # low above open
        dict(open=10, high=12, low=9, close=13),  # close above high
        dict(open=0, high=1, low=0, close=1),  # non-positive price
    ):
        with pytest.raises(ValidationError):
            Bar(ts=T0, **{k: D(v) for k, v in bad.items()})


def test_floats_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Bar(ts=T0, open=10.0, high=11, low=9, close=10)  # type: ignore[arg-type]


def test_bar_is_closed_at_open_plus_timeframe() -> None:
    b = bars_from_closes([10])[0]
    assert close_time(b, "1h") == T0 + timedelta(hours=1) and close_time(b, "1d") == T0 + timedelta(days=1)
    with pytest.raises(ValueError, match="unsupported timeframe"):
        timeframe_delta("5m")


def test_validate_rejects_unsorted_and_duplicate_bars() -> None:
    bars = bars_from_closes([10, 11, 12])
    with pytest.raises(DataQualityError):
        validate_bars([bars[1], bars[0]], "1h")
    with pytest.raises(DataQualityError):
        validate_bars([bars[0], bars[0]], "1h")
    assert validate_bars(bars, "1h")  # too short is only a warning


def test_validate_reports_gaps_without_failing() -> None:
    a = bars_from_closes(range(1, 60))
    b = bars_from_closes(range(60, 70), start=a[-1].ts + timedelta(hours=40))
    warnings = validate_bars([*a, *b], "1h")
    assert any("gap" in w for w in warnings)


def test_csv_roundtrip_is_exact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bars = bars_from_closes(["100.123456789", "101.5", "99.25"])
    to_csv(bars, tmp_path / "BTC-USD_1h.csv")
    assert load_csv(tmp_path / "BTC-USD_1h.csv") == bars
    src = CsvBars(tmp_path)
    got = src.bars("BTC-USD", "1h", T0, T0 + timedelta(hours=2))
    assert [b.close for b in got] == [D("100.123456789"), D("101.5")]  # start inclusive, end exclusive


def test_csv_accepts_unix_seconds_and_z_suffix(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "X_1h.csv"
    p.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1767625200,10,11,9,10.5,5\n"
        "2026-01-05T16:00:00Z,10.5,11,10,10.7,\n"
    )
    bars = load_csv(p)
    assert bars[0].ts == T0 and bars[1].ts == T0 + timedelta(hours=1) and bars[1].volume == 0


@pytest.mark.parametrize(
    "row",
    [
        "2026-01-05T15:00:00,10,11,9,10,1",  # no timezone
        "2026-01-05T15:00:00Z,10,9,9,10,1",  # impossible OHLC
        "2026-01-05T15:00:00Z,abc,11,9,10,1",  # not a number
    ],
)
def test_bad_csv_rows_name_the_line(tmp_path, row: str) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "X_1h.csv"
    p.write_text(f"timestamp,open,high,low,close,volume\n{row}\n")
    with pytest.raises(DataQualityError, match=":2:"):
        load_csv(p)


def test_missing_file_is_a_clear_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DataQualityError, match="no data file"):
        CsvBars(tmp_path).bars("NOPE", "1h", T0, T0 + timedelta(days=1))


def test_in_memory_source_filters_by_window() -> None:
    src = InMemoryBars({("A", "1h"): bars_from_closes(range(1, 11))})
    assert len(src.bars("A", "1h", T0 + timedelta(hours=2), T0 + timedelta(hours=5))) == 3
    assert src.bars("B", "1h", T0, T0 + timedelta(days=9)) == []
    assert isinstance(D(1), Decimal)


# ---- Alpaca historical bars + cache ------------------------------------------------------------------
def alpaca_bars(clock=None):  # type: ignore[no-untyped-def]
    from pydantic import SecretStr

    from engine.clock import FakeClock
    from engine.marketdata.alpaca import AlpacaBars
    from tests.fakes.alpaca_server import BASE, FakeAlpaca

    clock = clock or FakeClock(T0 + timedelta(days=30))
    fake = FakeAlpaca(clock)
    return (
        AlpacaBars(
            SecretStr("k"), SecretStr("s"), clock, base_url=BASE, client=fake.client, sleeper=lambda _s: None
        ),
        fake,
        clock,
    )


def test_alpaca_bars_follow_pagination_and_keep_decimals_exact() -> None:
    src, fake, _ = alpaca_bars()
    fake.history["AAPL"] = bars_from_closes(["100.1", "100.2", "100.3", "100.4", "100.5", "100.6", "100.7"])
    got = src.bars("AAPL", "1h", T0, T0 + timedelta(days=1))
    assert [b.close for b in got] == [D(f"100.{i}") for i in range(1, 8)]  # 3 pages of 3,3,1; no float noise
    assert sum(1 for m, p in fake.requests if p == "/v2/stocks/bars") == 3


def test_crypto_symbols_use_the_crypto_endpoint_with_a_slash() -> None:
    src, fake, _ = alpaca_bars()
    fake.history["BTC-USD"] = bars_from_closes([60000, 60100, 60200])
    got = src.bars("BTC-USD", "1h", T0, T0 + timedelta(days=1))
    assert len(got) == 3 and ("GET", "/v1beta3/crypto/us/bars") in fake.requests


def test_a_bar_that_has_not_closed_yet_is_never_returned() -> None:
    from engine.clock import FakeClock

    src, fake, _ = alpaca_bars(FakeClock(T0 + timedelta(hours=2, minutes=30)))
    fake.history["AAPL"] = bars_from_closes([1, 2, 3, 4, 5])  # the bars open at T0, T0+1h, T0+2h, ...
    got = src.bars("AAPL", "1h", T0, T0 + timedelta(days=1))
    assert [b.ts for b in got] == [T0, T0 + timedelta(hours=1)]  # the T0+2h bar closes at T0+3h > now


def test_bad_bar_data_fails_loudly() -> None:
    import pytest as _pytest

    from engine.marketdata.base import MarketDataError

    src, fake, _ = alpaca_bars()
    fake.history["AAPL"] = bars_from_closes([1, 2, 3])
    fake.outage = True
    with _pytest.raises(MarketDataError):
        src.bars("AAPL", "1h", T0, T0 + timedelta(days=1))
    with _pytest.raises(MarketDataError, match="unsupported"):
        src.bars("AAPL", "5m", T0, T0 + timedelta(days=1))


def test_the_cache_serves_covered_ranges_and_only_fetches_the_missing_tail(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from engine.marketdata.history import BarCache

    inner = InMemoryBars({("X", "1h"): bars_from_closes(range(1, 21))})
    cache = BarCache(inner, tmp_path)
    first = cache.bars("X", "1h", T0, T0 + timedelta(hours=10))
    assert len(first) == 10 and len(cache.fetches) == 1
    again = cache.bars("X", "1h", T0 + timedelta(hours=2), T0 + timedelta(hours=8))
    assert again == first[2:8] and len(cache.fetches) == 1  # no refetch
    more = cache.bars("X", "1h", T0, T0 + timedelta(hours=20))
    assert len(more) == 20 and len(cache.fetches) == 2
    assert cache.fetches[1][2] == T0 + timedelta(hours=9)  # only from the last cached bar onward
    assert len({b.ts for b in more}) == 20  # no duplicates after the merge
    assert CsvBars(tmp_path).bars("X", "1h", T0, T0 + timedelta(hours=20)) == more  # persisted on disk
