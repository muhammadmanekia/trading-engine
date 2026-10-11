"""Historical bars: the model, data-quality checks, and loaders.

A Bar's `ts` is its OPEN time; it is closed (and its values final) at `ts + timeframe`. Strategies and the
backtester only ever see closed bars, which is what makes lookahead bias structurally hard.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Protocol, Self

from pydantic import model_validator

from engine.domain.types import Frozen, NonNegDec, PosDec, UtcDatetime

TIMEFRAMES: dict[str, timedelta] = {"1h": timedelta(hours=1), "1d": timedelta(days=1)}


class DataQualityError(Exception):
    """The data cannot be trusted (unsorted, duplicated, impossible prices). Refuse to backtest on it."""


class Bar(Frozen):
    ts: UtcDatetime  # OPEN time
    open: PosDec
    high: PosDec
    low: PosDec
    close: PosDec
    volume: NonNegDec = Decimal(0)

    @model_validator(mode="after")
    def _ohlc(self) -> Self:
        if (
            self.low > self.high
            or self.low > min(self.open, self.close)
            or self.high < max(self.open, self.close)
        ):
            raise ValueError(f"inconsistent OHLC at {self.ts.isoformat()}")
        return self


def timeframe_delta(timeframe: str) -> timedelta:
    try:
        return TIMEFRAMES[timeframe]
    except KeyError:
        raise ValueError(f"unsupported timeframe {timeframe!r}; use one of {sorted(TIMEFRAMES)}") from None


def close_time(bar: Bar, timeframe: str) -> datetime:
    return bar.ts + timeframe_delta(timeframe)


def validate_bars(bars: Sequence[Bar], timeframe: str, *, gap_factor: int = 5) -> list[str]:
    """Raise DataQualityError for fatal problems; return human-readable warnings for the rest."""
    delta = timeframe_delta(timeframe)
    warnings: list[str] = []
    gaps = 0
    for prev, cur in zip(bars, bars[1:], strict=False):
        if cur.ts <= prev.ts:
            raise DataQualityError(f"bars not strictly increasing at {cur.ts.isoformat()}")
        if cur.ts - prev.ts > gap_factor * delta:
            gaps += 1
    if gaps:
        warnings.append(
            f"{gaps} gap(s) longer than {gap_factor} bars (weekends/holidays are normal for stocks)"
        )
    if bars and len(bars) < 50:
        warnings.append(f"only {len(bars)} bars: too short for most indicators")
    return warnings


class HistoricalData(Protocol):
    def bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[Bar]:
        """Closed bars with start <= ts < end, oldest first."""
        ...


class InMemoryBars:
    def __init__(self, data: Mapping[tuple[str, str], Sequence[Bar]] | None = None) -> None:
        self._data: dict[tuple[str, str], list[Bar]] = {k: list(v) for k, v in (data or {}).items()}

    def put(self, symbol: str, timeframe: str, bars: Sequence[Bar]) -> None:
        self._data[(symbol, timeframe)] = list(bars)

    def bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[Bar]:
        return [b for b in self._data.get((symbol, timeframe), []) if start <= b.ts < end]


class CsvBars:
    """Bars from `<dir>/<SYMBOL>_<timeframe>.csv` with columns: timestamp,open,high,low,close,volume.
    `timestamp` is ISO-8601 with a timezone, or unix seconds. Values are parsed as exact decimals."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self._dir / f"{symbol}_{timeframe}.csv"

    def bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[Bar]:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            raise DataQualityError(f"no data file for {symbol} {timeframe}: {path}")
        return [b for b in load_csv(path) if start <= b.ts < end]


def _parse_ts(raw: str) -> datetime:
    raw = raw.strip()
    if raw.replace(".", "", 1).isdigit():
        return datetime.fromtimestamp(int(Decimal(raw)), UTC)  # whole seconds; no floats
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise DataQualityError(f"timestamp without timezone: {raw!r}")
    return dt.astimezone(UTC)


def load_csv(path: str | Path) -> list[Bar]:
    bars: list[Bar] = []
    with open(path, newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            try:
                bars.append(
                    Bar(
                        ts=_parse_ts(row["timestamp"]),
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row.get("volume") or "0",
                    )
                )
            except (KeyError, ValueError, DataQualityError) as exc:
                raise DataQualityError(f"{path}:{n}: {exc}") from exc
    return bars


def to_csv(bars: Sequence[Bar], path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.ts.isoformat(), b.open, b.high, b.low, b.close, b.volume])


class BarCache:
    """Wraps any HistoricalData source with an on-disk CSV cache, so repeated backtests do not refetch.
    Only closed bars are ever cached; the newest cached bar is re-fetched on extension in case it was amended."""

    def __init__(self, inner: HistoricalData, directory: str | Path) -> None:
        self._inner = inner
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.fetches: list[tuple[str, str, datetime, datetime]] = []  # for tests and diagnostics

    def _path(self, symbol: str, timeframe: str) -> Path:
        return self._dir / f"{symbol}_{timeframe}.csv"

    def bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[Bar]:
        path = self._path(symbol, timeframe)
        cached = load_csv(path) if path.exists() else []
        delta = timeframe_delta(timeframe)
        covers_start = bool(cached) and cached[0].ts <= start
        covers_end = bool(cached) and cached[-1].ts + delta >= end
        if not (covers_start and covers_end):
            lo = cached[-1].ts if covers_start else start
            self.fetches.append((symbol, timeframe, lo, end))
            fresh = self._inner.bars(symbol, timeframe, lo, end)
            merged = {b.ts: b for b in cached}
            merged.update({b.ts: b for b in fresh})
            cached = [merged[k] for k in sorted(merged)]
            to_csv(cached, path)
        return [b for b in cached if start <= b.ts < end]
