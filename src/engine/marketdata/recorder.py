"""QuoteRecorder: keep what the market actually looked like (price and spread) while the service runs.

Backtests are only as honest as their cost model. Recording the spread Robinhood really quotes, minute by
minute, turns "a 200 bps placeholder" into a measured number, and gives the replay tool real conditions to
compare against. This data is not part of the audit log; losing it never affects trading.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol

import psycopg

from engine.domain.market import MarketSnapshot

log = logging.getLogger("engine.recorder")


class QuoteSink(Protocol):
    def due(self, account_id: str, now: datetime) -> bool: ...

    def store(self, account_id: str, snapshot: MarketSnapshot, now: datetime) -> int: ...


class _Throttle:
    def __init__(self, interval: timedelta) -> None:
        self._interval = interval
        self._last: dict[str, datetime] = {}

    def due(self, key: str, now: datetime) -> bool:
        last = self._last.get(key)
        return last is None or now - last >= self._interval

    def mark(self, key: str, now: datetime) -> None:
        self._last[key] = now


class InMemoryQuoteRecorder:
    def __init__(self, interval: timedelta = timedelta(seconds=60)) -> None:
        self._throttle = _Throttle(interval)
        self.rows: list[tuple[datetime, str, str, Decimal, Decimal | None]] = []

    def due(self, account_id: str, now: datetime) -> bool:
        return self._throttle.due(account_id, now)

    def store(self, account_id: str, snapshot: MarketSnapshot, now: datetime) -> int:
        self._throttle.mark(account_id, now)
        for sym, i in snapshot.instruments.items():
            self.rows.append((now, account_id, sym, i.price, i.est_slippage_bps))
        return len(snapshot.instruments)


class PostgresQuoteRecorder:
    def __init__(self, dsn: str, interval: timedelta = timedelta(seconds=60)) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._throttle = _Throttle(interval)

    def due(self, account_id: str, now: datetime) -> bool:
        return self._throttle.due(account_id, now)

    def store(self, account_id: str, snapshot: MarketSnapshot, now: datetime) -> int:
        self._throttle.mark(account_id, now)
        rows = [
            (now, account_id, sym, i.price, i.est_slippage_bps) for sym, i in snapshot.instruments.items()
        ]
        try:
            with self._conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO market_quotes (ts, account_id, symbol, price, spread_bps) VALUES (%s,%s,%s,%s,%s)",
                    rows,
                )
        except psycopg.Error:
            log.exception("could not record quotes (ignored: this data is optional)")
            return 0
        return len(rows)

    def close(self) -> None:
        self._conn.close()


def recorded_spread_bps(
    dsn: str, symbol: str, *, since: datetime | None = None, min_samples: int = 30
) -> Decimal | None:
    """Median recorded spread for a symbol, or None if there is not enough data to say."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            "SELECT count(spread_bps), percentile_cont(0.5) WITHIN GROUP (ORDER BY spread_bps) "
            "FROM market_quotes WHERE symbol = %s AND spread_bps IS NOT NULL AND (%s::timestamptz IS NULL OR ts >= %s)",
            (symbol, since, since),
        ).fetchone()
    if row is None or row[0] < min_samples or row[1] is None:
        return None
    return Decimal(str(row[1]))
