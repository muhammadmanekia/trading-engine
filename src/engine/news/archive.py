"""News archive: every item we have ever seen, with when we first saw it.

Building this from day one is what makes future news backtests honest: replaying `received_at` reproduces the
information a strategy actually had at each moment, instead of the (always earlier) publication time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

import psycopg
from psycopg.types.json import Jsonb

from engine.news.models import NewsItem, canonical_symbol


class NewsArchive(Protocol):
    def add(self, item: NewsItem) -> bool:
        """Store the item. False if we had already seen it (same source and id)."""
        ...

    def since(self, ts: datetime, symbols: Sequence[str] | None = None) -> list[NewsItem]:
        """Items received at or after `ts`, oldest first, optionally only those tagged with one of `symbols`."""
        ...


class InMemoryNewsArchive:
    def __init__(self) -> None:
        self._items: dict[str, NewsItem] = {}

    def add(self, item: NewsItem) -> bool:
        if item.key in self._items:
            return False
        self._items[item.key] = item
        return True

    def since(self, ts: datetime, symbols: Sequence[str] | None = None) -> list[NewsItem]:
        want = {canonical_symbol(s) for s in symbols} if symbols else None
        rows = [
            i
            for i in self._items.values()
            if i.received_at >= ts and (want is None or want & {canonical_symbol(s) for s in i.symbols})
        ]
        return sorted(rows, key=lambda i: (i.received_at, i.key))

    def __len__(self) -> int:
        return len(self._items)


class PostgresNewsArchive:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)

    def add(self, item: NewsItem) -> bool:
        cur = self._conn.execute(
            "INSERT INTO news_items (source, id, headline, summary, symbols, url, published_at, received_at, backfilled) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (source, id) DO NOTHING",
            (
                item.source,
                item.id,
                item.headline,
                item.summary,
                Jsonb(list(item.symbols)),
                item.url,
                item.published_at,
                item.received_at,
                item.backfilled,
            ),  # fmt: skip
        )
        return cur.rowcount == 1

    def since(self, ts: datetime, symbols: Sequence[str] | None = None) -> list[NewsItem]:
        rows = self._conn.execute(
            "SELECT source, id, headline, summary, symbols, url, published_at, received_at, backfilled "
            "FROM news_items WHERE received_at >= %s ORDER BY received_at, source, id",
            (ts,),
        ).fetchall()
        items = [
            NewsItem(
                source=r[0],
                id=r[1],
                headline=r[2],
                summary=r[3],
                symbols=tuple(r[4]),
                url=r[5],
                published_at=r[6],
                received_at=r[7],
                backfilled=r[8],
            )  # fmt: skip
            for r in rows
        ]
        if symbols:
            want = {canonical_symbol(s) for s in symbols}
            items = [i for i in items if want & {canonical_symbol(s) for s in i.symbols}]
        return items

    def close(self) -> None:
        self._conn.close()
