"""News archive: idempotent, point-in-time, on both implementations."""

from __future__ import annotations

from datetime import timedelta

import pytest

from engine.news.archive import InMemoryNewsArchive, NewsArchive, PostgresNewsArchive
from engine.news.models import NewsItem
from tests.helpers import T0


def n(i: str, sym: str = "AAPL", minutes: int = 0, **kw: object) -> NewsItem:
    return NewsItem.model_validate(
        {"id": i, "source": "benzinga", "headline": f"headline {i}", "symbols": (sym,),
         "published_at": T0, "received_at": T0 + timedelta(minutes=minutes), **kw}
    )  # fmt: skip


@pytest.fixture(params=["memory", "postgres"])
def archive(request: pytest.FixtureRequest) -> NewsArchive:
    if request.param == "memory":
        return InMemoryNewsArchive()
    return PostgresNewsArchive(request.getfixturevalue("pg_dsn"))


def test_add_is_idempotent_on_source_and_id(archive: NewsArchive) -> None:
    assert archive.add(n("1")) is True and archive.add(n("1")) is False
    assert archive.add(n("1").model_copy(update={"source": "other"})) is True  # same id, different source
    assert len(archive.since(T0)) == 2


def test_since_is_ordered_by_when_we_received_it_not_when_it_was_published(archive: NewsArchive) -> None:
    archive.add(n("late", minutes=10))
    archive.add(n("early", minutes=1))
    archive.add(n("mid", minutes=5))
    assert [i.id for i in archive.since(T0)] == ["early", "mid", "late"]
    assert [i.id for i in archive.since(T0 + timedelta(minutes=5))] == ["mid", "late"]


def test_since_can_filter_by_symbol_in_any_notation(archive: NewsArchive) -> None:
    archive.add(n("a", "AAPL"))
    archive.add(n("b", "BTC/USD"))
    assert [i.id for i in archive.since(T0, ["btc-usd"])] == ["b"]
    assert [i.id for i in archive.since(T0, ["AAPL", "MSFT"])] == ["a"]


def test_items_roundtrip_exactly_and_remember_if_they_were_backfilled(archive: NewsArchive) -> None:
    item = n("x", url="https://example.com/x", summary="body", backfilled=True)
    archive.add(item)
    assert archive.since(T0) == [item]
