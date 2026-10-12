"""News items. `received_at` is when WE first saw the item, not when it was published: that gap is the real-world
delay a strategy would have faced, so backtests built on this archive are point-in-time honest."""

from __future__ import annotations

from pydantic import Field

from engine.domain.types import Frozen, UtcDatetime


class NewsItem(Frozen):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    summary: str = ""
    symbols: tuple[str, ...] = ()
    url: str | None = None
    published_at: UtcDatetime
    received_at: UtcDatetime
    backfilled: bool = False  # fetched after the fact: received_at is NOT a true point-in-time

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    @property
    def text(self) -> str:
        return f"{self.headline}\n{self.summary}".strip()


def canonical_symbol(raw: str) -> str:
    """'BTC/USD' -> 'BTC-USD'; tickers are upper-cased."""
    return raw.strip().upper().replace("/", "-")
