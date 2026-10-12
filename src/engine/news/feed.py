"""The news-feed contract the listener depends on."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from engine.news.models import NewsItem


class NewsFeedError(Exception):
    pass


class NewsFeed(Protocol):
    name: str

    def fetch(self, start: datetime, symbols: Sequence[str], received_at: datetime) -> list[NewsItem]:
        """Items published at or after `start` that mention any of `symbols`, oldest first.
        `received_at` is stamped onto each (when WE saw it, which is what backtests must replay)."""
        ...
