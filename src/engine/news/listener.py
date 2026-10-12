"""Turns the raw feed into the few items worth showing a model.

Pipeline per poll: fetch -> archive (dedupe; every item is kept even if later dropped) -> injection screen
(blocked items alert a human and go no further) -> drop stale -> relevance to the universe -> novelty.
The listener never calls a model and never trades; it hands clean items to whoever asked.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from engine.clock import Clock
from engine.news.archive import NewsArchive
from engine.news.feed import NewsFeed, NewsFeedError
from engine.news.filters import NoveltyFilter, relevant_symbols, screen
from engine.news.models import NewsItem

log = logging.getLogger("engine.news")
Notify = Callable[[str, str, str, str | None], None]


class NewsListener:
    def __init__(
        self,
        feed: NewsFeed,
        archive: NewsArchive,
        clock: Clock,
        universe: Sequence[str],
        *,
        notify: Notify = lambda *a: None,
        max_age: timedelta = timedelta(hours=2),
        lookback: timedelta = timedelta(hours=1),
        novelty: NoveltyFilter | None = None,
    ) -> None:
        self._feed, self._archive, self._clock = feed, archive, clock
        self._universe = sorted(set(universe))
        self._notify, self._max_age, self._lookback = notify, max_age, lookback
        self._novelty = novelty or NoveltyFilter()
        self._cursor: datetime | None = None
        self.consecutive_failures = 0
        self.blocked_total = 0

    def poll(self) -> list[NewsItem]:
        """Fresh, safe, relevant, novel items since the last poll. A feed outage returns [] and alerts once."""
        now = self._clock.now()
        start = self._cursor or (now - self._lookback)
        try:
            items = self._feed.fetch(
                start - timedelta(minutes=2), self._universe, now
            )  # small overlap; archive dedupes
        except NewsFeedError as exc:
            self.consecutive_failures += 1
            log.warning("news poll failed (%d in a row): %s", self.consecutive_failures, exc)
            if self.consecutive_failures == 5:
                self._notify(
                    "WARNING", "News feed unavailable", f"5 failed polls in a row: {exc}", "news-down"
                )
            return []
        self.consecutive_failures = 0
        fresh: list[NewsItem] = []
        for item in items:
            if not self._archive.add(item):
                continue  # already seen
            checked = screen(item)
            if checked.item is None:
                self.blocked_total += 1
                self._notify(
                    "WARNING",
                    "Suspicious news item blocked",
                    f"{item.source}:{item.id} ({', '.join(checked.blocked)}). Not shown to any model.",
                    f"news-blocked:{item.key}",
                )
                continue
            clean = checked.item
            if now - clean.published_at > self._max_age:
                continue  # archived for research, but too old to trade on
            symbols = relevant_symbols(clean, self._universe)
            if not symbols or not self._novelty.is_new(clean, symbols):
                continue
            fresh.append(clean)
            self._cursor = max(self._cursor or clean.published_at, clean.published_at)
        if self._cursor is None or not items:
            self._cursor = self._cursor or start
        return fresh
