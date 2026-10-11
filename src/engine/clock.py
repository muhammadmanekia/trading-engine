"""Injectable time source so every time-dependent rule is deterministic under test."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 5, 15, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def set(self, when: datetime) -> None:
        self._now = when

    def advance(self, seconds: float = 0, **kwargs: float) -> datetime:
        self._now += timedelta(seconds=seconds, **kwargs)
        return self._now
