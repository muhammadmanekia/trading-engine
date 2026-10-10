"""Single-instance guard. Two execution services must never trade the same accounts."""

from __future__ import annotations

from typing import Protocol

import psycopg

LEADER_KEY = 7_345_002  # distinct from the event-store writer lock


class LeaderLock(Protocol):
    def acquire(self) -> bool: ...

    def still_held(self) -> bool: ...

    def release(self) -> None: ...


class PostgresLeaderLock:
    """Session-level advisory lock on a dedicated connection. If the connection dies the lock is gone, which
    `still_held` detects, and the executor stops trading rather than risk a second leader."""

    def __init__(self, dsn: str, key: int = LEADER_KEY) -> None:
        self._dsn, self._key = dsn, key
        self._conn: psycopg.Connection[tuple[object, ...]] | None = None

    def acquire(self) -> bool:
        conn = psycopg.connect(self._dsn, autocommit=True)
        row = conn.execute("SELECT pg_try_advisory_lock(%s)", (self._key,)).fetchone()
        if row is not None and row[0]:
            self._conn = conn
            return True
        conn.close()
        return False

    def still_held(self) -> bool:
        if self._conn is None:
            return False
        try:
            self._conn.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False

    def release(self) -> None:
        if self._conn is not None:
            try:
                self._conn.execute("SELECT pg_advisory_unlock(%s)", (self._key,))
            except psycopg.Error:
                pass
            finally:
                self._conn.close()
                self._conn = None


class InProcessLeaderLock:
    """For tests: a process-wide registry of held names."""

    _held: set[str] = set()

    def __init__(self, name: str = "executor") -> None:
        self._name = name
        self._mine = False
        self.lose_on_next_check = False

    def acquire(self) -> bool:
        if self._name in self._held:
            return False
        self._held.add(self._name)
        self._mine = True
        return True

    def still_held(self) -> bool:
        return self._mine and not self.lose_on_next_check

    def release(self) -> None:
        if self._mine:
            self._held.discard(self._name)
            self._mine = False
