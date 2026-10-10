"""Leader lock: two execution services must never trade the same accounts."""

from __future__ import annotations

import psycopg

from engine.service.leader import InProcessLeaderLock, PostgresLeaderLock


def test_in_process_lock_is_exclusive_and_releasable() -> None:
    a, b = InProcessLeaderLock("t1"), InProcessLeaderLock("t1")
    assert a.acquire() and not b.acquire()
    assert a.still_held() and not b.still_held()
    a.release()
    assert not a.still_held() and b.acquire()
    b.release()


def test_postgres_lock_excludes_a_second_instance(pg_dsn: str) -> None:
    a, b = PostgresLeaderLock(pg_dsn), PostgresLeaderLock(pg_dsn)
    assert a.acquire() and a.still_held()
    assert not b.acquire() and not b.still_held()
    a.release()
    assert b.acquire()
    b.release()


def test_losing_the_connection_loses_leadership(pg_dsn: str) -> None:
    a = PostgresLeaderLock(pg_dsn)
    assert a.acquire()
    with psycopg.connect(pg_dsn, autocommit=True) as admin:
        admin.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_locks WHERE locktype = 'advisory' AND granted AND pid <> pg_backend_pid()"
        )
    assert not a.still_held()  # the executor sees this and stops trading
    b = PostgresLeaderLock(pg_dsn)
    assert b.acquire()  # the dead leader's lock is gone with its session
    b.release()
