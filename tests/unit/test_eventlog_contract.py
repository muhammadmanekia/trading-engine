"""Contract tests: run against both the in-memory and Postgres stores."""

from __future__ import annotations

import pytest

from engine.eventlog.base import (
    ChainError,
    ConcurrencyError,
    DuplicateEvent,
    EventStore,
    NewEvent,
    verify_events,
)
from engine.eventlog.memory import InMemoryEventStore
from tests.helpers import T0


def ev(stream: str = "s", n: int = 1, key: str | None = None, account: str = "a") -> NewEvent:
    return NewEvent(
        stream_id=stream,
        type="AccountFunded",
        account_id=account,
        payload={"account_id": account, "amount": str(n)},
        ts=T0,
        idempotency_key=key,
    )


def test_append_assigns_sequence_and_stream_versions(store: EventStore) -> None:
    out = store.append([ev("a"), ev("a"), ev("b")])
    assert [e.seq for e in out] == sorted(e.seq for e in out) and len({e.seq for e in out}) == 3
    assert [e.stream_version for e in out] == [0, 1, 0]
    assert [e.stream_version for e in store.read_stream("a")] == [0, 1]


def test_read_all_after_seq_and_account_filter(store: EventStore) -> None:
    first = store.append([ev("a", account="x"), ev("b", account="y")])
    store.append([ev("c", account="x")])
    assert [e.account_id for e in store.read_all()] == ["x", "y", "x"]
    assert len(store.read_all(account_id="x")) == 2
    assert len(store.read_all(after_seq=first[0].seq)) == 2
    assert len(store.read_all(limit=1)) == 1


def test_payload_roundtrips_exactly(store: EventStore) -> None:
    e = NewEvent(
        "s", "AccountFunded", "a", {"account_id": "a", "amount": "0.10000000", "n": [1, {"k": "v"}]}, T0
    )
    stored = store.append([e])[0]
    assert store.read_stream("s")[0].payload == e.payload == stored.payload
    assert store.read_stream("s")[0].ts == T0


def test_optimistic_concurrency_conflict_writes_nothing(store: EventStore) -> None:
    store.append([ev("s")])
    with pytest.raises(ConcurrencyError):
        store.append([ev("s"), ev("other")], expected_versions={"s": -1})
    assert len(store.read_all()) == 1  # nothing from the failed batch
    store.append([ev("s")], expected_versions={"s": 0})
    assert len(store.read_stream("s")) == 2


def test_expected_version_minus_one_means_new_stream(store: EventStore) -> None:
    store.append([ev("new")], expected_versions={"new": -1})
    with pytest.raises(ConcurrencyError):
        store.append([ev("new")], expected_versions={"new": -1})


def test_idempotency_key_makes_duplicates_a_noop(store: EventStore) -> None:
    store.append([ev(key="k1")])
    with pytest.raises(DuplicateEvent):
        store.append([ev(key="k1")])
    assert len(store.read_all()) == 1


def test_duplicate_in_batch_is_atomic(store: EventStore) -> None:
    store.append([ev(key="taken")])
    with pytest.raises(DuplicateEvent):
        store.append([ev(key="fresh"), ev(key="taken")])
    store.append([ev(key="fresh")])  # 'fresh' was not consumed by the failed batch
    with pytest.raises(DuplicateEvent):
        store.append([ev(key="dup"), ev(key="dup")])


def test_hash_chain_verifies_and_links(store: EventStore) -> None:
    out = store.append([ev("a"), ev("b")])
    store.append([ev("a")])
    store.verify_chain()
    allv = store.read_all()
    assert allv[0].prev_hash == "0" * 64
    assert all(allv[i].prev_hash == allv[i - 1].hash for i in range(1, len(allv)))
    assert out[0].hash != out[1].hash


def test_empty_log_verifies(store: EventStore) -> None:
    store.verify_chain()
    assert store.read_all() == []


def test_tampering_is_detected_in_memory() -> None:
    s = InMemoryEventStore()
    s.append([ev(), ev(), ev()])
    s._tamper(1, payload={"account_id": "a", "amount": "999999"})
    with pytest.raises(ChainError, match="altered"):
        s.verify_chain()


def test_deleted_event_is_detected() -> None:
    s = InMemoryEventStore()
    s.append([ev(), ev(), ev()])
    events = s.read_all()
    with pytest.raises(ChainError, match="prev_hash"):
        verify_events([events[0], events[2]])


def test_postgres_rejects_update_delete_truncate(pg_dsn: str) -> None:
    import psycopg

    from engine.eventlog.postgres import PostgresEventStore

    s = PostgresEventStore(pg_dsn)
    s.append([ev()])
    with psycopg.connect(pg_dsn, autocommit=True) as c:
        for stmt in (
            "UPDATE events SET type = 'x'",
            "DELETE FROM events",
            "TRUNCATE events",
        ):
            with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
                c.execute(stmt)  # type: ignore[call-overload]
    assert len(s.read_all()) == 1
    s.verify_chain()
    s.close()


def test_postgres_detects_out_of_band_tampering(pg_dsn: str) -> None:
    """Even a superuser who disables the trigger cannot rewrite history undetected."""
    import psycopg

    from engine.eventlog.postgres import PostgresEventStore

    s = PostgresEventStore(pg_dsn)
    s.append([ev(), ev()])
    with psycopg.connect(pg_dsn, autocommit=True) as c:
        c.execute("ALTER TABLE events DISABLE TRIGGER events_no_update_delete")
        c.execute('UPDATE events SET payload = \'{"account_id": "a", "amount": "1000000"}\' WHERE seq = 1')
    with pytest.raises(ChainError):
        s.verify_chain()
    s.close()


def test_postgres_concurrent_writers_keep_a_linear_chain(pg_dsn: str) -> None:
    import threading

    from engine.eventlog.postgres import PostgresEventStore

    stores = [PostgresEventStore(pg_dsn) for _ in range(4)]
    errors: list[Exception] = []

    def work(s: PostgresEventStore, i: int) -> None:
        try:
            for j in range(15):
                s.append([ev(f"stream{i}", j)])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(s, i)) for i, s in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(stores[0].read_all()) == 60
    stores[0].verify_chain()
    for s in stores:
        s.close()
