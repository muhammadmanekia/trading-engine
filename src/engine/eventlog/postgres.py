"""Postgres event store. A single advisory lock serialises writers so the hash chain stays linear."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any

import psycopg
from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb

from engine.eventlog.base import (
    GENESIS_HASH,
    ConcurrencyError,
    DuplicateEvent,
    NewEvent,
    StoredEvent,
    compute_hash,
    verify_events,
)

_WRITER_LOCK = 7_345_001

_COLS = (
    "seq, event_id, stream_id, stream_version, type, account_id, schema_version, "
    "payload, ts, idempotency_key, prev_hash, hash"
)


def migrate(dsn: str) -> None:
    """Apply any unapplied *.sql migrations in filename order."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY)")
        applied = {r[0] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()}
        files = sorted(
            resources.files("engine.eventlog").joinpath("migrations").iterdir(),
            key=lambda p: p.name,
        )
        for f in files:
            if not f.name.endswith(".sql") or f.name in applied:
                continue
            with conn.transaction():
                conn.execute(f.read_text())
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (f.name,))


def _row(r: tuple[Any, ...]) -> StoredEvent:
    return StoredEvent(
        seq=r[0],
        event_id=str(r[1]),
        stream_id=r[2],
        stream_version=r[3],
        type=r[4],
        account_id=r[5],
        schema_version=r[6],
        payload=r[7],
        ts=r[8],
        idempotency_key=r[9],
        prev_hash=r[10].strip(),
        hash=r[11].strip(),
    )


class PostgresEventStore:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)

    def close(self) -> None:
        self._conn.close()

    def append(
        self, events: Sequence[NewEvent], expected_versions: Mapping[str, int] | None = None
    ) -> list[StoredEvent]:
        expected_versions = expected_versions or {}
        if not events:
            return []
        try:
            with self._conn.transaction():
                self._conn.execute("SELECT pg_advisory_xact_lock(%s)", (_WRITER_LOCK,))
                versions: dict[str, int] = {}

                def current(stream: str) -> int:
                    if stream not in versions:
                        row = self._conn.execute(
                            "SELECT COALESCE(MAX(stream_version), -1) FROM events WHERE stream_id = %s",
                            (stream,),
                        ).fetchone()
                        versions[stream] = int(row[0]) if row else -1
                    return versions[stream]

                for stream, want in expected_versions.items():
                    if current(stream) != want:
                        raise ConcurrencyError(f"{stream}: expected {want}, at {current(stream)}")

                keys = [e.idempotency_key for e in events if e.idempotency_key is not None]
                if len(set(keys)) != len(keys):
                    dup = next(k for k in keys if keys.count(k) > 1)
                    raise DuplicateEvent(dup)
                if keys:
                    hit = self._conn.execute(
                        "SELECT idempotency_key FROM events WHERE idempotency_key = ANY(%s) LIMIT 1",
                        (keys,),
                    ).fetchone()
                    if hit:
                        raise DuplicateEvent(hit[0])

                last = self._conn.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
                prev = last[0].strip() if last else GENESIS_HASH
                out: list[StoredEvent] = []
                for ev in events:
                    version = current(ev.stream_id) + 1
                    versions[ev.stream_id] = version
                    event_id = str(uuid.uuid4())
                    h = compute_hash(
                        prev,
                        event_id=event_id,
                        stream_id=ev.stream_id,
                        stream_version=version,
                        type=ev.type,
                        account_id=ev.account_id,
                        schema_version=ev.schema_version,
                        payload=ev.payload,
                        ts=ev.ts,
                        idempotency_key=ev.idempotency_key,
                    )
                    row = self._conn.execute(
                        "INSERT INTO events (event_id, stream_id, stream_version, type, account_id, "
                        "schema_version, payload, ts, idempotency_key, prev_hash, hash) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING " + _COLS,
                        (
                            event_id,
                            ev.stream_id,
                            version,
                            ev.type,
                            ev.account_id,
                            ev.schema_version,
                            Jsonb(ev.payload),
                            ev.ts,
                            ev.idempotency_key,
                            prev,
                            h,
                        ),
                    ).fetchone()
                    assert row is not None
                    out.append(_row(row))
                    prev = h
                return out
        except pg_errors.UniqueViolation as exc:
            constraint = exc.diag.constraint_name or ""
            if "idempotency" in constraint:
                raise DuplicateEvent(constraint) from exc
            raise ConcurrencyError(constraint) from exc

    def read_stream(self, stream_id: str) -> list[StoredEvent]:
        rows = self._conn.execute(
            f"SELECT {_COLS} FROM events WHERE stream_id = %s ORDER BY stream_version",  # noqa: S608
            (stream_id,),
        ).fetchall()
        return [_row(r) for r in rows]

    def read_all(
        self, after_seq: int = 0, account_id: str | None = None, limit: int | None = None
    ) -> list[StoredEvent]:
        sql = f"SELECT {_COLS} FROM events WHERE seq > %s"  # noqa: S608
        params: list[Any] = [after_seq]
        if account_id is not None:
            sql += " AND account_id = %s"
            params.append(account_id)
        sql += " ORDER BY seq"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        return [_row(r) for r in self._conn.execute(sql, params).fetchall()]

    def verify_chain(self) -> None:
        verify_events(self.read_all())
