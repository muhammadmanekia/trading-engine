"""In-memory event store. Same contract as the Postgres store; used for fast deterministic tests."""

from __future__ import annotations

import copy
import dataclasses
import threading
import uuid
from collections.abc import Mapping, Sequence

from engine.eventlog.base import (
    GENESIS_HASH,
    ConcurrencyError,
    DuplicateEvent,
    NewEvent,
    StoredEvent,
    compute_hash,
    verify_events,
)


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: list[StoredEvent] = []
        self._versions: dict[str, int] = {}
        self._keys: set[str] = set()
        self._lock = threading.Lock()

    def append(
        self, events: Sequence[NewEvent], expected_versions: Mapping[str, int] | None = None
    ) -> list[StoredEvent]:
        expected_versions = expected_versions or {}
        with self._lock:
            for stream, want in expected_versions.items():
                if self._versions.get(stream, -1) != want:
                    raise ConcurrencyError(f"{stream}: expected {want}, at {self._versions.get(stream, -1)}")
            seen = set(self._keys)
            for ev in events:
                if ev.idempotency_key is not None:
                    if ev.idempotency_key in seen:
                        raise DuplicateEvent(ev.idempotency_key)
                    seen.add(ev.idempotency_key)
            versions = dict(self._versions)
            prev = self._events[-1].hash if self._events else GENESIS_HASH
            seq = len(self._events)
            out: list[StoredEvent] = []
            for ev in events:
                seq += 1
                version = versions.get(ev.stream_id, -1) + 1
                versions[ev.stream_id] = version
                event_id = str(uuid.uuid4())
                payload = copy.deepcopy(ev.payload)
                h = compute_hash(
                    prev,
                    event_id=event_id,
                    stream_id=ev.stream_id,
                    stream_version=version,
                    type=ev.type,
                    account_id=ev.account_id,
                    schema_version=ev.schema_version,
                    payload=payload,
                    ts=ev.ts,
                    idempotency_key=ev.idempotency_key,
                )
                out.append(
                    StoredEvent(
                        seq=seq,
                        event_id=event_id,
                        stream_id=ev.stream_id,
                        stream_version=version,
                        type=ev.type,
                        account_id=ev.account_id,
                        schema_version=ev.schema_version,
                        payload=payload,
                        ts=ev.ts,
                        idempotency_key=ev.idempotency_key,
                        prev_hash=prev,
                        hash=h,
                    )
                )
                prev = h
            # commit only after every event validated
            self._events.extend(out)
            self._versions = versions
            self._keys = seen
            return list(out)

    def read_stream(self, stream_id: str) -> list[StoredEvent]:
        with self._lock:
            return [e for e in self._events if e.stream_id == stream_id]

    def read_all(
        self, after_seq: int = 0, account_id: str | None = None, limit: int | None = None
    ) -> list[StoredEvent]:
        with self._lock:
            rows = [
                e
                for e in self._events
                if e.seq > after_seq and (account_id is None or e.account_id == account_id)
            ]
            return rows[:limit] if limit is not None else rows

    def verify_chain(self) -> None:
        with self._lock:
            verify_events(self._events)

    def _tamper(self, index: int, **changes: object) -> None:
        """Test helper that simulates out-of-band tampering."""
        self._events[index] = dataclasses.replace(self._events[index], **changes)  # type: ignore[arg-type]
