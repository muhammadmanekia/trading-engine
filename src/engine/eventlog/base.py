"""Event store contract: append-only, hash-chained, optimistic-concurrency, idempotent."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from engine.domain.events import EVENT_MODELS
from engine.domain.types import Frozen

GENESIS_HASH = "0" * 64


class ConcurrencyError(Exception):
    """The stream's version did not match the caller's expectation."""


class DuplicateEvent(Exception):
    """An event with this idempotency key already exists; nothing was written."""

    def __init__(self, key: str) -> None:
        super().__init__(f"duplicate idempotency key: {key}")
        self.key = key


class ChainError(Exception):
    """The hash chain does not verify; the log has been tampered with or corrupted."""


@dataclass(frozen=True)
class NewEvent:
    stream_id: str
    type: str
    account_id: str
    payload: dict[str, Any]
    ts: datetime
    idempotency_key: str | None = None
    schema_version: int = 1


@dataclass(frozen=True)
class StoredEvent:
    seq: int
    event_id: str
    stream_id: str
    stream_version: int
    type: str
    account_id: str
    schema_version: int
    payload: dict[str, Any]
    ts: datetime
    idempotency_key: str | None
    prev_hash: str
    hash: str


def canonical_ts(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="microseconds")


def compute_hash(
    prev_hash: str,
    *,
    event_id: str,
    stream_id: str,
    stream_version: int,
    type: str,
    account_id: str,
    schema_version: int,
    payload: Mapping[str, Any],
    ts: datetime,
    idempotency_key: str | None,
) -> str:
    body = json.dumps(
        [
            event_id,
            stream_id,
            stream_version,
            type,
            account_id,
            schema_version,
            payload,
            canonical_ts(ts),
            idempotency_key,
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256((prev_hash + body).encode()).hexdigest()


def stored_hash(prev_hash: str, e: StoredEvent) -> str:
    return compute_hash(
        prev_hash,
        event_id=e.event_id,
        stream_id=e.stream_id,
        stream_version=e.stream_version,
        type=e.type,
        account_id=e.account_id,
        schema_version=e.schema_version,
        payload=e.payload,
        ts=e.ts,
        idempotency_key=e.idempotency_key,
    )


def verify_events(events: Sequence[StoredEvent]) -> None:
    """Verify a full, seq-ordered log from genesis. Raises ChainError on any inconsistency."""
    prev = GENESIS_HASH
    for e in events:
        if e.prev_hash != prev:
            raise ChainError(f"seq {e.seq}: prev_hash mismatch")
        if stored_hash(prev, e) != e.hash:
            raise ChainError(f"seq {e.seq}: hash mismatch (event altered)")
        prev = e.hash


class EventStore(Protocol):
    def append(
        self, events: Sequence[NewEvent], expected_versions: Mapping[str, int] | None = None
    ) -> list[StoredEvent]:
        """Atomically append. `expected_versions[stream]` is the version the stream must currently be at
        (-1 for an empty stream). Raises ConcurrencyError / DuplicateEvent and writes nothing."""
        ...

    def read_stream(self, stream_id: str) -> list[StoredEvent]: ...

    def read_all(
        self, after_seq: int = 0, account_id: str | None = None, limit: int | None = None
    ) -> list[StoredEvent]: ...

    def verify_chain(self) -> None: ...


# ---- typed helpers -----------------------------------------------------------------------------
def make_event(
    model: Frozen,
    *,
    stream_id: str,
    ts: datetime,
    idempotency_key: str | None = None,
) -> NewEvent:
    name = type(model).__name__
    if name not in EVENT_MODELS:
        raise ValueError(f"unregistered event model {name}")
    account_id = getattr(model, "account_id", None)
    if not isinstance(account_id, str) or not account_id:
        raise ValueError(f"{name} has no account_id")
    return NewEvent(
        stream_id=stream_id,
        type=name,
        account_id=account_id,
        payload=model.model_dump(mode="json"),
        ts=ts,
        idempotency_key=idempotency_key,
    )


def decode(e: StoredEvent) -> Frozen:
    model = EVENT_MODELS.get(e.type)
    if model is None:
        raise ValueError(f"unknown event type {e.type!r} (seq {e.seq})")
    return model.model_validate(e.payload)
