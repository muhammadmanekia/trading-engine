"""Intent inbox: the only way signal workers talk to the execution service.

Workers run with a Postgres role that can ONLY INSERT here (see migration 002). They hold no broker keys and
cannot write the event log. The executor validates every row against the Intent schema before it enters the
log, so a buggy or compromised worker can at worst submit intents, which the risk engine then judges.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from engine.clock import Clock
from engine.domain.events import IntentReceived
from engine.domain.intent import Intent
from engine.eventlog.base import DuplicateEvent, EventStore, make_event

log = logging.getLogger("engine.inbox")


@dataclass(frozen=True)
class InboxRow:
    seq: int
    intent_id: str
    payload: dict[str, Any]


class IntentInbox(Protocol):
    def fetch_pending(self, limit: int = 100) -> list[InboxRow]: ...

    def mark(self, seq: int, status: str, detail: str | None = None) -> None: ...


class InMemoryInbox:
    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def put(self, payload: dict[str, Any]) -> bool:
        """Producer side. Duplicate intent ids are ignored, like the Postgres unique constraint."""
        iid = str(payload.get("intent_id", ""))
        if any(r["intent_id"] == iid for r in self._rows):
            return False
        self._rows.append(
            {"seq": len(self._rows) + 1, "intent_id": iid, "payload": payload, "status": "pending"}
        )
        return True

    def fetch_pending(self, limit: int = 100) -> list[InboxRow]:
        return [
            InboxRow(r["seq"], r["intent_id"], r["payload"]) for r in self._rows if r["status"] == "pending"
        ][:limit]

    def mark(self, seq: int, status: str, detail: str | None = None) -> None:
        self._rows[seq - 1]["status"] = status
        self._rows[seq - 1]["detail"] = detail

    def status_of(self, intent_id: str) -> str | None:
        return next((r["status"] for r in self._rows if r["intent_id"] == intent_id), None)


class PostgresInbox:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)

    def fetch_pending(self, limit: int = 100) -> list[InboxRow]:
        rows = self._conn.execute(
            "SELECT seq, intent_id, payload FROM intent_inbox WHERE status = 'pending' ORDER BY seq LIMIT %s",
            (limit,),
        ).fetchall()
        return [InboxRow(r[0], r[1], r[2]) for r in rows]

    def mark(self, seq: int, status: str, detail: str | None = None) -> None:
        self._conn.execute(
            "UPDATE intent_inbox SET status = %s, detail = %s WHERE seq = %s", (status, detail, seq)
        )

    def close(self) -> None:
        self._conn.close()


def submit_to_inbox(dsn: str, intent: Intent) -> bool:
    """What a signal worker calls (connect with the intent_producer role). True if newly queued."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.execute(
            "INSERT INTO intent_inbox (intent_id, payload) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (intent.intent_id, Jsonb(intent.model_dump(mode="json"))),
        )
        return cur.rowcount == 1


def ingest(inbox: IntentInbox, store: EventStore, clock: Clock, accounts: set[str]) -> tuple[int, list[str]]:
    """Move valid pending intents into the event log. Returns (accepted, problems)."""
    accepted, problems = 0, []
    for row in inbox.fetch_pending():
        try:
            intent = Intent.model_validate(row.payload)
        except ValidationError as exc:
            inbox.mark(row.seq, "invalid", f"schema: {exc.error_count()} error(s)")
            problems.append(f"inbox intent {row.intent_id!r} failed validation")
            continue
        if intent.account_id not in accounts:
            inbox.mark(row.seq, "invalid", f"unknown account {intent.account_id!r}")
            problems.append(
                f"inbox intent {intent.intent_id!r} targets unknown account {intent.account_id!r}"
            )
            continue
        with contextlib.suppress(
            DuplicateEvent
        ):  # crashed after appending but before marking: safe to mark now
            store.append(
                [
                    make_event(
                        IntentReceived(account_id=intent.account_id, intent=intent),
                        stream_id=f"intent:{intent.intent_id}",
                        ts=clock.now(),
                        idempotency_key=f"intent:{intent.intent_id}",
                    )
                ]
            )
        inbox.mark(row.seq, "accepted")
        accepted += 1
    return accepted, problems
