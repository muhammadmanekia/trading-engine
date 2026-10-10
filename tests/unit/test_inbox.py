"""Intent inbox: the narrow door between signal workers (no keys) and the execution service."""

from __future__ import annotations

import psycopg
import pytest

from engine.clock import FakeClock
from engine.domain.intent import Intent
from engine.eventlog.base import EventStore
from engine.eventlog.memory import InMemoryEventStore
from engine.service.inbox import InMemoryInbox, PostgresInbox, ingest, submit_to_inbox
from tests.helpers import ACCOUNT, intent_now


def payload(clock: FakeClock, **kw: object) -> dict[str, object]:
    return intent_now(clock, **kw).model_dump(mode="json")


def test_valid_intents_enter_the_log_and_are_marked_accepted() -> None:
    clock, inbox, store = FakeClock(), InMemoryInbox(), InMemoryEventStore()
    p = payload(clock)
    inbox.put(p)
    accepted, problems = ingest(inbox, store, clock, {ACCOUNT})
    assert (accepted, problems) == (1, [])
    assert inbox.status_of(str(p["intent_id"])) == "accepted"
    assert [e.type for e in store.read_all()] == ["IntentReceived"]
    assert ingest(inbox, store, clock, {ACCOUNT}) == (0, [])  # nothing pending any more


def test_invalid_payloads_never_reach_the_log() -> None:
    clock, inbox, store = FakeClock(), InMemoryInbox(), InMemoryEventStore()
    bad = payload(clock)
    bad["quantity"] = 1.5  # float
    inbox.put(bad)
    no_stop_type = payload(clock)
    no_stop_type["surprise"] = "field"
    inbox.put(no_stop_type)
    accepted, problems = ingest(inbox, store, clock, {ACCOUNT})
    assert accepted == 0 and len(problems) == 2 and store.read_all() == []
    assert inbox.status_of(str(bad["intent_id"])) == "invalid"


def test_intents_for_unknown_accounts_are_refused() -> None:
    clock, inbox, store = FakeClock(), InMemoryInbox(), InMemoryEventStore()
    inbox.put(payload(clock, account_id="somebody-else"))
    accepted, problems = ingest(inbox, store, clock, {ACCOUNT})
    assert accepted == 0 and "unknown account" in problems[0] and store.read_all() == []


def test_duplicate_submission_is_ignored() -> None:
    clock, inbox = FakeClock(), InMemoryInbox()
    p = payload(clock)
    assert inbox.put(p) and not inbox.put(p)


def test_crash_between_append_and_mark_is_safe() -> None:
    """The row is still pending after a crash; re-ingesting must not duplicate the event."""
    clock, inbox, store = FakeClock(), InMemoryInbox(), InMemoryEventStore()
    p = payload(clock)
    inbox.put(p)
    ingest(inbox, store, clock, {ACCOUNT})
    inbox._rows[0]["status"] = "pending"  # simulate: the 'accepted' mark was lost
    assert ingest(inbox, store, clock, {ACCOUNT}) == (1, [])
    assert len(store.read_all()) == 1


def test_postgres_roundtrip_and_producer_cannot_touch_anything_else(pg_dsn: str, store: EventStore) -> None:
    clock = FakeClock()
    intent = Intent.model_validate(payload(clock))
    assert submit_to_inbox(pg_dsn, intent) and not submit_to_inbox(pg_dsn, intent)  # idempotent
    inbox = PostgresInbox(pg_dsn)
    rows = inbox.fetch_pending()
    assert [r.intent_id for r in rows] == [intent.intent_id]
    inbox.mark(rows[0].seq, "accepted")
    assert inbox.fetch_pending() == []
    inbox.close()


def test_producer_role_is_insert_only(pg_dsn: str) -> None:
    clock = FakeClock()
    with psycopg.connect(pg_dsn, autocommit=True) as c:
        c.execute("SET ROLE intent_producer")
        c.execute(  # the one thing a worker may do
            "INSERT INTO intent_inbox (intent_id, payload) VALUES ('x', '{}'::jsonb) ON CONFLICT DO NOTHING"
        )
        for forbidden in (
            "SELECT * FROM events",
            "INSERT INTO events (event_id, stream_id, stream_version, type, account_id, payload, ts, prev_hash, hash)"
            " VALUES (gen_random_uuid(), 's', 0, 't', 'a', '{}', now(), repeat('0',64), repeat('1',64))",
            "UPDATE intent_inbox SET status = 'accepted'",
            "DELETE FROM intent_inbox",
            "SELECT * FROM intent_inbox",
            "SELECT * FROM paper_broker_state",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                c.execute(forbidden)  # type: ignore[call-overload]
    assert clock is not None
