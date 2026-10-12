"""Propose-only mode: a model's trade idea becomes a Proposal that waits for a human.

Nothing here can place an order. Approving a proposal drops a fresh Intent into the intent inbox, the same door
every other signal uses, so the risk engine still has the final say. Proposals expire quickly because news goes
stale: approving a 3-hour-old headline trade is how people lose money.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

import psycopg
from psycopg.types.json import Jsonb

from engine.domain.intent import Intent

Status = Literal["pending", "approved", "rejected", "expired"]
APPROVED_INTENT_TTL = timedelta(minutes=10)


class ApprovalError(Exception):
    pass


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    account_id: str
    strategy_id: str
    intent: Intent
    thesis: str
    cited: tuple[str, ...]
    created_at: datetime
    expires_at: datetime
    status: Status = "pending"
    decided_at: datetime | None = None
    call_id: str | None = None


class ProposalStore(Protocol):
    def add(self, p: Proposal) -> bool: ...

    def get(self, proposal_id: str) -> Proposal | None: ...

    def pending(self, now: datetime, account_id: str | None = None) -> list[Proposal]: ...

    def count_since(self, ts: datetime, strategy_id: str) -> int: ...

    def decide(self, proposal_id: str, status: Status, now: datetime) -> bool:
        """Move a PENDING proposal to a final status. False if it was not pending (so it can only be decided once)."""
        ...


class InMemoryProposalStore:
    def __init__(self) -> None:
        self._items: dict[str, Proposal] = {}

    def add(self, p: Proposal) -> bool:
        if p.proposal_id in self._items:
            return False
        self._items[p.proposal_id] = p
        return True

    def get(self, proposal_id: str) -> Proposal | None:
        return self._items.get(proposal_id)

    def pending(self, now: datetime, account_id: str | None = None) -> list[Proposal]:
        return [
            p for p in self._items.values()
            if p.status == "pending" and p.expires_at > now and account_id in (None, p.account_id)
        ]  # fmt: skip

    def count_since(self, ts: datetime, strategy_id: str) -> int:
        return sum(1 for p in self._items.values() if p.strategy_id == strategy_id and p.created_at >= ts)

    def decide(self, proposal_id: str, status: Status, now: datetime) -> bool:
        p = self._items.get(proposal_id)
        if p is None or p.status != "pending":
            return False
        self._items[proposal_id] = Proposal(**{**p.__dict__, "status": status, "decided_at": now})
        return True


class PostgresProposalStore:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)

    @staticmethod
    def _row(r: tuple[Any, ...]) -> Proposal:
        return Proposal(
            proposal_id=r[0], account_id=r[1], strategy_id=r[2], intent=Intent.model_validate(r[3]), thesis=r[4],
            cited=tuple(r[5]), status=r[6], created_at=r[7], expires_at=r[8], decided_at=r[9], call_id=r[10],
        )  # fmt: skip

    _COLS = "proposal_id, account_id, strategy_id, intent, thesis, cited, status, created_at, expires_at, decided_at, call_id"

    def add(self, p: Proposal) -> bool:
        cur = self._conn.execute(
            f"INSERT INTO proposals ({self._COLS}) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",  # noqa: S608
            (p.proposal_id, p.account_id, p.strategy_id, Jsonb(json.loads(p.intent.model_dump_json())), p.thesis,
             Jsonb(list(p.cited)), p.status, p.created_at, p.expires_at, p.decided_at, p.call_id),
        )  # fmt: skip
        return cur.rowcount == 1

    def get(self, proposal_id: str) -> Proposal | None:
        row = self._conn.execute(
            f"SELECT {self._COLS} FROM proposals WHERE proposal_id = %s",  # noqa: S608
            (proposal_id,),
        ).fetchone()
        return self._row(row) if row else None

    def pending(self, now: datetime, account_id: str | None = None) -> list[Proposal]:
        rows = self._conn.execute(
            f"SELECT {self._COLS} FROM proposals WHERE status = 'pending' AND expires_at > %s "  # noqa: S608
            "AND (%s::text IS NULL OR account_id = %s) ORDER BY created_at",
            (now, account_id, account_id),
        ).fetchall()
        return [self._row(r) for r in rows]

    def count_since(self, ts: datetime, strategy_id: str) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM proposals WHERE strategy_id = %s AND created_at >= %s", (strategy_id, ts)
        ).fetchone()
        return int(row[0]) if row else 0

    def decide(self, proposal_id: str, status: Status, now: datetime) -> bool:
        cur = self._conn.execute(
            "UPDATE proposals SET status = %s, decided_at = %s WHERE proposal_id = %s AND status = 'pending'",
            (status, now, proposal_id),
        )
        return cur.rowcount == 1

    def close(self) -> None:
        self._conn.close()


def approve(
    store: ProposalStore, proposal_id: str, now: datetime, submit: Callable[[Intent], bool]
) -> Intent:
    """Approve a pending, unexpired proposal: queue a FRESH intent (new timestamps, same id and stop) for the
    execution service. The risk engine will still judge it against the market as it is right now."""
    p = store.get(proposal_id)
    if p is None:
        raise ApprovalError(f"no proposal {proposal_id!r}")
    if p.status != "pending":
        raise ApprovalError(f"proposal is already {p.status}")
    if now >= p.expires_at:
        store.decide(proposal_id, "expired", now)
        raise ApprovalError("this proposal has expired; the news is stale. Wait for a fresh one")
    if not store.decide(proposal_id, "approved", now):  # lost a race with another approver
        raise ApprovalError("proposal was decided by someone else")
    intent = p.intent.model_copy(update={"created_at": now, "expires_at": now + APPROVED_INTENT_TTL})
    submit(intent)
    return intent


def reject(store: ProposalStore, proposal_id: str, now: datetime) -> None:
    p = store.get(proposal_id)
    if p is None:
        raise ApprovalError(f"no proposal {proposal_id!r}")
    if not store.decide(proposal_id, "rejected", now):
        raise ApprovalError(f"proposal is already {p.status}")
