"""LLMCall records: the full, replayable account of every model call.

The authoritative audit trail is the hash-chained LlmDecisionMade event in the event log; it carries the hashes of
the prompt and response stored here. Together they let anyone reconstruct exactly what the model saw, what it
said, which model version said it, what it cost, and what the engine did with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb

ZERO = Decimal(0)


@dataclass(frozen=True)
class LLMCall:
    call_id: str
    ts: datetime
    account_id: str
    strategy_id: str
    mode: str  # "primary" | "shadow"
    provider: str
    model_requested: str
    model_reported: str | None
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    input_hash: str
    system_text: str
    user_text: str
    response_text: str | None
    valid: bool
    error: str | None
    decisions: list[dict[str, Any]] | None
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    latency_ms: int
    trigger: str = ""
    extra: dict[str, Any] = field(default_factory=dict)  # not persisted; for tests/diagnostics


class LLMRecorder(Protocol):
    def record(self, call: LLMCall) -> None: ...

    def find_by_input_hash(self, input_hash: str) -> LLMCall | None: ...

    def cost_since(self, ts: datetime, strategy_id: str | None = None) -> Decimal: ...

    def calls_since(self, ts: datetime, strategy_id: str | None = None) -> list[LLMCall]: ...


class InMemoryLLMRecorder:
    def __init__(self) -> None:
        self.calls: list[LLMCall] = []

    def record(self, call: LLMCall) -> None:
        self.calls.append(call)

    def find_by_input_hash(self, input_hash: str) -> LLMCall | None:
        return next((c for c in reversed(self.calls) if c.input_hash == input_hash and c.valid), None)

    def cost_since(self, ts: datetime, strategy_id: str | None = None) -> Decimal:
        return sum(
            (c.cost_usd for c in self.calls if c.ts >= ts and strategy_id in (None, c.strategy_id)), ZERO
        )

    def calls_since(self, ts: datetime, strategy_id: str | None = None) -> list[LLMCall]:
        return [c for c in self.calls if c.ts >= ts and strategy_id in (None, c.strategy_id)]


_COLS = (
    "call_id, ts, account_id, strategy_id, mode, provider, model_requested, model_reported, prompt_name, "
    "prompt_version, prompt_hash, input_hash, system_text, user_text, response_text, valid, error, decisions, "
    "input_tokens, output_tokens, cost_usd, latency_ms, trigger"
)


def _row(r: tuple[Any, ...]) -> LLMCall:
    return LLMCall(
        call_id=r[0], ts=r[1], account_id=r[2], strategy_id=r[3], mode=r[4], provider=r[5],
        model_requested=r[6], model_reported=r[7], prompt_name=r[8], prompt_version=r[9], prompt_hash=r[10],
        input_hash=r[11], system_text=r[12], user_text=r[13], response_text=r[14], valid=r[15], error=r[16],
        decisions=r[17], input_tokens=r[18], output_tokens=r[19], cost_usd=Decimal(r[20]),
        latency_ms=r[21], trigger=r[22],
    )  # fmt: skip


class PostgresLLMRecorder:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)

    def record(self, call: LLMCall) -> None:
        self._conn.execute(
            f"INSERT INTO llm_calls ({_COLS}) VALUES ({','.join(['%s'] * 23)}) ON CONFLICT (call_id) DO NOTHING",  # noqa: S608
            (call.call_id, call.ts, call.account_id, call.strategy_id, call.mode, call.provider, call.model_requested,
             call.model_reported, call.prompt_name, call.prompt_version, call.prompt_hash, call.input_hash,
             call.system_text, call.user_text, call.response_text, call.valid, call.error,
             Jsonb(json.loads(json.dumps(call.decisions, default=str))) if call.decisions is not None else None,
             call.input_tokens, call.output_tokens, call.cost_usd, call.latency_ms, call.trigger),
        )  # fmt: skip

    def find_by_input_hash(self, input_hash: str) -> LLMCall | None:
        row = self._conn.execute(
            f"SELECT {_COLS} FROM llm_calls WHERE input_hash = %s AND valid ORDER BY ts DESC LIMIT 1",  # noqa: S608
            (input_hash,),
        ).fetchone()
        return _row(row) if row else None

    def cost_since(self, ts: datetime, strategy_id: str | None = None) -> Decimal:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE ts >= %s AND (%s::text IS NULL OR strategy_id = %s)",
            (ts, strategy_id, strategy_id),
        ).fetchone()
        return Decimal(row[0]) if row else ZERO

    def calls_since(self, ts: datetime, strategy_id: str | None = None) -> list[LLMCall]:
        rows = self._conn.execute(
            f"SELECT {_COLS} FROM llm_calls WHERE ts >= %s AND (%s::text IS NULL OR strategy_id = %s) ORDER BY ts",  # noqa: S608
            (ts, strategy_id, strategy_id),
        ).fetchall()
        return [_row(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
