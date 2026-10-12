"""LLM provider contract. A provider turns (system prompt, user data, JSON schema) into a structured reply.

Rules every provider follows:
* it NEVER receives broker credentials, and API keys never leave its module or appear in errors/logs
* temperature is 0 and the reply is forced into the supplied JSON schema (tool call / json_schema)
* failures are typed: Unavailable is retryable, InvalidResponse means the model did not follow the schema,
  CredentialsError is fatal for that provider, BudgetExceeded is raised by the budget gate before any call
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol

import httpx
from pydantic import Field

from engine.domain.types import Frozen


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """Network trouble, timeout, 429 or 5xx after retries. Safe to try again later."""


class LLMInvalidResponse(LLMError):
    """The provider answered, but not with usable structured output."""


class LLMCredentialsError(LLMError):
    """The provider rejected our key. Never retried."""


class LLMBudgetExceeded(LLMError):
    """A spending cap would be crossed; no call was made."""


class LLMRequest(Frozen):
    system: str
    user: str
    schema_name: str = Field(pattern=r"^[A-Za-z0-9_]{1,40}$")
    json_schema: dict[str, Any]
    max_output_tokens: int = Field(default=800, ge=16, le=8000)


class LLMResponse(Frozen):
    text: str
    parsed: dict[str, Any] | None
    provider: str
    model_requested: str
    model_reported: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str | None = None


class LLMProvider(Protocol):
    name: str
    model: str
    knowledge_cutoff: date | None

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def input_hash(provider: str, model: str, prompt_hash: str, system: str, user: str, schema_name: str) -> str:
    """Identity of a call's inputs: same hash, same question. Used for the replay cache and the audit trail."""
    blob = json.dumps([provider, model, prompt_hash, system, user, schema_name], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def post_json(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    retries: int = 2,
    sleeper: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """POST with bounded retries for transient failures. Calling a model has no side effects, so retrying is
    safe; it only costs a little. Neither headers nor the body ever appear in an exception message."""
    last: str = "no attempt"
    for attempt in range(retries + 1):
        try:
            resp = client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        else:
            if resp.status_code in (401, 403):
                raise LLMCredentialsError(f"HTTP {resp.status_code}: the provider rejected the API key")
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
                try:
                    wait = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    wait = 0.0
                if wait and attempt < retries:
                    sleeper(min(wait, 10.0))
                    continue
            elif resp.status_code >= 400:
                raise LLMInvalidResponse(f"HTTP {resp.status_code}: {_brief(resp)}")
            else:
                return resp
        if attempt < retries:
            sleeper(min(0.5 * (2**attempt), 5.0))
    raise LLMUnavailable(f"provider unavailable ({last})")


def _brief(resp: httpx.Response) -> str:
    try:
        return json.dumps(resp.json())[:300]
    except ValueError:
        return resp.text[:300]
