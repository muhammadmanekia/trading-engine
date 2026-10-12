"""Anthropic and xAI adapters against fake HTTP servers: request shape, parsing, failure mapping, key hygiene."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from engine.llm.anthropic import AnthropicProvider
from engine.llm.provider import (
    LLMCredentialsError,
    LLMInvalidResponse,
    LLMRequest,
    LLMUnavailable,
)
from engine.llm.xai import XAIProvider
from engine.strategy.llm_schema import DECISIONS_JSON_SCHEMA, SCHEMA_NAME

KEY = "sk-super-secret-key-123"
REQ = LLMRequest(system="sys", user="usr", schema_name=SCHEMA_NAME, json_schema=DECISIONS_JSON_SCHEMA)
DECISIONS = {"decisions": []}


def client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def claude(handler: Any) -> AnthropicProvider:
    return AnthropicProvider(SecretStr(KEY), "claude-test", client=client(handler), sleeper=lambda s: None)


def grok(handler: Any) -> XAIProvider:
    return XAIProvider(SecretStr(KEY), "grok-test", client=client(handler), sleeper=lambda s: None)


def test_claude_forces_a_tool_call_and_parses_the_input() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["headers"], seen["body"] = req.headers, json.loads(req.content)
        return httpx.Response(200, json={
            "model": "claude-test-20260101", "stop_reason": "tool_use", "usage": {"input_tokens": 321, "output_tokens": 45},
            "content": [{"type": "text", "text": "thinking"}, {"type": "tool_use", "name": "submit_decisions", "input": DECISIONS}],
        })  # fmt: skip

    r = claude(handler).complete(REQ)
    assert (
        r.parsed == DECISIONS
        and r.input_tokens == 321
        and r.output_tokens == 45
        and r.model_reported == "claude-test-20260101"
    )
    assert seen["headers"]["x-api-key"] == KEY and seen["body"]["temperature"] == 0
    assert seen["body"]["tool_choice"] == {"type": "tool", "name": "submit_decisions"}
    assert seen["body"]["tools"][0]["input_schema"] == DECISIONS_JSON_SCHEMA and "tools" in seen["body"]


def test_grok_uses_strict_json_schema_and_parses_content() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["headers"], seen["body"] = req.headers, json.loads(req.content)
        return httpx.Response(200, json={
            "model": "grok-test-1", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(DECISIONS)}}],
        })  # fmt: skip

    r = grok(handler).complete(REQ)
    assert (
        r.parsed == DECISIONS
        and r.input_tokens == 100
        and r.output_tokens == 20
        and r.finish_reason == "stop"
    )
    assert seen["headers"]["authorization"] == f"Bearer {KEY}"
    assert (
        seen["body"]["response_format"]["json_schema"]["strict"] is True and seen["body"]["temperature"] == 0
    )


@pytest.mark.parametrize("make", [claude, grok])
def test_both_map_failures_to_typed_errors_and_never_leak_the_key(make: Any) -> None:
    for status, exc in [(401, LLMCredentialsError), (403, LLMCredentialsError), (400, LLMInvalidResponse),
                        (429, LLMUnavailable), (503, LLMUnavailable)]:  # fmt: skip
        with pytest.raises(exc) as e:
            make(lambda req, s=status: httpx.Response(s, json={"error": "nope"})).complete(REQ)
        assert KEY not in str(e.value) and "x-api-key" not in str(e.value).lower()


@pytest.mark.parametrize("make", [claude, grok])
def test_network_errors_are_unavailable_and_retried_a_bounded_number_of_times(make: Any) -> None:
    calls = []

    def boom(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("down", request=req)

    with pytest.raises(LLMUnavailable) as e:
        make(boom).complete(REQ)
    assert len(calls) == 3 and KEY not in str(e.value)


@pytest.mark.parametrize("make", [claude, grok])
def test_replies_without_usable_structure_are_invalid(make: Any) -> None:
    for payload in [{}, {"content": [{"type": "text", "text": "hi"}]}, {"choices": [{"message": {"content": "not json"}}]},
                    {"choices": [{"message": {"content": "[1,2]"}}]}]:  # fmt: skip
        with pytest.raises(LLMInvalidResponse):
            make(lambda req, p=payload: httpx.Response(200, json=p)).complete(REQ)
