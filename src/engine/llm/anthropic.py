"""Claude via the Messages API. Output is forced through a single tool call so the reply is schema-shaped JSON.

This module and xai.py are the only places an LLM API key is handled.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
from pydantic import SecretStr

from engine.llm.provider import LLMInvalidResponse, LLMRequest, LLMResponse, post_json

URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
TOOL_NAME = "submit_decisions"


class AnthropicProvider:
    name = "claude"

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        *,
        knowledge_cutoff: date | None = None,
        url: str = URL,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        timeout_s: float = 60.0,
        name: str = "claude",
    ) -> None:
        self._key, self.model, self.knowledge_cutoff = api_key, model, knowledge_cutoff
        self._url, self._sleep = url, sleeper
        self._client = client or httpx.Client(timeout=timeout_s)
        self.name = name

    def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "temperature": 0,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": "Submit your trading decisions in the required structure.",
                    "input_schema": request.json_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
        }
        headers = {"x-api-key": self._key.get_secret_value(), "anthropic-version": API_VERSION}
        started = time.monotonic()
        resp = post_json(self._client, self._url, headers, body, sleeper=self._sleep)
        latency = int((time.monotonic() - started) * 1000)
        try:
            data = resp.json()
            blocks = data.get("content") or []
            tool = next(b for b in blocks if b.get("type") == "tool_use" and b.get("name") == TOOL_NAME)
            parsed = tool["input"]
            usage = data.get("usage") or {}
        except (ValueError, StopIteration, KeyError, TypeError, AttributeError) as exc:
            raise LLMInvalidResponse(f"{self.name}: no usable tool call in reply") from exc
        if not isinstance(parsed, dict):
            raise LLMInvalidResponse(f"{self.name}: tool input was not an object")
        return LLMResponse(
            text=json.dumps(parsed, sort_keys=True),
            parsed=parsed,
            provider=self.name,
            model_requested=self.model,
            model_reported=data.get("model"),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            latency_ms=latency,
            finish_reason=data.get("stop_reason"),
        )
