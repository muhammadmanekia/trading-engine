"""Grok via xAI's OpenAI-compatible chat completions endpoint, with a strict json_schema response format.

This module and anthropic.py are the only places an LLM API key is handled.
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

URL = "https://api.x.ai/v1/chat/completions"


class XAIProvider:
    name = "grok"

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
        name: str = "grok",
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
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": request.schema_name, "schema": request.json_schema, "strict": True},
            },
        }
        headers = {"Authorization": f"Bearer {self._key.get_secret_value()}"}
        started = time.monotonic()
        resp = post_json(self._client, self._url, headers, body, sleeper=self._sleep)
        latency = int((time.monotonic() - started) * 1000)
        try:
            data = resp.json()
            choice = data["choices"][0]
            text = choice["message"]["content"]
            usage = data.get("usage") or {}
            parsed = json.loads(text)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponse(f"{self.name}: reply was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMInvalidResponse(f"{self.name}: reply was not a JSON object")
        return LLMResponse(
            text=text,
            parsed=parsed,
            provider=self.name,
            model_requested=self.model,
            model_reported=data.get("model"),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency,
            finish_reason=choice.get("finish_reason"),
        )
