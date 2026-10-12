"""A scripted provider for tests, dry runs and replay. It never touches the network."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

from engine.llm.provider import LLMInvalidResponse, LLMRequest, LLMResponse, LLMUnavailable

Responder = Callable[[LLMRequest], "dict[str, Any] | str | Exception"]


class ScriptedProvider:
    """`responder(request)` returns the reply: a dict (valid structured output), a str (raw text, possibly
    garbage), or an Exception instance to raise. Every request is kept for assertions."""

    def __init__(
        self,
        responder: Responder,
        *,
        name: str = "scripted",
        model: str = "scripted-1",
        knowledge_cutoff: date | None = None,
        input_tokens: int = 500,
        output_tokens: int = 120,
    ) -> None:
        self.name, self.model, self.knowledge_cutoff = name, model, knowledge_cutoff
        self._responder = responder
        self._tokens = (input_tokens, output_tokens)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        reply = self._responder(request)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            try:
                parsed = json.loads(reply, parse_float=Decimal)
            except ValueError:
                parsed = None
            if not isinstance(parsed, dict):
                raise LLMInvalidResponse("the model did not return a JSON object")
            text = reply
        else:
            text, parsed = (
                json.dumps(reply, default=str),
                json.loads(json.dumps(reply, default=str), parse_float=Decimal),
            )
        return LLMResponse(
            text=text,
            parsed=parsed,
            provider=self.name,
            model_requested=self.model,
            model_reported=self.model,
            input_tokens=self._tokens[0],
            output_tokens=self._tokens[1],
            latency_ms=1,
            finish_reason="stop",
        )

    @classmethod
    def always(cls, reply: dict[str, Any] | str | Exception, **kw: Any) -> ScriptedProvider:
        return cls(lambda _r: reply, **kw)

    @classmethod
    def down(cls, **kw: Any) -> ScriptedProvider:
        return cls(lambda _r: LLMUnavailable("scripted outage"), **kw)
