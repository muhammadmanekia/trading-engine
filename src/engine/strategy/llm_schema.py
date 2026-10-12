"""LLM strategy definition, the decision schema the model must follow, and the validator that stands between a
model's output and a trade.

This validator is the security boundary for LLM strategies. It assumes the model may be confused, wrong, or
fully controlled by an attacker, and only lets through decisions that satisfy every structural rule. Whatever it
lets through is still judged by the risk engine and, in propose-only mode, by a human.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from engine.domain.types import AssetClass, Dec, Frozen, PosDec
from engine.news.filters import sanitize
from engine.strategy.schema import StrategyDef, StrategyError, asset_class_of, parse_strategy

ZERO = Decimal(0)
ONE = Decimal(1)
THESIS_MAX = 400
INVALIDATION_MAX = 200
SCHEMA_NAME = "trading_decisions"

# The wire schema is hand-written (not generated) so it stays stable, strict-mode friendly (every property
# required, nullables explicit) and identical for every provider.
DECISIONS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "symbol",
                    "action",
                    "confidence",
                    "horizon",
                    "stop_loss_pct",
                    "thesis",
                    "invalidation",
                    "cited_ids",
                ],
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": ["enter", "exit", "hold"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "horizon": {"type": "string", "enum": ["intraday", "days", "weeks"]},
                    "stop_loss_pct": {"type": ["number", "null"], "minimum": 0.005, "maximum": 0.5},
                    "thesis": {"type": "string", "maxLength": THESIS_MAX},
                    "invalidation": {"type": "string", "maxLength": INVALIDATION_MAX},
                    "cited_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                },
            },
        }
    },
}


class LLMDecision(Frozen):
    symbol: str = Field(min_length=1, max_length=20)
    action: Literal["enter", "exit", "hold"]
    confidence: Dec = Field(ge=0, le=1)
    horizon: Literal["intraday", "days", "weeks"] = "days"
    stop_loss_pct: Dec | None = Field(default=None, gt=0, lt=1)
    thesis: str = Field(default="", max_length=THESIS_MAX)
    invalidation: str = Field(default="", max_length=INVALIDATION_MAX)
    cited_ids: tuple[str, ...] = Field(default=(), max_length=10)


def parse_decisions(parsed: Mapping[str, Any] | None) -> list[LLMDecision]:
    """Strict parse of a provider's structured output. Raises ValueError (including pydantic's) on anything off."""
    if (
        not isinstance(parsed, Mapping)
        or set(parsed) != {"decisions"}
        or not isinstance(parsed["decisions"], list)
    ):
        raise ValueError("reply is not exactly {'decisions': [...]}")
    return [LLMDecision.model_validate(d) for d in parsed["decisions"]]


# ---- the strategy file -------------------------------------------------------------------------------
class NewsConfig(Frozen):
    enabled: bool = True
    lookback_hours: int = Field(default=24, ge=1, le=168)
    max_items: int = Field(default=12, ge=1, le=40)
    min_citations: int = Field(default=1, ge=0, le=5)
    max_symbols_per_item: int = Field(default=4, ge=1, le=10)


class ModelsConfig(Frozen):
    primary: str
    shadow: tuple[str, ...] = ()
    mode: Literal["propose_only", "auto", "shadow_only"] = "propose_only"
    consensus: bool = False  # act only when every listed model agrees

    @model_validator(mode="after")
    def _check(self) -> ModelsConfig:
        if self.primary in self.shadow:
            raise ValueError("a provider cannot be both primary and shadow")
        if self.consensus and not self.shadow:
            raise ValueError("consensus needs at least one shadow provider to agree with")
        return self


class LLMLimits(Frozen):
    min_confidence: Dec = Field(default=Decimal("0.65"), gt=0, le=1)
    max_entries_per_day: int = Field(default=3, ge=0, le=50)
    min_stop_pct: Dec = Field(default=Decimal("0.01"), gt=0, lt=1)
    max_stop_pct: Dec = Field(default=Decimal("0.10"), gt=0, lt=1)
    max_calls_per_day: int = Field(default=60, ge=1, le=2000)
    daily_usd: PosDec | None = None
    max_consecutive_invalid: int = Field(default=5, ge=1, le=100)
    proposal_ttl_minutes: int = Field(default=30, ge=1, le=1440)

    @model_validator(mode="after")
    def _stops(self) -> LLMLimits:
        if self.min_stop_pct >= self.max_stop_pct:
            raise ValueError("min_stop_pct must be below max_stop_pct")
        return self


class LLMSizing(Frozen):
    max_notional: PosDec  # the largest position this strategy may ever take, in dollars
    tiers: tuple[tuple[Dec, Dec], ...] = (
        (Decimal("0.65"), Decimal("0.25")),
        (Decimal("0.80"), Decimal("0.50")),
        (Decimal("0.90"), Decimal("1.00")),
    )  # (confidence at least, fraction of max_notional)

    @field_validator("tiers")
    @classmethod
    def _tiers(cls, v: tuple[tuple[Decimal, Decimal], ...]) -> tuple[tuple[Decimal, Decimal], ...]:
        if not v:
            raise ValueError("at least one sizing tier is required")
        last = ZERO
        for conf, frac in v:
            if not (ZERO < conf <= ONE and ZERO < frac <= ONE) or conf <= last:
                raise ValueError(
                    "tiers must be ascending confidence thresholds in (0,1] with fractions in (0,1]"
                )
            last = conf
        return v


class LLMStrategyDef(Frozen):
    type: Literal["llm"]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")
    version: int = Field(default=1, ge=1)
    description: str = ""
    instructions: str = Field(min_length=20, max_length=4000)
    universe: tuple[str, ...] = Field(min_length=1, max_length=20)
    timeframe: Literal["1h", "1d"] = "1h"
    review_on_bar_close: bool = False  # also ask the model once per closed bar, not only when news arrives
    context_bars: int = Field(default=40, ge=20, le=200)
    prompt: str = "llm_strategy"
    prompt_version: str = "latest"
    news: NewsConfig = NewsConfig()
    models: ModelsConfig
    limits: LLMLimits = LLMLimits()
    sizing: LLMSizing

    @field_validator("universe")
    @classmethod
    def _symbols(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for s in v:
            if not re.match(r"^[A-Z0-9.\-]{1,15}$", s):
                raise ValueError(f"bad symbol {s!r}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate symbols in universe")
        if len({asset_class_of(s) for s in v}) > 1:
            raise ValueError("a strategy trades one asset class; split stocks and crypto")
        return v

    @property
    def asset_class(self) -> AssetClass:
        return asset_class_of(self.universe[0])

    @property
    def hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def parse_llm_strategy(raw: Mapping[str, Any]) -> LLMStrategyDef:
    try:
        return LLMStrategyDef.model_validate(_floats_to_strings(dict(raw)))
    except ValueError as exc:
        raise StrategyError(str(exc)) from exc


def _floats_to_strings(value: Any) -> Any:
    if isinstance(value, float):
        return str(value)
    if isinstance(value, dict):
        return {k: _floats_to_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_floats_to_strings(v) for v in value]
    return value


def load_any_strategy(path: str | Path) -> StrategyDef | LLMStrategyDef:
    """A strategy file is either a rule strategy (default) or `type = "llm"`."""
    raw = tomllib.loads(Path(path).read_text())
    return parse_llm_strategy(raw) if raw.get("type") == "llm" else parse_strategy(raw)


# ---- validation -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rejection:
    symbol: str | None
    reason: str


@dataclass(frozen=True)
class Checked:
    accepted: list[LLMDecision]
    holds: list[LLMDecision]
    rejected: list[Rejection]


def validate_decisions(
    decisions: Sequence[LLMDecision],
    strategy: LLMStrategyDef,
    *,
    supplied_ids: set[str],
    position_qty: Callable[[str], Decimal],
    pending_entry: Callable[[str], bool],
    pending_exit: Callable[[str], bool],
    entries_today: int,
    news_triggered: bool,
) -> Checked:
    """Pure function: which of the model's decisions may become an Intent? Everything else is dropped with a
    reason. The model's text is cleaned before it is stored or shown to a human."""
    accepted: list[LLMDecision] = []
    holds: list[LLMDecision] = []
    rejected: list[Rejection] = []
    universe = set(strategy.universe)

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.symbol] = counts.get(d.symbol, 0) + 1
    entries = entries_today

    for d in decisions:
        if counts[d.symbol] > 1:
            rejected.append(Rejection(d.symbol, "duplicate decisions for one symbol"))
            continue
        if d.symbol not in universe:
            rejected.append(Rejection(d.symbol, "symbol is not in the strategy universe"))
            continue
        clean = d.model_copy(
            update={
                "thesis": sanitize(d.thesis, THESIS_MAX),
                "invalidation": sanitize(d.invalidation, INVALIDATION_MAX),
            }
        )
        if d.action == "hold":
            holds.append(clean)
            continue
        held = position_qty(d.symbol) > 0
        if d.action == "enter":
            problem = _entry_problem(
                clean, strategy, supplied_ids, held, pending_entry(d.symbol), entries, news_triggered
            )
            if problem:
                rejected.append(Rejection(d.symbol, problem))
                continue
            entries += 1
        else:  # exit
            if not held:
                rejected.append(Rejection(d.symbol, "exit requested but nothing is held"))
                continue
            if pending_exit(d.symbol):
                rejected.append(Rejection(d.symbol, "an exit is already in flight"))
                continue
            if any(c not in supplied_ids for c in d.cited_ids):
                rejected.append(Rejection(d.symbol, "cites an id that was not supplied"))
                continue
        accepted.append(clean)
    return Checked(accepted, holds, rejected)


def _entry_problem(
    d: LLMDecision,
    strategy: LLMStrategyDef,
    supplied_ids: set[str],
    held: bool,
    pending: bool,
    entries_so_far: int,
    news_triggered: bool,
) -> str | None:
    lim = strategy.limits
    if held or pending:
        return "already holding or entering"
    if d.confidence < lim.min_confidence:
        return f"confidence {d.confidence} is below the minimum {lim.min_confidence}"
    if d.stop_loss_pct is None:
        return "an entry needs a stop_loss_pct"
    if not (lim.min_stop_pct <= d.stop_loss_pct <= lim.max_stop_pct):
        return f"stop_loss_pct {d.stop_loss_pct} is outside [{lim.min_stop_pct}, {lim.max_stop_pct}]"
    unknown = [c for c in d.cited_ids if c not in supplied_ids]
    if unknown:
        return "cites an id that was not supplied (fabricated source)"
    if (news_triggered or strategy.news.enabled) and len(set(d.cited_ids)) < strategy.news.min_citations:
        return f"an entry must cite at least {strategy.news.min_citations} supplied news item(s)"
    if entries_so_far >= lim.max_entries_per_day:
        return f"daily entry limit of {lim.max_entries_per_day} reached"
    return None


def tier_fraction(confidence: Decimal, sizing: LLMSizing) -> Decimal:
    """The position-size fraction for this confidence: the highest tier whose threshold it meets (0 if none)."""
    frac = ZERO
    for threshold, fraction in sizing.tiers:
        if confidence >= threshold:
            frac = fraction
    return frac


def quantity_for(decision: LLMDecision, sizing: LLMSizing, price: Decimal, qty_step: Decimal) -> Decimal:
    notional = sizing.max_notional * tier_fraction(decision.confidence, sizing)
    return (
        (notional / price / qty_step).to_integral_value(rounding=ROUND_DOWN) * qty_step if price > 0 else ZERO
    )
