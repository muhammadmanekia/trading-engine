"""Strategy definition: a declarative, validated, hashable description of a rule strategy.

There is deliberately no code in a strategy file. Conditions are a small tree (comparisons, crosses,
and/or/not) over named series (price fields and indicators) and numbers. Depth and size are bounded, names
are checked, and every strategy has a stable hash so results can be tied to the exact definition used.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from engine.broker.base import ProtectionMode
from engine.domain.types import AssetClass, Frozen, PosDec
from engine.strategy.indicators import IndicatorSpec

PRICE_FIELDS = ("open", "high", "low", "close", "volume")
_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
MAX_DEPTH = 8
MAX_NODES = 120


class StrategyError(ValueError):
    """The strategy definition is invalid or cannot run on the chosen venue."""


# ---- operands -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Num:
    value: Decimal


@dataclass(frozen=True)
class Ref:
    name: str
    lag: int = 0  # bars back: lag=1 is the previous closed bar


@dataclass(frozen=True)
class Arith:
    op: Literal["add", "sub", "mul", "div"]
    left: Operand
    right: Operand


Operand = Num | Ref | Arith


# ---- conditions ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Compare:
    op: Literal["gt", "gte", "lt", "lte", "eq"]
    left: Operand
    right: Operand


@dataclass(frozen=True)
class Cross:
    direction: Literal["above", "below"]
    left: Operand
    right: Operand


@dataclass(frozen=True)
class All:
    items: tuple[Condition, ...]


@dataclass(frozen=True)
class Any_:
    items: tuple[Condition, ...]


@dataclass(frozen=True)
class Not:
    item: Condition


Condition = Compare | Cross | All | Any_ | Not


class _Budget:
    def __init__(self) -> None:
        self.nodes = 0

    def spend(self, path: str) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise StrategyError(f"{path}: expression too large (max {MAX_NODES} nodes)")


def parse_operand(raw: Any, path: str, budget: _Budget, depth: int = 0) -> Operand:
    budget.spend(path)
    if depth > MAX_DEPTH:
        raise StrategyError(f"{path}: expression nested deeper than {MAX_DEPTH}")
    if isinstance(raw, bool):
        raise StrategyError(f"{path}: booleans are not valid operands")
    if isinstance(raw, int):
        return Num(Decimal(raw))
    if isinstance(raw, float):
        return Num(Decimal(str(raw)))  # TOML floats become the exact decimal the author wrote
    if isinstance(raw, str):
        if _NUMBER.match(raw):
            return Num(Decimal(raw))
        if not _NAME.match(raw):
            raise StrategyError(f"{path}: {raw!r} is not a valid series name")
        return Ref(raw)
    if isinstance(raw, dict):
        if "ref" in raw:
            extra = set(raw) - {"ref", "lag"}
            lag = raw.get("lag", 0)
            name = raw["ref"]
            if (
                extra
                or not isinstance(name, str)
                or not _NAME.match(name)
                or not isinstance(lag, int)
                or lag < 0
            ):
                raise StrategyError(f"{path}: bad ref {raw!r} (use {{ ref = 'name', lag = n }} with n >= 0)")
            return Ref(name, lag)
        if len(raw) == 1:
            ((op, args),) = raw.items()
            if op in ("add", "sub", "mul", "div") and isinstance(args, list) and len(args) == 2:
                return Arith(
                    op,
                    parse_operand(args[0], f"{path}.{op}[0]", budget, depth + 1),
                    parse_operand(args[1], f"{path}.{op}[1]", budget, depth + 1),
                )
    raise StrategyError(f"{path}: cannot parse operand {raw!r}")


def parse_condition(
    raw: Any, path: str = "condition", budget: _Budget | None = None, depth: int = 0
) -> Condition:
    budget = budget or _Budget()
    budget.spend(path)
    if depth > MAX_DEPTH:
        raise StrategyError(f"{path}: condition nested deeper than {MAX_DEPTH}")
    if not isinstance(raw, dict) or len(raw) != 1:
        raise StrategyError(f"{path}: a condition is a table with exactly one key, got {raw!r}")
    ((op, args),) = raw.items()
    if op in ("gt", "gte", "lt", "lte", "eq", "cross_above", "cross_below"):
        if not isinstance(args, list) or len(args) != 2:
            raise StrategyError(f"{path}.{op}: expected [left, right]")
        left = parse_operand(args[0], f"{path}.{op}[0]", budget)
        right = parse_operand(args[1], f"{path}.{op}[1]", budget)
        if op.startswith("cross_"):
            return Cross("above" if op == "cross_above" else "below", left, right)
        return Compare(op, left, right)
    if op in ("all", "any"):
        if not isinstance(args, list) or not args:
            raise StrategyError(f"{path}.{op}: expected a non-empty list of conditions")
        items = tuple(parse_condition(a, f"{path}.{op}[{i}]", budget, depth + 1) for i, a in enumerate(args))
        return All(items) if op == "all" else Any_(items)
    if op == "not":
        return Not(parse_condition(args, f"{path}.not", budget, depth + 1))
    raise StrategyError(f"{path}: unknown operator {op!r}")


def referenced_names(node: Operand | Condition) -> set[str]:
    match node:
        case Num():
            return set()
        case Ref():
            return {node.name}
        case Arith() | Compare() | Cross():
            return referenced_names(node.left) | referenced_names(node.right)
        case All() | Any_():
            return set().union(*(referenced_names(i) for i in node.items))
        case Not():
            return referenced_names(node.item)
    raise StrategyError(f"unknown node {node!r}")  # pragma: no cover


def max_lag(node: Operand | Condition) -> int:
    match node:
        case Num():
            return 0
        case Ref():
            return node.lag
        case Arith():
            return max(max_lag(node.left), max_lag(node.right))
        case Compare():
            return max(max_lag(node.left), max_lag(node.right))
        case Cross():
            return max(max_lag(node.left), max_lag(node.right)) + 1  # a cross looks one bar back
        case All() | Any_():
            return max(max_lag(i) for i in node.items)
        case Not():
            return max_lag(node.item)
    raise StrategyError(f"unknown node {node!r}")  # pragma: no cover


# ---- risk-related parts of a strategy -----------------------------------------------------------
class StopSpec(Frozen):
    type: Literal["atr", "percent"]
    multiple: PosDec | None = None  # atr: stop = close - multiple * ATR
    indicator: str = "atr"  # which ATR indicator to use
    pct: PosDec | None = Field(default=None, lt=1)  # percent: stop = close * (1 - pct)

    @model_validator(mode="after")
    def _check(self) -> StopSpec:
        if self.type == "atr" and self.multiple is None:
            raise ValueError("atr stop needs 'multiple'")
        if self.type == "percent" and self.pct is None:
            raise ValueError("percent stop needs 'pct' (a fraction below 1)")
        return self


class TakeProfitSpec(Frozen):
    type: Literal["r_multiple", "percent"]
    multiple: PosDec | None = None  # r_multiple: target = entry + multiple * (entry - stop)
    pct: PosDec | None = None

    @model_validator(mode="after")
    def _check(self) -> TakeProfitSpec:
        if self.type == "r_multiple" and self.multiple is None:
            raise ValueError("r_multiple take-profit needs 'multiple'")
        if self.type == "percent" and self.pct is None:
            raise ValueError("percent take-profit needs 'pct'")
        return self


class SizingSpec(Frozen):
    type: Literal["risk_fraction", "fixed_notional", "fixed_quantity"]
    risk_pct: PosDec | None = Field(default=None, le=1)  # risk this fraction of the budget per trade
    notional: PosDec | None = None
    quantity: PosDec | None = None

    @model_validator(mode="after")
    def _check(self) -> SizingSpec:
        need = {
            "risk_fraction": self.risk_pct,
            "fixed_notional": self.notional,
            "fixed_quantity": self.quantity,
        }
        if need[self.type] is None:
            raise ValueError(f"{self.type} sizing needs its parameter")
        return self


# ---- the strategy -------------------------------------------------------------------------------
class StrategyDef(Frozen):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")
    version: int = Field(default=1, ge=1)
    description: str = ""
    universe: tuple[str, ...] = Field(min_length=1)
    timeframe: Literal["1h", "1d"]
    indicators: dict[str, IndicatorSpec] = Field(default_factory=dict)
    entry: dict[str, Any]
    exit: dict[str, Any] | None = None
    stop: StopSpec
    take_profit: TakeProfitSpec | None = None
    sizing: SizingSpec
    cooldown_bars: int = Field(default=0, ge=0, le=1000)
    max_positions: int = Field(default=1, ge=1, le=50)
    lookback: int | None = Field(default=None, ge=20, le=5000)

    @field_validator("universe")
    @classmethod
    def _symbols(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for s in v:
            if not re.match(r"^[A-Z0-9.\-]{1,15}$", s):
                raise ValueError(f"bad symbol {s!r}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate symbols in universe")
        return v

    @field_validator("indicators")
    @classmethod
    def _indicator_names(cls, v: dict[str, IndicatorSpec]) -> dict[str, IndicatorSpec]:
        for name in v:
            if not _NAME.match(name) or name in PRICE_FIELDS:
                raise ValueError(f"indicator name {name!r} is invalid or shadows a price field")
        return v

    @model_validator(mode="after")
    def _semantics(self) -> StrategyDef:
        known = set(PRICE_FIELDS) | set(self.indicators)
        for label, raw in (("entry", self.entry), ("exit", self.exit)):
            if raw is None:
                continue
            missing = referenced_names(parse_condition(raw, label)) - known
            if missing:
                raise ValueError(f"{label} refers to undefined series {sorted(missing)}")
        if self.stop.type == "atr":
            spec = self.indicators.get(self.stop.indicator)
            if spec is None or spec.type != "atr":
                raise ValueError(f"atr stop needs an 'atr' indicator named {self.stop.indicator!r}")
        classes = {asset_class_of(s) for s in self.universe}
        if len(classes) > 1:
            raise ValueError("a strategy trades one asset class; split stocks and crypto into two strategies")
        return self

    # ---- derived ----
    @property
    def asset_class(self) -> AssetClass:
        return asset_class_of(self.universe[0])

    @property
    def entry_condition(self) -> Condition:
        return parse_condition(self.entry, "entry")

    @property
    def exit_condition(self) -> Condition | None:
        return parse_condition(self.exit, "exit") if self.exit is not None else None

    @property
    def warmup(self) -> int:
        """Bars of history needed before every referenced value (and the prior bar, for crosses) exists."""
        base = max((s.warmup for s in self.indicators.values()), default=1)
        lag = max_lag(self.entry_condition)
        if self.exit_condition is not None:
            lag = max(lag, max_lag(self.exit_condition))
        return base + lag

    @property
    def window(self) -> int:
        """Fixed trailing window every evaluation uses (identical in backtest and live)."""
        return self.lookback or min(max(self.warmup * 4, 300), 5000)

    @property
    def hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def asset_class_of(symbol: str) -> AssetClass:
    """Crypto pairs are written 'BTC-USD'; stock tickers have no dash."""
    return AssetClass.CRYPTO if "-" in symbol else AssetClass.STOCK


def validate_for_venue(strategy: StrategyDef, protection: ProtectionMode, asset_class: AssetClass) -> None:
    """Reject at load time what the venue cannot do, instead of discovering it as a rejected order."""
    if strategy.asset_class is not asset_class:
        raise StrategyError(
            f"strategy {strategy.id!r} trades {strategy.asset_class} but the venue trades {asset_class}"
        )
    if strategy.take_profit is not None and protection is not ProtectionMode.NATIVE_BRACKET:
        raise StrategyError(
            f"strategy {strategy.id!r} defines a take_profit, but this venue can only hold a stop-loss "
            "(a second sell order would reserve the same shares/coins). "
            "Remove take_profit or add an exit rule."
        )


def parse_strategy(raw: Mapping[str, Any]) -> StrategyDef:
    try:
        return StrategyDef.model_validate(_normalise(dict(raw)))
    except StrategyError:
        raise
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise StrategyError(str(exc)) from exc


def load_strategy(path: str | Path) -> StrategyDef:
    return parse_strategy(tomllib.loads(Path(path).read_text()))


def _normalise(value: Any) -> Any:
    """TOML floats -> strings so exact decimals survive validation and hashing."""
    if isinstance(value, float):
        return str(value)
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def with_overrides(strategy: StrategyDef, overrides: Mapping[str, Any]) -> StrategyDef:
    """A copy with dotted-path parameters replaced, e.g. {'indicators.fast.period': 10, 'stop.multiple': 3}.
    Re-validated, so an override can never produce an invalid strategy."""
    raw = strategy.model_dump(mode="json")
    for path, value in overrides.items():
        node: Any = raw
        *parents, leaf = path.split(".")
        for p in parents:
            node = _step(node, p, path)
        _set(node, leaf, _normalise(value), path)
    return parse_strategy(raw)


def _step(node: Any, key: str, path: str) -> Any:
    """Walk one path segment: a dict key, or a list index (e.g. 'entry.all.0.gt.1')."""
    try:
        return node[int(key)] if isinstance(node, list) else node[key]
    except (KeyError, IndexError, ValueError, TypeError):
        raise StrategyError(f"override path {path!r} does not exist") from None


def _set(node: Any, key: str, value: Any, path: str) -> None:
    try:
        if isinstance(node, list):
            node[int(key)] = value
        elif isinstance(node, dict) and key in node:
            node[key] = value
        else:
            raise KeyError(key)
    except (KeyError, IndexError, ValueError):
        raise StrategyError(f"override path {path!r} does not exist") from None
