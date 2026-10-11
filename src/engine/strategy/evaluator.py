"""Evaluate a strategy on a window of CLOSED bars. A pure function: same bars in, same signal out.

Live trading and the backtester call exactly this function with the same trailing window, which is what
guarantees they agree. It sees nothing after the last bar of the window, so lookahead is impossible by
construction (and a property test proves signals do not change when future bars are appended).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Literal

from engine.marketdata.history import Bar, close_time
from engine.strategy import indicators as ind
from engine.strategy.schema import (
    All,
    Any_,
    Arith,
    Compare,
    Condition,
    Cross,
    Not,
    Num,
    Operand,
    Ref,
    StrategyDef,
)

Series = dict[str, list[Decimal | None]]
Tri = bool | None  # True / False / None (unknown: a value was still warming up)


@dataclass(frozen=True)
class Signal:
    kind: Literal["enter", "exit", "hold"]
    reason: str
    bar_close: datetime
    values: dict[str, Decimal]  # indicator values at the decision bar, for the audit trail


def build_series(strategy: StrategyDef, window: Sequence[Bar]) -> Series:
    series: Series = {
        "open": [b.open for b in window],
        "high": [b.high for b in window],
        "low": [b.low for b in window],
        "close": [b.close for b in window],
        "volume": [b.volume for b in window],
    }
    for name, spec in strategy.indicators.items():
        series[name] = ind.compute(spec, window)
    return series


def operand_value(op: Operand, series: Series, i: int) -> Decimal | None:
    match op:
        case Num():
            return op.value
        case Ref():
            j = i - op.lag
            return series[op.name][j] if j >= 0 else None
        case Arith():
            a, b = operand_value(op.left, series, i), operand_value(op.right, series, i)
            if a is None or b is None:
                return None
            try:
                return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b if b != 0 else None}[op.op]
            except (DivisionByZero, InvalidOperation):
                return None
    raise TypeError(f"unknown operand {op!r}")  # pragma: no cover


def eval_condition(cond: Condition, series: Series, i: int) -> Tri:
    match cond:
        case Compare():
            a, b = operand_value(cond.left, series, i), operand_value(cond.right, series, i)
            if a is None or b is None:
                return None
            return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b, "eq": a == b}[cond.op]
        case Cross():
            la, lb = operand_value(cond.left, series, i), operand_value(cond.right, series, i)
            pa, pb = operand_value(cond.left, series, i - 1), operand_value(cond.right, series, i - 1)
            if None in (la, lb, pa, pb):
                return None
            assert la is not None and lb is not None and pa is not None and pb is not None
            return (pa <= pb and la > lb) if cond.direction == "above" else (pa >= pb and la < lb)
        case All():
            results = [eval_condition(c, series, i) for c in cond.items]
            if any(r is False for r in results):
                return False
            return True if all(r is True for r in results) else None
        case Any_():
            results = [eval_condition(c, series, i) for c in cond.items]
            if any(r is True for r in results):
                return True
            return None if any(r is None for r in results) else False
        case Not():
            inner = eval_condition(cond.item, series, i)
            return None if inner is None else not inner
    raise TypeError(f"unknown condition {cond!r}")  # pragma: no cover


def evaluate(
    strategy: StrategyDef,
    bars: Sequence[Bar],
    *,
    in_position: bool,
    bars_since_exit: int | None = None,
) -> Signal:
    """`bars` are closed bars, oldest first, the last one being the bar that just closed."""
    if not bars:
        raise ValueError("no bars")
    window = list(bars[-strategy.window :])
    when = close_time(window[-1], strategy.timeframe)
    if len(window) < strategy.warmup:
        return Signal("hold", f"warming up ({len(window)}/{strategy.warmup} bars)", when, {})
    series = build_series(strategy, window)
    i = len(window) - 1
    values = {name: v for name in strategy.indicators if (v := series[name][i]) is not None}

    if in_position:
        exit_cond = strategy.exit_condition
        if exit_cond is not None and eval_condition(exit_cond, series, i) is True:
            return Signal("exit", "exit rule", when, values)
        return Signal("hold", "in position", when, values)

    if bars_since_exit is not None and bars_since_exit < strategy.cooldown_bars:
        return Signal("hold", f"cooldown ({bars_since_exit}/{strategy.cooldown_bars} bars)", when, values)
    result = eval_condition(strategy.entry_condition, series, i)
    if result is True:
        return Signal("enter", "entry rule", when, values)
    return Signal("hold", "indicators not ready" if result is None else "no entry", when, values)
