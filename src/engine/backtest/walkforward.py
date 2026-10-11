"""Guardrails against fooling yourself: walk-forward testing and parameter-sensitivity checks.

A single backtest over all the data answers "what would have been best in hindsight". Walk-forward asks the
question that matters: when parameters are chosen ONLY from the past, how does the strategy do on the data that
comes next? A large gap between in-sample and out-of-sample results, parameters that jump around from window to
window, or a result that collapses when a parameter moves 20%, all mean the edge is probably an artefact.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from engine.backtest.engine import BacktestConfig, BacktestEngine
from engine.backtest.metrics import Metrics, compute_metrics
from engine.marketdata.history import Bar
from engine.strategy.schema import StrategyDef, with_overrides
from engine.strategy.source import RuleStrategySource

ZERO = Decimal(0)
Objective = Literal["sharpe", "total_return", "avg_r"]
MAX_GRID = 400


@dataclass(frozen=True)
class WindowResult:
    train: tuple[datetime, datetime]
    test: tuple[datetime, datetime]
    params: Mapping[str, Any]
    in_sample: Decimal | None  # objective on the training window
    out_of_sample: Decimal | None  # objective on the unseen test window
    test_return: Decimal
    test_trades: int


@dataclass
class WalkForwardReport:
    objective: Objective
    windows: list[WindowResult]
    oos_return: Decimal  # test-window returns compounded (each window restarts from the same capital)
    oos_trades: int
    efficiency: (
        Decimal | None
    )  # mean out-of-sample / mean in-sample objective: ~1 is honest, ~0 or negative is overfit
    stability: Decimal  # share of windows that chose the most common parameter set
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SensitivityPoint:
    path: str
    value: Any
    objective: Decimal | None
    total_return: Decimal
    trades: int


@dataclass
class SensitivityReport:
    base: SensitivityPoint
    neighbours: list[SensitivityPoint]
    profitable_share: Decimal
    warnings: list[str] = field(default_factory=list)


# ---- helpers ------------------------------------------------------------------------------------
def objective_value(m: Metrics, objective: Objective) -> Decimal | None:
    return {"sharpe": m.sharpe, "total_return": m.total_return, "avg_r": m.avg_r}[objective]


def backtest(
    strategy: StrategyDef,
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    *,
    trade_from: datetime | None = None,
) -> tuple[Any, Metrics]:
    engine = BacktestEngine([RuleStrategySource(strategy, config.account_id)], data, config)
    result = engine.run(trade_from=trade_from)
    return result, compute_metrics(result)


def parameter_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    size = 1
    for values in grid.values():
        size *= len(values)
    if size > MAX_GRID:
        raise ValueError(f"parameter grid has {size} combinations (max {MAX_GRID}); a bigger search overfits")
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))]


def _slice(
    data: Mapping[str, Sequence[Bar]], start: datetime, end: datetime, *, inclusive: bool = False
) -> dict[str, list[Bar]]:
    return {
        sym: [b for b in bars if start <= b.ts and (b.ts <= end if inclusive else b.ts < end)]
        for sym, bars in data.items()
    }


# ---- walk-forward -------------------------------------------------------------------------------
def walk_forward(
    strategy: StrategyDef,
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    train_bars: int,
    test_bars: int,
    objective: Objective = "sharpe",
    min_trades: int = 10,
) -> WalkForwardReport:
    """Slide a (train, test) pair along the timeline. On each train window pick the best parameter set that made
    at least `min_trades` trades, then run ONLY that set on the following unseen test window."""
    lead = strategy.universe[0]
    timeline = [b.ts for b in data[lead]]
    warm = strategy.window
    candidates = parameter_grid(param_grid)
    windows: list[WindowResult] = []
    start = 0
    while start + train_bars + test_bars <= len(timeline):
        train_lo, train_hi = timeline[start], timeline[start + train_bars]
        test_lo = timeline[start + train_bars]
        test_hi = timeline[start + train_bars + test_bars - 1]  # last bar of the test window (inclusive)
        best: tuple[dict[str, Any], Decimal | None] | None = None
        for params in candidates:
            variant = with_overrides(strategy, params) if params else strategy
            _, m = backtest(variant, _slice(data, train_lo, train_hi), config)
            score = objective_value(m, objective)
            if (
                m.trades >= min_trades
                and score is not None
                and (best is None or best[1] is None or score > best[1])
            ):
                best = (params, score)
        if best is None:
            windows.append(WindowResult((train_lo, train_hi), (test_lo, test_hi), {}, None, None, ZERO, 0))
        else:
            variant = with_overrides(strategy, best[0]) if best[0] else strategy
            # indicators get history before the test window, but nothing trades until test_lo
            context_lo = timeline[max(0, start + train_bars - warm)]
            _, tm = backtest(
                variant, _slice(data, context_lo, test_hi, inclusive=True), config, trade_from=test_lo
            )
            windows.append(
                WindowResult(
                    (train_lo, train_hi), (test_lo, test_hi), best[0], best[1],
                    objective_value(tm, objective), tm.total_return, tm.trades,
                )
            )  # fmt: skip
        start += test_bars
    return _summarise(objective, windows, min_trades)


def _summarise(objective: Objective, windows: list[WindowResult], min_trades: int) -> WalkForwardReport:
    compounded = Decimal(1)
    for w in windows:
        compounded *= 1 + w.test_return
    traded = [w for w in windows if w.in_sample is not None]
    ins = [w.in_sample for w in traded if w.in_sample is not None]
    oos = [w.out_of_sample for w in traded if w.out_of_sample is not None]
    efficiency = None
    if ins and oos and sum(ins, ZERO) != 0:
        efficiency = (sum(oos, ZERO) / len(oos)) / (sum(ins, ZERO) / len(ins))
    common = Counter(tuple(sorted(w.params.items())) for w in traded)
    stability = Decimal(common.most_common(1)[0][1]) / len(traded) if traded else ZERO

    rep = WalkForwardReport(
        objective=objective,
        windows=windows,
        oos_return=compounded - 1,
        oos_trades=sum(w.test_trades for w in windows),
        efficiency=efficiency,
        stability=stability,
    )
    if not windows:
        rep.warnings.append("not enough data for even one train+test window")
    if windows and not traded:
        rep.warnings.append(f"no parameter set reached {min_trades} trades on any training window")
    if len(windows) < 4:
        rep.warnings.append(f"only {len(windows)} windows: weak evidence either way")
    if efficiency is not None and efficiency < Decimal("0.5"):
        rep.warnings.append(
            f"out-of-sample {objective} is only {efficiency:.0%} of in-sample: the edge probably does not survive"
        )
    if traded and stability < Decimal("0.4"):
        rep.warnings.append(
            f"the best parameters changed in most windows (stability {stability:.0%}): fragile"
        )
    if rep.oos_trades < 30:
        rep.warnings.append(f"only {rep.oos_trades} out-of-sample trades: too few to trust")
    if traded and sum(1 for w in traded if w.test_return > 0) * 2 < len(traded):
        rep.warnings.append("fewer than half the unseen windows were profitable")
    return rep


# ---- sensitivity --------------------------------------------------------------------------------
def _get_path(raw: Any, path: str) -> Any:
    node = raw
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def sensitivity(
    strategy: StrategyDef,
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    paths: Sequence[str],
    *,
    objective: Objective = "sharpe",
    shift: Decimal = Decimal("0.2"),
) -> SensitivityReport:
    """Nudge each numeric parameter by +/- `shift` and see whether the result survives. A robust edge sits on a
    plateau; an overfit one sits on a needle."""
    raw = strategy.model_dump(mode="json")
    _, base_m = backtest(strategy, data, config)
    base = SensitivityPoint(
        "base", None, objective_value(base_m, objective), base_m.total_return, base_m.trades
    )
    points: list[SensitivityPoint] = []
    for path in paths:
        current = _get_path(raw, path)
        for factor in (1 - shift, 1 + shift):
            value = _scaled(current, factor)
            if value == current:
                continue
            _, m = backtest(with_overrides(strategy, {path: value}), data, config)
            points.append(
                SensitivityPoint(path, value, objective_value(m, objective), m.total_return, m.trades)
            )
    profitable = Decimal(sum(1 for p in points if p.total_return > 0)) / len(points) if points else ZERO
    rep = SensitivityReport(base, points, profitable)
    if base_m.total_return <= 0:
        rep.warnings.append("the base case is not profitable; sensitivity says little")
    elif points and profitable < Decimal("0.7"):
        rep.warnings.append(f"only {profitable:.0%} of nearby parameter values are profitable: fragile")
    flips = [p for p in points if p.total_return < 0 < base_m.total_return]
    if flips:
        rep.warnings.append(f"the sign flips for {', '.join(sorted({p.path for p in flips}))}")
    return rep


def _scaled(value: Any, factor: Decimal) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(1, int((Decimal(value) * factor).to_integral_value()))
    return str((Decimal(str(value)) * factor).normalize())
