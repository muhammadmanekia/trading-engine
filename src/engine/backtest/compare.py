"""Compare strategies the same way, against buy-and-hold, without fooling ourselves.

Protocol (see docs/superpowers/specs/2026-10-13-strategy-evaluation-design.md):
* rolling windows BEFORE the exam start: choose parameters on the preceding `train_bars`, score on the next
  `test_bars`, then step forward by `test_bars`;
* one final exam from the exam start to the end of the data, run only on request and recorded in a ledger so a
  strategy edited after seeing its exam is flagged;
* a strategy "beats buy-and-hold" only if it wins the exam, wins at least two thirds of the windows, trades enough
  to be more than luck, and still wins with spread, slippage and fees doubled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Any

from engine.backtest.benchmark import equal_weight_return
from engine.backtest.engine import BacktestConfig
from engine.backtest.walkforward import Objective, backtest, objective_value, parameter_grid
from engine.marketdata.history import Bar, timeframe_delta
from engine.strategy.schema import StrategyDef, with_overrides

ZERO = Decimal(0)


@dataclass(frozen=True)
class Protocol:
    train_bars: int = 504  # about two years of daily bars
    test_bars: int = 126  # about six months
    min_trades: int = 30  # across all scored periods, or the sample is too thin to trust
    tune_min_trades: int = 10  # a parameter set must trade this much on its training window to be eligible
    drawdown_flag: Decimal = Decimal("0.40")  # shown, never a pass/fail condition
    cost_multiplier: Decimal = Decimal(2)  # the cost-stress rerun


@dataclass(frozen=True)
class WindowSpan:
    train_lo: int  # indexes into the strategy's bar timeline
    test_lo: int
    test_hi: int  # exclusive


def required_wins(windows: int) -> int:
    """The smallest number of winning windows that is at least two thirds of `windows`."""
    return (2 * windows + 2) // 3


def rolling_spans(bars_before_exam: int, p: Protocol) -> list[WindowSpan]:
    spans: list[WindowSpan] = []
    test_lo = p.train_bars
    while test_lo + p.test_bars <= bars_before_exam:
        spans.append(WindowSpan(test_lo - p.train_bars, test_lo, test_lo + p.test_bars))
        test_lo += p.test_bars
    return spans


@dataclass(frozen=True)
class WindowScore:
    label: str
    start: datetime
    end: datetime  # exclusive end of the scored period (for the open-ended exam: the last bar's close time)
    strategy_return: Decimal
    benchmark_return: Decimal
    trades: int
    max_drawdown: Decimal
    params: Mapping[str, Any]
    sat_out: bool = False  # no parameter set qualified on the training window, so the strategy held cash

    @property
    def excess(self) -> Decimal:
        return self.strategy_return - self.benchmark_return


@dataclass(frozen=True)
class Evaluation:
    strategy_id: str
    strategy_hash: str
    windows: tuple[WindowScore, ...]
    exam: WindowScore | None


def stressed(config: BacktestConfig, factor: Decimal) -> BacktestConfig:
    v = config.venue
    return replace(
        config,
        venue=v.with_costs(
            spread_bps=v.spread_bps * factor,
            slippage_bps=v.slippage_bps * factor,
            fee_rate=v.fee_rate * factor,
        ),
    )


def _cut(data: Mapping[str, Sequence[Bar]], lo: datetime, hi: datetime | None) -> dict[str, list[Bar]]:
    return {s: [b for b in bars if b.ts >= lo and (hi is None or b.ts < hi)] for s, bars in data.items()}


def _choose(
    strategy: StrategyDef,
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    lo: datetime,
    hi: datetime,
    grid: Mapping[str, Sequence[Any]],
    objective: Objective,
    min_trades: int,
) -> dict[str, Any] | None:
    """The best parameter set on [lo, hi), or None if none made `min_trades`. Without a grid it returns {}
    (the strategy as written) and never sits a window out."""
    candidates = parameter_grid(grid)
    if len(candidates) == 1:
        return candidates[0]
    best: tuple[dict[str, Any], Decimal] | None = None
    for params in candidates:
        variant = with_overrides(strategy, params) if params else strategy
        _, m = backtest(variant, _cut(data, lo, hi), config)
        score = objective_value(m, objective)
        if m.trades >= min_trades and score is not None and (best is None or score > best[1]):
            best = (params, score)
    return best[0] if best else None


def score_span(
    label: str,
    strategy: StrategyDef,
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    timeline: Sequence[datetime],
    span: WindowSpan,
    grid: Mapping[str, Sequence[Any]],
    objective: Objective,
    p: Protocol,
) -> WindowScore:
    train_lo, test_lo = timeline[span.train_lo], timeline[span.test_lo]
    test_hi = timeline[span.test_hi] if span.test_hi < len(timeline) else None
    end = test_hi or timeline[-1] + timeframe_delta(strategy.timeframe)
    try:
        bench = equal_weight_return(_cut(data, test_lo, test_hi), config.venue)
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc
    params = _choose(strategy, data, config, train_lo, test_lo, grid, objective, p.tune_min_trades)
    if params is None:
        return WindowScore(label, test_lo, end, ZERO, bench, 0, ZERO, {}, sat_out=True)
    variant = with_overrides(strategy, params) if params else strategy
    # The first decision is taken at the close of the bar before the window, so its fill is the window's
    # first open, the same price the benchmark pays. That decision only uses past bars.
    context_lo = timeline[max(0, span.test_lo - variant.window)]  # indicators warm up before the window
    _, m = backtest(variant, _cut(data, context_lo, test_hi), config, trade_from=timeline[span.test_lo - 1])
    return WindowScore(label, test_lo, end, m.total_return, bench, m.trades, m.max_drawdown, params)


def evaluate(
    strategy: StrategyDef,
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    p: Protocol,
    exam_start: datetime,
    *,
    run_exam: bool,
    grid: Mapping[str, Sequence[Any]] | None = None,
    objective: Objective = "sharpe",
) -> Evaluation:
    own = {s: data[s] for s in strategy.universe}
    timeline = sorted({b.ts for bars in own.values() for b in bars})
    exam_idx = next((i for i, ts in enumerate(timeline) if ts >= exam_start), None)
    if exam_idx is None:
        raise ValueError("the exam start is after the last bar")
    spans = rolling_spans(exam_idx, p)
    if not spans:
        raise ValueError(
            f"not enough history before the exam: need {p.train_bars + p.test_bars} bars, have {exam_idx}"
        )
    chosen = grid or {}
    windows = tuple(
        score_span(f"window {i}", strategy, own, config, timeline, sp, chosen, objective, p)
        for i, sp in enumerate(spans, start=1)
    )
    exam = None
    if run_exam:
        span = WindowSpan(exam_idx - p.train_bars, exam_idx, len(timeline))
        exam = score_span("final exam", strategy, own, config, timeline, span, chosen, objective, p)
    return Evaluation(strategy.id, strategy.hash, windows, exam)
