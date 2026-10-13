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

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

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


ExamStatus = Literal["not run", "new", "clean", "seen by an earlier version"]


def exam_fingerprint(**parts: Any) -> str:
    """A digest of every setting that shapes an exam result (grid, objective, protocol, dates, risk, costs, cash),
    so rerunning the same file under different settings cannot pass for a first, clean exam."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class ExamLedger:
    """Which strategy versions have had their final exam, and under which settings. An entry matches only the same
    strategy hash AND the same settings fingerprint. The first exam of a strategy id is 'clean'; every later one
    (an edited file, or the same file under different settings) is permanently 'seen by an earlier version',
    because its author had already looked at an exam result."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, list[dict[str, Any]]] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                if not isinstance(loaded, dict):
                    raise ValueError("not an object")
                self._data = loaded
            except ValueError as exc:
                raise ValueError(f"exam ledger {path} is unreadable ({exc}); fix or delete it") from exc

    def _match(self, strategy_id: str, strategy_hash: str, fingerprint: str) -> dict[str, Any] | None:
        return next(
            (
                e
                for e in self._data.get(strategy_id, [])
                if e["hash"] == strategy_hash and e["fingerprint"] == fingerprint
            ),
            None,
        )

    def status(self, strategy_id: str, strategy_hash: str, fingerprint: str) -> ExamStatus:
        entry = self._match(strategy_id, strategy_hash, fingerprint)
        if entry is not None:
            return "clean" if entry["clean"] else "seen by an earlier version"
        return "seen by an earlier version" if self._data.get(strategy_id) else "new"

    def record(
        self,
        strategy_id: str,
        strategy_hash: str,
        fingerprint: str,
        exam_return: Decimal,
        benchmark: Decimal,
        now: datetime,
    ) -> None:
        if self._match(strategy_id, strategy_hash, fingerprint) is not None:
            return
        entries = self._data.setdefault(strategy_id, [])
        entries.append(
            {
                "hash": strategy_hash,
                "fingerprint": fingerprint,
                "clean": not entries,
                "recorded": now.isoformat(),
                "exam_return": str(exam_return),
                "benchmark_return": str(benchmark),
            }
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(self._data, fh, indent=2)
        os.replace(tmp, self._path)  # atomic: a crash never leaves a half-written ledger


@dataclass(frozen=True)
class Verdict:
    label: Literal["PASS", "NOT PROVEN", "EXAM NOT RUN", "EXAM ALREADY SEEN"]
    checks: tuple[
        tuple[str, bool | None, str], ...
    ]  # (condition, met?, detail); None = could not be assessed
    flags: tuple[str, ...]


def _won(windows: Sequence[WindowScore]) -> int:
    return sum(1 for w in windows if w.excess > 0)


def judge(base: Evaluation, stress: Evaluation, p: Protocol, exam_status: ExamStatus) -> Verdict:
    n = len(base.windows)
    need = required_wins(n)
    won, stress_won = _won(base.windows), _won(stress.windows)
    trades = sum(w.trades for w in base.windows) + (base.exam.trades if base.exam else 0)

    exam_ok: bool | None = None
    exam_detail = "not run (use --final-exam)"
    if base.exam is not None:
        exam_ok = base.exam.excess > 0
        exam_detail = f"{base.exam.strategy_return:.2%} vs buy-and-hold {base.exam.benchmark_return:.2%}"
    # without an exam there is nothing to stress on the exam side, so the check cannot be assessed
    stress_ok: bool | None = None if stress.exam is None else (stress_won >= need and stress.exam.excess > 0)
    stress_detail = f"{stress_won}/{n} windows" + (
        f", exam {stress.exam.strategy_return:.2%} vs {stress.exam.benchmark_return:.2%}"
        if stress.exam
        else ""
    )
    checks = (
        ("beats buy-and-hold on the final exam", exam_ok, exam_detail),
        (f"wins at least {need} of {n} rolling windows", won >= need, f"won {won}/{n}"),
        (f"at least {p.min_trades} trades", trades >= p.min_trades, f"{trades} trades"),
        (f"still wins with {p.cost_multiplier}x costs", stress_ok, stress_detail),
    )

    scored = [*base.windows, *([base.exam] if base.exam else [])]
    flags = [
        f"{w.label}: drawdown {w.max_drawdown:.0%} exceeds {p.drawdown_flag:.0%}"
        for w in scored
        if w.max_drawdown > p.drawdown_flag
    ]
    sat = [w.label for w in scored if w.sat_out]
    if sat:
        flags.append(f"{', '.join(sat)}: no parameter set qualified, so the strategy held cash")

    label: Literal["PASS", "NOT PROVEN", "EXAM NOT RUN", "EXAM ALREADY SEEN"]
    if exam_status == "seen by an earlier version":
        label = "EXAM ALREADY SEEN"
    elif base.exam is None:
        label = "EXAM NOT RUN"
    else:
        label = "PASS" if all(ok for _, ok, _ in checks) else "NOT PROVEN"
    return Verdict(label, checks, tuple(flags))


def _pct(x: Decimal) -> str:
    return f"{x:+.1%}"


def _line(w: WindowScore) -> str:
    return (
        f"    {w.label:<10} {w.start:%Y-%m-%d}..{w.end:%Y-%m-%d}  strategy {_pct(w.strategy_return):>8}  "
        f"buy&hold {_pct(w.benchmark_return):>8}  excess {_pct(w.excess):>8}  trades {w.trades:>3}  "
        f"drawdown {w.max_drawdown:.0%}" + ("  (held cash)" if w.sat_out else "")
    )


def format_scoreboard(rows: Sequence[tuple[Evaluation, Verdict]], *, risk_note: str) -> str:
    out = [
        f"Strategy comparison vs equal-weight buy-and-hold of each strategy's own stocks  (risk config: {risk_note})"
    ]
    for ev, v in rows:
        out += ["", f"{ev.strategy_id}@{ev.strategy_hash}  ->  {v.label}"]
        out += [_line(w) for w in ev.windows]
        if ev.exam is not None:
            out.append(_line(ev.exam))
        for name, ok, detail in v.checks:
            mark = "?" if ok is None else ("x" if ok else " ")
            out.append(f"  [{mark}] {name}: {detail}")
        out += [f"  note: {f}" for f in v.flags]
    out += [
        "",
        "Cost-stress doubles the benchmark's entry cost as well as the strategy's costs.",
        "A PASS is evidence, not proof: it earns a paper run, nothing more.",
    ]
    return "\n".join(out)


def _score_json(w: WindowScore) -> dict[str, Any]:
    return {
        "label": w.label, "start": w.start.isoformat(), "end": w.end.isoformat(),
        "strategy_return": str(w.strategy_return), "benchmark_return": str(w.benchmark_return),
        "excess": str(w.excess), "trades": w.trades, "max_drawdown": str(w.max_drawdown),
        "params": dict(w.params), "sat_out": w.sat_out,
    }  # fmt: skip


def scoreboard_json(rows: Sequence[tuple[Evaluation, Verdict]]) -> list[dict[str, Any]]:
    return [
        {
            "strategy": ev.strategy_id, "hash": ev.strategy_hash, "verdict": v.label,
            "windows": [_score_json(w) for w in ev.windows],
            "exam": _score_json(ev.exam) if ev.exam else None,
            "checks": [{"condition": n, "met": ok, "detail": d} for n, ok, d in v.checks],
            "flags": list(v.flags),
        }
        for ev, v in rows
    ]  # fmt: skip
