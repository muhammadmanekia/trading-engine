# Strategy Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `python -m engine.backtest compare` so any number of rule strategies can be scored the same way against an equal-weight buy-and-hold of their own stock universe, using rolling windows plus one locked, once-only final exam.

**Architecture:** Two small pure modules in `engine/backtest/` (`benchmark.py`, `compare.py`) built on the existing `BacktestEngine` and the walk-forward helpers (`backtest`, `parameter_grid`, `objective_value`). The CLI command lives in the service layer file `service/backtest_cli.py` (which already reads data keys from the environment). An `ExamLedger` JSON file records each strategy hash's exam so edits after peeking are flagged.

**Tech Stack:** Python 3.12, Decimal arithmetic, pytest, mypy `--strict`, ruff (line length 110).

Spec: `docs/superpowers/specs/2026-10-13-strategy-evaluation-design.md`.

**Deviations from the spec (decided while planning, reflected back into the spec in Task 9):**
1. Each strategy is benchmarked against buy-and-hold of **its own** universe (a 3-stock strategy is not compared to a 25-stock basket). With the shared 25-stock universe this is identical to the spec.
2. The spec test "a buy-and-hold strategy scores about zero excess return" is replaced by two checks that hold under the engine's position caps: a never-trading strategy has `excess == -benchmark`, and a hold strategy's return is positive but below the benchmark's (it is capped below 100% invested).
3. The risk preset halts a strategy after 8 consecutive losses, which silently ruined an earlier backtest (`HALTED_STRATEGY`). Research runs use a tracked `config/research.toml` with the breakers loosened, and the scoreboard prints which risk config it used.

## File map

| File | Responsibility |
|---|---|
| `src/engine/backtest/report.py` (modify) | Label the engine's built-in benchmark honestly ("first symbol only") |
| `src/engine/backtest/benchmark.py` (create) | `equal_weight_return(data, venue)` |
| `src/engine/backtest/compare.py` (create) | Protocol, windows, scoring, `evaluate`, `judge`, `ExamLedger`, scoreboard formatting |
| `src/engine/service/backtest_cli.py` (modify) | `compare` subcommand |
| `config/research.toml` (create) | Risk layer for research runs (breakers loosened) |
| `tests/backtest/test_benchmark.py` (create) | Benchmark math |
| `tests/backtest/test_compare.py` (create) | Windows, scoring, judging, ledger, scoreboard |
| `tests/backtest/test_cli.py` (modify) | `compare` end to end on CSV data |
| `docs/strategies.md` (modify) | How to run and read `compare` |

Run all commands from `trading-engine/` with the venv active (`source .venv/bin/activate`).

---

### Task 1: Stop the built-in benchmark from misleading

The engine's `benchmark_return` is buy-and-hold of the **first symbol only** (`engine.py` line ~219). The summary prints it as plain "buy & hold", which made a 25-stock strategy look 185 points behind. Relabel it.

**Files:**
- Modify: `src/engine/backtest/report.py:30`
- Test: `tests/backtest/test_engine.py`

- [ ] **Step 1: Write the failing test** (append to `tests/backtest/test_engine.py`)

```python
def test_the_summary_says_the_built_in_benchmark_covers_only_the_first_symbol() -> None:
    r = run(HOLD, closes(100, 101, 103, 102, 105, 108))
    text = format_summary(r, compute_metrics(r))
    assert "first symbol only" in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/backtest/test_engine.py::test_the_summary_says_the_built_in_benchmark_covers_only_the_first_symbol -o addopts="" -q`
Expected: FAIL (`assert 'first symbol only' in ...`)

- [ ] **Step 3: Change the label.** In `src/engine/backtest/report.py` replace line 30 so it reads:

```python
        f"  Return        {_pct(m.total_return):>10}   (buy & hold of the first symbol only: {_pct(m.benchmark_return)}, excess {_pct(m.excess_return)})",
```

- [ ] **Step 4: Run the backtest tests**

Run: `pytest tests/backtest -o addopts="" -q`
Expected: all pass (no test asserted the old wording; if one does, update its expected text to the new label).

- [ ] **Step 5: Commit**

```bash
git add src/engine/backtest/report.py tests/backtest/test_engine.py
git commit -m "Backtest summary: label the built-in benchmark as first-symbol-only

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The equal-weight benchmark

**Files:**
- Create: `src/engine/backtest/benchmark.py`
- Test: `tests/backtest/test_benchmark.py`

- [ ] **Step 1: Write the failing tests** (`tests/backtest/test_benchmark.py`)

```python
"""The yardstick: equal dollars in every stock, bought at the first open, never sold."""

from __future__ import annotations

import pytest

from engine.backtest.benchmark import equal_weight_return
from engine.backtest.venue import ALPACA_STOCKS
from tests.helpers import D, bars_from_closes

FREE = ALPACA_STOCKS.with_costs(spread_bps=D(0), slippage_bps=D(0), fee_rate=D(0))


def test_equal_dollars_means_an_average_of_the_stock_returns() -> None:
    data = {"A": bars_from_closes([100, 105, 110]), "B": bars_from_closes([50, 45, 40])}
    # A: +10%, B: -20% -> half the money in each -> (1.1 + 0.8) / 2 - 1 = -5%
    assert equal_weight_return(data, FREE) == D("-0.05")


def test_costs_are_paid_once_on_the_way_in() -> None:
    venue = ALPACA_STOCKS.with_costs(spread_bps=D(20), slippage_bps=D(0), fee_rate=D("0.001"))
    data = {"A": bars_from_closes([100, 110])}
    paid = (1 + venue.side_cost) * (1 + D("0.001"))
    assert equal_weight_return(data, venue) == D("1.1") / paid - 1


def test_a_stock_with_no_bars_is_an_error_not_a_silent_skip() -> None:
    with pytest.raises(ValueError, match="no bars for B"):
        equal_weight_return({"A": bars_from_closes([100, 101]), "B": []}, FREE)
    with pytest.raises(ValueError, match="no symbols"):
        equal_weight_return({}, FREE)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/backtest/test_benchmark.py -o addopts="" -q`
Expected: FAIL (`ModuleNotFoundError: engine.backtest.benchmark`)

- [ ] **Step 3: Implement** (`src/engine/backtest/benchmark.py`)

```python
"""The yardstick every strategy is judged against: equal dollars in every stock, bought once, never sold.

Costs: the purchase pays the venue's half-spread, slippage and fee, exactly like a strategy's entry. Like a
strategy's open positions at the end of a backtest, the holding is marked to the last close with no exit cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from engine.backtest.venue import VenueProfile
from engine.marketdata.history import Bar

ZERO = Decimal(0)
ONE = Decimal(1)


def equal_weight_return(data: Mapping[str, Sequence[Bar]], venue: VenueProfile) -> Decimal:
    """Total return of splitting cash equally across `data`'s symbols at each symbol's first open and holding
    to its last close. `data` must already be cut to the period being measured."""
    if not data:
        raise ValueError("no symbols to benchmark")
    legs: list[tuple[Decimal, Decimal]] = []
    for symbol, bars in sorted(data.items()):
        if not bars:
            raise ValueError(f"no bars for {symbol} in the benchmark period")
        legs.append((bars[0].open, bars[-1].close))
    paid = (ONE + venue.side_cost) * (ONE + venue.fee_rate)
    value = sum((close / (opened * paid) for opened, close in legs), ZERO) / len(legs)
    return value - ONE
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/backtest/test_benchmark.py -o addopts="" -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/engine/backtest/benchmark.py tests/backtest/test_benchmark.py
git commit -m "Backtest: equal-weight buy-and-hold benchmark

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Protocol, rolling windows, the two-thirds rule

**Files:**
- Create: `src/engine/backtest/compare.py`
- Test: `tests/backtest/test_compare.py`

- [ ] **Step 1: Write the failing tests** (`tests/backtest/test_compare.py`)

```python
"""The comparison protocol: windows never leak, the rule is exact at its boundaries, the exam cannot be peeked."""

from __future__ import annotations

import pytest

from engine.backtest.compare import Protocol, WindowSpan, required_wins, rolling_spans


def test_four_windows_fit_between_mid_2020_and_the_exam() -> None:
    spans = rolling_spans(1008, Protocol(train_bars=504, test_bars=126))
    assert spans == [
        WindowSpan(0, 504, 630),
        WindowSpan(126, 630, 756),
        WindowSpan(252, 756, 882),
        WindowSpan(378, 882, 1008),
    ]


def test_windows_never_overlap_never_reach_the_exam_and_train_before_they_test() -> None:
    for exam_idx in (700, 1000, 1008, 1300):
        spans = rolling_spans(exam_idx, Protocol(train_bars=504, test_bars=126))
        for a, b in zip(spans, spans[1:], strict=False):
            assert a.test_hi <= b.test_lo  # scored periods are disjoint
        assert all(s.test_hi <= exam_idx and s.train_lo + 504 == s.test_lo for s in spans)


def test_no_windows_when_history_before_the_exam_is_too_short() -> None:
    assert rolling_spans(600, Protocol(train_bars=504, test_bars=126)) == []


@pytest.mark.parametrize(("windows", "needed"), [(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (6, 4), (9, 6)])
def test_two_thirds_rounds_up(windows: int, needed: int) -> None:
    assert required_wins(windows) == needed
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/backtest/test_compare.py -o addopts="" -q`
Expected: FAIL (`ModuleNotFoundError` / `ImportError`)

- [ ] **Step 3: Implement** (`src/engine/backtest/compare.py`, first part)

```python
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
from engine.marketdata.history import Bar
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/backtest/test_compare.py -o addopts="" -q`
Expected: 4 passed (the parametrized test counts as 7 cases: 10 passed total)

- [ ] **Step 5: Commit**

```bash
git add src/engine/backtest/compare.py tests/backtest/test_compare.py
git commit -m "Compare: protocol, rolling windows, two-thirds rule

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Scoring a window and evaluating a strategy

**Files:**
- Modify: `src/engine/backtest/compare.py` (append)
- Test: `tests/backtest/test_compare.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/backtest/test_compare.py`)

```python
import random
from datetime import timedelta
from typing import Any

from engine.backtest.compare import Evaluation, evaluate, stressed
from engine.backtest.engine import BacktestConfig
from engine.backtest.venue import ALPACA_STOCKS
from engine.risk.config import disposable_money_preset
from engine.strategy.schema import parse_strategy
from tests.helpers import D, T0, bars_from_closes

FREE = ALPACA_STOCKS.with_costs(spread_bps=D(0), slippage_bps=D(0), fee_rate=D(0), adv_usd=D(10**9))
COSTLY = ALPACA_STOCKS.with_costs(spread_bps=D(40), slippage_bps=D(10), fee_rate=D(0), adv_usd=D(10**9))
LOOSE = disposable_money_preset().model_copy(update={"max_consecutive_losses": 1000, "daily_loss_cap_pct": D("0.9")})
PROTO = Protocol(train_bars=120, test_bars=60, min_trades=5, tune_min_trades=3)
SPEC: dict[str, Any] = {
    "id": "sma-cross", "universe": ["X"], "timeframe": "1h", "lookback": 60,
    "indicators": {"fast": {"type": "sma", "period": 3}, "slow": {"type": "sma", "period": 8}, "atr": {"type": "atr", "period": 5}},
    "entry": {"cross_above": ["fast", "slow"]}, "exit": {"cross_below": ["fast", "slow"]},
    "stop": {"type": "atr", "multiple": 3}, "sizing": {"type": "fixed_quantity", "quantity": 50},
}  # fmt: skip
NEVER: dict[str, Any] = {**SPEC, "id": "never", "entry": {"gt": ["close", 10_000_000]}}
HOLD: dict[str, Any] = {
    "id": "hold", "universe": ["X"], "timeframe": "1h", "entry": {"gt": ["close", 0]},
    "stop": {"type": "percent", "pct": 0.5}, "sizing": {"type": "fixed_quantity", "quantity": 100},
}  # fmt: skip


def walk(n: int, seed: int = 1, drift: float = 0.0015, vol: float = 0.02) -> list[Any]:
    rng, price, out = random.Random(seed), 100.0, []
    for _ in range(n):
        price = max(5.0, price * (1 + drift + rng.gauss(0, vol)))
        out.append(round(price, 2))
    return bars_from_closes(out, spread="0.02", volume="1000000")


def cfg(venue: Any = FREE) -> BacktestConfig:
    return BacktestConfig(venue=venue, starting_cash=D(100_000), risk=LOOSE, account_id="bt")


DATA = {"X": walk(600)}
EXAM = DATA["X"][480].ts  # six scored windows (120..480), then a 120-bar final exam


def test_every_window_is_scored_against_the_benchmark_and_stops_before_the_exam() -> None:
    ev = evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, EXAM, run_exam=False)
    assert len(ev.windows) == 6 and ev.exam is None
    assert all(w.end <= EXAM for w in ev.windows)
    assert all(w.excess == w.strategy_return - w.benchmark_return for w in ev.windows)
    assert [w.label for w in ev.windows][:2] == ["window 1", "window 2"]


def test_running_the_exam_does_not_change_the_rolling_results() -> None:
    without = evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, EXAM, run_exam=False)
    with_exam = evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, EXAM, run_exam=True)
    assert without.windows == with_exam.windows
    assert with_exam.exam is not None and with_exam.exam.label == "final exam" and with_exam.exam.start == EXAM


def test_a_strategy_that_never_trades_earns_zero_and_loses_to_a_rising_benchmark() -> None:
    ev = evaluate(parse_strategy(NEVER), DATA, cfg(), PROTO, EXAM, run_exam=True)
    assert all(w.strategy_return == 0 and w.trades == 0 for w in ev.windows)
    assert all(w.excess == -w.benchmark_return for w in ev.windows)


def test_a_hold_strategy_is_positive_but_capped_below_the_benchmark_when_the_market_rises() -> None:
    up = {"X": bars_from_closes([100 + i * 0.5 for i in range(600)], spread="0.02", volume="1000000")}
    ev = evaluate(parse_strategy(HOLD), up, cfg(), PROTO, up["X"][480].ts, run_exam=True)
    assert ev.exam is not None
    assert 0 < ev.exam.strategy_return < ev.exam.benchmark_return


def test_the_cost_stress_config_doubles_spread_slippage_and_fees() -> None:
    base = cfg(ALPACA_STOCKS.with_costs(spread_bps=D(3), slippage_bps=D(2), fee_rate=D("0.0005")))
    s = stressed(base, D(2))
    assert (s.venue.spread_bps, s.venue.slippage_bps, s.venue.fee_rate) == (D(6), D(4), D("0.001"))
    assert base.venue.spread_bps == D(3)  # the original is untouched


def test_costs_only_ever_reduce_the_scored_returns() -> None:
    base = evaluate(parse_strategy(SPEC), DATA, cfg(COSTLY), PROTO, EXAM, run_exam=False)
    worse = evaluate(parse_strategy(SPEC), DATA, stressed(cfg(COSTLY), D(2)), PROTO, EXAM, run_exam=False)
    assert sum(w.strategy_return for w in worse.windows) <= sum(w.strategy_return for w in base.windows)


def test_a_parameter_grid_is_tuned_on_the_training_window_only() -> None:
    grid = {"indicators.fast.period": [2, 3, 4]}
    ev = evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, EXAM, run_exam=False, grid=grid)
    assert len(ev.windows) == 6
    assert all(w.sat_out or "indicators.fast.period" in w.params for w in ev.windows)


def test_not_enough_history_before_the_exam_is_an_error() -> None:
    with pytest.raises(ValueError, match="not enough history before the exam"):
        evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, DATA["X"][150].ts, run_exam=False)
    with pytest.raises(ValueError, match="after the last bar"):
        evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, DATA["X"][-1].ts + timedelta(days=9), run_exam=False)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/backtest/test_compare.py -o addopts="" -q`
Expected: FAIL (`ImportError: cannot import name 'Evaluation'`)

- [ ] **Step 3: Implement** (append to `src/engine/backtest/compare.py`)

```python
@dataclass(frozen=True)
class WindowScore:
    label: str
    start: datetime
    end: datetime  # exclusive end of the scored period (the last bar's time for the open-ended exam)
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
            spread_bps=v.spread_bps * factor, slippage_bps=v.slippage_bps * factor, fee_rate=v.fee_rate * factor
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
    """The best parameter set on [lo, hi), or None if none made `min_trades`. Without a grid this is {}."""
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
    end = test_hi or timeline[-1]
    bench = equal_weight_return(_cut(data, test_lo, test_hi), config.venue)
    params = _choose(strategy, data, config, train_lo, test_lo, grid, objective, p.tune_min_trades)
    if params is None:
        return WindowScore(label, test_lo, end, ZERO, bench, 0, ZERO, {}, sat_out=True)
    variant = with_overrides(strategy, params) if params else strategy
    context_lo = timeline[max(0, span.test_lo - strategy.window)]  # indicators warm up before the window
    _, m = backtest(variant, _cut(data, context_lo, test_hi), config, trade_from=test_lo)
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/backtest/test_compare.py -o addopts="" -q`
Expected: all pass. If `test_a_hold_strategy_is_positive...` fails because the hold strategy was fully refused by the risk engine, print `ev.exam` and check `LOOSE`; the preset cap is 30% per position, so the return is positive and below the benchmark.

- [ ] **Step 5: Type-check and commit**

Run: `mypy` — Expected: `Success: no issues found`

```bash
git add src/engine/backtest/compare.py tests/backtest/test_compare.py
git commit -m "Compare: score rolling windows and the final exam against the benchmark

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The exam ledger

**Files:**
- Modify: `src/engine/backtest/compare.py` (append)
- Test: `tests/backtest/test_compare.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
from engine.backtest.compare import ExamLedger


def test_first_exam_is_clean_and_rerunning_the_same_strategy_stays_clean(tmp_path: Any) -> None:
    ledger = ExamLedger(tmp_path / "ledger.json")
    assert ledger.status("s", "h1") == "new"
    ledger.record("s", "h1", D("0.2"), D("0.1"), T0)
    assert ledger.status("s", "h1") == "clean"
    ledger.record("s", "h1", D("0.2"), D("0.1"), T0)  # same hash again changes nothing
    assert len(ExamLedger(tmp_path / "ledger.json")._data["s"]) == 1


def test_editing_a_strategy_after_its_exam_makes_the_new_version_not_clean(tmp_path: Any) -> None:
    ledger = ExamLedger(tmp_path / "ledger.json")
    ledger.record("s", "h1", D("0.2"), D("0.1"), T0)
    assert ledger.status("s", "h2") == "seen by an earlier version"
    ledger.record("s", "h2", D("0.5"), D("0.1"), T0)
    assert ledger.status("s", "h2") == "seen by an earlier version"  # recording does not launder it
    assert ledger.status("s", "h1") == "clean"
    assert ledger.status("other", "h9") == "new"


def test_the_ledger_survives_a_restart_and_rejects_a_corrupt_file(tmp_path: Any) -> None:
    path = tmp_path / "sub" / "ledger.json"
    ExamLedger(path).record("s", "h1", D("0.2"), D("0.1"), T0)
    assert ExamLedger(path).status("s", "h1") == "clean"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="exam ledger"):
        ExamLedger(path)
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/backtest/test_compare.py -o addopts="" -q` → FAIL (`ImportError: ExamLedger`)

- [ ] **Step 3: Implement** (append to `compare.py`)

```python
ExamStatus = Literal["not run", "new", "clean", "seen by an earlier version"]


class ExamLedger:
    """Which strategy versions have had their final exam. One entry per strategy hash; the first version of a
    strategy id to be examined is 'clean', every later edited version is permanently 'seen by an earlier
    version', because its author had already looked at the exam result."""

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

    def status(self, strategy_id: str, strategy_hash: str) -> ExamStatus:
        entries = self._data.get(strategy_id, [])
        for e in entries:
            if e["hash"] == strategy_hash:
                return "clean" if e["clean"] else "seen by an earlier version"
        return "new" if not entries else "seen by an earlier version"

    def record(
        self, strategy_id: str, strategy_hash: str, exam_return: Decimal, benchmark: Decimal, now: datetime
    ) -> None:
        entries = self._data.setdefault(strategy_id, [])
        if any(e["hash"] == strategy_hash for e in entries):
            return
        entries.append(
            {
                "hash": strategy_hash,
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
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/backtest/test_compare.py -o addopts="" -q` → all pass

- [ ] **Step 5: Commit**

```bash
git add src/engine/backtest/compare.py tests/backtest/test_compare.py
git commit -m "Compare: exam ledger that flags strategies edited after their exam

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The verdict (the four-part rule)

**Files:**
- Modify: `src/engine/backtest/compare.py` (append)
- Test: `tests/backtest/test_compare.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
from engine.backtest.compare import Verdict, WindowScore, judge


def score(label: str, excess: str, trades: int = 8, drawdown: str = "0.1", sat_out: bool = False) -> WindowScore:
    bench = D("0.05")
    return WindowScore(label, T0, T0, bench + D(excess), bench, trades, D(drawdown), {}, sat_out)


def evaluation(window_excess: list[str], exam_excess: str | None, trades: int = 8, drawdown: str = "0.1") -> Evaluation:
    windows = tuple(score(f"window {i}", e, trades, drawdown) for i, e in enumerate(window_excess, start=1))
    return Evaluation("s", "h", windows, score("final exam", exam_excess, trades) if exam_excess is not None else None)


def passing() -> Evaluation:
    return evaluation(["0.02", "0.01", "-0.01", "0.03"], "0.04")  # 3 of 4 windows, exam won, 5 x 8 = 40 trades


def check(v: Verdict, name_part: str) -> bool | None:
    return next(ok for name, ok, _ in v.checks if name_part in name)


def test_all_four_conditions_met_is_a_pass() -> None:
    v = judge(passing(), passing(), PROTO_RULES, "clean")
    assert v.label == "PASS" and all(ok for _, ok, _ in v.checks)


def test_the_windows_rule_is_exact_at_its_boundary() -> None:
    two_of_four = evaluation(["0.02", "0.01", "-0.01", "-0.03"], "0.04")
    v = judge(two_of_four, two_of_four, PROTO_RULES, "clean")
    assert check(v, "windows") is False and v.label == "NOT PROVEN"


def test_the_trade_count_rule_is_exact_at_its_boundary() -> None:
    def with_trades(total: int) -> Evaluation:  # 4 windows + exam = 5 periods
        per, extra = divmod(total, 5)
        ev = evaluation(["0.02", "0.01", "0.03", "0.02"], "0.04", trades=per)
        last = ev.exam
        assert last is not None
        return Evaluation(ev.strategy_id, ev.strategy_hash, ev.windows, WindowScore(**{**last.__dict__, "trades": last.trades + extra}))

    assert check(judge(with_trades(29), with_trades(29), PROTO_RULES, "clean"), "trades") is False
    assert check(judge(with_trades(30), with_trades(30), PROTO_RULES, "clean"), "trades") is True


def test_losing_the_exam_or_the_cost_stress_is_not_proven() -> None:
    lost_exam = evaluation(["0.02", "0.01", "0.03", "0.02"], "-0.01")
    assert judge(lost_exam, passing(), PROTO_RULES, "clean").label == "NOT PROVEN"
    fragile = evaluation(["0.02", "-0.01", "-0.03", "-0.02"], "0.04")
    v = judge(passing(), fragile, PROTO_RULES, "clean")
    assert check(v, "x costs") is False and v.label == "NOT PROVEN"


def test_an_exam_that_was_not_run_cannot_pass() -> None:
    no_exam = evaluation(["0.02", "0.01", "0.03", "0.02"], None)
    v = judge(no_exam, no_exam, PROTO_RULES, "not run")
    assert v.label == "EXAM NOT RUN" and check(v, "final exam") is None


def test_a_strategy_edited_after_its_exam_never_passes() -> None:
    v = judge(passing(), passing(), PROTO_RULES, "seen by an earlier version")
    assert v.label == "EXAM ALREADY SEEN"


def test_big_drawdowns_and_sat_out_windows_are_flagged_but_do_not_decide() -> None:
    deep = evaluation(["0.02", "0.01", "0.03", "0.02"], "0.04", drawdown="0.45")
    v = judge(deep, deep, PROTO_RULES, "clean")
    assert v.label == "PASS" and any("drawdown 45%" in f for f in v.flags)
    sat = Evaluation("s", "h", (score("window 1", "0.0", sat_out=True),), None)
    assert any("held cash" in f for f in judge(sat, sat, PROTO_RULES, "not run").flags)


PROTO_RULES = Protocol()
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/backtest/test_compare.py -o addopts="" -q` → FAIL (`ImportError: Verdict`)

- [ ] **Step 3: Implement** (append to `compare.py`)

```python
@dataclass(frozen=True)
class Verdict:
    label: Literal["PASS", "NOT PROVEN", "EXAM NOT RUN", "EXAM ALREADY SEEN"]
    checks: tuple[tuple[str, bool | None, str], ...]  # (condition, met?, detail); None = could not be assessed
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
    stress_ok = stress_won >= need and (stress.exam is None or stress.exam.excess > 0)
    stress_detail = f"{stress_won}/{n} windows" + (
        f", exam {stress.exam.strategy_return:.2%} vs {stress.exam.benchmark_return:.2%}" if stress.exam else ""
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
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/backtest/test_compare.py -o addopts="" -q` → all pass

- [ ] **Step 5: Type-check and commit**

Run: `mypy` → `Success: no issues found`

```bash
git add src/engine/backtest/compare.py tests/backtest/test_compare.py
git commit -m "Compare: the four-part rule for beating buy-and-hold

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The scoreboard text and JSON

**Files:**
- Modify: `src/engine/backtest/compare.py` (append)
- Test: `tests/backtest/test_compare.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
import json as jsonlib

from engine.backtest.compare import format_scoreboard, scoreboard_json


def test_the_scoreboard_names_the_verdict_every_check_and_every_window() -> None:
    ev = passing()
    v = judge(ev, ev, PROTO_RULES, "clean")
    text = format_scoreboard([(ev, v)], risk_note="config/research.toml")
    assert "s@h" in text and "PASS" in text and "config/research.toml" in text
    assert "[x] wins at least 3 of 4 rolling windows" in text and "window 4" in text and "final exam" in text
    bad = judge(evaluation(["-0.02"] * 4, "-0.01"), ev, PROTO_RULES, "clean")
    assert "[ ] " in format_scoreboard([(ev, bad)], risk_note="x") and "NOT PROVEN" in format_scoreboard([(ev, bad)], risk_note="x")


def test_the_json_scoreboard_keeps_exact_decimals_as_strings() -> None:
    ev = passing()
    out = scoreboard_json([(ev, judge(ev, ev, PROTO_RULES, "clean"))])
    parsed = jsonlib.loads(jsonlib.dumps(out))
    assert parsed[0]["strategy"] == "s" and parsed[0]["verdict"] == "PASS"
    assert parsed[0]["exam"]["excess"] == str(ev.exam.excess) if ev.exam else False
```

- [ ] **Step 2: Run to verify failure** — FAIL (`ImportError: format_scoreboard`)

- [ ] **Step 3: Implement** (append to `compare.py`)

```python
def _pct(x: Decimal) -> str:
    return f"{x:+.1%}"


def _line(w: WindowScore) -> str:
    return (
        f"    {w.label:<10} {w.start:%Y-%m-%d}..{w.end:%Y-%m-%d}  strategy {_pct(w.strategy_return):>8}  "
        f"buy&hold {_pct(w.benchmark_return):>8}  excess {_pct(w.excess):>8}  trades {w.trades:>3}  "
        f"drawdown {w.max_drawdown:.0%}" + ("  (held cash)" if w.sat_out else "")
    )


def format_scoreboard(rows: Sequence[tuple[Evaluation, Verdict]], *, risk_note: str) -> str:
    out = [f"Strategy comparison vs equal-weight buy-and-hold of each strategy's own stocks  (risk config: {risk_note})"]
    for ev, v in rows:
        out += ["", f"{ev.strategy_id}@{ev.strategy_hash}  ->  {v.label}"]
        out += [_line(w) for w in ev.windows]
        if ev.exam is not None:
            out.append(_line(ev.exam))
        for name, ok, detail in v.checks:
            mark = "?" if ok is None else ("x" if ok else " ")
            out.append(f"  [{mark}] {name}: {detail}")
        out += [f"  note: {f}" for f in v.flags]
    out += ["", "A PASS is evidence, not proof: it earns a paper run, nothing more."]
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
```

- [ ] **Step 4: Run to verify pass** — all pass. Then `ruff format src tests/backtest && ruff check src tests/backtest && mypy`.

- [ ] **Step 5: Commit**

```bash
git add src/engine/backtest/compare.py tests/backtest/test_compare.py
git commit -m "Compare: scoreboard text and JSON

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The `compare` command and the research risk config

**Files:**
- Create: `config/research.toml`
- Modify: `src/engine/service/backtest_cli.py` (parser around line 79-96, dispatch around line 148)
- Test: `tests/backtest/test_cli.py` (append)

- [ ] **Step 1: Create `config/research.toml`**

```toml
# Risk layer for RESEARCH backtests only (python -m engine.backtest compare --config config/research.toml).
# The live presets halt a strategy after 8 losing trades in a row, which is right for real money but silently
# ends a six-year backtest early (it showed up as HALTED_STRATEGY). Here the breakers are loosened so a strategy
# is judged on its own behaviour. Position, exposure and cash-reserve limits are unchanged.
[risk]
max_consecutive_losses = 1000
daily_loss_cap_pct = 0.9
strategy_drawdown_limit_pct = 0.9

[[accounts]]
id = "research"
broker = "alpaca"
```

- [ ] **Step 2: Write the failing CLI test** (append to `tests/backtest/test_cli.py`)

```python
from datetime import timedelta

from tests.helpers import T0


def test_compare_prints_a_scoreboard_and_locks_the_exam_until_asked(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exam = (T0 + timedelta(hours=150)).strftime("%Y-%m-%dT%H:%M:%S")
    ledger = workdir / "ledger.json"
    common = args(workdir, "compare", "--train-bars", "60", "--test-bars", "30", "--exam-start", exam,
                  "--ledger", str(ledger), "--out", str(workdir / "cmp"))  # fmt: skip
    assert main(common, {}) == 0
    out = capsys.readouterr().out
    assert "Strategy comparison" in out and "wave@" in out and "EXAM NOT RUN" in out and "window 3" in out
    assert not ledger.exists()  # looking at the rolling windows does not use up the exam

    assert main([*common, "--final-exam"], {}) == 0
    out = capsys.readouterr().out
    assert "final exam" in out and "EXAM NOT RUN" not in out and ledger.exists()
    assert {p.name for p in (workdir / "cmp").iterdir()} == {"scoreboard.txt", "scoreboard.json"}

    (workdir / "wave.toml").write_text((workdir / "wave.toml").read_text().replace("quantity = 5", "quantity = 6"))
    assert main([*common, "--final-exam"], {}) == 0
    assert "EXAM ALREADY SEEN" in capsys.readouterr().out  # edited after its exam: never a clean pass


def test_compare_refuses_an_exam_start_with_too_little_history_before_it(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exam = (T0 + timedelta(hours=40)).strftime("%Y-%m-%dT%H:%M:%S")
    assert main(args(workdir, "compare", "--train-bars", "60", "--test-bars", "30", "--exam-start", exam), {}) == 2
    assert "not enough history before the exam" in capsys.readouterr().err
```

- [ ] **Step 3: Run to verify failure** — `pytest tests/backtest/test_cli.py -o addopts="" -q` → FAIL (`unrecognized arguments` / invalid command)

- [ ] **Step 4: Implement in `src/engine/service/backtest_cli.py`**

Add imports:

```python
from engine.backtest.compare import (
    ExamLedger,
    ExamStatus,
    Protocol,
    evaluate,
    format_scoreboard,
    judge,
    scoreboard_json,
    stressed,
)
import json
```

Add the subcommand in `_parser()` after the `sens` block:

```python
    cmp_ = sub.add_parser("compare", help="score strategies against buy-and-hold: rolling windows + a locked final exam")
    common(cmp_)
    cmp_.add_argument("--train-bars", type=int, default=504)
    cmp_.add_argument("--test-bars", type=int, default=126)
    cmp_.add_argument("--exam-start", default="2024-07-01", help="first bar of the final exam (YYYY-MM-DD[THH:MM:SS])")
    cmp_.add_argument("--final-exam", action="store_true", help="also run the final exam (recorded in the ledger)")
    cmp_.add_argument("--ledger", default="runs/exam-ledger.json")
    cmp_.add_argument("--grid", action="append", default=[], metavar="PATH=V1,V2,...", help="parameters tuned on each training window")
    cmp_.add_argument("--objective", choices=["sharpe", "total_return", "avg_r"], default="sharpe")
```

In `_dispatch`, immediately after `timeframe = timeframes.pop()` and before `symbols = ...`, default the start for compare:

```python
    if args.command == "compare" and args.start is None:
        args.start = "2020-01-01"  # the rolling windows need years of history, not the 1-year default
```

and right after `data = _load(args, env, symbols, timeframe)` add the branch (before `if args.command == "run":`):

```python
    if args.command == "compare":
        return _compare(args, strategies, data, config, risk_note=args.config or "built-in preset")
```

Add the function near the other helpers:

```python
def _compare(
    args: argparse.Namespace,
    strategies: Sequence[StrategyDef],
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    *,
    risk_note: str,
) -> int:
    protocol = Protocol(train_bars=args.train_bars, test_bars=args.test_bars)
    exam_start = _parse_date(args.exam_start, datetime.now(UTC))
    ledger = ExamLedger(Path(args.ledger))
    grid = _parse_grid(args.grid)
    parameter_grid(grid)  # fail fast on a too-large search
    now = datetime.now(UTC)
    rows = []
    for strategy in strategies:
        base = evaluate(strategy, data, config, protocol, exam_start, run_exam=args.final_exam, grid=grid, objective=args.objective)
        stress = evaluate(
            strategy, data, stressed(config, protocol.cost_multiplier), protocol, exam_start,
            run_exam=args.final_exam, grid=grid, objective=args.objective,
        )  # fmt: skip
        status: ExamStatus = ledger.status(strategy.id, strategy.hash) if args.final_exam else "not run"
        if args.final_exam and base.exam is not None:
            ledger.record(strategy.id, strategy.hash, base.exam.strategy_return, base.exam.benchmark_return, now)
        rows.append((base, judge(base, stress, protocol, status)))
    text = format_scoreboard(rows, risk_note=risk_note)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scoreboard.txt").write_text(text + "\n")
        (out / "scoreboard.json").write_text(json.dumps(scoreboard_json(rows), indent=2))
    return 0
```

Note: `status` for a first exam is `"new"`, which `judge` treats like clean (only `"seen by an earlier version"` blocks a PASS). The ledger records after judging inputs are fixed, so the first run is clean and every later edited version is flagged.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/backtest/test_cli.py -o addopts="" -q` → all pass. Then the whole suite: `pytest -o addopts="" -q` → all pass; `ruff format src tests/backtest && ruff check src tests/backtest && mypy` → clean.

- [ ] **Step 6: Commit**

```bash
git add config/research.toml src/engine/service/backtest_cli.py tests/backtest/test_cli.py
git commit -m "Backtest CLI: compare command with a locked, once-only final exam

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Docs, spec touch-ups, and the first real scoreboard

**Files:**
- Modify: `docs/strategies.md` (new section 4b after "Do not fool yourself")
- Modify: `docs/superpowers/specs/2026-10-13-strategy-evaluation-design.md`

- [ ] **Step 1: Add to `docs/strategies.md`** (after section 4)

````markdown
## 4b. Compare strategies against buy-and-hold

```bash
python -m engine.backtest compare \
    --strategy strategies/examples/core-trend-hold.toml --strategy strategies/examples/rsi-dip-wide.toml \
    --venue alpaca-stocks --data alpaca --config config/research.toml --out runs/compare
# only when you are ready to spend each strategy's one final exam:
python -m engine.backtest compare ... --final-exam
```
Each strategy is scored on rolling 6-month windows (parameters chosen on the preceding 2 years), then once on a
final exam (default: 2024-07-01 to now). It must beat an equal-weight buy-and-hold of its own stocks on the exam,
win at least two thirds of the windows, make 30+ trades, and still win with costs doubled. Editing a strategy
after its exam marks the new version `EXAM ALREADY SEEN` for good. Use `config/research.toml` so the live
circuit breakers do not end a long backtest early. A PASS earns a paper run, not real money.
````

- [ ] **Step 2: Update the spec** in `docs/superpowers/specs/2026-10-13-strategy-evaluation-design.md`:
  - Under "The yardstick", add: "Each strategy is compared with buy-and-hold of **its own** universe."
  - In "Testing", replace the bullet about a buy-and-hold strategy scoring about zero excess return with: "A never-trading strategy has excess equal to minus the benchmark; a hold strategy is positive but below the benchmark (position caps keep it under 100% invested)."
  - Add to "Known limits": "The engine's older built-in benchmark covers only the first symbol; `compare` does not use it."

- [ ] **Step 3: Run the first real scoreboard** (needs the Alpaca keys in `.env`; takes several minutes):

```bash
python -m engine.backtest compare \
  --strategy strategies/examples/core-trend-hold.toml --strategy strategies/examples/rsi-dip-wide.toml \
  --venue alpaca-stocks --data alpaca --start 2020-01-01 --config config/research.toml --out runs/compare
```
Expected: a scoreboard with `EXAM NOT RUN` and four-ish windows per strategy. Read it before spending any exam.

- [ ] **Step 4: Full verification and commit**

Run: `pytest -o addopts="" -q && ruff check src tests/backtest && mypy` → all pass / clean.

```bash
git add docs/strategies.md docs/superpowers/specs/2026-10-13-strategy-evaluation-design.md
git commit -m "Docs: how to compare strategies against buy-and-hold

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git push origin HEAD
```

---

## Self-review against the spec

| Spec requirement | Task |
|---|---|
| Equal-weight buy-and-hold benchmark, same costs, paid once | 2 |
| Per-window and exam metrics: return, excess, drawdown, trades (Sharpe/exposure) | 4, 7. Sharpe and exposure are not in the scoreboard to keep it readable; the full `run` command still reports them. |
| Rolling windows: tune 2 years, score 6 months, step 6 months, never overlap | 3, 4 |
| Four-part rule: exam, two-thirds windows, 30 trades, doubled costs | 6 |
| Drawdown over 40% flagged, not gating | 6 |
| Exam locked behind `--final-exam`; ledger; edited strategies flagged | 5, 8 |
| Stop with a clear message on missing data, too few bars, bad strategy | 2, 4 (benchmark and evaluate raise); strategy-load errors come from the existing CLI error path |
| `compare` CLI writing JSON and text under `runs/` | 7, 8 |
| Reuses the real backtest engine | 4 (`backtest` from walkforward) |

No placeholders remain. Names used across tasks: `Protocol`, `WindowSpan`, `rolling_spans`, `required_wins`, `WindowScore`, `Evaluation`, `stressed`, `score_span`, `evaluate`, `ExamLedger`, `ExamStatus`, `Verdict`, `judge`, `format_scoreboard`, `scoreboard_json`, `equal_weight_return`.

One gap, deliberately left open: Sharpe and exposure are computed by the existing metrics but not shown per window; add them to `WindowScore` only if the first real scoreboard shows they are needed.
