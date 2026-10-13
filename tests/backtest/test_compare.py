"""The comparison protocol: windows never leak, the rule is exact at its boundaries, the exam cannot be peeked."""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

import pytest

from engine.backtest.compare import Protocol, WindowSpan, evaluate, required_wins, rolling_spans, stressed
from engine.backtest.engine import BacktestConfig
from engine.backtest.venue import ALPACA_STOCKS
from engine.risk.config import disposable_money_preset
from engine.strategy.schema import parse_strategy
from tests.helpers import D, bars_from_closes


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


FREE = ALPACA_STOCKS.with_costs(spread_bps=D(0), slippage_bps=D(0), fee_rate=D(0), adv_usd=D(10**9))
COSTLY = ALPACA_STOCKS.with_costs(spread_bps=D(40), slippage_bps=D(10), fee_rate=D(0), adv_usd=D(10**9))
LOOSE = disposable_money_preset().model_copy(
    update={"max_consecutive_losses": 1000, "daily_loss_cap_pct": D("0.9")}
)
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
    rng, price, out = random.Random(seed), 100.0, []  # noqa: S311 - seeded test data
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
    assert (
        with_exam.exam is not None and with_exam.exam.label == "final exam" and with_exam.exam.start == EXAM
    )


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
        evaluate(
            parse_strategy(SPEC), DATA, cfg(), PROTO, DATA["X"][-1].ts + timedelta(days=9), run_exam=False
        )
