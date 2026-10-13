"""The comparison protocol: windows never leak, the rule is exact at its boundaries, the exam cannot be peeked."""

from __future__ import annotations

import json
import random
from datetime import timedelta
from typing import Any

import pytest

from engine.backtest.compare import (
    Evaluation,
    ExamLedger,
    Protocol,
    Verdict,
    WindowScore,
    WindowSpan,
    evaluate,
    format_scoreboard,
    judge,
    required_wins,
    rolling_spans,
    scoreboard_json,
    stressed,
)
from engine.backtest.engine import BacktestConfig
from engine.backtest.venue import ALPACA_STOCKS
from engine.backtest.walkforward import backtest
from engine.risk.config import disposable_money_preset
from engine.strategy.schema import parse_strategy
from tests.helpers import T0, D, bars_from_closes


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
PROTO_RULES = Protocol()
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


def score(
    label: str, excess: str, trades: int = 8, drawdown: str = "0.1", sat_out: bool = False
) -> WindowScore:
    bench = D("0.05")
    return WindowScore(label, T0, T0, bench + D(excess), bench, trades, D(drawdown), {}, sat_out)


def evaluation(
    window_excess: list[str], exam_excess: str | None, trades: int = 8, drawdown: str = "0.1"
) -> Evaluation:
    windows = tuple(score(f"window {i}", e, trades, drawdown) for i, e in enumerate(window_excess, start=1))
    exam = score("final exam", exam_excess, trades) if exam_excess is not None else None
    return Evaluation("s", "h", windows, exam)


def passing() -> Evaluation:
    return evaluation(
        ["0.02", "0.01", "-0.01", "0.03"], "0.04"
    )  # 3 of 4 windows, exam won, 5 x 8 = 40 trades


def check(v: Verdict, name_part: str) -> bool | None:
    return next(ok for name, ok, _ in v.checks if name_part in name)


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
    assert ev.exam is not None and ev.exam.strategy_return == 0 and ev.exam.trades == 0
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


def test_a_parameter_grid_records_the_chosen_parameters_per_window() -> None:
    grid = {"indicators.fast.period": [2, 3, 4]}
    ev = evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, EXAM, run_exam=False, grid=grid)
    assert len(ev.windows) == 6
    assert all(w.sat_out or "indicators.fast.period" in w.params for w in ev.windows)


def perturbed(data: dict[str, list[Any]], from_ts: Any) -> dict[str, list[Any]]:
    """The same timestamps, but every bar at or after `from_ts` replaced with different, wild prices."""
    out = {}
    for sym, bars in data.items():
        junk = walk(len(bars), seed=99, drift=-0.01, vol=0.15)
        out[sym] = [
            j.model_copy(update={"ts": b.ts}) if b.ts >= from_ts else b
            for b, j in zip(bars, junk, strict=True)
        ]
    return out


def test_nothing_at_or_after_the_exam_start_can_change_the_rolling_windows() -> None:
    before = evaluate(parse_strategy(SPEC), DATA, cfg(COSTLY), PROTO, EXAM, run_exam=False)
    after = evaluate(parse_strategy(SPEC), perturbed(DATA, EXAM), cfg(COSTLY), PROTO, EXAM, run_exam=False)
    assert before.windows == after.windows


def test_a_window_choice_and_earlier_windows_ignore_everything_from_its_test_start_on() -> None:
    grid = {"indicators.fast.period": [2, 3, 4, 5]}
    k = 3
    base = evaluate(parse_strategy(SPEC), DATA, cfg(COSTLY), PROTO, EXAM, run_exam=False, grid=grid)
    test_start = base.windows[k - 1].start
    after = evaluate(
        parse_strategy(SPEC), perturbed(DATA, test_start), cfg(COSTLY), PROTO, EXAM, run_exam=False, grid=grid
    )
    assert base.windows[: k - 1] == after.windows[: k - 1]
    assert (base.windows[k - 1].params, base.windows[k - 1].sat_out) == (
        after.windows[k - 1].params,
        after.windows[k - 1].sat_out,
    )
    # teeth: the grid does pick different parameters in different windows, and the perturbation does matter
    assert len({tuple(w.params.items()) for w in base.windows}) > 1
    assert base.windows[k - 1] != after.windows[k - 1]


def test_the_strategy_first_fills_at_the_window_first_open_like_the_benchmark() -> None:
    up = {"X": bars_from_closes([100 + i * 0.5 for i in range(600)], spread="0.02", volume="1000000")}
    timeline = [b.ts for b in up["X"]]
    test_lo = 480
    result, _ = backtest(parse_strategy(HOLD), {"X": up["X"][400:]}, cfg(), trade_from=timeline[test_lo - 1])
    fills = [e for e in result.events if e.type == "OrderFilled"]
    assert fills and fills[0].ts == timeline[test_lo]


def test_a_symbol_with_no_bars_in_a_window_names_the_window() -> None:
    spec = {**SPEC, "universe": ["X", "Y"]}
    short = {"X": DATA["X"], "Y": DATA["X"][:100]}
    with pytest.raises(ValueError, match=r"window \d+: no bars for Y"):
        evaluate(parse_strategy(spec), short, cfg(), PROTO, EXAM, run_exam=False)


def test_not_enough_history_before_the_exam_is_an_error() -> None:
    with pytest.raises(ValueError, match="not enough history before the exam"):
        evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, DATA["X"][150].ts, run_exam=False)
    with pytest.raises(ValueError, match="after the last bar"):
        evaluate(
            parse_strategy(SPEC), DATA, cfg(), PROTO, DATA["X"][-1].ts + timedelta(days=9), run_exam=False
        )


def test_score_span_lets_the_strategy_decide_at_the_close_before_the_window_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards score_span's own trade_from: the benchmark buys at the window's first open, so the strategy must be
    allowed to act at the previous bar's close (otherwise it gets there one bar late)."""
    import engine.backtest.compare as compare

    seen: list[Any] = []
    real = compare.backtest

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("trade_from"))
        return real(*args, **kwargs)

    monkeypatch.setattr(compare, "backtest", spy)
    ev = evaluate(parse_strategy(SPEC), DATA, cfg(), PROTO, EXAM, run_exam=True)
    times = [b.ts for b in DATA["X"]]
    expected = [times[times.index(w.start) - 1] for w in (*ev.windows, ev.exam) if w is not None]
    assert seen == expected


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


def test_recording_writes_valid_json_in_a_new_nested_directory_and_leaves_no_temp_files(
    tmp_path: Any,
) -> None:
    path = tmp_path / "a" / "b" / "ledger.json"
    ExamLedger(path).record("s", "h1", D("0.2"), D("0.1"), T0)
    assert json.loads(path.read_text())["s"][0]["hash"] == "h1"
    assert [p.name for p in path.parent.iterdir()] == ["ledger.json"]


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
        bumped = WindowScore(**{**last.__dict__, "trades": last.trades + extra})
        return Evaluation(ev.strategy_id, ev.strategy_hash, ev.windows, bumped)

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
    assert check(v, "x costs") is None  # nothing was stressed on the exam, so it is not reported as met


def test_a_strategy_edited_after_its_exam_never_passes() -> None:
    v = judge(passing(), passing(), PROTO_RULES, "seen by an earlier version")
    assert v.label == "EXAM ALREADY SEEN"


def test_big_drawdowns_and_sat_out_windows_are_flagged_but_do_not_decide() -> None:
    deep = evaluation(["0.02", "0.01", "0.03", "0.02"], "0.04", drawdown="0.45")
    v = judge(deep, deep, PROTO_RULES, "clean")
    assert v.label == "PASS" and any("drawdown 45%" in f for f in v.flags)
    sat = Evaluation("s", "h", (score("window 1", "0.0", sat_out=True),), None)
    assert any("held cash" in f for f in judge(sat, sat, PROTO_RULES, "not run").flags)


def test_the_scoreboard_names_the_verdict_every_check_and_every_window() -> None:
    ev = passing()
    v = judge(ev, ev, PROTO_RULES, "clean")
    text = format_scoreboard([(ev, v)], risk_note="config/research.toml")
    assert "s@h" in text and "PASS" in text and "config/research.toml" in text
    assert "[x] wins at least 3 of 4 rolling windows" in text and "window 4" in text and "final exam" in text
    assert "Cost-stress doubles the benchmark's entry cost as well as the strategy's costs." in text
    bad = judge(evaluation(["-0.02"] * 4, "-0.01"), ev, PROTO_RULES, "clean")
    bad_text = format_scoreboard([(ev, bad)], risk_note="x")
    assert "[ ] " in bad_text and "NOT PROVEN" in bad_text


def test_the_json_scoreboard_keeps_exact_decimals_as_strings() -> None:
    ev = passing()
    out = scoreboard_json([(ev, judge(ev, ev, PROTO_RULES, "clean"))])
    parsed = json.loads(json.dumps(out))
    assert parsed[0]["strategy"] == "s" and parsed[0]["verdict"] == "PASS"
    assert ev.exam is not None
    assert parsed[0]["exam"]["excess"] == str(ev.exam.excess)
