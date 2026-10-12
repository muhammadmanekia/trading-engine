"""Forward scoring is honest: next-open entries, baseline-relative, leak-flagged, no verdict on thin data."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from engine.llm.evaluate import MIN_SAMPLES, compare_models, format_report, score_calls, summarise
from engine.llm.recorder import LLMCall
from tests.helpers import T0, D, bars_from_closes


def call(
    i: int,
    provider: str = "claude",
    symbol: str = "AAPL",
    action: str = "enter",
    at=None,
    valid: bool = True,
    trigger: str | None = None,
):  # type: ignore[no-untyped-def]
    return LLMCall(
        call_id=f"c{i}-{provider}", ts=at or T0 + timedelta(hours=i), account_id="a", strategy_id="s", mode="primary",
        provider=provider, model_requested="m", model_reported="m", prompt_name="p", prompt_version="v1", prompt_hash="h",
        input_hash=f"i{i}-{provider}", system_text="", user_text="", response_text="", valid=valid, error=None,
        decisions=[{"symbol": symbol, "action": action, "confidence": "0.8"}] if valid else None,
        input_tokens=100, output_tokens=20, cost_usd=D("0.01"), latency_ms=200 + i, trigger=trigger or f"news:{i}",
    )  # fmt: skip


RISING = {"AAPL": bars_from_closes(range(100, 160))}  # +1 per bar


def test_entry_is_the_next_bar_open_and_exit_is_the_horizon_close() -> None:
    (s,), pending = score_calls([call(0)], RISING, horizon_bars=3)
    # call at T0 -> first bar with ts >= T0 is bar 0 (opens 100); exit = close of bar 2 (102)
    assert s.entry_price == D(100) and s.exit_price == D(102) and s.ret == D("0.02") and pending == 0


def test_decisions_without_enough_future_are_pending_not_scored() -> None:
    scored, pending = score_calls([call(58)], RISING, horizon_bars=5)
    assert scored == [] and pending == 1


def test_only_valid_enter_decisions_are_scored() -> None:
    scored, _ = score_calls(
        [call(0, action="hold"), call(1, valid=False), call(2, symbol="TSLA")], RISING, horizon_bars=2
    )
    assert scored == []


def test_a_rising_market_makes_every_buy_look_good_so_the_baseline_cancels_it() -> None:
    steady = {
        "AAPL": bars_from_closes([D(100) * D("1.01") ** i for i in range(80)], spread="0")
    }  # +1% every bar
    scored, _ = score_calls([call(i) for i in range(40)], steady, horizon_bars=3)
    (row,) = summarise(scored)
    assert row.hit_rate == 1 and row.mean_return > D("0.02")
    assert abs(row.mean_excess) < D("0.001")  # a buy-anything model earned the market, nothing extra


def test_thin_data_gets_no_verdict_and_leaked_periods_are_excluded() -> None:
    few, _ = score_calls([call(i) for i in range(5)], RISING, horizon_bars=2)
    assert "insufficient data" in summarise(few)[0].verdict
    scored, _ = score_calls(
        [call(i) for i in range(40)], RISING, horizon_bars=2, cutoffs={"claude": date(2030, 1, 1)}
    )
    (row,) = summarise(scored)
    assert row.n == 0 and row.contaminated == 40 and "no clean" in row.verdict


def test_a_genuinely_better_model_is_distinguished_from_a_coin_flip() -> None:
    wiggly = {"AAPL": bars_from_closes([100 + (i % 7) * 3 - (i % 3) * 2 for i in range(200)])}
    # a model that only buys right before known up-moves vs one buying right before down-moves
    ups = [i for i in range(190) if wiggly["AAPL"][i + 1].close > wiggly["AAPL"][i].open]
    downs = [i for i in range(190) if wiggly["AAPL"][i + 1].close < wiggly["AAPL"][i].open]
    good, bad = (
        [call(i, "good") for i in ups[: MIN_SAMPLES + 10]],
        [call(i, "bad") for i in downs[: MIN_SAMPLES + 10]],
    )
    scored, _ = score_calls(good + bad, wiggly, horizon_bars=1)
    rows = {r.provider: r for r in summarise(scored)}
    assert "beats" in rows["good"].verdict and "WORSE" in rows["bad"].verdict


def test_model_comparison_reports_cost_latency_validity_and_agreement() -> None:
    calls = [call(1, "claude", trigger="t1"), call(1, "grok", trigger="t1"),
             call(2, "claude", trigger="t2"), call(2, "grok", action="hold", trigger="t2"),
             call(3, "grok", valid=False, trigger="t3")]  # fmt: skip
    stats, agreement = compare_models(calls)
    by = {s.provider: s for s in stats}
    assert by["grok"].invalid == 1 and by["claude"].total_cost_usd == D("0.02") and by["claude"].enters == 2
    assert agreement[("claude", "grok")] == (1, 2)  # same on t1, different on t2
    text = format_report(summarise([]), stats, agreement, 3)
    assert "claude vs grok: agreed on 1/2" in text and Decimal("0.02") == D("0.02")
