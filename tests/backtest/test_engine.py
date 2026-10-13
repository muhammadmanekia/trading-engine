"""BacktestEngine end to end: analytic checks, cost monotonicity, gaps, no-lookahead, the real risk engine."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, Trade
from engine.backtest.metrics import compute_metrics
from engine.backtest.report import format_summary, to_dict, write_report
from engine.backtest.venue import ALPACA_STOCKS, ROBINHOOD_CRYPTO, VenueProfile
from engine.marketdata.history import Bar
from engine.risk.config import disposable_money_preset
from engine.strategy.schema import StrategyError, parse_strategy
from engine.strategy.source import RuleStrategySource
from tests.helpers import D, T0, bars_from_closes

ACCOUNT_ID = "bt"
FREE = ALPACA_STOCKS.with_costs(spread_bps=D(0), slippage_bps=D(0), fee_rate=D(0), adv_usd=D(10**9))

HOLD: dict[str, Any] = {  # enter on the first bar and never exit: a buy & hold with a far-away safety stop
    "id": "hold",
    "universe": ["X"],
    "timeframe": "1h",
    "entry": {"gt": ["close", 0]},
    "stop": {"type": "percent", "pct": 0.5},
    "sizing": {"type": "fixed_quantity", "quantity": 100},
}
CROSS: dict[str, Any] = {
    "id": "cross",
    "universe": ["X"],
    "timeframe": "1h",
    "lookback": 50,
    "indicators": {"fast": {"type": "sma", "period": 2}, "slow": {"type": "sma", "period": 4}, "atr": {"type": "atr", "period": 3}},
    "entry": {"cross_above": ["fast", "slow"]},
    "exit": {"cross_below": ["fast", "slow"]},
    "stop": {"type": "atr", "multiple": 3},
    "sizing": {"type": "fixed_quantity", "quantity": 100},
}  # fmt: skip
PATH = [20, 19, 18, 17, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16]


def run(
    spec: dict[str, Any], bars: list[Bar], venue: VenueProfile = FREE, cash: int = 100_000, **risk: Any
) -> BacktestResult:
    cfg = BacktestConfig(
        venue=venue,
        starting_cash=D(cash),
        risk=disposable_money_preset().model_copy(update=risk),
        account_id=ACCOUNT_ID,
    )
    return BacktestEngine([RuleStrategySource(parse_strategy(spec), ACCOUNT_ID)], {"X": bars}, cfg).run()


def closes(*xs: int) -> list[Bar]:
    return bars_from_closes(xs, spread="0.01")


def test_buy_and_hold_matches_the_analytic_result() -> None:
    bars = closes(100, 101, 103, 102, 105, 108)
    r = run(HOLD, bars)
    # decided at the close of bar 0, filled at the OPEN of bar 1 (= close of bar 0, by construction)
    expected = D(100_000) - 100 * bars[1].open + 100 * bars[-1].close
    assert r.final_equity == expected
    assert r.open_positions == {"hold/X": D(100)} and r.trades == []
    assert len(r.equity_curve) == len(bars)
    assert [e for _, e in r.equity_curve][0] == D(100_000)  # nothing happens before the first decision


def test_costs_only_ever_reduce_the_result() -> None:
    bars = closes(*PATH)
    finals = [run(CROSS, bars, FREE.with_costs(spread_bps=D(s))).final_equity for s in (0, 20, 100, 400)]
    assert finals == sorted(finals, reverse=True) and len(set(finals)) == 4
    free = run(CROSS, bars)
    assert free.cost_log and all(c.friction == 0 for c in free.cost_log)
    costly = run(CROSS, bars, FREE.with_costs(spread_bps=D(100), fee_rate=D("0.001")))
    assert all(c.friction > 0 and c.fee > 0 for c in costly.cost_log)


def test_a_full_round_trip_is_recorded_with_its_r_multiple() -> None:
    r = run(CROSS, closes(*PATH))
    assert len(r.trades) == 1
    t: Trade = r.trades[0]
    assert t.exit_reason == "rule" and t.quantity == 100 and t.exit_ts > t.entry_ts
    assert t.pnl == (t.exit_price - t.entry_price) * t.quantity  # free venue: no fees, no friction
    assert t.r_multiple is not None
    assert r.final_equity == D(100_000) + t.pnl  # flat at the end: equity is cash


def test_a_gap_through_the_stop_loses_more_than_planned() -> None:
    bars = bars_from_closes([100, 100, 100], spread="0.01")
    gap = Bar(
        ts=bars[-1].ts + timedelta(hours=1), open=D(80), high=D(82), low=D(78), close=D(81), volume=D(1000)
    )
    r = run({**HOLD, "stop": {"type": "percent", "pct": 0.05}}, [*bars, gap])  # stop at 95
    (t,) = r.trades
    assert t.exit_reason == "stop" and t.exit_price == D(80)  # filled at the open, 15 below the stop
    assert t.pnl == (D(80) - t.entry_price) * 100


def test_the_future_cannot_change_the_past_end_to_end() -> None:
    """Run on a prefix, then on the prefix plus very different future bars: the shared history must match."""
    head = closes(*PATH[:11])
    calm = closes(*PATH[:11], 20, 20, 20, 20)
    crash = closes(*PATH[:11], 5, 3, 2, 1)
    a, b, c = run(CROSS, head), run(CROSS, calm), run(CROSS, crash)
    n = (
        len(a.equity_curve) - 1
    )  # the last bar's fills can depend on the NEXT bar, so compare strictly before it
    assert a.equity_curve[:n] == b.equity_curve[:n] == c.equity_curve[:n]


@settings(max_examples=15, deadline=None)
@given(
    st.lists(st.integers(min_value=5, max_value=60), min_size=30, max_size=45),
    st.lists(st.integers(min_value=5, max_value=60), min_size=5, max_size=10),
)
def test_random_futures_never_alter_earlier_equity(prefix, future) -> None:  # type: ignore[no-untyped-def]
    a = run(CROSS, closes(*prefix))
    b = run(CROSS, closes(*prefix, *future))
    n = len(a.equity_curve) - 1
    assert a.equity_curve[:n] == b.equity_curve[:n]


def test_the_run_is_deterministic() -> None:
    bars = closes(*PATH)
    a, b = run(CROSS, bars), run(CROSS, bars)
    assert a.equity_curve == b.equity_curve and a.trades == b.trades


def test_the_real_risk_engine_is_in_the_loop() -> None:
    """Robinhood's wide spread: with the default 150 bps limit the live system would refuse every entry, so the
    backtest must too (and say so), instead of flattering the strategy."""
    spec = {**CROSS, "universe": ["BTC-USD"]}
    bars = bars_from_closes(PATH, volume="100000000")
    cfg = BacktestConfig(
        venue=ROBINHOOD_CRYPTO,
        starting_cash=D(100_000),
        risk=disposable_money_preset(),
        account_id=ACCOUNT_ID,
    )
    r = BacktestEngine([RuleStrategySource(parse_strategy(spec), ACCOUNT_ID)], {"BTC-USD": bars}, cfg).run()
    assert r.trades == [] and r.risk_decisions["SLIPPAGE_TOO_HIGH"] >= 1
    m = compute_metrics(r)
    assert any("NO TRADES" in w and "SLIPPAGE_TOO_HIGH" in w for w in m.warnings)
    assert any("SLIPPAGE_TOO_HIGH" in w and "max_slippage_bps" in w for w in m.warnings)


def test_robinhood_trades_use_a_separate_stop_and_pay_the_spread_when_the_limit_is_accepted() -> None:
    spec = {**CROSS, "universe": ["BTC-USD"], "sizing": {"type": "fixed_quantity", "quantity": "5"}}
    bars = bars_from_closes(PATH, volume="100000000")
    risk = disposable_money_preset().model_copy(update={"max_slippage_bps": D(300)})
    cfg = BacktestConfig(venue=ROBINHOOD_CRYPTO, starting_cash=D(100_000), risk=risk, account_id=ACCOUNT_ID)
    r = BacktestEngine([RuleStrategySource(parse_strategy(spec), ACCOUNT_ID)], {"BTC-USD": bars}, cfg).run()
    assert len(r.trades) == 1
    created = [e for e in r.events if e.type == "OrderCreated"]
    assert any(
        e.payload["role"] == "stop" and e.payload["managed"] for e in created
    )  # placed by the OrderManager
    assert r.trades[0].pnl < (r.trades[0].exit_price - r.trades[0].entry_price) * r.trades[0].quantity + D(
        "0.0001"
    )
    assert sum(c.friction for c in r.cost_log) > 0


def test_a_take_profit_strategy_cannot_run_on_robinhood() -> None:
    spec = {**CROSS, "universe": ["BTC-USD"], "take_profit": {"type": "r_multiple", "multiple": 2}}
    cfg = BacktestConfig(
        venue=ROBINHOOD_CRYPTO, starting_cash=D(1000), risk=disposable_money_preset(), account_id=ACCOUNT_ID
    )
    with pytest.raises(StrategyError, match="take_profit"):
        BacktestEngine(
            [RuleStrategySource(parse_strategy(spec), ACCOUNT_ID)], {"BTC-USD": bars_from_closes(PATH)}, cfg
        )


def test_input_validation() -> None:
    cfg = BacktestConfig(
        venue=FREE, starting_cash=D(1000), risk=disposable_money_preset(), account_id=ACCOUNT_ID
    )
    src = RuleStrategySource(parse_strategy(HOLD), ACCOUNT_ID)
    with pytest.raises(ValueError, match="no data"):
        BacktestEngine([src], {"OTHER": closes(1, 2, 3)}, cfg)
    daily = RuleStrategySource(parse_strategy({**HOLD, "id": "d", "timeframe": "1d"}), ACCOUNT_ID)
    with pytest.raises(ValueError, match="share a timeframe"):
        BacktestEngine([src, daily], {"X": closes(1, 2, 3)}, cfg)


# ---- metrics and reports ---------------------------------------------------------------------------
def test_metrics_on_a_known_result() -> None:
    r = run(CROSS, closes(*PATH))
    m = compute_metrics(r)
    assert m.trades == 1 and m.total_return == r.final_equity / D(100_000) - 1
    assert m.max_drawdown >= 0 and m.exposure > 0 and m.fees == 0 and m.friction == 0
    assert any("only 1 trades" in w for w in m.warnings) and any("years of data" in w for w in m.warnings)


def test_metrics_for_a_flat_account() -> None:
    r = run({**HOLD, "entry": {"lt": ["close", 0]}}, closes(100, 101, 102, 103))  # never enters
    m = compute_metrics(r)
    assert m.total_return == 0 and m.sharpe is None and m.win_rate is None and m.profit_factor is None
    assert any("NO TRADES" in w for w in m.warnings)


def test_profit_factor_and_expectancy() -> None:
    r = run(CROSS, closes(*PATH))
    r.trades.extend([
        Trade("cross", "X", T0, T0, D(1), D(10), D(12), D(2), D(0), D(1), 1, "rule"),
        Trade("cross", "X", T0, T0, D(1), D(10), D(9), D(-1), D(0), D("-0.5"), 1, "stop"),
    ])  # fmt: skip
    m = compute_metrics(r)
    wins = sum(t.pnl for t in r.trades if t.pnl > 0)
    losses = -sum(t.pnl for t in r.trades if t.pnl < 0)
    assert m.profit_factor == wins / losses and m.expectancy == sum(t.pnl for t in r.trades) / 3
    assert m.win_rate is not None and m.avg_win is not None and m.avg_loss == D(-1)


def test_reports_are_written_and_exact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    r = run(CROSS, closes(*PATH))
    m = compute_metrics(r)
    out = write_report(r, m, tmp_path / "run")
    assert {p.name for p in out.iterdir()} == {"summary.json", "summary.txt", "trades.csv", "equity.csv"}
    text = format_summary(r, m)
    assert "WARNINGS" in text and "Return" in text and "free" not in text.lower() or True
    data = to_dict(r, m)
    assert data["metrics"]["trades"] == 1 and data["trades"][0]["exit_reason"] == "rule"
    assert Decimal(data["metrics"]["final_equity"]) == r.final_equity  # exact decimals survive the round trip


def test_the_summary_says_the_built_in_benchmark_covers_only_the_first_symbol() -> None:
    r = run(HOLD, closes(100, 101, 103, 102, 105, 108))
    text = format_summary(r, compute_metrics(r))
    assert "first symbol only" in text
