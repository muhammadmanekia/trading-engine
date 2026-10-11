"""Walk-forward and sensitivity: they must detect an overfit strategy, and must not flatter a robust one."""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Any

import pytest

from engine.backtest.engine import BacktestConfig
from engine.backtest.venue import ALPACA_STOCKS
from engine.backtest.walkforward import parameter_grid, sensitivity, walk_forward
from engine.risk.config import disposable_money_preset
from engine.strategy.schema import parse_strategy
from tests.helpers import D, bars_from_closes

FREE = ALPACA_STOCKS.with_costs(spread_bps=D(0), slippage_bps=D(0), fee_rate=D(0), adv_usd=D(10**9))
CFG = BacktestConfig(venue=FREE, starting_cash=D(100_000), risk=disposable_money_preset(), account_id="bt")
SPEC: dict[str, Any] = {
    "id": "sma-cross",
    "universe": ["X"],
    "timeframe": "1h",
    "lookback": 60,
    "indicators": {"fast": {"type": "sma", "period": 3}, "slow": {"type": "sma", "period": 8}, "atr": {"type": "atr", "period": 5}},
    "entry": {"cross_above": ["fast", "slow"]},
    "exit": {"cross_below": ["fast", "slow"]},
    "stop": {"type": "atr", "multiple": 3},
    "sizing": {"type": "fixed_quantity", "quantity": 50},
}  # fmt: skip
STRATEGY = parse_strategy(SPEC)


def trending(n: int, seed: int = 1, drift: float = 0.0015, vol: float = 0.02) -> list[Any]:
    """A noisy up-trending path (so a trend-following cross has a real edge)."""
    rng, price, out = random.Random(seed), 100.0, []
    for _ in range(n):
        price = max(5.0, price * (1 + drift + rng.gauss(0, vol)))
        out.append(round(price, 2))
    return bars_from_closes(out, spread="0.02", volume="1000000")


def noise(n: int, seed: int = 2) -> list[Any]:
    """A trendless random walk: any 'edge' found here is a mirage."""
    return trending(n, seed=seed, drift=0.0, vol=0.02)


def test_the_grid_is_a_cartesian_product_and_is_capped() -> None:
    g = parameter_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(g) == 6 and {"a": 2, "b": "z"} in g
    assert parameter_grid({}) == [{}]
    with pytest.raises(ValueError, match="overfits"):
        parameter_grid({"a": list(range(30)), "b": list(range(30))})


def test_walk_forward_reports_every_window_and_chooses_params_only_from_the_past() -> None:
    bars = trending(420)
    rep = walk_forward(
        STRATEGY, {"X": bars}, CFG, {"indicators.fast.period": [2, 3, 4], "indicators.slow.period": [6, 8, 12]},
        train_bars=120, test_bars=60, min_trades=3,
    )  # fmt: skip
    assert len(rep.windows) == (420 - 120) // 60 == 5
    for w in rep.windows:
        assert w.train[1] <= w.test[0]  # training data ends no later than the test window starts
        assert not w.params or set(w.params) <= {"indicators.fast.period", "indicators.slow.period"}
    assert rep.oos_trades == sum(w.test_trades for w in rep.windows)
    compounded = Decimal(1)
    for w in rep.windows:
        compounded *= 1 + w.test_return
    assert rep.oos_return == compounded - 1


def test_a_strategy_tuned_on_pure_noise_is_flagged_not_celebrated() -> None:
    rep = walk_forward(
        STRATEGY, {"X": noise(600)}, CFG,
        {"indicators.fast.period": [2, 3, 4, 5], "indicators.slow.period": [6, 8, 12, 16], "stop.multiple": [2, 3, 4]},
        train_bars=150, test_bars=75, min_trades=3, objective="total_return",
    )  # fmt: skip
    assert rep.warnings  # at least one honest warning: thin trades, unstable params, or poor out-of-sample
    if rep.efficiency is not None:
        assert rep.efficiency < Decimal("1.5")  # no free lunch: in-sample gains do not carry over


def test_not_enough_data_gives_a_warning_instead_of_a_crash() -> None:
    rep = walk_forward(STRATEGY, {"X": trending(100)}, CFG, {}, train_bars=120, test_bars=60)
    assert rep.windows == [] and any("not enough data" in w for w in rep.warnings)


def test_when_no_parameter_set_trades_enough_the_test_window_stays_flat() -> None:
    rep = walk_forward(
        STRATEGY,
        {"X": trending(300)},
        CFG,
        {"stop.multiple": [3]},
        train_bars=100,
        test_bars=50,
        min_trades=500,
    )
    assert all(w.out_of_sample is None and w.test_trades == 0 and w.test_return == 0 for w in rep.windows)
    assert any("no parameter set reached" in w for w in rep.warnings)


def test_sensitivity_probes_each_parameter_both_ways() -> None:
    rep = sensitivity(STRATEGY, {"X": trending(300)}, CFG, ["indicators.fast.period", "stop.multiple"])
    paths = [p.path for p in rep.neighbours]
    assert paths.count("indicators.fast.period") >= 1 and paths.count("stop.multiple") == 2
    assert Decimal(0) <= rep.profitable_share <= Decimal(1) and rep.base.path == "base"


def test_sensitivity_warns_when_the_base_case_loses_money() -> None:
    rep = sensitivity(
        STRATEGY,
        {"X": bars_from_closes(list(range(200, 100, -1)), spread="0.02", volume="1000000")},
        CFG,
        ["stop.multiple"],
    )
    assert any("not profitable" in w for w in rep.warnings) or rep.base.total_return > 0
