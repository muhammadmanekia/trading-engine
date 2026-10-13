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
