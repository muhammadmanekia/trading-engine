"""BacktestBroker fill rules: next-open fills, gaps, costs, and the stop-before-target convention."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from engine.backtest.broker import BacktestBroker
from engine.backtest.engine import SimClock
from engine.backtest.venue import ALPACA_STOCKS, ROBINHOOD_CRYPTO, VenueProfile
from engine.broker.base import BrokerStatus, OrderRequest, OrderType
from engine.domain.types import AssetClass, Side
from engine.marketdata.history import Bar
from tests.helpers import D, T0

FREE = ALPACA_STOCKS.with_costs(spread_bps=D(0), slippage_bps=D(0), fee_rate=D(0))


def bar(i: int, o: float, h: float, l: float, c: float) -> Bar:  # noqa: E741
    return Bar(
        ts=T0 + timedelta(hours=i),
        open=D(str(o)),
        high=D(str(h)),
        low=D(str(l)),
        close=D(str(c)),
        volume=D(1000),
    )


def make(venue: VenueProfile = FREE) -> BacktestBroker:
    b = BacktestBroker(SimClock(T0), venue, starting_cash=D(1_000_000))
    b.prices["X"] = D(100)
    return b


def req(cid: str, side: Side = Side.BUY, qty: int = 10, **kw: object) -> OrderRequest:
    return OrderRequest.model_validate(
        {
            "client_order_id": cid,
            "symbol": "X",
            "asset_class": AssetClass.STOCK,
            "side": side,
            "quantity": D(qty),
            **kw,
        }
    )


def run_bar(b: BacktestBroker, bar_: Bar) -> None:
    b.begin_bar({"X": bar_})
    b.run_phase("open")
    b.run_phase("range")
    b.end_bar()


def test_orders_never_fill_on_submission_only_on_the_next_bar() -> None:
    b = make()
    o = b.submit(req("m"))
    assert (
        o.status is BrokerStatus.ACCEPTED and b.get_positions() == {}
    )  # placed at a bar close: not filled yet
    run_bar(b, bar(1, 102, 105, 99, 103))
    got = b.get_order("m")
    assert got is not None and got.status is BrokerStatus.FILLED and got.fills[0].price == D(102)  # next OPEN


def test_buys_pay_and_sells_receive_the_half_spread_plus_slippage_against_them() -> None:
    venue = ALPACA_STOCKS.with_costs(spread_bps=D(100), slippage_bps=D(50), fee_rate=D("0.001"))
    assert venue.side_cost == D("0.01")  # 50bps half-spread + 50bps slippage
    b = make(venue)
    b.submit(req("buy"))
    run_bar(b, bar(1, 100, 101, 99, 100))
    buy = b.get_order("buy")
    assert (
        buy is not None and buy.fills[0].price == D("101") and buy.fills[0].fee == D("1.01")
    )  # 10 * 101 * 0.001
    b.submit(req("sell", Side.SELL))
    run_bar(b, bar(2, 100, 101, 99, 100))
    sell = b.get_order("sell")
    assert sell is not None and sell.fills[0].price == D("99")
    friction = sum(c.friction for c in b.cost_log)
    assert friction == D(10) * (D("101") - D(100)) + D(10) * (D(100) - D("99"))  # 20, never negative
    assert all(c.friction >= 0 for c in b.cost_log)


def test_native_stop_triggers_at_the_stop_price_when_the_bar_trades_through_it() -> None:
    b = make()
    b.submit(req("e", stop_loss_price=D(90)))
    run_bar(b, bar(1, 100, 101, 99, 100))  # entry fills at the open; the stop leg arms
    run_bar(b, bar(2, 95, 96, 85, 88))  # opens above the stop, trades down through it
    stop = b.get_order("e:stop")
    assert stop is not None and stop.status is BrokerStatus.FILLED and stop.fills[0].price == D(90)
    assert b.get_positions() == {}


def test_a_gap_below_the_stop_fills_at_the_open_not_the_stop() -> None:
    b = make()
    b.submit(req("e", stop_loss_price=D(90)))
    run_bar(b, bar(1, 100, 101, 99, 100))
    run_bar(b, bar(2, 80, 82, 78, 81))  # opens 10 below the stop
    stop = b.get_order("e:stop")
    assert stop is not None and stop.fills[0].price == D(
        80
    )  # the loss is bigger than planned, as in real life


def test_when_stop_and_target_are_both_reachable_in_one_bar_the_stop_wins() -> None:
    b = make()
    b.submit(req("e", stop_loss_price=D(90), take_profit_price=D(110)))
    run_bar(b, bar(1, 100, 101, 99, 100))
    run_bar(b, bar(2, 100, 115, 85, 100))  # touches both 85 and 115
    legs = {c.client_order_id: c.status for c in b.get_order("e").children}  # type: ignore[union-attr]
    assert legs["e:stop"] is BrokerStatus.FILLED and legs["e:tp"] is BrokerStatus.CANCELLED


def test_take_profit_fills_at_the_target_or_a_better_open() -> None:
    b = make()
    b.submit(req("e", stop_loss_price=D(90), take_profit_price=D(110)))
    run_bar(b, bar(1, 100, 101, 99, 100))
    run_bar(b, bar(2, 112, 118, 111, 115))  # gaps above the target
    tp = [c for c in b.get_order("e").children if c.client_order_id == "e:tp"][0]  # type: ignore[union-attr]
    assert tp.status is BrokerStatus.FILLED and tp.fills[0].price == D(112)


def test_limit_buy_fills_at_the_limit_or_a_better_open_only_if_the_range_reaches_it() -> None:
    b = make()
    b.submit(req("lim", limit_price=D(95)))
    run_bar(b, bar(1, 100, 102, 96, 99))  # never reaches 95
    assert b.get_order("lim").status is BrokerStatus.ACCEPTED  # type: ignore[union-attr]
    run_bar(b, bar(2, 97, 98, 94, 96))  # reaches 95
    got = b.get_order("lim")
    assert got is not None and got.fills[0].price == D(95)
    b.submit(req("gap", limit_price=D(95)))
    run_bar(b, bar(3, 90, 93, 89, 92))  # opens below the limit: price improvement
    assert b.get_order("gap").fills[0].price == D(90)  # type: ignore[union-attr]


def test_separate_order_venues_hold_a_standalone_stop() -> None:
    venue = ROBINHOOD_CRYPTO.with_costs(spread_bps=D(0), slippage_bps=D(0))
    b = make(venue)
    b.submit(req("e").model_copy(update={"asset_class": AssetClass.CRYPTO}))
    run_bar(b, bar(1, 100, 101, 99, 100))
    b.submit(
        req("s", Side.SELL, order_type=OrderType.STOP, stop_price=D(95)).model_copy(
            update={"asset_class": AssetClass.CRYPTO}
        )
    )
    run_bar(b, bar(2, 98, 99, 94, 96))
    s = b.get_order("s")
    assert s is not None and s.status is BrokerStatus.FILLED and s.fills[0].price == D(95)
    assert b.get_positions() == {}


def test_the_broker_never_fills_a_bar_it_has_not_been_given() -> None:
    b = make()
    b.submit(req("m"))
    b.run_phase("open")  # no begin_bar: there is no market to trade against
    b.run_phase("range")
    assert b.get_order("m").status is BrokerStatus.ACCEPTED  # type: ignore[union-attr]
    assert isinstance(D(1), Decimal)
