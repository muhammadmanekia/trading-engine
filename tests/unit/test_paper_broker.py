"""PaperBroker: a simulated exchange whose state must survive restarts."""

from __future__ import annotations

from engine.broker.base import BrokerStatus, OrderRequest, OrderType
from engine.broker.paper import InMemoryPaperState, PaperBroker, PostgresPaperState
from engine.clock import FakeClock
from engine.core import TradingCore
from engine.domain.types import AssetClass, Side
from engine.eventlog.memory import InMemoryEventStore
from engine.orders.state import OrderStatus as S
from tests.helpers import ACCOUNT, D, build_core, intent_now

SYM = "BTC-USD"


def req(cid: str, side: Side = Side.BUY, qty: str = "0.5", **kw: object) -> OrderRequest:
    return OrderRequest.model_validate(
        {
            "client_order_id": cid,
            "symbol": SYM,
            "asset_class": AssetClass.CRYPTO,
            "side": side,
            "quantity": D(qty),
            **kw,
        }
    )


def test_state_survives_a_restart() -> None:
    store, clock = InMemoryPaperState(), FakeClock()
    a = PaperBroker(clock, "p1", store, starting_cash=D(100_000))
    a.tick({SYM: D(60_000)})
    a.submit(req("buy"))
    a.submit(req("stop", Side.SELL, order_type=OrderType.STOP, stop_price=55_000))
    b = PaperBroker(clock, "p1", store, starting_cash=D(999))  # starting_cash ignored: state restored
    assert b.get_positions() == {SYM: D("0.5")}
    assert b.get_cash() == a.get_cash() != D(999)
    got = b.get_order("stop")
    assert got is not None and got.status is BrokerStatus.ACCEPTED
    b.tick({SYM: D(54_000)})  # the restored resting stop still triggers
    after = b.get_order("stop")
    assert after is not None and after.status is BrokerStatus.FILLED and b.get_positions() == {}


def test_an_ambiguous_timeout_does_not_lose_the_order() -> None:
    store, clock = InMemoryPaperState(), FakeClock()
    a = PaperBroker(clock, "p1", store, starting_cash=D(100_000))
    a.tick({SYM: D(60_000)})
    a.fail_next_submit("unavailable_after")
    try:
        a.submit(req("buy"))
    except Exception:  # noqa: S110
        pass
    assert PaperBroker(clock, "p1", store, starting_cash=D(1)).get_order("buy") is not None


def test_accounts_are_isolated() -> None:
    store, clock = InMemoryPaperState(), FakeClock()
    a = PaperBroker(clock, "p1", store, starting_cash=D(100_000))
    PaperBroker(clock, "p2", store, starting_cash=D(5))
    a.tick({SYM: D(1)})
    assert PaperBroker(clock, "p2", store, starting_cash=D(1)).get_cash() == 5


def test_postgres_persistence(pg_dsn: str) -> None:
    store, clock = PostgresPaperState(pg_dsn), FakeClock()
    a = PaperBroker(clock, "p1", store, starting_cash=D(100_000))
    a.tick({SYM: D(60_000)})
    a.submit(req("buy"))
    b = PaperBroker(clock, "p1", PostgresPaperState(pg_dsn), starting_cash=D(1))
    assert b.get_positions() == {SYM: D("0.5")} and b.get_cash() == a.get_cash()
    store.close()


def test_core_recovers_on_top_of_a_restarted_paper_broker() -> None:
    """Whole-system restart: event log in memory, paper state in its store, nothing else survives."""
    log, paper_state, clock = InMemoryEventStore(), InMemoryPaperState(), FakeClock()
    broker = PaperBroker(clock, ACCOUNT, paper_state, starting_cash=D(10_000))
    core, _, _ = build_core(log, clock, broker, prices={"AAPL": "100"})
    i = intent_now(clock)
    oid = core.submit_intent(i).order_id
    assert oid

    reborn_broker = PaperBroker(clock, ACCOUNT, paper_state, starting_cash=D(1))
    survivor: TradingCore = build_core(log, clock, reborn_broker, fund=False)[0]
    survivor.recover()
    st = survivor.projector.refresh()
    assert st.orders[oid].status is S.FILLED and st.anomalies == []
    assert st.position_qty("s1", "AAPL") == reborn_broker.positions["AAPL"] == 10
