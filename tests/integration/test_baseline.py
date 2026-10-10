"""Opening balances: new accounts, pre-existing accounts (unmanaged holdings), and paper funding."""

from __future__ import annotations

from engine.domain.events import BaselinePosition
from engine.domain.types import Side
from engine.eventlog.base import EventStore
from tests.helpers import D, alpaca_runtime, build_core, intent_now


def test_new_alpaca_account_is_imported_from_the_broker_once(store: EventStore) -> None:
    rt, fake, _ = alpaca_runtime(store)
    rt.bootstrap()
    rt.bootstrap()  # idempotent
    st = rt.core.projector.refresh()
    assert st.cash == D(100_000) and st.funded and st.positions == {}
    assert [e.type for e in store.read_all()].count("BaselineImported") == 1


def test_existing_holdings_are_imported_as_unmanaged_and_counted_in_risk(store: EventStore) -> None:
    rt, fake, _ = alpaca_runtime(store, symbols=("AAPL", "MSFT"))
    fake.mock.prices["MSFT"] = D(300)
    fake.mock.positions["MSFT"] = D(100)  # $30,000 already held at the broker
    fake.mock.cash = D(70_000)
    rt.bootstrap()
    st = rt.core.projector.refresh()
    pos = st.positions[("unmanaged", "MSFT")]
    assert pos.quantity == 100 and pos.avg_cost == 300  # no cost basis available: priced at the market
    assert st.equity({"MSFT": D(300)}) == D(100_000)
    assert rt.reconciler.run().ok  # the import matches the broker exactly


def test_strategies_can_never_sell_unmanaged_holdings(store: EventStore) -> None:
    rt, fake, _ = alpaca_runtime(store, symbols=("MSFT",))
    fake.mock.prices["MSFT"] = D(300)
    fake.mock.positions["MSFT"] = D(100)
    rt.bootstrap()
    res = rt.core.submit_intent(
        intent_now(rt.core.clock, symbol="MSFT", side=Side.SELL, stop_loss_price=None)
    )  # type: ignore[arg-type]
    assert res.decision is not None and res.decision.code == "NO_POSITION"
    assert fake.mock.positions["MSFT"] == 100


def test_unmanaged_exposure_limits_new_entries(store: EventStore) -> None:
    """$30k of MSFT out of $100k equity leaves less room under the per-asset cap for MSFT than for AAPL."""
    rt, fake, _ = alpaca_runtime(store, symbols=("AAPL", "MSFT"))
    fake.mock.prices["MSFT"] = D(300)
    fake.mock.positions["MSFT"] = D(100)
    fake.mock.cash = D(70_000)
    rt.bootstrap()
    res = rt.core.submit_intent(intent_now(rt.core.clock, symbol="MSFT", quantity=50, stop_loss_price=250))  # type: ignore[arg-type]
    assert res.decision is not None and res.decision.code == "NO_CAPACITY"  # already at the 30% position cap


def test_unmanaged_positions_are_not_given_protective_stops(store: EventStore) -> None:
    rt, fake, _ = alpaca_runtime(store, symbols=("MSFT",))
    fake.mock.prices["MSFT"] = D(300)
    fake.mock.positions["MSFT"] = D(10)
    rt.bootstrap()
    rt.core.manager.sync()
    assert fake.mock.placed_orders == []


def test_paper_funding_happens_exactly_once(store: EventStore) -> None:
    core, _, _ = build_core(store, fund=False)
    assert core.control.fund_once(D(5_000)) and not core.control.fund_once(D(5_000))
    assert core.projector.refresh().cash == 5_000


def test_baseline_import_with_explicit_positions(store: EventStore) -> None:
    core, _, _ = build_core(store, fund=False)
    assert core.control.import_baseline(
        D(1_000), [BaselinePosition(symbol="BTC-USD", quantity=D("0.5"), avg_cost=D(60_000))]
    )
    assert not core.control.import_baseline(D(1_000))  # a second import is refused
    st = core.projector.refresh()
    assert st.cash == 1_000 and st.positions[("unmanaged", "BTC-USD")].quantity == D("0.5")
