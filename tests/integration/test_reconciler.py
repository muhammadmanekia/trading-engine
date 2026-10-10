"""Reconciler: detects broker/ledger disagreement and fails closed, without editing the ledger itself."""

from __future__ import annotations

import pytest

from engine.broker.base import BrokerUnavailable, OrderRequest
from engine.domain.types import AssetClass, Side
from engine.eventlog.base import EventStore
from engine.reconciler import Reconciler, ReconcilerPolicy
from engine.service.runtime import AccountRuntime
from tests.fakes.alpaca_server import FakeAlpaca
from tests.helpers import D, T0, alpaca_runtime, intent_now


def boot(store: EventStore) -> tuple[AccountRuntime, FakeAlpaca]:
    rt, fake, clock = alpaca_runtime(store)
    rt.bootstrap()
    return rt, fake


def trade(rt: AccountRuntime) -> None:
    clock = rt.core.clock
    res = rt.core.submit_intent(intent_now(clock))  # type: ignore[arg-type]
    assert res.order_id and res.decision is not None and res.decision.allowed


def events(store: EventStore, kind: str) -> list[dict]:  # type: ignore[type-arg]
    return [e.payload for e in store.read_all() if e.type == kind]


def test_clean_account_reports_ok_and_records_the_run(store: EventStore) -> None:
    rt, _ = boot(store)
    trade(rt)
    res = rt.reconciler.run()
    assert res.ok and not res.halted
    done = events(store, "ReconciliationCompleted")
    assert done and done[-1]["ok"] is True and events(store, "DriftDetected") == []


def test_position_drift_halts_only_after_it_persists(store: EventStore) -> None:
    rt, fake = boot(store)
    trade(rt)
    fake.mock.positions["AAPL"] += D(5)  # someone bought 5 shares behind our back
    first = rt.reconciler.run()
    assert [d.kind for d in first.drifts] == ["POSITION_MISMATCH"] and not first.halted  # could be a race
    assert not rt.core.projector.refresh().kill_switch
    second = rt.reconciler.run()
    assert second.halted and rt.core.projector.refresh().kill_switch
    assert len(events(store, "DriftDetected")) == 1  # recorded when it appeared, not every run


def test_a_halted_account_blocks_new_entries(store: EventStore) -> None:
    rt, fake = boot(store)
    trade(rt)
    fake.mock.positions["AAPL"] += D(5)
    rt.reconciler.run()
    rt.reconciler.run()
    res = rt.core.submit_intent(intent_now(rt.core.clock))  # type: ignore[arg-type]
    assert res.decision is not None and res.decision.code == "KILL_SWITCH"


def test_transient_drift_that_clears_never_halts(store: EventStore) -> None:
    rt, fake = boot(store)
    trade(rt)
    fake.mock.positions["AAPL"] += D(5)
    assert rt.reconciler.run().drifts
    fake.mock.positions["AAPL"] -= D(5)  # it was only a fill in flight
    res = rt.reconciler.run()
    assert res.ok and not rt.core.projector.refresh().kill_switch
    assert events(store, "ReconciliationCompleted")[-1]["summary"] == "recovered"
    fake.mock.positions["AAPL"] += D(5)
    assert not rt.reconciler.run().halted  # the streak restarted from zero


def test_small_cash_difference_only_warns(store: EventStore) -> None:
    rt, fake = boot(store)
    fake.mock.cash -= D(800)  # 0.8% of equity: above the 0.5% tolerance, far below the 5% critical line
    res = rt.reconciler.run()
    assert [(d.kind, d.severity) for d in res.drifts] == [("CASH_MISMATCH", "warning")]
    rt.reconciler.run()
    rt.reconciler.run()
    assert not rt.core.projector.refresh().kill_switch  # warnings never halt


def test_large_cash_difference_is_critical(store: EventStore) -> None:
    rt, fake = boot(store)
    fake.mock.cash -= D(20_000)
    rt.reconciler.run()
    assert rt.reconciler.run().halted


def test_unknown_working_order_at_the_broker_is_flagged_not_fatal(store: EventStore) -> None:
    rt, fake = boot(store)
    fake.mock.submit(  # a manual order placed in the broker's own app
        OrderRequest(
            client_order_id="manual-1",
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            side=Side.BUY,
            quantity=D(1),
            limit_price=D(50),
        )  # fmt: skip
    )
    res = rt.reconciler.run()
    assert [(d.kind, d.severity) for d in res.drifts] == [("ORPHAN_BROKER_ORDER", "warning")]
    assert not res.halted


def test_working_ledger_order_missing_at_the_broker_is_flagged_after_the_grace_period(
    store: EventStore,
) -> None:
    rt, fake = boot(store)
    res = rt.core.submit_intent(intent_now(rt.core.clock, limit_price=95))  # type: ignore[arg-type]
    assert res.order_id
    del fake.mock._orders[res.order_id]  # the broker 'lost' it
    assert rt.reconciler.run().ok  # inside the grace period it may simply be in flight
    rt.core.clock.advance(minutes=5)  # type: ignore[attr-defined]
    drifts = rt.reconciler.run().drifts
    assert [d.kind for d in drifts] == ["ORDER_MISSING_AT_BROKER"]


def test_unreadable_broker_is_unknown_not_fine(store: EventStore) -> None:
    rt, fake = boot(store)
    fake.outage = True
    before = len(store.read_all())
    with pytest.raises(BrokerUnavailable):
        rt.reconciler.run()
    assert len(store.read_all()) == before  # no 'all clear' recorded, nothing halted
    assert not rt.core.projector.refresh().kill_switch


def test_operator_adjustment_resolves_a_legitimate_drift(store: EventStore) -> None:
    rt, fake = boot(store)
    fake.mock.cash += D(5_000)  # the owner deposited money at the broker
    assert rt.reconciler.run().drifts
    rt.core.control.adjust("deposit of $5000 on the broker side", cash_delta=D(5_000))
    assert rt.reconciler.run().ok
    with pytest.raises(ValueError):
        rt.core.control.adjust("  ")  # an adjustment must be explained


def test_policy_confirm_runs_is_configurable(store: EventStore) -> None:
    rt, fake = boot(store)
    trade(rt)
    fake.mock.positions["AAPL"] += D(1)
    strict = Reconciler(rt.core, rt.broker, rt.core.clock, ReconcilerPolicy(confirm_runs=1))
    assert strict.run().halted
    assert T0 is not None
