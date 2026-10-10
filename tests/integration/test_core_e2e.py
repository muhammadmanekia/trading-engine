"""End to end: Intent -> risk -> order -> mock broker -> fills -> projection, plus the control plane."""

from __future__ import annotations

import pytest

from engine.broker.mock import MockBroker
from engine.clock import FakeClock
from engine.domain.types import Mode
from engine.eventlog.base import EventStore
from engine.orders.state import OrderStatus as S
from engine.projections.account import replay
from tests.helpers import ACCOUNT, D, build_core, intent_now


def test_full_flow_is_fully_audited_in_order(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    i = intent_now(clock)
    res = core.submit_intent(i)
    assert res.decision is not None and res.decision.kind == "approve" and res.order_id
    types = [e.type for e in store.read_all()]
    assert types[:4] == ["AccountFunded", "IntentReceived", "RiskDecisionMade", "OrderCreated"]
    assert types.index("OrderSubmitAttempted") < types.index("OrderAcked") < types.index("OrderFilled")
    decision_event = next(e for e in store.read_all() if e.type == "RiskDecisionMade")
    assert decision_event.payload["decision"] == "approve" and decision_event.payload["checks_passed"]
    store.verify_chain()


def test_state_from_replay_equals_live_state(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    core.submit_intent(intent_now(clock))
    broker.set_price("AAPL", D(85))
    core.manager.sync()
    live = core.projector.refresh()
    fresh = replay(ACCOUNT, store.read_all(account_id=ACCOUNT))
    assert (live.cash, live.realized_total, dict(live.positions), live.consecutive_losses["s1"]) == (
        fresh.cash,
        fresh.realized_total,
        dict(fresh.positions),
        fresh.consecutive_losses["s1"],
    )
    assert {k: v.status for k, v in live.orders.items()} == {k: v.status for k, v in fresh.orders.items()}


def test_rejected_intent_is_logged_with_reason_and_never_reaches_the_broker(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    res = core.submit_intent(intent_now(clock, stop_loss_price=None))
    assert res.decision is not None and res.decision.code == "NO_STOP_LOSS" and res.order_id is None
    assert broker.submit_calls == 0
    logged = [e for e in store.read_all() if e.type == "RiskDecisionMade"]
    assert logged[0].payload["code"] == "NO_STOP_LOSS" and logged[0].payload["decision"] == "reject"


def test_resized_intent_trades_the_smaller_quantity(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    res = core.submit_intent(intent_now(clock, quantity=100))
    assert res.decision is not None and res.decision.kind == "resize"
    assert core.projector.refresh().position_qty("s1", "AAPL") == 30


def test_duplicate_delivery_is_a_noop(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    i = intent_now(clock)
    first = core.submit_intent(i)
    second = core.submit_intent(i)
    assert not first.duplicate and second.duplicate and second.order_id == first.order_id
    assert len(broker.placed_orders) == 1


def test_provider_failure_fails_closed(store: EventStore) -> None:
    core, broker, clock = build_core(store)

    def boom() -> object:
        raise ConnectionError("price feed down")

    core._market = boom  # type: ignore[assignment]
    res = core.submit_intent(intent_now(clock))
    assert res.decision is not None and res.decision.code == "PROVIDER_ERROR" and res.order_id is None
    assert broker.submit_calls == 0


def test_account_mismatch_is_refused_without_logging(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    before = len(store.read_all())
    res = core.submit_intent(intent_now(clock, account_id="other"))
    assert res.decision is not None and res.decision.code == "ACCOUNT_MISMATCH"
    assert len(store.read_all()) == before


# ---- control plane -----------------------------------------------------------------------------
def test_kill_switch_blocks_entries_cancels_resting_entries_but_keeps_stops_and_exits(
    store: EventStore,
) -> None:
    core, broker, clock = build_core(store)
    held = core.submit_intent(intent_now(clock, symbol="AAPL")).order_id  # filled, stop working
    resting = core.submit_intent(intent_now(clock, limit_price=95, strategy_id="s2")).order_id
    assert held and resting

    cancelled = core.activate_kill_switch("manual test")
    st = core.projector.refresh()
    assert cancelled == [resting] and st.orders[resting].status is S.CANCELLED
    assert st.orders[f"{held}:stop"].status is S.OPEN  # protection stays live

    blocked = core.submit_intent(intent_now(clock))
    assert blocked.decision is not None and blocked.decision.code == "KILL_SWITCH"
    exit_ = core.submit_intent(intent_now(clock, side="SELL", stop_loss_price=None))
    assert exit_.decision is not None and exit_.decision.allowed
    assert core.projector.refresh().position_qty("s1", "AAPL") == 0

    core.control.set_kill_switch(False)
    assert core.submit_intent(intent_now(clock)).decision.allowed  # type: ignore[union-attr]


def test_live_mode_needs_confirmation_and_must_match_the_broker(store: EventStore) -> None:
    core, broker, clock = build_core(store)
    with pytest.raises(PermissionError):
        core.control.set_mode(Mode.LIVE)
    core.control.set_mode(Mode.LIVE, confirmed=True)  # account says LIVE, broker is still PAPER
    res = core.submit_intent(intent_now(clock))
    assert res.decision is not None and res.decision.code == "MODE_MISMATCH" and broker.submit_calls == 0


def test_paper_and_live_share_one_code_path(store: EventStore) -> None:
    """Identical inputs, only the broker's mode flag differs -> identical outcomes."""
    results = []
    for mode in (Mode.PAPER, Mode.LIVE):
        from engine.eventlog.memory import InMemoryEventStore

        s = InMemoryEventStore()
        clock = FakeClock()
        core, broker, _ = build_core(s, clock, MockBroker(clock, mode=mode))
        core.control.set_mode(mode, confirmed=True)
        res = core.submit_intent(intent_now(clock, intent_id="same"))
        st = core.projector.refresh()
        results.append((res.decision.kind, st.cash, st.position_qty("s1", "AAPL")))  # type: ignore[union-attr]
    assert results[0] == results[1]


def test_daily_loss_trips_a_persisted_breaker_until_reset(store: EventStore) -> None:
    core, broker, clock = build_core(store, cash=1_000)
    core.submit_intent(intent_now(clock, quantity=30, stop_loss_price=70))
    broker.set_price("AAPL", D(10))  # gap far through the stop: the position cap sized this to 3 shares
    core.manager.sync()
    broker.set_price("AAPL", D(100))
    core.projector.refresh()
    tripped = core.submit_intent(intent_now(clock))
    # loss ~ 3 * 90 = $270 of ~$1000 starting equity: over the 25% daily cap
    assert tripped.decision is not None and tripped.decision.code == "DAILY_LOSS_CAP"
    assert core.projector.refresh().halted_global
    again = core.submit_intent(intent_now(clock))
    assert again.decision is not None and again.decision.code == "HALTED_GLOBAL"
    core.control.reset_breaker("global")
    assert not core.projector.refresh().halted_global


def test_strategy_breaker_reset_requires_strategy_id(store: EventStore) -> None:
    core, _, _ = build_core(store)
    with pytest.raises(ValueError):
        core.control.reset_breaker("strategy")
