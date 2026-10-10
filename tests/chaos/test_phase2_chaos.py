"""Whole-service chaos: random outages, timeouts, rejections, price moves, partial fills and restarts.

Whatever the schedule, once the faults stop the system must converge to: one broker order per intent, a
ledger that matches the broker, every position protected, no anomalies, an intact hash chain, and no
reconciler-triggered halt.
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from engine.eventlog.base import EventStore
from engine.orders.state import OrderStatus as S
from engine.service.alerts import RecordingAlertSink
from engine.service.executor import Executor, Intervals
from engine.service.inbox import InMemoryInbox
from engine.service.leader import InProcessLeaderLock
from tests.fakes.alpaca_server import FakeAlpaca
from tests.helpers import D, alpaca_runtime, intent_payload

STEPS = 60


@pytest.fixture(autouse=True)
def _release_locks() -> object:
    yield
    InProcessLeaderLock._held.clear()


def boot(store: EventStore, clock, fake: FakeAlpaca | None = None):  # type: ignore[no-untyped-def]
    rt, fake, clock = alpaca_runtime(store, clock, fake=fake)
    return rt, fake, clock


@pytest.mark.parametrize("seed", range(8))
def test_random_faults_and_restarts_never_make_ledger_and_broker_diverge(
    store: EventStore, seed: int
) -> None:
    rng = random.Random(seed)
    rt, fake, clock = alpaca_runtime(store)
    inbox, alerts = InMemoryInbox(), RecordingAlertSink()
    intervals = Intervals(sync=clock.now() - clock.now(), reconcile=clock.now() - clock.now())  # every tick
    ex = Executor(store, [rt], inbox, InProcessLeaderLock("chaos"), alerts, clock, intervals=intervals)
    assert ex.start()
    submitted: list[str] = []
    outage_left = 0

    for _ in range(STEPS):
        roll = rng.random()
        if roll < 0.35:
            price = fake.mock.prices["AAPL"]
            payload = intent_payload(
                clock,
                quantity=rng.randint(1, 30),
                stop_loss_price=str((price * D("0.9")).quantize(D("0.01"))),
            )
            inbox.put(payload)
            submitted.append(str(payload["intent_id"]))
        elif roll < 0.50:
            move = D(rng.uniform(-0.06, 0.06)).quantize(D("0.0001"))
            fake.set_price("AAPL", max(D("5"), (fake.mock.prices["AAPL"] * (1 + move)).quantize(D("0.01"))))
        elif roll < 0.60:
            outage_left = rng.randint(1, 3)
        elif roll < 0.72:
            fake.fail_next_submit(rng.choice(["reject", "unavailable_before", "unavailable_after"]))  # type: ignore[arg-type]
        elif roll < 0.80:
            ex.stop()  # crash/restart: a brand-new process over the same log and the same broker
            rt, fake, clock = boot(store, clock, fake)
            ex = Executor(
                store, [rt], inbox, InProcessLeaderLock("chaos"), alerts, clock, intervals=intervals
            )
            assert ex.start()
        elif roll < 0.88:
            fake.mock.fill_fraction = rng.choice([D("0.5"), D(1)])
        fake.outage = outage_left > 0
        outage_left = max(0, outage_left - 1)
        clock.advance(seconds=rng.randint(2, 10))
        assert ex.tick()  # the loop itself never dies

    # ---- the faults stop; the system must settle ------------------------------------------
    fake.outage = False
    fake.mock.fill_fraction = D(1)
    fake._faults.clear()
    for _ in range(8):
        clock.advance(seconds=5)
        fake.mock.step()
        assert ex.tick()

    st = rt.core.projector.refresh()
    entries = [cid for cid in fake.mock.placed_orders if cid.startswith("ord-")]
    assert len(entries) == len(set(entries)), "an intent produced two broker orders"
    assert set(entries) <= {f"ord-{i}" for i in submitted}
    assert not st.kill_switch, [a for a in alerts.alerts if "Kill" in a.title or "halted" in a.title]
    assert st.anomalies == []
    assert rt.reconciler.run().ok
    for symbol, qty in fake.mock.get_positions().items():
        assert st.symbol_qty(symbol) == qty
    # every held share is covered by a working stop
    protected = sum(
        (
            o.quantity - o.filled_quantity
            for o in st.orders.values()
            if o.role == "stop" and not o.is_terminal
        ),
        Decimal(0),
    )
    assert protected >= st.symbol_qty("AAPL")
    # nothing the ledger believes is open is missing at the broker
    assert all(
        fake.mock.get_order(o.order_id) is not None
        for o in st.orders.values()
        if o.status in (S.OPEN, S.PARTIAL) and o.parent_order_id is None
    )
    store.verify_chain()
