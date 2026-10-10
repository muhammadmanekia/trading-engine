from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from engine.domain.events import (
    AccountFunded,
    CircuitBreakerReset,
    CircuitBreakerTripped,
    BreakerTrip,
    KillSwitchSet,
    ModeSet,
    OrderCreated,
    OrderFilled,
)
from engine.domain.types import AssetClass, Mode, Side
from engine.eventlog.base import make_event
from engine.eventlog.memory import InMemoryEventStore
from engine.projections.account import AccountProjector, replay
from tests.helpers import ACCOUNT, T0, D


def emit(store: InMemoryEventStore, model, minutes: int = 0, key: str | None = None):  # type: ignore[no-untyped-def]
    oid = getattr(model, "order_id", None)
    stream = f"order:{oid}" if oid else f"account:{ACCOUNT}"
    return store.append(
        [make_event(model, stream_id=stream, ts=T0 + timedelta(minutes=minutes), idempotency_key=key)]
    )


def order(store: InMemoryEventStore, oid: str, side: Side = Side.BUY, qty: str = "10") -> None:
    emit(
        store,
        OrderCreated(
            account_id=ACCOUNT,
            order_id=oid,
            intent_id=oid,
            strategy_id="s1",
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            side=side,
            quantity=D(qty),
            role="entry" if side is Side.BUY else "exit",
        ),
    )


def fill(
    store: InMemoryEventStore,
    oid: str,
    side: Side,
    qty: str,
    price: str,
    fee: str = "0",
    fid: str = "f1",
    minutes: int = 0,
) -> None:
    emit(
        store,
        OrderFilled(
            account_id=ACCOUNT,
            order_id=oid,
            strategy_id="s1",
            symbol="AAPL",
            side=side,
            quantity=D(qty),
            price=D(price),
            fee=D(fee),
            fill_id=fid,
        ),
        minutes,
    )


def test_funding_and_control_flags() -> None:
    s = InMemoryEventStore()
    emit(s, AccountFunded(account_id=ACCOUNT, amount=D(1000)))
    emit(s, KillSwitchSet(account_id=ACCOUNT, active=True, reason="test"))
    emit(s, ModeSet(account_id=ACCOUNT, mode=Mode.LIVE))
    st_ = replay(ACCOUNT, s.read_all())
    assert st_.cash == 1000 and st_.kill_switch and st_.mode is Mode.LIVE


def test_buy_then_sell_realizes_pnl_and_updates_cash() -> None:
    s = InMemoryEventStore()
    emit(s, AccountFunded(account_id=ACCOUNT, amount=D(10_000)))
    order(s, "o1")
    fill(s, "o1", Side.BUY, "10", "100", fee="1")
    order(s, "o2", Side.SELL)
    fill(s, "o2", Side.SELL, "10", "110", fee="1", fid="f2")
    st_ = replay(ACCOUNT, s.read_all())
    # cost basis includes the buy fee: (1000+1)/10 = 100.1 ; pnl = (110-100.1)*10 - 1 = 98
    assert st_.realized_total == D(98)
    assert st_.cash == D(10_000) - 1001 + 1100 - 1
    assert st_.position_qty("s1", "AAPL") == 0
    assert st_.consecutive_losses["s1"] == 0
    assert st_.orders["o1"].status.value == "FILLED"


def test_losses_track_consecutive_count_and_drawdown() -> None:
    s = InMemoryEventStore()
    emit(s, AccountFunded(account_id=ACCOUNT, amount=D(10_000)))
    for n, price in enumerate(["95", "90", "99"], start=1):
        order(s, f"b{n}")
        fill(s, f"b{n}", Side.BUY, "10", "100", fid=f"bf{n}")
        order(s, f"s{n}", Side.SELL)
        fill(s, f"s{n}", Side.SELL, "10", price, fid=f"sf{n}")
    st_ = replay(ACCOUNT, s.read_all())
    assert st_.consecutive_losses["s1"] == 3
    assert st_.strategy_realized["s1"] == D(-50 - 100 - 10)
    assert st_.drawdown("s1") == D(160)  # peak stayed at 0


def test_a_win_resets_the_loss_streak() -> None:
    s = InMemoryEventStore()
    emit(s, AccountFunded(account_id=ACCOUNT, amount=D(10_000)))
    for n, price in enumerate(["90", "110"], start=1):
        order(s, f"b{n}")
        fill(s, f"b{n}", Side.BUY, "1", "100", fid=f"bf{n}")
        order(s, f"s{n}", Side.SELL, qty="1")
        fill(s, f"s{n}", Side.SELL, "1", price, fid=f"sf{n}")
    assert replay(ACCOUNT, s.read_all()).consecutive_losses["s1"] == 0


def test_duplicate_fill_does_not_double_count() -> None:
    s = InMemoryEventStore()
    emit(s, AccountFunded(account_id=ACCOUNT, amount=D(10_000)))
    order(s, "o1")
    fill(s, "o1", Side.BUY, "5", "100", fid="same")
    fill(s, "o1", Side.BUY, "5", "100", fid="same")
    st_ = replay(ACCOUNT, s.read_all())
    assert st_.position_qty("s1", "AAPL") == 5 and st_.cash == D(9_500)
    assert any("duplicate fill" in a for a in st_.anomalies)


def test_oversell_is_clamped_and_flagged_never_negative() -> None:
    s = InMemoryEventStore()
    order(s, "o1", Side.SELL)
    fill(s, "o1", Side.SELL, "10", "100")
    st_ = replay(ACCOUNT, s.read_all())
    assert st_.position_qty("s1", "AAPL") == 0
    assert any("exceeds position" in a for a in st_.anomalies)


def test_breaker_trip_and_reset() -> None:
    s = InMemoryEventStore()
    emit(s, CircuitBreakerTripped(account_id=ACCOUNT, trip=BreakerTrip(scope="global", code="X")))
    emit(
        s,
        CircuitBreakerTripped(
            account_id=ACCOUNT, trip=BreakerTrip(scope="strategy", strategy_id="s1", code="Y")
        ),
    )
    st_ = replay(ACCOUNT, s.read_all())
    assert st_.halted_global and st_.halted_strategies == {"s1"}
    emit(s, CircuitBreakerReset(account_id=ACCOUNT, scope="global"))
    emit(s, CircuitBreakerReset(account_id=ACCOUNT, scope="strategy", strategy_id="s1"))
    st_ = replay(ACCOUNT, s.read_all())
    assert not st_.halted_global and not st_.halted_strategies


def test_replay_ignores_other_accounts_via_projector() -> None:
    s = InMemoryEventStore()
    emit(s, AccountFunded(account_id=ACCOUNT, amount=D(100)))
    emit(s, AccountFunded(account_id="other", amount=D(999)))
    p = AccountProjector(s, ACCOUNT)
    assert p.refresh().cash == 100


@settings(max_examples=60, deadline=None)
@given(
    trades=st.lists(
        st.tuples(st.integers(1, 20), st.integers(50, 150), st.integers(50, 150), st.booleans()),
        min_size=1,
        max_size=12,
    ),
    cut=st.integers(0, 100),
)
def test_incremental_equals_full_replay_for_any_prefix(trades, cut) -> None:  # type: ignore[no-untyped-def]
    """Applying events incrementally in two steps equals replaying the whole log (determinism)."""
    s = InMemoryEventStore()
    emit(s, AccountFunded(account_id=ACCOUNT, amount=D(1_000_000)))
    for n, (qty, buy, sell, fee_on) in enumerate(trades):
        fee = "0.5" if fee_on else "0"
        order(s, f"b{n}", qty=str(qty))
        fill(s, f"b{n}", Side.BUY, str(qty), str(buy), fee=fee, fid=f"bf{n}", minutes=n)
        order(s, f"s{n}", Side.SELL, qty=str(qty))
        fill(s, f"s{n}", Side.SELL, str(qty), str(sell), fee=fee, fid=f"sf{n}", minutes=n)
    events = s.read_all()
    full = replay(ACCOUNT, events)
    k = cut % (len(events) + 1)
    inc = replay(ACCOUNT, events[:k])
    for e in events[k:]:
        inc.apply(e)
    assert (inc.cash, inc.realized_total, dict(inc.positions)) == (
        full.cash,
        full.realized_total,
        dict(full.positions),
    )
    # conservation: cash change equals realized pnl when flat (all positions closed)
    assert all(p.quantity == 0 for p in full.positions.values())
    assert full.cash == D(1_000_000) + full.realized_total
    assert isinstance(full.cash, Decimal)
