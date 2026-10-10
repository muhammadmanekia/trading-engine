"""Deterministic projection of the event log into account + order state.

`replay(events)` is a pure fold. Applying a prefix incrementally gives the same result as replaying it.
The fold never raises on historical data: inconsistencies are recorded in `anomalies`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from engine.domain.events import (
    AccountFunded,
    AnomalyDetected,
    CircuitBreakerReset,
    CircuitBreakerTripped,
    IntentAbandoned,
    IntentReceived,
    KillSwitchSet,
    ModeSet,
    OrderAcked,
    OrderCancelled,
    OrderCreated,
    OrderExpired,
    OrderFilled,
    OrderRejected,
    OrderSubmitAttempted,
    RiskDecisionMade,
)
from engine.domain.intent import Intent
from engine.domain.types import Mode, Side
from engine.eventlog.base import EventStore, StoredEvent, decode
from engine.orders.state import IllegalTransition, OrderState, OrderStatus, transition

ZERO = Decimal(0)


@dataclass
class Position:
    quantity: Decimal = ZERO
    avg_cost: Decimal = ZERO


@dataclass
class IntentRecord:
    intent: Intent
    decision: str | None = None  # approve | resize | reject
    approved_quantity: Decimal | None = None
    order_id: str | None = None
    abandoned: bool = False


@dataclass
class AccountState:
    account_id: str
    cash: Decimal = ZERO
    mode: Mode = Mode.PAPER
    kill_switch: bool = False
    halted_global: bool = False
    halted_strategies: set[str] = field(default_factory=set)
    positions: dict[tuple[str, str], Position] = field(default_factory=dict)  # (strategy, symbol)
    realized_total: Decimal = ZERO
    realized_by_day: dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: ZERO))
    strategy_realized: dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: ZERO))
    strategy_peak: dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: ZERO))
    consecutive_losses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    intents: dict[str, IntentRecord] = field(default_factory=dict)
    order_times: list[datetime] = field(default_factory=list)  # creation time of entry/exit orders
    orders: dict[str, OrderState] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    last_seq: int = 0

    # ---- queries ----------------------------------------------------------------------------
    def position_qty(self, strategy_id: str, symbol: str) -> Decimal:
        p = self.positions.get((strategy_id, symbol))
        return p.quantity if p else ZERO

    def symbol_qty(self, symbol: str) -> Decimal:
        return sum((p.quantity for (_, s), p in self.positions.items() if s == symbol), ZERO)

    def _value(self, qty: Decimal, symbol: str, avg: Decimal, prices: Mapping[str, Decimal]) -> Decimal:
        return qty * prices.get(symbol, avg)

    def exposure(self, prices: Mapping[str, Decimal]) -> Decimal:
        return sum(
            (self._value(p.quantity, s, p.avg_cost, prices) for (_, s), p in self.positions.items()), ZERO
        )

    def symbol_exposure(self, symbol: str, prices: Mapping[str, Decimal]) -> Decimal:
        return sum(
            (
                self._value(p.quantity, s, p.avg_cost, prices)
                for (_, s), p in self.positions.items()
                if s == symbol
            ),
            ZERO,
        )

    def strategy_exposure(self, strategy_id: str, prices: Mapping[str, Decimal]) -> Decimal:
        return sum(
            (
                self._value(p.quantity, s, p.avg_cost, prices)
                for (st, s), p in self.positions.items()
                if st == strategy_id
            ),
            ZERO,
        )

    def equity(self, prices: Mapping[str, Decimal]) -> Decimal:
        return self.cash + self.exposure(prices)

    def drawdown(self, strategy_id: str) -> Decimal:
        return self.strategy_peak[strategy_id] - self.strategy_realized[strategy_id]

    def orders_since(self, since: datetime) -> int:
        return sum(1 for t in self.order_times if t >= since)

    def open_orders(self) -> list[OrderState]:
        return [o for o in self.orders.values() if not o.is_terminal]

    # ---- fold -------------------------------------------------------------------------------
    def apply(self, stored: StoredEvent) -> None:
        self.last_seq = stored.seq
        ev = decode(stored)
        match ev:
            case AccountFunded():
                self.cash += ev.amount
            case KillSwitchSet():
                self.kill_switch = ev.active
            case ModeSet():
                self.mode = ev.mode
            case CircuitBreakerTripped():
                if ev.trip.scope == "global":
                    self.halted_global = True
                elif ev.trip.strategy_id:
                    self.halted_strategies.add(ev.trip.strategy_id)
            case CircuitBreakerReset():
                if ev.scope == "global":
                    self.halted_global = False
                elif ev.strategy_id:
                    self.halted_strategies.discard(ev.strategy_id)
                    self.consecutive_losses[ev.strategy_id] = 0
                    # restart the drawdown baseline from the current level
                    self.strategy_peak[ev.strategy_id] = self.strategy_realized[ev.strategy_id]
            case IntentReceived():
                self.intents.setdefault(ev.intent.intent_id, IntentRecord(intent=ev.intent))
            case RiskDecisionMade():
                rec = self.intents.get(ev.intent_id)
                if rec is None:
                    self.anomalies.append(f"decision for unknown intent {ev.intent_id}")
                else:
                    rec.decision = ev.decision
                    rec.approved_quantity = ev.approved_quantity
            case IntentAbandoned():
                if (rec2 := self.intents.get(ev.intent_id)) is not None:
                    rec2.abandoned = True
            case OrderCreated():
                self._order_created(ev, stored.ts)
            case OrderSubmitAttempted() | OrderAcked() | OrderRejected() | OrderCancelled() | OrderExpired():
                self._order_lifecycle(ev)
            case OrderFilled():
                self._order_filled(ev, stored.ts)
            case AnomalyDetected():
                self.anomalies.append(f"{ev.code}: {ev.detail}")

    def _order_created(self, ev: OrderCreated, ts: datetime) -> None:
        if ev.order_id in self.orders:
            self.anomalies.append(f"duplicate OrderCreated {ev.order_id}")
            return
        self.orders[ev.order_id] = OrderState(
            order_id=ev.order_id,
            intent_id=ev.intent_id,
            strategy_id=ev.strategy_id,
            symbol=ev.symbol,
            side=ev.side,
            asset_class=ev.asset_class,
            role=ev.role,
            quantity=ev.quantity,
            parent_order_id=ev.parent_order_id,
            limit_price=ev.limit_price,
            stop_loss_price=ev.stop_loss_price,
            take_profit_price=ev.take_profit_price,
        )
        if ev.role in ("entry", "exit"):
            self.order_times.append(ts)
            if len(self.order_times) > 2000:
                del self.order_times[:1000]
            if (rec := self.intents.get(ev.intent_id)) is not None:
                rec.order_id = ev.order_id

    def _order_lifecycle(
        self, ev: OrderSubmitAttempted | OrderAcked | OrderRejected | OrderCancelled | OrderExpired
    ) -> None:
        order = self.orders.get(ev.order_id)
        if order is None:
            self.anomalies.append(f"{type(ev).__name__} for unknown order {ev.order_id}")
            return
        try:
            order.status = transition(order.status, type(ev).__name__)
        except IllegalTransition as exc:
            self.anomalies.append(f"{ev.order_id}: {exc}")
            return
        if isinstance(ev, OrderSubmitAttempted):
            order.attempts = max(order.attempts, ev.attempt)
        elif isinstance(ev, OrderAcked):
            order.broker_order_id = ev.broker_order_id
        elif isinstance(ev, OrderRejected):
            order.reject_code = ev.code

    def _order_filled(self, ev: OrderFilled, ts: datetime) -> None:
        order = self.orders.get(ev.order_id)
        if order is None:
            self.anomalies.append(f"fill {ev.fill_id} for unknown order {ev.order_id}")
        elif ev.fill_id in order.fill_ids:
            self.anomalies.append(f"duplicate fill {ev.fill_id} on {ev.order_id}")
            return  # never double-count economics
        else:
            new_filled = order.filled_quantity + ev.quantity
            try:
                order.status = transition(
                    order.status, "OrderFilled", fully_filled=new_filled >= order.quantity
                )
            except IllegalTransition as exc:
                self.anomalies.append(f"{ev.order_id}: {exc}")
            if new_filled > order.quantity:
                self.anomalies.append(f"{ev.order_id}: overfill {new_filled} > {order.quantity}")
            order.fill_ids.add(ev.fill_id)
            order.filled_quantity = new_filled
        self._apply_fill_economics(ev, ts)

    def _apply_fill_economics(self, ev: OrderFilled, ts: datetime) -> None:
        key = (ev.strategy_id, ev.symbol)
        pos = self.positions.setdefault(key, Position())
        notional = ev.quantity * ev.price
        if ev.side is Side.BUY:
            new_qty = pos.quantity + ev.quantity
            pos.avg_cost = (pos.quantity * pos.avg_cost + notional + ev.fee) / new_qty
            pos.quantity = new_qty
            self.cash -= notional + ev.fee
            return
        sold = min(ev.quantity, pos.quantity)
        if sold < ev.quantity:
            self.anomalies.append(f"sell of {ev.quantity} {ev.symbol} exceeds position {pos.quantity}")
        pnl = (ev.price - pos.avg_cost) * sold - ev.fee
        pos.quantity -= sold
        if pos.quantity == 0:
            pos.avg_cost = ZERO
        self.cash += notional - ev.fee
        self.realized_total += pnl
        self.realized_by_day[ts.date().isoformat()] += pnl
        s = ev.strategy_id
        self.strategy_realized[s] += pnl
        self.strategy_peak[s] = max(self.strategy_peak[s], self.strategy_realized[s])
        if pnl < 0:
            self.consecutive_losses[s] += 1
        elif pnl > 0:
            self.consecutive_losses[s] = 0


def replay(account_id: str, events: Iterable[StoredEvent]) -> AccountState:
    state = AccountState(account_id=account_id)
    for e in events:
        state.apply(e)
    return state


class AccountProjector:
    """Keeps an AccountState current by tailing the store from its last applied seq."""

    def __init__(self, store: EventStore, account_id: str) -> None:
        self._store = store
        self.state = AccountState(account_id=account_id)

    def refresh(self) -> AccountState:
        for e in self._store.read_all(after_seq=self.state.last_seq, account_id=self.state.account_id):
            self.state.apply(e)
        return self.state

    @property
    def order_status(self) -> dict[str, OrderStatus]:
        return {k: v.status for k, v in self.state.orders.items()}
