"""Order manager: turns approved intents into broker orders, safely and idempotently.

Write-ahead discipline: the ledger records intent-to-act (OrderCreated, OrderSubmitAttempted) BEFORE
the broker is called, and the broker call is keyed by a deterministic client_order_id, so a crash at any
point is recoverable by `sync()` without ever creating a second order.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from engine.broker.base import (
    Broker,
    BrokerOrder,
    BrokerRejected,
    BrokerStatus,
    BrokerUnavailable,
    OrderRequest,
)
from engine.clock import Clock
from engine.domain.events import (
    AnomalyDetected,
    OrderAcked,
    OrderCancelled,
    OrderCreated,
    OrderExpired,
    OrderFilled,
    OrderRejected,
    OrderSubmitAttempted,
)
from engine.domain.intent import Intent
from engine.domain.types import Frozen, Side
from engine.eventlog.base import DuplicateEvent, EventStore, make_event
from engine.orders.state import OrderState, OrderStatus
from engine.projections.account import AccountProjector

Hook = Callable[[str], None]


def order_id_for(intent_id: str) -> str:
    return f"ord-{intent_id}"


class OrderManager:
    def __init__(
        self,
        store: EventStore,
        broker: Broker,
        clock: Clock,
        projector: AccountProjector,
        *,
        max_submit_attempts: int = 5,
        hook: Hook | None = None,
    ) -> None:
        self.store = store
        self.broker = broker
        self.clock = clock
        self.projector = projector
        self.max_submit_attempts = max_submit_attempts
        self._hook = hook

    # ---- public -----------------------------------------------------------------------------
    @property
    def account_id(self) -> str:
        return self.projector.state.account_id

    def place(self, intent: Intent, qty: Decimal) -> str:
        """Create the order record and drive it to the broker. Safe to call twice for one intent."""
        order_id = order_id_for(intent.intent_id)
        role = "entry" if intent.side is Side.BUY else "exit"
        self._emit(
            OrderCreated(
                account_id=self.account_id,
                order_id=order_id,
                intent_id=intent.intent_id,
                strategy_id=intent.strategy_id,
                symbol=intent.symbol,
                asset_class=intent.asset_class,
                side=intent.side,
                quantity=qty,
                role=role,
                limit_price=intent.limit_price,
                stop_loss_price=intent.stop_loss_price,
                take_profit_price=intent.take_profit_price,
            ),
            key=f"order-created:{order_id}",
        )
        self._crash_point("after_created")
        wants_protection = intent.stop_loss_price is not None or intent.take_profit_price is not None
        if role == "entry" and wants_protection and not self.broker.supports_protective_orders:
            self._emit(
                OrderRejected(
                    account_id=self.account_id,
                    order_id=order_id,
                    code="NO_PROTECTION",
                    detail="broker cannot attach a protective order",
                ),
                key=f"order-rejected:{order_id}:NO_PROTECTION",
            )
            return order_id
        if role == "exit":
            self._release_protection(intent.strategy_id, intent.symbol)
        self._drive(order_id)
        return order_id

    def sync(self) -> None:
        """Reconcile every unfinished order with the broker. Also the crash-recovery entry point."""
        self.projector.refresh()
        for order in list(self.projector.state.orders.values()):
            if order.parent_order_id is not None:
                continue  # protective legs are reconciled through their parent
            if not order.is_terminal or self._has_open_children(order.order_id):
                self._drive(order.order_id)

    def cancel(self, order_id: str) -> bool:
        order = self.projector.state.orders.get(order_id)
        if order is None or order.is_terminal:
            return False
        try:
            self.broker.cancel(order_id)
        except BrokerUnavailable:
            return False
        self._drive(order_id)
        return True

    def cancel_open_entries(self) -> list[str]:
        """Used by the kill switch: stop new risk from building. Exits and protective legs stay live."""
        self.projector.refresh()
        out = []
        for o in list(self.projector.state.orders.values()):
            if o.role == "entry" and not o.is_terminal and self.cancel(o.order_id):
                out.append(o.order_id)
        return out

    def _release_protection(self, strategy_id: str, symbol: str) -> None:
        """A deliberate exit must not leave stop/take-profit legs working against a closed position."""
        parents: set[str] = set()
        for o in list(self.projector.state.orders.values()):
            if (
                o.role in ("stop", "take_profit")
                and not o.is_terminal
                and o.strategy_id == strategy_id
                and o.symbol == symbol
                and o.parent_order_id is not None
            ):
                try:
                    self.broker.cancel(o.order_id)
                except BrokerUnavailable:
                    continue
                parents.add(o.parent_order_id)
        for parent_id in parents:
            self._drive(parent_id)

    # ---- internals --------------------------------------------------------------------------
    def _crash_point(self, name: str) -> None:
        if self._hook is not None:
            self._hook(name)

    def _emit(self, model: Frozen, *, key: str | None = None) -> bool:
        """Append one event; returns False if its idempotency key already existed."""
        stream = f"order:{getattr(model, 'order_id', None) or 'misc'}"
        try:
            self.store.append([make_event(model, stream_id=stream, ts=self.clock.now(), idempotency_key=key)])
        except DuplicateEvent:
            self.projector.refresh()
            return False
        self.projector.refresh()
        return True

    def _order(self, order_id: str) -> OrderState:
        return self.projector.state.orders[order_id]

    def _has_open_children(self, order_id: str) -> bool:
        return any(
            o.parent_order_id == order_id and not o.is_terminal for o in self.projector.state.orders.values()
        )

    def _request(self, o: OrderState) -> OrderRequest:
        return OrderRequest(
            client_order_id=o.order_id,
            symbol=o.symbol,
            asset_class=o.asset_class,
            side=o.side,
            quantity=o.quantity,
            limit_price=o.limit_price,
            stop_loss_price=o.stop_loss_price,
            take_profit_price=o.take_profit_price,
        )

    def _drive(self, order_id: str) -> None:
        o = self._order(order_id)
        try:
            bo = self.broker.get_order(order_id)
        except BrokerUnavailable:
            return  # try again on the next sync; never guess
        if bo is not None:
            self._apply_broker_order(order_id, bo)
            return
        if o.status not in (OrderStatus.CREATED, OrderStatus.SUBMITTING):
            if not o.is_terminal:
                self._emit(
                    AnomalyDetected(
                        account_id=self.account_id,
                        order_id=order_id,
                        code="ORDER_MISSING_AT_BROKER",
                        detail=f"status {o.status}",
                    )
                )
            return
        # The broker definitively has no such order: it is safe to (re)submit.
        if o.attempts >= self.max_submit_attempts:
            self._emit(
                OrderRejected(account_id=self.account_id, order_id=order_id, code="SUBMIT_FAILED"),
                key=f"order-rejected:{order_id}:SUBMIT_FAILED",
            )
            return
        attempt = o.attempts + 1
        self._emit(
            OrderSubmitAttempted(account_id=self.account_id, order_id=order_id, attempt=attempt),
            key=f"submit:{order_id}:{attempt}",
        )
        self._crash_point("after_attempt")
        try:
            bo = self.broker.submit(self._request(self._order(order_id)))
        except BrokerRejected as exc:
            self._emit(
                OrderRejected(
                    account_id=self.account_id, order_id=order_id, code=exc.code, detail=exc.detail
                ),
                key=f"order-rejected:{order_id}:{exc.code}",
            )
            return
        except BrokerUnavailable:
            return  # ambiguous: the next sync() asks the broker before doing anything else
        self._crash_point("after_broker_submit")
        self._apply_broker_order(order_id, bo)

    def _apply_broker_order(self, order_id: str, bo: BrokerOrder) -> None:
        o = self._order(order_id)
        if (
            o.status in (OrderStatus.CREATED, OrderStatus.SUBMITTING)
            and bo.status is not BrokerStatus.REJECTED
        ):
            self._emit(
                OrderAcked(account_id=self.account_id, order_id=order_id, broker_order_id=bo.broker_order_id),
                key=f"ack:{order_id}",
            )
            self._crash_point("after_ack")

        for f in bo.fills:
            o = self._order(order_id)
            if f.fill_id in o.fill_ids:
                continue  # duplicate delivery
            if o.filled_quantity + f.quantity > o.quantity:
                self._emit(
                    AnomalyDetected(
                        account_id=self.account_id,
                        order_id=order_id,
                        code="OVERFILL",
                        detail=f"fill {f.fill_id} would exceed order quantity",
                    ),
                    key=f"overfill:{order_id}:{f.fill_id}",
                )
                continue
            self._emit(
                OrderFilled(
                    account_id=self.account_id,
                    order_id=order_id,
                    strategy_id=o.strategy_id,
                    symbol=o.symbol,
                    side=o.side,
                    quantity=f.quantity,
                    price=f.price,
                    fee=f.fee,
                    fill_id=f.fill_id,
                ),
                key=f"fill:{order_id}:{f.fill_id}",
            )

        o = self._order(order_id)
        if not o.is_terminal:
            if bo.status is BrokerStatus.REJECTED:
                self._emit(
                    OrderRejected(
                        account_id=self.account_id,
                        order_id=order_id,
                        code=bo.reject_reason or "BROKER_REJECTED",
                    ),
                    key=f"order-rejected:{order_id}:{bo.reject_reason or 'BROKER_REJECTED'}",
                )
            elif bo.status is BrokerStatus.CANCELLED:
                self._emit(
                    OrderCancelled(account_id=self.account_id, order_id=order_id), key=f"cancelled:{order_id}"
                )
            elif bo.status is BrokerStatus.EXPIRED:
                self._emit(
                    OrderExpired(account_id=self.account_id, order_id=order_id), key=f"expired:{order_id}"
                )

        # Protective legs enter the ledger once the parent's size is final.
        if self._order(order_id).is_terminal and bo.fills:
            for child in bo.children:
                self._ensure_child(order_id, child)
                self._apply_broker_order(child.client_order_id, child)

    def _ensure_child(self, parent_id: str, child: BrokerOrder) -> None:
        if child.client_order_id in self.projector.state.orders:
            return
        parent = self._order(parent_id)
        role = "stop" if child.client_order_id.endswith(":stop") else "take_profit"
        self._emit(
            OrderCreated(
                account_id=self.account_id,
                order_id=child.client_order_id,
                intent_id=parent.intent_id,
                strategy_id=parent.strategy_id,
                symbol=parent.symbol,
                asset_class=parent.asset_class,
                side=Side.SELL,
                quantity=child.quantity,
                role=role,
                parent_order_id=parent_id,
            ),
            key=f"order-created:{child.client_order_id}",
        )
