"""Order manager: turns approved intents into broker orders, safely and idempotently.

Write-ahead discipline: the ledger records intent-to-act (OrderCreated, OrderSubmitAttempted) BEFORE
the broker is called, and the broker call is keyed by a deterministic client_order_id, so a crash at any
point is recoverable by `sync()` without ever creating a second order.

Protection invariant (checked after every fill and on every sync): the ledger position of each
(strategy, symbol) is covered by a working stop-loss or a pending exit order. If it is not, the manager
places a replacement stop. A stop the broker rejects triggers an emergency flatten and trips the
strategy's breaker, because an unprotected position is a position the risk engine did not approve.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from engine.broker.base import (
    Broker,
    BrokerFill,
    BrokerOrder,
    BrokerRejected,
    BrokerStatus,
    BrokerUnavailable,
    OrderRequest,
    OrderType,
    ProtectionMode,
)
from engine.clock import Clock
from engine.domain.events import (
    AnomalyDetected,
    BreakerTrip,
    CircuitBreakerTripped,
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
ZERO = Decimal(0)
MAX_REJECTED_STOPS = 3  # stop retrying protection for a position after this many broker rejections


def order_id_for(intent_id: str) -> str:
    return f"ord-{intent_id}"


def _plain(d: Decimal) -> str:
    return format(d.normalize(), "f")


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
        reject: tuple[str, str] | None = None
        if role == "entry" and (intent.stop_loss_price is not None or intent.take_profit_price is not None):
            prot = self.broker.protection
            if prot is ProtectionMode.NONE:
                reject = ("NO_PROTECTION", "broker cannot attach a protective order")
            elif prot is ProtectionMode.SEPARATE_ORDERS and (
                intent.take_profit_price is not None or intent.stop_loss_price is None
            ):
                reject = ("TP_NOT_SUPPORTED", "this broker can hold only a stop-loss, not a take-profit")
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
        if reject is not None:
            self._emit(
                OrderRejected(
                    account_id=self.account_id, order_id=order_id, code=reject[0], detail=reject[1]
                ),
                key=f"order-rejected:{order_id}:{reject[0]}",
            )
            return order_id
        if role == "exit":
            self._release_protection(intent.strategy_id, intent.symbol)
        self._drive(order_id)
        self._enforce_protection()
        return order_id

    def sync(self) -> None:
        """Reconcile every unfinished order with the broker. Also the crash-recovery entry point."""
        self.projector.refresh()
        for order in list(self.projector.state.orders.values()):
            if order.parent_order_id is not None and not order.managed:
                continue  # native protective legs are reconciled through their parent
            if not order.is_terminal or self._has_open_children(order.order_id):
                self._drive(order.order_id)
        self._enforce_protection()

    def cancel(self, order_id: str) -> bool:
        order = self.projector.state.orders.get(order_id)
        if order is None or order.is_terminal:
            return False
        try:
            self.broker.cancel(order_id)
        except BrokerUnavailable:
            return False
        self._drive(self._root_id(order_id))
        return True

    def cancel_open_entries(self) -> list[str]:
        """Used by the kill switch: stop new risk from building. Exits and protective legs stay live."""
        self.projector.refresh()
        out = []
        for o in list(self.projector.state.orders.values()):
            if o.role == "entry" and not o.is_terminal and self.cancel(o.order_id):
                out.append(o.order_id)
        return out

    # ---- exits: free the shares reserved by protective legs ---------------------------------
    def _release_protection(self, strategy_id: str, symbol: str) -> None:
        """A deliberate exit must not leave stop legs working against a closed position (and brokers
        will not let two sells reserve the same shares). The protection invariant restores a stop if the
        exit later fails."""
        roots: set[str] = set()
        for o in list(self.projector.state.orders.values()):
            if (
                o.role in ("stop", "take_profit")
                and not o.is_terminal
                and o.strategy_id == strategy_id
                and o.symbol == symbol
            ):
                try:
                    self.broker.cancel(o.order_id)
                except BrokerUnavailable:
                    continue
                roots.add(self._root_id(o.order_id))
        for root in roots:
            self._drive(root)

    # ---- protection invariant ---------------------------------------------------------------
    def _enforce_protection(self) -> None:
        self.projector.refresh()
        self._handle_failed_protection()
        state = self.projector.state
        pairs = {(s, sym) for (s, sym), p in state.positions.items() if p.quantity > 0 and s != "unmanaged"}
        for strategy_id, symbol in sorted(pairs):
            self._ensure_stop(strategy_id, symbol)

    def _ensure_stop(self, strategy_id: str, symbol: str) -> None:
        state = self.projector.state
        orders = list(state.orders.values())
        same = [o for o in orders if o.strategy_id == strategy_id and o.symbol == symbol]
        entries = [o for o in same if o.role == "entry" and o.filled_quantity > 0 and o.stop_loss_price]
        if not entries:
            return
        native = self.broker.protection is ProtectionMode.NATIVE_BRACKET
        covered = ZERO
        for o in same:
            if o.role == "stop" and not o.is_terminal or o.role == "exit" and not o.is_terminal:
                covered += o.quantity - o.filled_quantity
            elif native and o.role == "entry" and not o.is_terminal:
                covered += o.filled_quantity  # bracket legs activate when the entry completes
        need = state.position_qty(strategy_id, symbol) - covered
        if need <= 0:
            return
        if (
            sum(1 for o in same if o.role == "stop" and o.status is OrderStatus.REJECTED)
            >= MAX_REJECTED_STOPS
        ):
            return
        base = max(entries, key=lambda o: o.order_id)
        n = sum(1 for o in orders if o.parent_order_id == base.order_id and o.managed and o.role == "stop")
        leg_id = f"{base.order_id}:stop:{n}"
        self._emit(
            OrderCreated(
                account_id=self.account_id,
                order_id=leg_id,
                intent_id=base.intent_id,
                strategy_id=strategy_id,
                symbol=symbol,
                asset_class=base.asset_class,
                side=Side.SELL,
                quantity=need,
                role="stop",
                parent_order_id=base.order_id,
                order_type="stop",
                stop_price=base.stop_loss_price,
                managed=True,
            ),
            key=f"order-created:{leg_id}",
        )
        self._crash_point("after_leg_created")
        self._drive(leg_id)

    def _handle_failed_protection(self) -> None:
        """A rejected stop means an unprotected position: flatten it and stop the strategy."""
        for leg in list(self.projector.state.orders.values()):
            if leg.managed and leg.role == "stop" and leg.status is OrderStatus.REJECTED:
                flat_id = f"{leg.order_id}:flatten"
                if flat_id in self.projector.state.orders:
                    continue
                self._emit(
                    AnomalyDetected(
                        account_id=self.account_id,
                        order_id=leg.order_id,
                        code="PROTECTION_FAILED",
                        detail=f"stop rejected ({leg.reject_code}); flattening {leg.symbol}",
                    ),
                    key=f"anomaly:{flat_id}",
                )
                self._emit(
                    CircuitBreakerTripped(
                        account_id=self.account_id,
                        trip=BreakerTrip(
                            scope="strategy",
                            strategy_id=leg.strategy_id,
                            code="PROTECTION_FAILED",
                            detail=leg.order_id,
                        ),
                    ),
                    key=f"trip:{flat_id}",
                )
                self._emit(
                    OrderCreated(
                        account_id=self.account_id,
                        order_id=flat_id,
                        intent_id=leg.intent_id,
                        strategy_id=leg.strategy_id,
                        symbol=leg.symbol,
                        asset_class=leg.asset_class,
                        side=Side.SELL,
                        quantity=leg.quantity,
                        role="exit",
                        parent_order_id=leg.parent_order_id,
                        order_type="market",
                        managed=True,
                    ),
                    key=f"order-created:{flat_id}",
                )
                self._drive(flat_id)
        for o in list(self.projector.state.orders.values()):
            if o.managed and o.role == "exit" and o.status is OrderStatus.REJECTED:
                self._emit(
                    AnomalyDetected(
                        account_id=self.account_id,
                        order_id=o.order_id,
                        code="FLATTEN_FAILED",
                        detail=f"{o.symbol} may be unprotected; manual action required ({o.reject_code})",
                    ),
                    key=f"anomaly-flatten-failed:{o.order_id}",
                )

    # ---- internals --------------------------------------------------------------------------
    def _crash_point(self, name: str) -> None:
        if self._hook is not None:
            self._hook(name)

    def _emit(self, model: Frozen, *, key: str | None = None) -> bool:
        """Append one event; returns False if its idempotency key already existed."""
        oid = getattr(model, "order_id", None)
        stream = f"order:{oid}" if oid else f"account:{self.account_id}"
        try:
            self.store.append([make_event(model, stream_id=stream, ts=self.clock.now(), idempotency_key=key)])
        except DuplicateEvent:
            self.projector.refresh()
            return False
        self.projector.refresh()
        return True

    def _order(self, order_id: str) -> OrderState:
        return self.projector.state.orders[order_id]

    def _root_id(self, order_id: str) -> str:
        """Native legs are reconciled via their parent; managed legs are their own root."""
        o = self._order(order_id)
        return o.parent_order_id if (o.parent_order_id and not o.managed) else order_id

    def _has_open_children(self, order_id: str) -> bool:
        return any(
            o.parent_order_id == order_id and not o.managed and not o.is_terminal
            for o in self.projector.state.orders.values()
        )

    def _request(self, o: OrderState) -> OrderRequest:
        bracket = o.role == "entry" and self.broker.protection is ProtectionMode.NATIVE_BRACKET
        return OrderRequest(
            client_order_id=o.order_id,
            symbol=o.symbol,
            asset_class=o.asset_class,
            side=o.side,
            quantity=o.quantity,
            order_type=OrderType(o.order_type.upper()) if o.order_type else None,
            limit_price=o.limit_price,
            stop_price=o.stop_price,
            stop_loss_price=o.stop_loss_price if bracket else None,
            take_profit_price=o.take_profit_price if bracket else None,
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

    def _new_fills(self, o: OrderState, bo: BrokerOrder) -> list[BrokerFill]:
        if bo.fills:
            return list(bo.fills)
        if bo.filled_quantity and bo.avg_fill_price:
            delta = bo.filled_quantity - o.filled_quantity
            if delta > 0:
                price = (bo.filled_quantity * bo.avg_fill_price - o.filled_notional) / delta
                if price <= 0:
                    price = bo.avg_fill_price
                return [
                    BrokerFill(
                        fill_id=f"{o.order_id}-cum-{_plain(bo.filled_quantity)}",
                        quantity=delta,
                        price=price,
                        fee=ZERO,
                        ts=self.clock.now(),
                    )
                ]
        return []

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

        for f in self._new_fills(o, bo):
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
        self._crash_point("after_fills")

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

        # Native protective legs enter the ledger once the parent's size is final.
        if self._order(order_id).is_terminal and (bo.fills or bo.filled_quantity):
            for child in bo.children:
                self._ensure_child(order_id, child)
                self._apply_broker_order(child.client_order_id, child)
        if o.role == "entry":
            self._enforce_protection()

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
