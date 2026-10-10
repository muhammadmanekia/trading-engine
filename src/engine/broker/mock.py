"""Deterministic in-memory broker with fault injection, for CI and for the Phase 2 contract tests.

Bracket behaviour mirrors typical brokers: protective stop / take-profit legs become active once the
entry has filled (or been cancelled with a partial fill, in which case they are sized to the filled
quantity). The two legs are OCO.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from engine.broker.base import (
    BrokerFill,
    BrokerOrder,
    BrokerRejected,
    BrokerStatus,
    BrokerUnavailable,
    OrderRequest,
)
from engine.clock import Clock
from engine.domain.types import Mode, Side

SubmitFault = Literal["reject", "unavailable_before", "unavailable_after"]


@dataclass
class _Order:
    req: OrderRequest
    broker_order_id: str
    quantity: Decimal
    status: BrokerStatus = BrokerStatus.ACCEPTED
    fills: list[BrokerFill] = field(default_factory=list)
    trigger: Literal["stop", "tp"] | None = None
    parent: str | None = None
    active: bool = True
    children: list[str] = field(default_factory=list)

    @property
    def filled(self) -> Decimal:
        return sum((f.quantity for f in self.fills), Decimal(0))

    @property
    def open(self) -> bool:
        return self.status in (BrokerStatus.ACCEPTED, BrokerStatus.PARTIALLY_FILLED)


class MockBroker:
    supports_protective_orders: bool

    def __init__(
        self,
        clock: Clock,
        *,
        mode: Mode = Mode.PAPER,
        fee_rate: Decimal = Decimal("0.001"),
        fill_fraction: Decimal = Decimal(1),
        supports_protective_orders: bool = True,
    ) -> None:
        self.clock = clock
        self.mode = mode
        self.fee_rate = fee_rate
        self.fill_fraction = fill_fraction  # fraction of the order filled per matching step
        self.supports_protective_orders = supports_protective_orders
        self.prices: dict[str, Decimal] = {}
        self.positions: dict[str, Decimal] = {}
        self._orders: dict[str, _Order] = {}
        self._seq = 0
        # fault injection
        self.outage = False
        self.duplicate_fills = False
        self._submit_faults: deque[SubmitFault] = deque()
        self.submit_calls = 0

    # ---- test controls ----------------------------------------------------------------------
    def fail_next_submit(self, kind: SubmitFault, times: int = 1) -> None:
        self._submit_faults.extend([kind] * times)

    def set_price(self, symbol: str, price: Decimal) -> None:
        self.prices[symbol] = price
        self.step()

    def step(self) -> None:
        """Advance matching: fill marketable parents and trigger protective legs."""
        for order in list(self._orders.values()):
            if order.open and order.active:
                self._match(order)

    @property
    def placed_orders(self) -> list[str]:
        """Client ids of top-level (non-protective) orders actually accepted by the broker."""
        return [cid for cid, o in self._orders.items() if o.parent is None]

    # ---- Broker interface -------------------------------------------------------------------
    def healthy(self) -> bool:
        return not self.outage

    def submit(self, req: OrderRequest) -> BrokerOrder:
        self.submit_calls += 1
        if self.outage:
            raise BrokerUnavailable("outage")
        fault = self._submit_faults.popleft() if self._submit_faults else None
        if fault == "reject":
            raise BrokerRejected("MOCK_REJECT", "injected")
        if fault == "unavailable_before":
            raise BrokerUnavailable("timeout before the order was placed")
        existing = self._orders.get(req.client_order_id)
        if existing is None:
            self._place(req)
        if fault == "unavailable_after":
            raise BrokerUnavailable("timeout after the order was placed")
        return self._view(self._orders[req.client_order_id])

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        if self.outage:
            raise BrokerUnavailable("outage")
        order = self._orders.get(client_order_id)
        return self._view(order) if order else None

    def cancel(self, client_order_id: str) -> None:
        if self.outage:
            raise BrokerUnavailable("outage")
        order = self._orders.get(client_order_id)
        if order is None or not order.open:
            return
        order.status = BrokerStatus.CANCELLED
        if order.parent is None:
            self._activate_children(order)  # protect whatever did fill
            if order.filled == 0:
                self._cancel_children(order)
        else:
            self._cancel_siblings(order)

    # ---- internals --------------------------------------------------------------------------
    def _next_id(self) -> str:
        self._seq += 1
        return f"mock-{self._seq}"

    def _place(self, req: OrderRequest) -> None:
        if req.symbol not in self.prices:
            raise BrokerRejected("NO_PRICE", req.symbol)
        if req.side is Side.SELL and req.quantity > self.positions.get(req.symbol, Decimal(0)):
            raise BrokerRejected("INSUFFICIENT_POSITION", req.symbol)
        order = _Order(req=req, broker_order_id=self._next_id(), quantity=req.quantity)
        self._orders[req.client_order_id] = order
        if req.side is Side.BUY:
            for suffix, trig, price in (
                ("stop", "stop", req.stop_loss_price),
                ("tp", "tp", req.take_profit_price),
            ):
                if price is None:
                    continue
                cid = f"{req.client_order_id}:{suffix}"
                child_req = req.model_copy(
                    update={
                        "client_order_id": cid,
                        "side": Side.SELL,
                        "limit_price": None,
                        "stop_loss_price": price if trig == "stop" else None,
                        "take_profit_price": price if trig == "tp" else None,
                    }
                )
                self._orders[cid] = _Order(
                    req=child_req,
                    broker_order_id=self._next_id(),
                    quantity=req.quantity,
                    trigger=trig,  # type: ignore[arg-type]
                    parent=req.client_order_id,
                    active=False,
                )
                order.children.append(cid)
        self._match(order)

    def _marketable(self, o: _Order) -> bool:
        price = self.prices[o.req.symbol]
        if o.trigger == "stop":
            assert o.req.stop_loss_price is not None
            return price <= o.req.stop_loss_price
        if o.trigger == "tp":
            assert o.req.take_profit_price is not None
            return price >= o.req.take_profit_price
        limit = o.req.limit_price
        if limit is None:
            return True
        return price <= limit if o.req.side is Side.BUY else price >= limit

    def _match(self, o: _Order) -> None:
        if not (o.open and o.active) or not self._marketable(o):
            return
        remaining = o.quantity - o.filled
        qty = remaining if o.trigger else min(remaining, o.quantity * self.fill_fraction)
        if qty <= 0:
            return
        price = self.prices[o.req.symbol]
        if o.req.side is Side.SELL:
            qty = min(qty, self.positions.get(o.req.symbol, Decimal(0)))
            if qty <= 0:
                return
        fill = BrokerFill(
            fill_id=f"{o.req.client_order_id}-f{len(o.fills) + 1}",
            quantity=qty,
            price=price,
            fee=qty * price * self.fee_rate,
            ts=self.clock.now(),
        )
        o.fills.append(fill)
        delta = qty if o.req.side is Side.BUY else -qty
        self.positions[o.req.symbol] = self.positions.get(o.req.symbol, Decimal(0)) + delta
        if o.filled >= o.quantity:
            o.status = BrokerStatus.FILLED
            if o.parent is None:
                self._activate_children(o)
            else:
                self._cancel_siblings(o)
        else:
            o.status = BrokerStatus.PARTIALLY_FILLED

    def _activate_children(self, parent: _Order) -> None:
        for cid in parent.children:
            child = self._orders[cid]
            if child.open and not child.active and parent.filled > 0:
                child.quantity = parent.filled
                child.active = True
                self._match(child)  # a gap through the trigger fires immediately

    def _cancel_children(self, parent: _Order) -> None:
        for cid in parent.children:
            self._orders[cid].status = BrokerStatus.CANCELLED

    def _cancel_siblings(self, filled_child: _Order) -> None:
        assert filled_child.parent is not None
        for cid in self._orders[filled_child.parent].children:
            sib = self._orders[cid]
            if sib is not filled_child and sib.open:
                sib.status = BrokerStatus.CANCELLED

    def _view(self, o: _Order) -> BrokerOrder:
        fills = list(o.fills)
        if self.duplicate_fills:
            fills = [f for f in fills for _ in (0, 1)]
        return BrokerOrder(
            client_order_id=o.req.client_order_id,
            broker_order_id=o.broker_order_id,
            symbol=o.req.symbol,
            side=o.req.side,
            status=o.status,
            quantity=o.quantity,
            fills=tuple(fills),
            children=tuple(self._view(self._orders[c]) for c in o.children),
        )
