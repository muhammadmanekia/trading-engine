"""Deterministic in-memory broker with fault injection, used in CI, as the contract-suite reference, and as
the matching engine inside PaperBroker and the fake Alpaca/Robinhood servers.

Behaviour mirrors typical brokers:
* NATIVE_BRACKET: protective stop / take-profit legs activate once the entry has filled (or been cancelled
  with a partial fill, in which case they are sized to the filled quantity). The two legs are OCO.
* Standalone STOP orders (SEPARATE_ORDERS) trigger when the price crosses the stop.
* A standalone SELL cannot reserve shares already reserved by another open SELL (like real brokers).
* Buys that exceed cash are rejected (when `starting_cash` is given).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from engine.broker.base import (
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
from engine.domain.types import Mode, Side

SubmitFault = Literal["reject", "unavailable_before", "unavailable_after"]
ZERO = Decimal(0)


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
        return sum((f.quantity for f in self.fills), ZERO)

    @property
    def open(self) -> bool:
        return self.status in (BrokerStatus.ACCEPTED, BrokerStatus.PARTIALLY_FILLED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "req": self.req.model_dump(mode="json"),
            "broker_order_id": self.broker_order_id,
            "quantity": str(self.quantity),
            "status": self.status.value,
            "fills": [f.model_dump(mode="json") for f in self.fills],
            "trigger": self.trigger,
            "parent": self.parent,
            "active": self.active,
            "children": list(self.children),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> _Order:
        return _Order(
            req=OrderRequest.model_validate(d["req"]),
            broker_order_id=d["broker_order_id"],
            quantity=Decimal(d["quantity"]),
            status=BrokerStatus(d["status"]),
            fills=[BrokerFill.model_validate(f) for f in d["fills"]],
            trigger=d["trigger"],
            parent=d["parent"],
            active=d["active"],
            children=list(d["children"]),
        )


class MockBroker:
    def __init__(
        self,
        clock: Clock,
        *,
        mode: Mode = Mode.PAPER,
        fee_rate: Decimal = Decimal("0.001"),
        fill_fraction: Decimal = Decimal(1),
        protection: ProtectionMode = ProtectionMode.NATIVE_BRACKET,
        supports_protective_orders: bool | None = None,
        starting_cash: Decimal | None = None,
        cumulative_only: bool = False,
    ) -> None:
        if supports_protective_orders is False:
            protection = ProtectionMode.NONE
        self.clock = clock
        self.mode = mode
        self.protection = protection
        self.fee_rate = fee_rate
        self.fill_fraction = fill_fraction  # fraction of the order filled per matching step
        self.cumulative_only = cumulative_only  # report fills as cumulative qty/avg only
        self.enforce_cash = starting_cash is not None
        self.cash: Decimal = starting_cash if starting_cash is not None else ZERO
        self.prices: dict[str, Decimal] = {}
        self.positions: dict[str, Decimal] = {}
        self._orders: dict[str, _Order] = {}
        self._seq = 0
        # fault injection
        self.outage = False
        self.duplicate_fills = False
        self._submit_faults: deque[SubmitFault] = deque()
        self.submit_calls = 0
        # rules return a reject code to refuse a matching request (e.g. 'refuse every STOP order')
        self.reject_rules: list[Callable[[OrderRequest], str | None]] = []

    @property
    def supports_protective_orders(self) -> bool:
        return self.protection is not ProtectionMode.NONE

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

    # ---- persistence (PaperBroker) ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "cash": str(self.cash),
            "prices": {k: str(v) for k, v in self.prices.items()},
            "positions": {k: str(v) for k, v in self.positions.items()},
            "seq": self._seq,
            "orders": {k: o.to_dict() for k, o in self._orders.items()},
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.cash = Decimal(data["cash"])
        self.prices = {k: Decimal(v) for k, v in data["prices"].items()}
        self.positions = {k: Decimal(v) for k, v in data["positions"].items()}
        self._seq = int(data["seq"])
        self._orders = {k: _Order.from_dict(v) for k, v in data["orders"].items()}

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
            for rule in self.reject_rules:
                if (code := rule(req)) is not None:
                    raise BrokerRejected(code, "rule")
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

    def get_positions(self) -> dict[str, Decimal]:
        if self.outage:
            raise BrokerUnavailable("outage")
        return {s: q for s, q in self.positions.items() if q != 0}

    def get_cash(self) -> Decimal:
        if self.outage:
            raise BrokerUnavailable("outage")
        return self.cash

    def list_open_orders(self) -> list[BrokerOrder]:
        if self.outage:
            raise BrokerUnavailable("outage")
        out = []
        for o in self._orders.values():
            if o.parent is not None:
                continue
            if o.open or any(self._orders[c].open and self._orders[c].active for c in o.children):
                out.append(self._view(o))
        return out

    # ---- internals --------------------------------------------------------------------------
    def _next_id(self) -> str:
        self._seq += 1
        return f"mock-{self._seq}"

    def _reserved_sell_qty(self, symbol: str) -> Decimal:
        """Shares committed to open standalone SELL orders (native OCO legs reserve as one unit)."""
        return sum(
            (
                o.quantity - o.filled
                for o in self._orders.values()
                if o.open and o.parent is None and o.req.side is Side.SELL and o.req.symbol == symbol
            ),
            ZERO,
        )

    def _place(self, req: OrderRequest) -> None:
        if req.symbol not in self.prices:
            raise BrokerRejected("NO_PRICE", req.symbol)
        if req.side is Side.SELL:
            available = self.positions.get(req.symbol, ZERO) - self._reserved_sell_qty(req.symbol)
            if req.quantity > available:
                raise BrokerRejected("INSUFFICIENT_POSITION", req.symbol)
        elif self.enforce_cash:
            px = req.limit_price if req.limit_price is not None else self.prices[req.symbol]
            if req.quantity * px > self.cash:
                raise BrokerRejected("INSUFFICIENT_FUNDS", req.symbol)
        order = _Order(req=req, broker_order_id=self._next_id(), quantity=req.quantity)
        self._orders[req.client_order_id] = order
        if req.side is Side.BUY and self.protection is ProtectionMode.NATIVE_BRACKET:
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

    def _execution_price(self, o: _Order) -> Decimal:
        """The price a marketable order fills at. The backtest broker overrides this (open, gaps, costs)."""
        return self.prices[o.req.symbol]

    def _marketable(self, o: _Order) -> bool:
        price = self.prices[o.req.symbol]
        if o.trigger == "stop":
            assert o.req.stop_loss_price is not None
            return price <= o.req.stop_loss_price
        if o.trigger == "tp":
            assert o.req.take_profit_price is not None
            return price >= o.req.take_profit_price
        if o.req.kind is OrderType.STOP:
            assert o.req.stop_price is not None
            return price <= o.req.stop_price if o.req.side is Side.SELL else price >= o.req.stop_price
        limit = o.req.limit_price
        if limit is None:
            return True
        return price <= limit if o.req.side is Side.BUY else price >= limit

    def _match(self, o: _Order) -> None:
        if not (o.open and o.active) or not self._marketable(o):
            return
        remaining = o.quantity - o.filled
        whole = o.trigger is not None or o.req.kind is OrderType.STOP
        qty = remaining if whole else min(remaining, o.quantity * self.fill_fraction)
        if qty <= 0:
            return
        price = self._execution_price(o)
        if o.req.side is Side.SELL:
            qty = min(qty, self.positions.get(o.req.symbol, ZERO))
            if qty <= 0:
                return
        fee = qty * price * self.fee_rate
        fill = BrokerFill(
            fill_id=f"{o.req.client_order_id}-f{len(o.fills) + 1}",
            quantity=qty,
            price=price,
            fee=fee,
            ts=self.clock.now(),
        )
        o.fills.append(fill)
        if o.req.side is Side.BUY:
            self.positions[o.req.symbol] = self.positions.get(o.req.symbol, ZERO) + qty
            self.cash -= qty * price + fee
        else:
            self.positions[o.req.symbol] = self.positions.get(o.req.symbol, ZERO) - qty
            self.cash += qty * price - fee
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
        filled_quantity = None
        avg = None
        if self.cumulative_only:
            filled_quantity = o.filled
            if o.fills:
                avg = sum((f.quantity * f.price for f in o.fills), ZERO) / o.filled
            fills = []
        return BrokerOrder(
            client_order_id=o.req.client_order_id,
            broker_order_id=o.broker_order_id,
            symbol=o.req.symbol,
            side=o.req.side,
            status=o.status,
            quantity=o.quantity,
            fills=tuple(fills),
            filled_quantity=filled_quantity,
            avg_fill_price=avg,
            children=tuple(self._view(self._orders[c]) for c in o.children),
        )
