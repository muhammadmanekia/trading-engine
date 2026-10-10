"""Order lifecycle: a pure transition table plus the per-order state folded from events.

Late fills on CANCELLED/EXPIRED orders are legitimate broker races and are accepted; fills on a
FILLED or REJECTED order, and any overfill, are illegal and surface as anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from engine.domain.events import OrderRole
from engine.domain.types import AssetClass, Side


class OrderStatus(StrEnum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


TERMINAL = frozenset({OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED})

# Order-lifecycle event type names this module understands.
_ALLOWED: dict[OrderStatus, frozenset[str]] = {
    OrderStatus.CREATED: frozenset(
        {"OrderSubmitAttempted", "OrderAcked", "OrderFilled", "OrderRejected", "OrderCancelled"}
    ),
    OrderStatus.SUBMITTING: frozenset(
        {"OrderSubmitAttempted", "OrderAcked", "OrderFilled", "OrderRejected", "OrderCancelled"}
    ),
    OrderStatus.OPEN: frozenset({"OrderFilled", "OrderCancelled", "OrderExpired", "OrderRejected"}),
    OrderStatus.PARTIAL: frozenset({"OrderFilled", "OrderCancelled", "OrderExpired"}),
    OrderStatus.CANCELLED: frozenset({"OrderFilled"}),
    OrderStatus.EXPIRED: frozenset({"OrderFilled"}),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


class IllegalTransition(Exception):
    pass


def transition(status: OrderStatus, event_type: str, *, fully_filled: bool = False) -> OrderStatus:
    """Return the next status or raise IllegalTransition. `fully_filled` applies to OrderFilled."""
    if event_type not in _ALLOWED[status]:
        raise IllegalTransition(f"{event_type} not allowed in {status}")
    match event_type:
        case "OrderSubmitAttempted":
            return OrderStatus.SUBMITTING
        case "OrderAcked":
            return OrderStatus.OPEN
        case "OrderFilled":
            if status in (OrderStatus.CANCELLED, OrderStatus.EXPIRED):
                return status
            return OrderStatus.FILLED if fully_filled else OrderStatus.PARTIAL
        case "OrderRejected":
            return OrderStatus.REJECTED
        case "OrderCancelled":
            return OrderStatus.CANCELLED
        case "OrderExpired":
            return OrderStatus.EXPIRED
    raise IllegalTransition(f"unknown order event {event_type}")  # pragma: no cover


@dataclass
class OrderState:
    order_id: str
    intent_id: str
    strategy_id: str
    symbol: str
    side: Side
    asset_class: AssetClass
    role: OrderRole
    quantity: Decimal
    parent_order_id: str | None
    limit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    order_type: str | None = None
    stop_price: Decimal | None = None
    managed: bool = False  # a leg the OrderManager placed itself (not a native broker leg)
    created_at: datetime | None = None
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: Decimal = Decimal(0)
    filled_notional: Decimal = Decimal(0)
    fill_ids: set[str] = field(default_factory=set)
    attempts: int = 0
    broker_order_id: str | None = None
    reject_code: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def remaining(self) -> Decimal:
        return self.quantity - self.filled_quantity
