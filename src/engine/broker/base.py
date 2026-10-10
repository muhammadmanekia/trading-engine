"""Broker adapter contract (v2). Every adapter must pass tests/contract/test_broker_contract.py.

Semantics every adapter must provide:
* `submit` is idempotent on `client_order_id`: resubmitting returns the existing order, never a second one.
* `BrokerRejected` means the order definitely was NOT placed.
* `BrokerUnavailable` is ambiguous: the order may or may not have been placed. Callers must reconcile via
  `get_order(client_order_id)` before retrying.
* `CredentialsError` is fatal (bad/expired keys): nothing should be retried and trading must halt.
* `get_order` returns None only when the broker is reachable and has no such order.
* Fills are reported either individually (`fills`, with stable ids) or cumulatively
  (`filled_quantity` + `avg_fill_price`), for brokers that do not expose executions.
* `protection` says how a stop-loss is attached: natively with the entry (bracket), as a separate order
  placed by the OrderManager, or not at all.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from engine.domain.types import AssetClass, Dec, Frozen, Mode, NonNegDec, PosDec, Side, UtcDatetime


class BrokerError(Exception):
    pass


class BrokerRejected(BrokerError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class BrokerUnavailable(BrokerError):
    pass


class CredentialsError(BrokerError):
    """Authentication failed. Never retry; the executor halts trading for the account."""


class BrokerStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ProtectionMode(StrEnum):
    NATIVE_BRACKET = "NATIVE_BRACKET"  # stop (and take-profit) attached to the entry at the broker
    SEPARATE_ORDERS = "SEPARATE_ORDERS"  # OrderManager places a stop order after the entry fills
    NONE = "NONE"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderRequest(Frozen):
    client_order_id: str
    symbol: str
    asset_class: AssetClass
    side: Side
    quantity: PosDec
    order_type: OrderType | None = None  # None: LIMIT if limit_price else MARKET
    limit_price: PosDec | None = None
    stop_price: PosDec | None = None  # for standalone STOP orders
    stop_loss_price: PosDec | None = None  # bracket params (NATIVE_BRACKET only)
    take_profit_price: PosDec | None = None

    @property
    def kind(self) -> OrderType:
        if self.order_type is not None:
            return self.order_type
        return OrderType.LIMIT if self.limit_price is not None else OrderType.MARKET


class BrokerFill(Frozen):
    fill_id: str
    quantity: PosDec
    price: PosDec
    fee: NonNegDec
    ts: UtcDatetime


class BrokerOrder(Frozen):
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: Side
    status: BrokerStatus
    quantity: PosDec
    fills: tuple[BrokerFill, ...] = ()
    # Cumulative reporting for brokers without per-execution detail (used when `fills` is empty).
    filled_quantity: NonNegDec | None = None
    avg_fill_price: Dec | None = None
    children: tuple[BrokerOrder, ...] = ()  # native protective stop / take-profit legs
    reject_reason: str | None = None


class Broker(Protocol):
    mode: Mode
    protection: ProtectionMode

    def submit(self, req: OrderRequest) -> BrokerOrder: ...

    def get_order(self, client_order_id: str) -> BrokerOrder | None: ...

    def cancel(self, client_order_id: str) -> None: ...

    def healthy(self) -> bool: ...

    def get_positions(self) -> dict[str, Decimal]:
        """Non-zero holdings by canonical symbol (e.g. 'AAPL', 'BTC-USD')."""
        ...

    def get_cash(self) -> Decimal:
        """Spendable USD cash."""
        ...

    def list_open_orders(self) -> list[BrokerOrder]:
        """Every working order (top-level, with native legs nested)."""
        ...
