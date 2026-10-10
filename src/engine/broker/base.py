"""Broker adapter contract. Real adapters (Phase 2) must pass the same contract tests as MockBroker.

Semantics every adapter must provide:
* `submit` is idempotent on `client_order_id`: resubmitting returns the existing order, never a second one.
* `BrokerRejected` means the order definitely was NOT placed.
* `BrokerUnavailable` is ambiguous: the order may or may not have been placed. Callers must reconcile via
  `get_order(client_order_id)` before retrying.
* `get_order` returns None only when the broker is reachable and has no such order.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from engine.domain.types import AssetClass, Frozen, Mode, NonNegDec, PosDec, Side, UtcDatetime


class BrokerError(Exception):
    pass


class BrokerRejected(BrokerError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class BrokerUnavailable(BrokerError):
    pass


class BrokerStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderRequest(Frozen):
    client_order_id: str
    symbol: str
    asset_class: AssetClass
    side: Side
    quantity: PosDec
    limit_price: PosDec | None = None
    stop_loss_price: PosDec | None = None
    take_profit_price: PosDec | None = None


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
    children: tuple[BrokerOrder, ...] = ()  # protective stop / take-profit legs
    reject_reason: str | None = None


class Broker(Protocol):
    mode: Mode
    supports_protective_orders: bool

    def submit(self, req: OrderRequest) -> BrokerOrder: ...

    def get_order(self, client_order_id: str) -> BrokerOrder | None: ...

    def cancel(self, client_order_id: str) -> None: ...

    def healthy(self) -> bool: ...
