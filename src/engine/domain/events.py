"""Typed event payloads. The event type name is the class name; every event is account-scoped."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from engine.domain.intent import Intent
from engine.domain.types import AssetClass, Dec, Frozen, Mode, NonNegDec, PosDec, Side

OrderRole = Literal["entry", "exit", "stop", "take_profit"]


class BreakerTrip(Frozen):
    scope: Literal["global", "strategy"]
    strategy_id: str | None = None
    code: str
    detail: str = ""


# ---- account / control -------------------------------------------------------------------------
class AccountFunded(Frozen):
    account_id: str
    amount: PosDec
    note: str = ""


class KillSwitchSet(Frozen):
    account_id: str
    active: bool
    reason: str = ""


class ModeSet(Frozen):
    account_id: str
    mode: Mode


class CircuitBreakerTripped(Frozen):
    account_id: str
    trip: BreakerTrip


class CircuitBreakerReset(Frozen):
    account_id: str
    scope: Literal["global", "strategy"]
    strategy_id: str | None = None


# ---- intent / risk -----------------------------------------------------------------------------
class IntentReceived(Frozen):
    account_id: str
    intent: Intent


class RiskDecisionMade(Frozen):
    account_id: str
    intent_id: str
    decision: Literal["approve", "resize", "reject"]
    approved_quantity: PosDec | None = None
    code: str
    detail: str = ""
    checks_passed: tuple[str, ...] = ()
    trips: tuple[BreakerTrip, ...] = ()
    reference_price: Dec | None = None


class IntentAbandoned(Frozen):
    account_id: str
    intent_id: str
    reason: str


# ---- orders ------------------------------------------------------------------------------------
class OrderCreated(Frozen):
    account_id: str
    order_id: str
    intent_id: str
    strategy_id: str
    symbol: str
    asset_class: AssetClass
    side: Side
    quantity: PosDec
    role: OrderRole
    limit_price: PosDec | None = None
    stop_loss_price: PosDec | None = None
    take_profit_price: PosDec | None = None
    parent_order_id: str | None = None


class OrderSubmitAttempted(Frozen):
    account_id: str
    order_id: str
    attempt: int = Field(ge=1)


class OrderAcked(Frozen):
    account_id: str
    order_id: str
    broker_order_id: str


class OrderFilled(Frozen):
    account_id: str
    order_id: str
    strategy_id: str
    symbol: str
    side: Side
    quantity: PosDec
    price: PosDec
    fee: NonNegDec
    fill_id: str


class OrderRejected(Frozen):
    account_id: str
    order_id: str
    code: str
    detail: str = ""


class OrderCancelled(Frozen):
    account_id: str
    order_id: str
    reason: str = ""


class OrderExpired(Frozen):
    account_id: str
    order_id: str
    reason: str = ""


class AnomalyDetected(Frozen):
    account_id: str
    order_id: str | None = None
    code: str
    detail: str = ""


EVENT_MODELS: dict[str, type[Frozen]] = {
    cls.__name__: cls
    for cls in (
        AccountFunded,
        KillSwitchSet,
        ModeSet,
        CircuitBreakerTripped,
        CircuitBreakerReset,
        IntentReceived,
        RiskDecisionMade,
        IntentAbandoned,
        OrderCreated,
        OrderSubmitAttempted,
        OrderAcked,
        OrderFilled,
        OrderRejected,
        OrderCancelled,
        OrderExpired,
        AnomalyDetected,
    )
}
