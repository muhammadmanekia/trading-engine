"""Typed event payloads. The event type name is the class name; every event is account-scoped."""

from __future__ import annotations

from decimal import Decimal
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
    # Protective legs placed by the OrderManager itself (SEPARATE_ORDERS brokers) carry these.
    order_type: Literal["market", "limit", "stop"] | None = None
    stop_price: PosDec | None = None
    managed: bool = False


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


# ---- reconciliation / baseline ----------------------------------------------------------------
class BaselinePosition(Frozen):
    symbol: str
    quantity: PosDec
    avg_cost: PosDec


class BaselineImported(Frozen):
    """Opening balances of a pre-existing broker account. Positions are 'unmanaged': counted in exposure
    and equity, but never sold by a strategy."""

    account_id: str
    cash: NonNegDec
    positions: tuple[BaselinePosition, ...] = ()
    note: str = ""


class PositionDelta(Frozen):
    strategy_id: str
    symbol: str
    quantity: Dec
    avg_cost: PosDec | None = None


class LedgerAdjusted(Frozen):
    """Operator-approved correction after a reconciliation drift (deposit, manual trade, ...)."""

    account_id: str
    cash_delta: Dec = Decimal(0)
    position_deltas: tuple[PositionDelta, ...] = ()
    reason: str


class ReconciliationCompleted(Frozen):
    account_id: str
    ok: bool
    drift_count: int
    summary: str = ""


class DriftDetected(Frozen):
    account_id: str
    kind: str
    severity: Literal["warning", "critical"]
    symbol: str | None = None
    detail: str = ""
    ledger_value: str | None = None
    broker_value: str | None = None


class LlmDecisionMade(Frozen):
    """Audit record of one model decision. The full prompt and reply live in the llm_calls table; this event
    carries their hashes, so the hash-chained log proves which prompt and reply led to what."""

    account_id: str
    strategy_id: str
    call_id: str
    provider: str
    model: str
    mode: Literal["primary", "shadow"]
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    input_hash: str
    response_hash: str | None = None
    symbol: str | None = None
    action: str | None = None
    confidence: Dec | None = None
    outcome: Literal["intent", "proposal", "shadow", "hold", "rejected", "invalid", "error"]
    reason: str = ""
    cost_usd: NonNegDec = Decimal(0)


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
        BaselineImported,
        LedgerAdjusted,
        ReconciliationCompleted,
        DriftDetected,
        LlmDecisionMade,
    )
}
