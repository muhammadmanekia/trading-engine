"""TradingCore: Intent -> risk engine -> order manager, with every step recorded in the event log.

The only path from a signal to a broker order goes through `submit_intent`. Restart safety comes from
`recover()`, which rebuilds state from the log and finishes whatever was in flight.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

from engine.broker.base import Broker
from engine.clock import Clock
from engine.control import Control
from engine.domain.events import (
    CircuitBreakerTripped,
    IntentAbandoned,
    IntentReceived,
    RiskDecisionMade,
)
from engine.domain.intent import Intent
from engine.domain.market import MarketSnapshot, SystemState
from engine.eventlog.base import DuplicateEvent, EventStore, make_event
from engine.orders.manager import Hook, OrderManager
from engine.projections.account import AccountProjector, AccountState
from engine.risk.config import RiskConfig
from engine.risk.decision import Decision
from engine.risk.engine import evaluate


@dataclass(frozen=True)
class SubmitResult:
    decision: Decision | None  # None when the intent was already decided (duplicate delivery)
    order_id: str | None
    duplicate: bool = False


class TradingCore:
    def __init__(
        self,
        store: EventStore,
        broker: Broker,
        clock: Clock,
        account_id: str,
        *,
        market: Callable[[], MarketSnapshot],
        system: Callable[[], SystemState],
        config: Callable[[Intent], RiskConfig],
        hook: Hook | None = None,
        max_submit_attempts: int = 5,
    ) -> None:
        self.store = store
        self.clock = clock
        self.account_id = account_id
        self._market, self._system, self._config = market, system, config
        self._hook = hook
        self.projector = AccountProjector(store, account_id)
        self.control = Control(store, clock, account_id)
        self.manager = OrderManager(
            store, broker, clock, self.projector, max_submit_attempts=max_submit_attempts, hook=hook
        )

    @property
    def state(self) -> AccountState:
        return self.projector.state

    # ---- the one path to a broker -----------------------------------------------------------
    def submit_intent(self, intent: Intent) -> SubmitResult:
        if intent.account_id != self.account_id:
            return SubmitResult(Decision.reject(intent.intent_id, "ACCOUNT_MISMATCH"), None)
        with contextlib.suppress(DuplicateEvent):  # redelivery, or a crash before the decision
            self.store.append(
                [
                    make_event(
                        IntentReceived(account_id=self.account_id, intent=intent),
                        stream_id=f"intent:{intent.intent_id}",
                        ts=self.clock.now(),
                        idempotency_key=f"intent:{intent.intent_id}",
                    )
                ]
            )
        self._crash_point("after_received")
        state = self.projector.refresh()
        if state.intents[intent.intent_id].decision is not None:
            rec = state.intents[intent.intent_id]
            return SubmitResult(None, rec.order_id, duplicate=True)

        try:
            decision = evaluate(intent, state, self._market(), self._system(), self._config(intent))
        except Exception as exc:  # providers failing is itself a reason not to trade
            decision = Decision.reject(intent.intent_id, "PROVIDER_ERROR", f"{type(exc).__name__}: {exc}")

        events = [
            make_event(
                RiskDecisionMade(
                    account_id=self.account_id,
                    intent_id=intent.intent_id,
                    decision=decision.kind,
                    approved_quantity=decision.quantity,
                    code=decision.code,
                    detail=decision.detail,
                    checks_passed=decision.checks_passed,
                    trips=decision.trips,
                    reference_price=decision.reference_price,
                ),
                stream_id=f"intent:{intent.intent_id}",
                ts=self.clock.now(),
                idempotency_key=f"decision:{intent.intent_id}",
            )
        ]
        events += [
            make_event(
                CircuitBreakerTripped(account_id=self.account_id, trip=t),
                stream_id=f"account:{self.account_id}",
                ts=self.clock.now(),
            )
            for t in decision.trips
        ]
        try:
            self.store.append(events)
        except DuplicateEvent:
            self.projector.refresh()
            return SubmitResult(None, None, duplicate=True)
        self.projector.refresh()
        self._crash_point("after_decision")

        if not decision.allowed or decision.quantity is None:
            return SubmitResult(decision, None)
        return SubmitResult(decision, self.manager.place(intent, decision.quantity))

    # ---- restart / recovery -----------------------------------------------------------------
    def recover(self) -> None:
        """Rebuild from the log, finish in-flight intents, and reconcile open orders with the broker."""
        self.process_pending()
        self.manager.sync()

    def process_pending(self) -> None:
        """Decide intents that were received but not yet decided, and place approved-but-unplaced ones."""
        state = self.projector.refresh()
        now = self.clock.now()
        for rec in list(state.intents.values()):
            if rec.abandoned:
                continue
            if rec.decision is None:
                self.submit_intent(rec.intent)  # re-evaluated against current state
            elif rec.decision in ("approve", "resize") and rec.order_id is None:
                if now >= rec.intent.expires_at or rec.approved_quantity is None:
                    self.store.append(
                        [
                            make_event(
                                IntentAbandoned(
                                    account_id=self.account_id,
                                    intent_id=rec.intent.intent_id,
                                    reason="approved but never ordered; expired before recovery",
                                ),
                                stream_id=f"intent:{rec.intent.intent_id}",
                                ts=now,
                            )
                        ]
                    )
                else:
                    self.manager.place(rec.intent, rec.approved_quantity)

    # ---- kill switch ------------------------------------------------------------------------
    def activate_kill_switch(self, reason: str = "", *, cancel_open_entries: bool = True) -> list[str]:
        """Block all new entries immediately. Exits and protective stops remain live."""
        self.control.set_kill_switch(True, reason)
        self.projector.refresh()
        return self.manager.cancel_open_entries() if cancel_open_entries else []

    def _crash_point(self, name: str) -> None:
        if self._hook is not None:
            self._hook(name)
