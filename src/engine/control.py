"""Control plane: funding, kill switch, paper/live mode and breaker resets. Every change is an event."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

from engine.clock import Clock
from engine.domain.events import (
    AccountFunded,
    BaselineImported,
    BaselinePosition,
    CircuitBreakerReset,
    KillSwitchSet,
    LedgerAdjusted,
    ModeSet,
    PositionDelta,
)
from engine.domain.types import Frozen, Mode
from engine.eventlog.base import DuplicateEvent, EventStore, make_event


class Control:
    def __init__(self, store: EventStore, clock: Clock, account_id: str) -> None:
        self.store = store
        self.clock = clock
        self.account_id = account_id

    def _emit(self, model: Frozen, key: str | None = None) -> bool:
        """Append an account-level event; False if its idempotency key already existed."""
        try:
            self.store.append(
                [
                    make_event(
                        model,
                        stream_id=f"account:{self.account_id}",
                        ts=self.clock.now(),
                        idempotency_key=key,
                    )
                ]
            )
        except DuplicateEvent:
            return False
        return True

    def fund(self, amount: Decimal, note: str = "") -> None:
        self._emit(AccountFunded(account_id=self.account_id, amount=amount, note=note))

    def fund_once(self, amount: Decimal, note: str = "") -> bool:
        """Initial paper funding. A restart must not fund twice."""
        return self._emit(
            AccountFunded(account_id=self.account_id, amount=amount, note=note), key=f"fund:{self.account_id}"
        )

    def set_kill_switch(self, active: bool, reason: str = "") -> None:
        self._emit(KillSwitchSet(account_id=self.account_id, active=active, reason=reason))

    def set_mode(self, mode: Mode, *, confirmed: bool = False) -> None:
        """Live trading is off by default and needs an explicit confirmation."""
        if mode is Mode.LIVE and not confirmed:
            raise PermissionError("switching to LIVE requires confirmed=True")
        self._emit(ModeSet(account_id=self.account_id, mode=mode))

    def import_baseline(
        self, cash: Decimal, positions: Sequence[BaselinePosition] = (), note: str = ""
    ) -> bool:
        """One-time import of a pre-existing broker account's balances. Returns False if already imported."""
        return self._emit(
            BaselineImported(account_id=self.account_id, cash=cash, positions=tuple(positions), note=note),
            key=f"baseline:{self.account_id}",
        )

    def adjust(
        self,
        reason: str,
        cash_delta: Decimal = Decimal(0),
        position_deltas: Sequence[PositionDelta] = (),
    ) -> None:
        """Operator-approved ledger correction after a reconciliation drift (deposit, manual trade, ...)."""
        if not reason.strip():
            raise ValueError("an adjustment needs a reason")
        self._emit(
            LedgerAdjusted(
                account_id=self.account_id,
                cash_delta=cash_delta,
                position_deltas=tuple(position_deltas),
                reason=reason,
            )
        )

    def reset_breaker(self, scope: Literal["global", "strategy"], strategy_id: str | None = None) -> None:
        if scope == "strategy" and not strategy_id:
            raise ValueError("strategy_id is required for a strategy breaker")
        self._emit(CircuitBreakerReset(account_id=self.account_id, scope=scope, strategy_id=strategy_id))
