"""Control plane: funding, kill switch, paper/live mode and breaker resets. Every change is an event."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from engine.clock import Clock
from engine.domain.events import (
    AccountFunded,
    CircuitBreakerReset,
    KillSwitchSet,
    ModeSet,
)
from engine.domain.types import Frozen, Mode
from engine.eventlog.base import EventStore, make_event


class Control:
    def __init__(self, store: EventStore, clock: Clock, account_id: str) -> None:
        self.store = store
        self.clock = clock
        self.account_id = account_id

    def _emit(self, model: Frozen) -> None:
        self.store.append([make_event(model, stream_id=f"account:{self.account_id}", ts=self.clock.now())])

    def fund(self, amount: Decimal, note: str = "") -> None:
        self._emit(AccountFunded(account_id=self.account_id, amount=amount, note=note))

    def set_kill_switch(self, active: bool, reason: str = "") -> None:
        self._emit(KillSwitchSet(account_id=self.account_id, active=active, reason=reason))

    def set_mode(self, mode: Mode, *, confirmed: bool = False) -> None:
        """Live trading is off by default and needs an explicit confirmation."""
        if mode is Mode.LIVE and not confirmed:
            raise PermissionError("switching to LIVE requires confirmed=True")
        self._emit(ModeSet(account_id=self.account_id, mode=mode))

    def reset_breaker(self, scope: Literal["global", "strategy"], strategy_id: str | None = None) -> None:
        if scope == "strategy" and not strategy_id:
            raise ValueError("strategy_id is required for a strategy breaker")
        self._emit(CircuitBreakerReset(account_id=self.account_id, scope=scope, strategy_id=strategy_id))
