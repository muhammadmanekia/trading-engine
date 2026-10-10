"""Reconciler: compares what the broker says with what the ledger believes, and fails closed on drift.

Runs inside the execution service (the only process allowed to hold broker keys). It never edits the
ledger:
disagreement is recorded as events and, if serious and persistent, the account's kill switch is thrown. A
human then fixes the cause and, if the broker is right, records an explicit `Control.adjust(...)`.

Severity:
* critical: position quantities differ, or cash differs by more than `cash_critical_pct` of equity
* warning : small cash difference (fees, spread), orders the ledger does not know (e.g. a manual order in the
            same account), ledger orders that are working but absent at the broker
Only CRITICAL drift that persists for `confirm_runs` consecutive runs trips the kill switch, so a fill that
lands between two API calls does not halt trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from typing import Literal

from engine.broker.base import Broker, BrokerOrder
from engine.clock import Clock
from engine.core import TradingCore
from engine.domain.events import DriftDetected, ReconciliationCompleted
from engine.domain.types import Frozen
from engine.eventlog.base import NewEvent, make_event
from engine.orders.state import OrderStatus
from engine.projections.account import AccountState

log = logging.getLogger("engine.reconciler")
ZERO = Decimal(0)


@dataclass(frozen=True)
class Drift:
    kind: str
    severity: Literal["warning", "critical"]
    symbol: str | None
    detail: str
    ledger_value: str | None = None
    broker_value: str | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.symbol or ''}"


@dataclass(frozen=True)
class ReconcilerPolicy:
    qty_tolerance: Decimal = Decimal("0.00000001")
    cash_tolerance_abs: Decimal = Decimal(5)
    cash_tolerance_pct: Decimal = Decimal("0.005")
    cash_critical_pct: Decimal = Decimal("0.05")
    order_grace: timedelta = timedelta(seconds=120)
    confirm_runs: int = 2


@dataclass
class ReconciliationResult:
    drifts: list[Drift] = field(default_factory=list)
    halted: bool = False

    @property
    def ok(self) -> bool:
        return not self.drifts


class Reconciler:
    def __init__(
        self, core: TradingCore, broker: Broker, clock: Clock, policy: ReconcilerPolicy | None = None
    ) -> None:
        self.core = core
        self.broker = broker
        self.clock = clock
        self.policy = policy or ReconcilerPolicy()
        self._streak: dict[str, int] = {}
        self._had_drift = False

    def run(self) -> ReconciliationResult:
        """Raises BrokerError if the broker cannot be read (the caller treats that as unknown, not fine)."""
        state = self.core.projector.refresh()
        positions = self.broker.get_positions()
        cash = self.broker.get_cash()
        open_orders = self.broker.list_open_orders()

        drifts = [
            *self._positions(state, positions),
            *self._orders(state, open_orders),
            *self._cash(state, cash),
        ]
        keys = {d.key for d in drifts}
        self._streak = {k: self._streak.get(k, 0) + 1 for k in keys}  # clean keys drop out -> streak resets

        events = []
        for d in drifts:
            if self._streak[d.key] == 1:  # record a drift when it appears, not on every run
                drift_event = DriftDetected(
                    account_id=state.account_id,
                    kind=d.kind,
                    severity=d.severity,
                    symbol=d.symbol,
                    detail=d.detail,
                    ledger_value=d.ledger_value,
                    broker_value=d.broker_value,
                )
                events.append(self._event(drift_event))
        summary = "recovered" if (not drifts and self._had_drift) else ", ".join(sorted(keys))
        events.append(
            self._event(
                ReconciliationCompleted(
                    account_id=state.account_id, ok=not drifts, drift_count=len(drifts), summary=summary
                )
            )
        )
        self.core.store.append(events)
        self._had_drift = bool(drifts)

        result = ReconciliationResult(drifts=drifts)
        persistent = [
            d for d in drifts if d.severity == "critical" and self._streak[d.key] >= self.policy.confirm_runs
        ]
        if persistent and not state.kill_switch:
            reason = "reconciliation drift: " + ", ".join(sorted({d.key for d in persistent}))
            log.critical("%s — activating kill switch for %s", reason, state.account_id)
            self.core.activate_kill_switch(reason)
            result.halted = True
        return result

    # ---- comparisons ------------------------------------------------------------------------
    def _positions(self, state: AccountState, broker_positions: dict[str, Decimal]) -> list[Drift]:
        ledger: dict[str, Decimal] = {}
        for (_strategy, symbol), p in state.positions.items():
            if p.quantity != 0:
                ledger[symbol] = ledger.get(symbol, ZERO) + p.quantity
        out = []
        for symbol in sorted(set(ledger) | set(broker_positions)):
            mine, theirs = ledger.get(symbol, ZERO), broker_positions.get(symbol, ZERO)
            if abs(mine - theirs) > self.policy.qty_tolerance:
                out.append(
                    Drift(
                        "POSITION_MISMATCH",
                        "critical",
                        symbol,
                        f"ledger holds {mine}, broker holds {theirs}",
                        str(mine),
                        str(theirs),
                    )  # fmt: skip
                )
        return out

    def _orders(self, state: AccountState, open_orders: list[BrokerOrder]) -> list[Drift]:
        at_broker: set[str] = set()

        def collect(o: BrokerOrder) -> None:
            at_broker.add(o.client_order_id)
            for c in o.children:
                if c.status.value in ("ACCEPTED", "PARTIALLY_FILLED"):
                    collect(c)

        for o in open_orders:
            collect(o)
        out = []
        known = set(state.orders)
        for cid in sorted(at_broker - known):
            detail = f"broker has working order {cid} unknown to the ledger"
            out.append(Drift("ORPHAN_BROKER_ORDER", "warning", None, detail, None, cid))
        now = self.clock.now()
        for lo in state.orders.values():
            if lo.status not in (OrderStatus.OPEN, OrderStatus.PARTIAL) or lo.order_id in at_broker:
                continue
            if lo.created_at is not None and now - lo.created_at < self.policy.order_grace:
                continue  # may be mid-flight
            detail = f"ledger order {lo.order_id} is working but not at the broker"
            out.append(Drift("ORDER_MISSING_AT_BROKER", "warning", lo.symbol, detail, lo.order_id, None))
        return out

    def _cash(self, state: AccountState, broker_cash: Decimal) -> list[Drift]:
        diff = abs(state.cash - broker_cash)
        equity = max(abs(state.equity({})), Decimal(1))
        tolerance = max(self.policy.cash_tolerance_abs, self.policy.cash_tolerance_pct * equity)
        if diff <= tolerance:
            return []
        severity: Literal["warning", "critical"] = (
            "critical" if diff > self.policy.cash_critical_pct * equity else "warning"
        )
        return [
            Drift(
                "CASH_MISMATCH", severity, None, f"cash differs by {diff}", str(state.cash), str(broker_cash)
            )
        ]

    def _event(self, model: Frozen) -> NewEvent:
        return make_event(model, stream_id=f"account:{self.core.account_id}", ts=self.clock.now())
