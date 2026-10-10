"""PaperBroker: a simulated exchange driven by live quotes, with state that survives restarts.

Robinhood Crypto has no sandbox, so crypto "paper mode" runs the same matching engine as MockBroker against
real quotes. It mirrors Robinhood's order model (SEPARATE_ORDERS protection) so paper and live exercise the
same OrderManager code path. Its state is persisted after every mutation: a restart must not forget orders
the ledger already knows about.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb

from engine.broker.base import BrokerOrder, OrderRequest, ProtectionMode
from engine.broker.mock import MockBroker
from engine.clock import Clock
from engine.domain.types import Mode


class PaperStateStore(Protocol):
    def load(self, account_id: str) -> dict[str, Any] | None: ...

    def save(self, account_id: str, state: dict[str, Any]) -> None: ...


class InMemoryPaperState:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def load(self, account_id: str) -> dict[str, Any] | None:
        return self._data.get(account_id)

    def save(self, account_id: str, state: dict[str, Any]) -> None:
        self._data[account_id] = state


class PostgresPaperState:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)

    def load(self, account_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT state FROM paper_broker_state WHERE account_id = %s", (account_id,)
        ).fetchone()
        return row[0] if row else None

    def save(self, account_id: str, state: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO paper_broker_state (account_id, state, updated_at) VALUES (%s, %s, now()) "
            "ON CONFLICT (account_id) DO UPDATE SET state = EXCLUDED.state, updated_at = now()",
            (account_id, Jsonb(state)),
        )

    def close(self) -> None:
        self._conn.close()


class PaperBroker(MockBroker):
    def __init__(
        self,
        clock: Clock,
        account_id: str,
        state_store: PaperStateStore,
        *,
        starting_cash: Decimal,
        fee_rate: Decimal = Decimal("0.0025"),
        protection: ProtectionMode = ProtectionMode.SEPARATE_ORDERS,
    ) -> None:
        super().__init__(
            clock, mode=Mode.PAPER, fee_rate=fee_rate, protection=protection, starting_cash=starting_cash
        )
        self.account_id = account_id
        self._state_store = state_store
        existing = state_store.load(account_id)
        if existing is not None:
            self.restore(existing)
        else:
            self._persist()

    def _persist(self) -> None:
        self._state_store.save(self.account_id, self.snapshot())

    def tick(self, prices: Mapping[str, Decimal]) -> None:
        """Feed fresh quotes; resting limit/stop orders may trigger."""
        for symbol, price in prices.items():
            self.prices[symbol] = price
        self.step()
        self._persist()

    def submit(self, req: OrderRequest) -> BrokerOrder:
        try:
            return super().submit(req)
        finally:
            self._persist()  # even an 'unavailable_after' must not lose the order

    def cancel(self, client_order_id: str) -> None:
        try:
            super().cancel(client_order_id)
        finally:
            self._persist()
