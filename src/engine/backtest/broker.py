"""BacktestBroker: the Phase 1 matching engine driven by historical bars instead of live prices.

Conventions (all deliberately pessimistic; documented because they decide whether a backtest can be trusted):
* An order placed when a bar CLOSES is first eligible on the NEXT bar. Market orders fill at that bar's OPEN.
* Stops trigger when the bar's low reaches the stop. If the bar OPENS beyond the stop (a gap), the fill is
  at the open, not at the stop price. Take-profits fill at the target, or the open if already beyond it.
* If a stop and a take-profit are both reachable inside one bar, the stop wins (we cannot know the order).
* Every fill pays the venue's half-spread plus slippage against us, and the commission.
* Within a bar the phases are: (1) market orders fill at the open, (2) the OrderManager places protection,
  (3) resting orders are tested against the bar's range.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from engine.backtest.venue import VenueProfile
from engine.broker.base import OrderType
from engine.broker.mock import MockBroker, _Order
from engine.clock import Clock
from engine.domain.types import Mode, Side
from engine.marketdata.history import Bar

Phase = Literal["open", "range"]


@dataclass(frozen=True)
class CostRecord:
    ts: datetime
    symbol: str
    side: Side
    quantity: Decimal
    reference_price: Decimal  # where the fill would have been with no spread/slippage
    fill_price: Decimal
    fee: Decimal

    @property
    def friction(self) -> Decimal:
        """Money lost to spread + slippage on this fill (always >= 0)."""
        diff = self.fill_price - self.reference_price
        return (diff if self.side is Side.BUY else -diff) * self.quantity


class BacktestBroker(MockBroker):
    def __init__(self, clock: Clock, venue: VenueProfile, *, starting_cash: Decimal) -> None:
        super().__init__(
            clock,
            mode=Mode.PAPER,
            fee_rate=venue.fee_rate,
            protection=venue.protection,
            starting_cash=starting_cash,
        )
        self.venue = venue
        self.phase: Phase | None = None
        self.bars: dict[str, Bar] = {}
        self.cost_log: list[CostRecord] = []
        self._reference: Decimal = Decimal(0)

    # ---- driven by the engine ---------------------------------------------------------------
    def begin_bar(self, bars: Mapping[str, Bar]) -> None:
        self.bars = dict(bars)

    def run_phase(self, phase: Phase) -> None:
        self.phase = phase
        try:
            self.step()
        finally:
            self.phase = None

    def end_bar(self) -> None:
        for symbol, bar in self.bars.items():
            self.prices[symbol] = bar.close
        self.bars = {}

    # ---- fill rules -------------------------------------------------------------------------
    def step(self) -> None:
        """Stops before everything else, so a bar that touches both stop and target resolves to the stop."""

        def rank(o: _Order) -> int:
            if o.trigger == "stop" or o.req.kind is OrderType.STOP:
                return 0
            return 1 if o.trigger == "tp" else 2

        for order in sorted((o for o in self._orders.values() if o.open and o.active), key=rank):
            if order.open and order.active:
                self._match(order)

    def _marketable(self, o: _Order) -> bool:
        bar = self.bars.get(o.req.symbol)
        if self.phase is None or bar is None:
            return False
        req = o.req
        if o.trigger == "stop":
            return (
                self.phase == "range" and req.stop_loss_price is not None and bar.low <= req.stop_loss_price
            )
        if o.trigger == "tp":
            return (
                self.phase == "range"
                and req.take_profit_price is not None
                and bar.high >= req.take_profit_price
            )
        kind = req.kind
        if kind is OrderType.MARKET:
            return self.phase == "open"
        if self.phase != "range":
            return False
        if kind is OrderType.STOP:
            assert req.stop_price is not None
            return bar.low <= req.stop_price if req.side is Side.SELL else bar.high >= req.stop_price
        assert req.limit_price is not None
        return bar.low <= req.limit_price if req.side is Side.BUY else bar.high >= req.limit_price

    def _execution_price(self, o: _Order) -> Decimal:
        bar = self.bars[o.req.symbol]
        req = o.req
        if o.trigger == "stop":
            assert req.stop_loss_price is not None
            raw = min(bar.open, req.stop_loss_price)  # a gap through the stop fills at the open
        elif o.trigger == "tp":
            assert req.take_profit_price is not None
            raw = max(bar.open, req.take_profit_price)
        elif req.kind is OrderType.STOP:
            assert req.stop_price is not None
            raw = min(bar.open, req.stop_price) if req.side is Side.SELL else max(bar.open, req.stop_price)
        elif req.kind is OrderType.LIMIT:
            assert req.limit_price is not None
            raw = min(bar.open, req.limit_price) if req.side is Side.BUY else max(bar.open, req.limit_price)
        else:
            raw = bar.open
        self._reference = raw
        c = self.venue.side_cost
        return raw * (1 + c) if req.side is Side.BUY else raw * (1 - c)

    def _match(self, o: _Order) -> None:
        before = len(o.fills)
        super()._match(o)
        for fill in o.fills[before:]:
            self.cost_log.append(
                CostRecord(
                    fill.ts, o.req.symbol, o.req.side, fill.quantity, self._reference, fill.price, fill.fee
                )
            )
