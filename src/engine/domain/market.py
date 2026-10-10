"""Inputs to the risk engine that are not part of the ledger."""

from __future__ import annotations

from decimal import Decimal

from engine.domain.types import AssetClass, Frozen, Mode, NonNegDec, PosDec, UtcDatetime


class InstrumentInfo(Frozen):
    symbol: str
    asset_class: AssetClass
    price: PosDec
    quote_ts: UtcDatetime
    tradable: bool = True
    halted: bool = False
    qty_step: PosDec = Decimal("0.00000001")
    avg_daily_volume_usd: NonNegDec | None = None
    market_cap_usd: NonNegDec | None = None
    robinhood_listed: bool | None = None  # crypto only
    est_slippage_bps: NonNegDec | None = None  # crypto only


class MarketSnapshot(Frozen):
    as_of: UtcDatetime
    instruments: dict[str, InstrumentInfo]

    def price_map(self) -> dict[str, Decimal]:
        return {s: i.price for s, i in self.instruments.items()}


class SystemState(Frozen):
    now: UtcDatetime
    broker_mode: Mode
    broker_healthy: bool
    feeds_ok: bool
