"""Venue profiles: everything about a market that affects what a backtest should charge and allow.

Costs are applied adversely on every fill: buying at the ask side, selling at the bid side. The profiles are
deliberately pessimistic; a strategy that only works with optimistic costs does not work. The Robinhood spread
is a placeholder (200 bps total) until real quotes recorded by the execution service replace it
(`python -m engine.service quotes` shows the live number).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from engine.broker.base import ProtectionMode
from engine.domain.types import AssetClass

BPS = Decimal(10_000)


@dataclass(frozen=True)
class VenueProfile:
    name: str
    asset_class: AssetClass
    protection: ProtectionMode
    qty_step: Decimal
    fee_rate: Decimal = Decimal(0)  # commission as a fraction of notional
    spread_bps: Decimal = Decimal(0)  # FULL bid-ask spread; half is paid on each side
    slippage_bps: Decimal = Decimal(0)  # extra adverse move per side
    market_cap_usd: Decimal | None = None  # crypto liquidity facts the risk engine checks
    adv_usd: Decimal | None = None  # override the average daily dollar volume (else computed from bars)
    periods_per_year_1d: int = 252
    periods_per_year_1h: int = 1764

    @property
    def side_cost(self) -> Decimal:
        """Fraction of price lost on each side of a trade to spread + slippage."""
        return (self.spread_bps / 2 + self.slippage_bps) / BPS

    @property
    def round_trip_cost(self) -> Decimal:
        return 2 * self.side_cost + 2 * self.fee_rate

    def periods_per_year(self, timeframe: str) -> int:
        return self.periods_per_year_1h if timeframe == "1h" else self.periods_per_year_1d

    def with_costs(self, **changes: Any) -> VenueProfile:
        return replace(self, **changes)


ALPACA_STOCKS = VenueProfile(
    name="alpaca-stocks",
    asset_class=AssetClass.STOCK,
    protection=ProtectionMode.NATIVE_BRACKET,
    qty_step=Decimal(1),  # whole shares: fractional shares cannot carry stops
    spread_bps=Decimal(3),
    slippage_bps=Decimal(2),
)

ROBINHOOD_CRYPTO = VenueProfile(
    name="robinhood-crypto",
    asset_class=AssetClass.CRYPTO,
    protection=ProtectionMode.SEPARATE_ORDERS,  # a stop-loss only; no take-profit
    qty_step=Decimal("0.00001"),
    spread_bps=Decimal(
        200
    ),  # placeholder: tighter than this and the live engine would not have rejected the trade
    slippage_bps=Decimal(5),
    market_cap_usd=Decimal("1000000000000"),
    periods_per_year_1d=365,
    periods_per_year_1h=24 * 365,
)

VENUES: dict[str, VenueProfile] = {v.name: v for v in (ALPACA_STOCKS, ROBINHOOD_CRYPTO)}


def get_venue(name: str, **overrides: Any) -> VenueProfile:
    try:
        base = VENUES[name]
    except KeyError:
        raise ValueError(f"unknown venue {name!r}; choose from {sorted(VENUES)}") from None
    return base.with_costs(**overrides) if overrides else base
