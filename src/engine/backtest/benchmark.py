"""The yardstick every strategy is judged against: equal dollars in every stock, bought once, never sold.

Costs: the purchase pays the venue's half-spread, slippage and fee, exactly like a strategy's entry. Like a
strategy's open positions at the end of a backtest, the holding is marked to the last close with no exit cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from engine.backtest.venue import VenueProfile
from engine.marketdata.history import Bar

ZERO = Decimal(0)
ONE = Decimal(1)


def equal_weight_return(data: Mapping[str, Sequence[Bar]], venue: VenueProfile) -> Decimal:
    """Total return of splitting cash equally across `data`'s symbols at each symbol's first open and holding
    to its last close. `data` must already be cut to the period being measured."""
    if not data:
        raise ValueError("no symbols to benchmark")
    legs: list[tuple[Decimal, Decimal]] = []
    for symbol, bars in sorted(data.items()):
        if not bars:
            raise ValueError(f"no bars for {symbol} in the benchmark period")
        legs.append((bars[0].open, bars[-1].close))
    paid = (ONE + venue.side_cost) * (ONE + venue.fee_rate)
    value = sum((close / (opened * paid) for opened, close in legs), ZERO) / len(legs)
    return value - ONE
