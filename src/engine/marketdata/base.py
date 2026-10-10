"""Market data contract. Providers turn broker/exchange data into the InstrumentInfo the risk engine consumes.

A provider must fail closed: if quotes cannot be fetched or parsed it raises MarketDataError, and the
TradingCore turns that into a PROVIDER_ERROR rejection. It must never invent a price.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from engine.domain.market import MarketSnapshot


class MarketDataError(Exception):
    pass


class MarketDataProvider(Protocol):
    def snapshot(self, symbols: Sequence[str]) -> MarketSnapshot: ...


def parse_rfc3339(ts: str) -> datetime:
    """Parse timestamps with up to 9 fractional digits (Python's fromisoformat takes 6)."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    head, sep, tail = ts.partition(".")
    if sep:
        digits = ""
        rest = ""
        for i, ch in enumerate(tail):
            if ch.isdigit():
                digits += ch
            else:
                rest = tail[i:]
                break
        ts = f"{head}.{digits[:6].ljust(6, '0')}{rest}"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        raise MarketDataError(f"timestamp without timezone: {ts!r}")
    return dt.astimezone(UTC)
