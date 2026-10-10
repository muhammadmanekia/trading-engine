"""Robinhood crypto market data: live best bid/ask for an allowlist of coins.

Only coins in the allowlist file are tradable. That file also supplies the facts Robinhood's API does not give
us (market cap, daily volume, size/price increments), which doubles as a safety net: the bot can never trade
a token nobody explicitly approved. Quotes are parsed defensively: if the response is not recognisably a
bid/ask pair the provider raises, so a wrong assumption stops trading instead of mispricing it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Any

from engine.broker.base import BrokerError, CredentialsError
from engine.broker.robinhood_crypto import PairInfo, RobinhoodApi
from engine.clock import Clock
from engine.domain.market import InstrumentInfo, MarketSnapshot
from engine.domain.types import AssetClass
from engine.marketdata.base import MarketDataError, parse_rfc3339

_BID_KEYS = ("bid_inclusive_of_sell_spread", "bid_price", "bid")
_ASK_KEYS = ("ask_inclusive_of_buy_spread", "ask_price", "ask")
_TS_KEYS = ("timestamp", "updated_at")
BPS = Decimal(10_000)


class RobinhoodMarketData:
    def __init__(
        self,
        api: RobinhoodApi,
        pairs: dict[str, PairInfo],
        clock: Clock,
        *,
        slippage_floor_bps: Decimal = Decimal(10),
        max_quote_age: timedelta = timedelta(minutes=10),
    ) -> None:
        self._api = api
        self._pairs = pairs
        self._clock = clock
        self._floor = slippage_floor_bps
        self._max_age = max_quote_age

    def snapshot(self, symbols: Sequence[str]) -> MarketSnapshot:
        wanted = sorted(set(symbols))
        if not wanted:
            return MarketSnapshot(as_of=self._clock.now(), instruments={})
        unknown = [s for s in wanted if s not in self._pairs]
        if unknown:
            raise MarketDataError(f"not in the crypto allowlist: {unknown}")
        try:
            quotes = self._quotes(wanted)
        except (MarketDataError, CredentialsError):
            raise  # bad keys must reach the executor intact so it halts the account
        except BrokerError as exc:
            raise MarketDataError(f"robinhood quotes unavailable: {exc}") from exc
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            raise MarketDataError(f"unparseable robinhood quote: {type(exc).__name__}: {exc}") from exc
        instruments = {}
        for s in wanted:
            if s not in quotes:
                raise MarketDataError(f"no quote for {s}")
            bid, ask, ts = quotes[s]
            mid = (bid + ask) / 2
            spread_bps = (ask - bid) / mid * BPS
            pair = self._pairs[s]
            instruments[s] = InstrumentInfo(
                symbol=s,
                asset_class=AssetClass.CRYPTO,
                price=mid,
                quote_ts=ts,
                qty_step=pair.asset_increment,
                avg_daily_volume_usd=pair.avg_daily_volume_usd,
                market_cap_usd=pair.market_cap_usd,
                robinhood_listed=True,
                est_slippage_bps=max(spread_bps, self._floor),
            )
        return MarketSnapshot(as_of=self._clock.now(), instruments=instruments)

    def _quotes(self, symbols: list[str]) -> dict[str, tuple[Decimal, Decimal, Any]]:
        resp = self._api.http.request(
            "GET", "/api/v1/crypto/marketdata/best_bid_ask/", params=[("symbol", s) for s in symbols]
        )
        if resp.status_code != 200:
            raise MarketDataError(f"best_bid_ask: HTTP {resp.status_code}")
        out: dict[str, tuple[Decimal, Decimal, Any]] = {}
        for row in resp.json().get("results", []):
            bid, ask = _first(row, _BID_KEYS), _first(row, _ASK_KEYS)
            if bid is None or ask is None:
                raise MarketDataError(f"quote for {row.get('symbol')!r} has no recognisable bid/ask fields")
            bid_d, ask_d = Decimal(str(bid)), Decimal(str(ask))
            if bid_d <= 0 or ask_d <= 0 or ask_d < bid_d:
                raise MarketDataError(f"nonsensical quote for {row.get('symbol')!r}: bid={bid_d} ask={ask_d}")
            raw_ts = _first(row, _TS_KEYS)
            ts = (
                parse_rfc3339(str(raw_ts)) if raw_ts else self._clock.now()
            )  # no timestamp: stamp it as fetched now
            if self._clock.now() - ts > self._max_age:
                raise MarketDataError(f"quote for {row.get('symbol')!r} is stale")
            out[str(row["symbol"])] = (bid_d, ask_d, ts)
        return out


def _first(row: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    return None
