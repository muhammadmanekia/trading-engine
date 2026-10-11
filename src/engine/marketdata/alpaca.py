"""Alpaca market data for US stocks: latest quotes, 20-day average dollar volume, tradability, market clock.

Caveat (documented in docs/phase2-api-notes.md): the free IEX feed reports only IEX's share of volume, which
understates real volume many times over. Tune `min_avg_daily_volume_usd` accordingly, or set `volume_scale`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
from pydantic import SecretStr

from engine.broker.alpaca import AlpacaBroker
from engine.broker.base import BrokerError, CredentialsError
from engine.broker.http import BrokerHttp
from engine.clock import Clock
from engine.domain.market import InstrumentInfo, MarketSnapshot
from engine.domain.types import AssetClass
from engine.marketdata.base import MarketDataError, parse_rfc3339
from engine.marketdata.history import Bar, timeframe_delta, validate_bars

DATA_URL = "https://data.alpaca.markets"
ZERO = Decimal(0)


class AlpacaMarketData:
    def __init__(
        self,
        key_id: SecretStr,
        secret: SecretStr,
        trading: AlpacaBroker,
        clock: Clock,
        *,
        feed: str = "iex",
        base_url: str = DATA_URL,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        volume_scale: Decimal = Decimal(1),
        bars_ttl: timedelta = timedelta(hours=6),
        asset_ttl: timedelta = timedelta(hours=1),
        clock_ttl: timedelta = timedelta(seconds=30),
    ) -> None:
        self._trading = trading
        self._clock = clock
        self._feed = feed
        self._volume_scale = volume_scale
        self._http = BrokerHttp(
            base_url,
            name="alpaca-data",
            headers_for=lambda m, t, b: {
                "APCA-API-KEY-ID": key_id.get_secret_value(),
                "APCA-API-SECRET-KEY": secret.get_secret_value(),
            },
            client=client,
            sleeper=sleeper,
        )
        self._ttl = {"bars": bars_ttl, "asset": asset_ttl, "clock": clock_ttl}
        self._cache: dict[tuple[str, str], tuple[Any, Any]] = {}  # (kind, key) -> (expires_at, value)

    # ---- provider ---------------------------------------------------------------------------
    def snapshot(self, symbols: Sequence[str]) -> MarketSnapshot:
        syms = sorted({s for s in symbols})
        if not syms:
            return MarketSnapshot(as_of=self._clock.now(), instruments={})
        try:
            quotes = self._latest_quotes(syms)
            adv = self._avg_dollar_volume(syms)
            is_open = self._market_open()
            instruments: dict[str, InstrumentInfo] = {}
            for s in syms:
                if s not in quotes:
                    raise MarketDataError(f"no quote for {s}")
                price, ts = quotes[s]
                asset = self._asset(s)
                instruments[s] = InstrumentInfo(
                    symbol=s,
                    asset_class=AssetClass.STOCK,
                    price=price,
                    quote_ts=ts,
                    tradable=bool(asset.get("tradable")) and asset.get("status") == "active",
                    halted=False,
                    market_open=is_open,
                    qty_step=Decimal(1),  # whole shares only: fractional cannot carry stops
                    avg_daily_volume_usd=adv.get(s),
                )
            return MarketSnapshot(as_of=self._clock.now(), instruments=instruments)
        except (MarketDataError, CredentialsError):
            raise  # bad keys must reach the executor intact so it halts the account
        except BrokerError as exc:
            raise MarketDataError(f"alpaca data unavailable: {exc}") from exc
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            raise MarketDataError(f"unparseable alpaca data: {type(exc).__name__}: {exc}") from exc

    # ---- pieces -----------------------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.request("GET", path, params=params)
        if resp.status_code != 200:
            raise MarketDataError(f"{path}: HTTP {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    def _latest_quotes(self, syms: list[str]) -> dict[str, tuple[Decimal, Any]]:
        joined = ",".join(syms)
        quotes = self._get("/v2/stocks/quotes/latest", {"symbols": joined, "feed": self._feed}).get(
            "quotes", {}
        )
        trades: dict[str, Any] | None = None
        out: dict[str, tuple[Decimal, Any]] = {}
        for s in syms:
            q = quotes.get(s)
            if q and Decimal(str(q.get("ap", 0))) > 0 and Decimal(str(q.get("bp", 0))) > 0:
                mid = (Decimal(str(q["ap"])) + Decimal(str(q["bp"]))) / 2
                out[s] = (mid, parse_rfc3339(q["t"]))
                continue
            if trades is None:
                trades = self._get("/v2/stocks/trades/latest", {"symbols": joined, "feed": self._feed}).get(
                    "trades", {}
                )
            t = trades.get(s)
            if t and Decimal(str(t.get("p", 0))) > 0:
                out[s] = (Decimal(str(t["p"])), parse_rfc3339(t["t"]))
        return out

    def _avg_dollar_volume(self, syms: list[str]) -> dict[str, Decimal]:
        result: dict[str, Decimal] = {}
        missing = []
        for s in syms:
            hit = self._cached("bars", s)
            if hit is not None:
                result[s] = hit
            else:
                missing.append(s)
        if missing:
            start = (self._clock.now() - timedelta(days=45)).date().isoformat()
            bars = self._get(
                "/v2/stocks/bars",
                {
                    "symbols": ",".join(missing),
                    "timeframe": "1Day",
                    "start": start,
                    "limit": 1000,
                    "feed": self._feed,
                },
            ).get("bars", {})
            for s in missing:
                rows = (bars.get(s) or [])[-20:]
                if not rows:
                    continue  # unknown volume -> the risk engine rejects (fails closed)
                dollars = [Decimal(str(r["c"])) * Decimal(str(r["v"])) for r in rows]
                value = (sum(dollars, ZERO) / len(dollars)) * self._volume_scale
                self._store("bars", s, value)
                result[s] = value
        return result

    def _asset(self, symbol: str) -> dict[str, Any]:
        hit = self._cached("asset", symbol)
        if hit is not None:
            return hit  # type: ignore[no-any-return]
        asset = self._trading.get_asset(symbol)  # asset metadata lives on the trading API host
        self._store("asset", symbol, asset)
        return asset

    def _market_open(self) -> bool:
        hit = self._cached("clock", "clock")
        if hit is not None:
            return bool(hit)
        value = bool(self._trading.market_clock()["is_open"])
        self._store("clock", "clock", value)
        return value

    # ---- tiny TTL cache (driven by the injected clock) --------------------------------------
    def _cached(self, kind: str, key: str) -> Any | None:
        entry = self._cache.get((kind, key))
        if entry is None or entry[0] <= self._clock.now():
            return None
        return entry[1]

    def _store(self, kind: str, key: str, value: Any) -> None:
        self._cache[(kind, key)] = (self._clock.now() + self._ttl[kind], value)


# ---- historical bars -----------------------------------------------------------------------------
_TF = {"1h": "1Hour", "1d": "1Day"}
CRYPTO_BARS_PATH = "/v1beta3/crypto/us/bars"
STOCK_BARS_PATH = "/v2/stocks/bars"


class AlpacaBars:
    """Closed historical bars from Alpaca: stocks (split-adjusted) and crypto ('BTC-USD' is requested as 'BTC/USD').

    Bar timestamps are OPEN times. A bar that has not finished yet is never returned. Numbers are parsed straight
    from the JSON text into Decimal, never through a float.
    """

    def __init__(
        self,
        key_id: SecretStr,
        secret: SecretStr,
        clock: Clock,
        *,
        base_url: str = DATA_URL,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        feed: str = "iex",
    ) -> None:
        self._clock = clock
        self._feed = feed
        self._http = BrokerHttp(
            base_url,
            name="alpaca-bars",
            headers_for=lambda m, t, b: {
                "APCA-API-KEY-ID": key_id.get_secret_value(),
                "APCA-API-SECRET-KEY": secret.get_secret_value(),
            },
            client=client,
            sleeper=sleeper,
        )

    def bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[Bar]:
        if timeframe not in _TF:
            raise MarketDataError(f"unsupported timeframe {timeframe!r}")
        crypto = "-" in symbol
        wire = symbol.replace("-", "/")
        path = CRYPTO_BARS_PATH if crypto else STOCK_BARS_PATH
        params: dict[str, Any] = {
            "symbols": wire,
            "timeframe": _TF[timeframe],
            "start": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,
            "sort": "asc",
        }
        if not crypto:
            params.update(adjustment="split", feed=self._feed)
        out: list[Bar] = []
        delta = timeframe_delta(timeframe)
        try:
            for _ in range(500):  # pagination guard
                resp = self._http.request("GET", path, params=params)
                if resp.status_code != 200:
                    raise MarketDataError(f"{path}: HTTP {resp.status_code}")
                body = json.loads(resp.text, parse_float=Decimal)
                for raw in (body.get("bars") or {}).get(wire, []):
                    bar = Bar(
                        ts=parse_rfc3339(raw["t"]),
                        open=Decimal(str(raw["o"])),
                        high=Decimal(str(raw["h"])),
                        low=Decimal(str(raw["l"])),
                        close=Decimal(str(raw["c"])),
                        volume=Decimal(str(raw.get("v", 0))),
                    )
                    if bar.ts + delta <= self._clock.now() and start <= bar.ts < end:
                        out.append(bar)
                token = body.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
            else:
                raise MarketDataError(f"{path}: pagination did not terminate")
        except CredentialsError:
            raise
        except BrokerError as exc:
            raise MarketDataError(f"alpaca bars unavailable: {exc}") from exc
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            raise MarketDataError(f"unparseable alpaca bars: {type(exc).__name__}: {exc}") from exc
        validate_bars(out, timeframe)  # raises on unsorted/duplicate data
        return out
