"""Robinhood Crypto Trading API adapter.

What the official spec establishes (see docs/phase2-api-notes.md) and how we handle it:
* auth: x-api-key, x-timestamp (unix seconds), x-signature = base64 Ed25519 over
  api_key+timestamp+path+method+body
* client_order_id must be a UUID and repeating it returns the original order -> we derive a deterministic UUID
  (uuid5) from our order id, so a retry can never create a second order
* there is no lookup by client id -> we find orders by id (remembered) or by scanning open + recent orders
* executions have no id and no fee -> fill ids are derived from position, time and quantity
* no brackets/OCO and no sandbox -> protection is SEPARATE_ORDERS (OrderManager places a stop_loss order);
  crypto 'paper mode' is the PaperBroker simulator, not this class
Details the spec summary did not give (trading-pairs and best-bid-ask field names) are parsed defensively and
configured through the crypto allowlist; anything unparseable fails closed.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Callable, Iterable
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from engine.broker.base import (
    BrokerFill,
    BrokerOrder,
    BrokerRejected,
    BrokerStatus,
    BrokerUnavailable,
    OrderRequest,
    OrderType,
    ProtectionMode,
)
from engine.broker.http import BrokerHttp, TokenBucket, conservative_bucket
from engine.domain.types import Frozen, Mode, NonNegDec, PosDec, Side
from engine.marketdata.base import parse_rfc3339

BASE_URL = "https://trading.robinhood.com"
_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # fixed; changing it would orphan old orders
_P = "/api/v1/crypto/trading"


def wire_id(client_order_id: str) -> str:
    """Deterministic UUID for a client order id (Robinhood requires UUIDs)."""
    return str(uuid.uuid5(_NAMESPACE, f"trading-engine:{client_order_id}"))


class PairInfo(Frozen):
    """Per-symbol trading rules and liquidity facts (from the allowlist file)."""

    asset_increment: PosDec
    price_increment: PosDec
    min_order_size: PosDec = Decimal("0.00000001")
    max_order_size: PosDec = Decimal("1000000000")
    market_cap_usd: NonNegDec | None = None
    avg_daily_volume_usd: NonNegDec | None = None


def check_order_against_pairs(pairs: dict[str, PairInfo], req: OrderRequest) -> PairInfo:
    """Rules Robinhood enforces. Shared with PaperBroker so paper and live reject the same orders."""
    pair = pairs.get(req.symbol)
    if pair is None:
        raise BrokerRejected("UNSUPPORTED_SYMBOL", f"{req.symbol} is not in the crypto allowlist")
    if req.stop_loss_price is not None or req.take_profit_price is not None:
        raise BrokerRejected("BRACKET_NOT_SUPPORTED", "Robinhood has no bracket orders")
    if req.quantity % pair.asset_increment != 0:
        raise BrokerRejected("BAD_INCREMENT", f"quantity must be a multiple of {pair.asset_increment}")
    if not (pair.min_order_size <= req.quantity <= pair.max_order_size):
        raise BrokerRejected("SIZE_OUT_OF_RANGE", f"{pair.min_order_size}..{pair.max_order_size}")
    return pair


def load_pairs(path: str) -> dict[str, PairInfo]:
    with open(path) as fh:
        raw = json.load(fh, parse_float=str)  # floats stay exact strings
    return {sym: PairInfo.model_validate(info) for sym, info in raw.items()}


class RobinhoodApi:
    """Authenticated HTTP access shared by the broker and the market data provider."""

    def __init__(
        self,
        api_key: SecretStr,
        private_key_b64: SecretStr,
        *,
        base_url: str = BASE_URL,
        client: httpx.Client | None = None,
        time_source: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        bucket: TokenBucket | None = None,
    ) -> None:
        self._api_key = api_key
        raw = base64.b64decode(private_key_b64.get_secret_value())
        if len(raw) not in (32, 64):
            raise ValueError(
                "ROBINHOOD_PRIVATE_KEY must be a base64 Ed25519 key (32-byte seed or 64-byte pair)"
            )
        self._key = Ed25519PrivateKey.from_private_bytes(raw[:32])
        self._time = time_source
        self.http = BrokerHttp(
            base_url,
            name="robinhood",
            headers_for=self._headers,
            client=client,
            sleeper=sleeper,
            bucket=bucket if bucket is not None else conservative_bucket(sleeper),
        )

    def sign(self, timestamp: int, path: str, method: str, body: str) -> str:
        message = f"{self._api_key.get_secret_value()}{timestamp}{path}{method}{body}"
        return base64.b64encode(self._key.sign(message.encode())).decode()

    def _headers(self, method: str, target: str, body: str) -> dict[str, str]:
        ts = int(self._time())
        return {
            "x-api-key": self._api_key.get_secret_value(),
            "x-timestamp": str(ts),
            "x-signature": self.sign(ts, target, method, body),
        }

    def get_paged(
        self, path: str, params: dict[str, Any] | None = None, *, max_pages: int = 20
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        query = dict(params or {})
        for _ in range(max_pages):
            resp = self.http.request("GET", path, params=query)
            if resp.status_code != 200:
                raise BrokerUnavailable(f"GET {path}: HTTP {resp.status_code}")
            data = resp.json()
            results.extend(data.get("results", []))
            nxt = data.get("next")
            if not nxt:
                return results
            cursor = httpx.URL(nxt).params.get("cursor")
            if not cursor:
                return results
            query["cursor"] = cursor
        raise BrokerUnavailable(f"GET {path}: pagination did not terminate")


class RobinhoodCryptoBroker:
    protection = ProtectionMode.SEPARATE_ORDERS
    mode = Mode.LIVE

    def __init__(
        self,
        api: RobinhoodApi,
        pairs: dict[str, PairInfo],
        *,
        lookback_hours: int = 24,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._api = api
        self._pairs = pairs
        self._lookback_hours = lookback_hours
        self._now = now
        self._wire_to_cid: dict[str, str] = {}
        self._rh_id: dict[str, str] = {}  # our cid -> Robinhood order id
        self._qty: dict[str, Decimal] = {}  # wire id -> ordered quantity (the order object may not echo it)

    # ---- bookkeeping ------------------------------------------------------------------------
    def register_client_ids(self, client_order_ids: Iterable[str]) -> None:
        """Teach the adapter the ledger's order ids after a restart, so orders map back to them."""
        for cid in client_order_ids:
            self._wire_to_cid[wire_id(cid)] = cid

    def healthy(self) -> bool:
        return self._api.http.healthy()

    # ---- Broker interface -------------------------------------------------------------------
    def submit(self, req: OrderRequest) -> BrokerOrder:
        body = self._order_body(req)
        self._wire_to_cid[body["client_order_id"]] = req.client_order_id
        self._qty[body["client_order_id"]] = req.quantity
        resp = self._api.http.request("POST", f"{_P}/orders/", body=json.dumps(body, separators=(",", ":")))
        if resp.status_code in (200, 201):
            order = self._to_order(resp.json())
            self._rh_id[req.client_order_id] = order.broker_order_id
            return order
        raise BrokerRejected(
            "INVALID_ORDER" if resp.status_code == 400 else f"HTTP_{resp.status_code}", _detail(resp)
        )

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        raw = self._find(client_order_id)
        return None if raw is None else self._to_order(raw)

    def cancel(self, client_order_id: str) -> None:
        raw = self._find(client_order_id)
        if raw is None:
            return
        resp = self._api.http.request("POST", f"{_P}/orders/{raw['id']}/cancel/")
        if resp.status_code == 200:
            return
        if resp.status_code in (400, 404):  # probably already terminal: verify rather than assume
            again = self._find(client_order_id)
            if again is None or again.get("state") not in ("open", "partially_filled"):
                return
        raise BrokerUnavailable(f"cancel not accepted: HTTP {resp.status_code}")

    def get_positions(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for h in self._api.get_paged(f"{_P}/holdings/"):
            qty = Decimal(str(h["total_quantity"]))
            if qty != 0:
                out[f"{h['asset_code']}-USD"] = qty
        return out

    def get_cash(self) -> Decimal:
        resp = self._api.http.request("GET", f"{_P}/accounts/")
        if resp.status_code != 200:
            raise BrokerUnavailable(f"accounts: HTTP {resp.status_code}")
        acct = resp.json()
        if str(acct.get("buying_power_currency", "USD")) != "USD":
            raise BrokerUnavailable("account buying power is not in USD")
        return Decimal(str(acct["buying_power"]))

    def list_open_orders(self) -> list[BrokerOrder]:
        out: dict[str, BrokerOrder] = {}
        for state in ("open", "partially_filled"):
            for raw in self._api.get_paged(f"{_P}/orders/", {"state": state}):
                o = self._to_order(raw)
                out[o.broker_order_id] = o
        return list(out.values())

    # ---- lookup -----------------------------------------------------------------------------
    def _find(self, client_order_id: str) -> dict[str, Any] | None:
        wire = wire_id(client_order_id)
        self._wire_to_cid.setdefault(wire, client_order_id)
        known = self._rh_id.get(client_order_id)
        if known is not None:
            resp = self._api.http.request("GET", f"{_P}/orders/{known}/")
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                return data
            if resp.status_code != 404:
                raise BrokerUnavailable(f"order lookup: HTTP {resp.status_code}")
        for state in ("open", "partially_filled"):
            for raw in self._api.get_paged(f"{_P}/orders/", {"state": state}):
                if raw.get("client_order_id") == wire:
                    self._rh_id[client_order_id] = raw["id"]
                    return raw
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._now() - self._lookback_hours * 3600))
        for raw in self._api.get_paged(f"{_P}/orders/", {"created_at_start": since}):
            if raw.get("client_order_id") == wire:
                self._rh_id[client_order_id] = raw["id"]
                return raw
        return None

    # ---- translation ------------------------------------------------------------------------
    def _order_body(self, req: OrderRequest) -> dict[str, Any]:
        pair = check_order_against_pairs(self._pairs, req)
        side = req.side.value.lower()
        config_key, config = {
            OrderType.MARKET: ("market_order_config", {"asset_quantity": _q(req.quantity)}),
        }.get(req.kind, (None, None))
        kind = req.kind
        if kind is OrderType.LIMIT:
            assert req.limit_price is not None
            rounding = ROUND_DOWN if req.side is Side.BUY else ROUND_UP
            config_key = "limit_order_config"
            config = {
                "asset_quantity": _q(req.quantity),
                "limit_price": _q(_round(req.limit_price, pair.price_increment, rounding)),
                "time_in_force": "gtc",
            }
        elif kind is OrderType.STOP:
            assert req.stop_price is not None
            config_key = "stop_loss_order_config"
            config = {
                "asset_quantity": _q(req.quantity),
                "stop_price": _q(_round(req.stop_price, pair.price_increment, ROUND_UP)),  # earlier trigger
                "time_in_force": "gtc",
            }
        assert config_key is not None and config is not None
        return {
            "client_order_id": wire_id(req.client_order_id),
            "side": side,
            "type": "stop_loss" if kind is OrderType.STOP else kind.value.lower(),
            "symbol": req.symbol,
            config_key: config,
        }

    def _to_order(self, d: dict[str, Any]) -> BrokerOrder:
        filled = Decimal(str(d.get("filled_asset_quantity") or "0"))
        state = str(d.get("state", "open"))
        status = {
            "open": BrokerStatus.PARTIALLY_FILLED if filled > 0 else BrokerStatus.ACCEPTED,
            "partially_filled": BrokerStatus.PARTIALLY_FILLED,
            "filled": BrokerStatus.FILLED,
            "canceled": BrokerStatus.CANCELLED,
            "failed": BrokerStatus.REJECTED,
        }.get(state, BrokerStatus.ACCEPTED)
        wire = d["client_order_id"]
        cid = self._wire_to_cid.get(wire, wire)
        fills = tuple(
            BrokerFill(
                fill_id=f"{wire}:{i}:{ex['timestamp']}:{ex['quantity']}",
                quantity=Decimal(str(ex["quantity"])),
                price=Decimal(str(ex["effective_price"])),
                fee=Decimal(0),  # the spread is inside effective_price; v1 'non-fee tier' has no separate fee
                ts=parse_rfc3339(ex["timestamp"]),
            )
            for i, ex in enumerate(d.get("executions") or [])
        )
        # The spec's Order object does not list the ordered quantity. OrderManager never uses it, so when
        # it is unknown (e.g. after a restart) report what is filled, or a harmless minimum.
        quantity = _order_quantity(d) or self._qty.get(wire) or filled or Decimal("0.00000001")
        return BrokerOrder(
            client_order_id=cid,
            broker_order_id=d["id"],
            symbol=d["symbol"],
            side=Side(str(d["side"]).upper()),
            status=status,
            quantity=quantity,
            fills=fills,
            reject_reason="FAILED" if status is BrokerStatus.REJECTED else None,
        )


def _order_quantity(d: dict[str, Any]) -> Decimal | None:
    for key in (
        "market_order_config",
        "limit_order_config",
        "stop_loss_order_config",
        "stop_limit_order_config",
    ):
        cfg = d.get(key)
        if cfg and cfg.get("asset_quantity"):
            return Decimal(str(cfg["asset_quantity"]))
    return None


def _q(d: Decimal) -> str:
    return format(d.normalize(), "f")


def _round(value: Decimal, increment: Decimal, rounding: str) -> Decimal:
    return (value / increment).to_integral_value(rounding=rounding) * increment


def _detail(resp: httpx.Response) -> str:
    try:
        return json.dumps(resp.json())[:300]
    except ValueError:
        return resp.text[:300]
