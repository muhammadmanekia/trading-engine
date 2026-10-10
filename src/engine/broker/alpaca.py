"""Alpaca (US stocks) adapter.

Verified against Alpaca's docs (see docs/phase2-api-notes.md):
* brackets need BOTH legs, so an entry with only a stop uses an OTO order
* fractional shares cannot carry brackets/stops -> whole shares only (the risk engine floors to qty_step=1)
* only cumulative fill data (filled_qty, filled_avg_price) is on the order; OrderManager derives fills
* orders can be fetched by client id; bracket legs get broker-generated ids, which we re-key to
  '<client_order_id>:stop' / ':tp' so the ledger can refer to them deterministically
* 401 = bad keys (fatal); 403 = insufficient buying power/shares (a rejection, not an auth failure)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import httpx
from pydantic import SecretStr

from engine.broker.base import (
    BrokerOrder,
    BrokerRejected,
    BrokerStatus,
    BrokerUnavailable,
    OrderRequest,
    OrderType,
    ProtectionMode,
)
from engine.broker.http import BrokerHttp
from engine.domain.types import Mode, Side

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

_STATUS = {
    "new": BrokerStatus.ACCEPTED,
    "accepted": BrokerStatus.ACCEPTED,
    "pending_new": BrokerStatus.ACCEPTED,
    "accepted_for_bidding": BrokerStatus.ACCEPTED,
    "held": BrokerStatus.ACCEPTED,  # bracket leg waiting for the entry
    "replaced": BrokerStatus.ACCEPTED,
    "pending_replace": BrokerStatus.ACCEPTED,
    "pending_cancel": BrokerStatus.ACCEPTED,  # still working until it is actually canceled
    "calculated": BrokerStatus.ACCEPTED,
    "stopped": BrokerStatus.ACCEPTED,
    "done_for_day": BrokerStatus.ACCEPTED,
    "partially_filled": BrokerStatus.PARTIALLY_FILLED,
    "filled": BrokerStatus.FILLED,
    "canceled": BrokerStatus.CANCELLED,
    "expired": BrokerStatus.EXPIRED,
    "rejected": BrokerStatus.REJECTED,
    "suspended": BrokerStatus.REJECTED,
}


def _price(p: Decimal) -> str:
    step = Decimal("0.01") if p >= 1 else Decimal("0.0001")  # Alpaca's sub-penny rule
    return format(p.quantize(step, rounding=ROUND_HALF_UP), "f")


def _dec(v: Any) -> Decimal | None:
    return None if v in (None, "") else Decimal(str(v))


class AlpacaBroker:
    protection = ProtectionMode.NATIVE_BRACKET

    def __init__(
        self,
        key_id: SecretStr,
        secret: SecretStr,
        mode: Mode,
        *,
        http: BrokerHttp | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.mode = mode
        self._key_id, self._secret = key_id, secret
        self._http = http or BrokerHttp(
            base_url or (LIVE_URL if mode is Mode.LIVE else PAPER_URL),
            name=f"alpaca-{mode.value.lower()}",
            headers_for=self._headers,
            client=client,
            sleeper=sleeper,
        )

    def _headers(self, method: str, target: str, body: str) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key_id.get_secret_value(),
            "APCA-API-SECRET-KEY": self._secret.get_secret_value(),
        }

    # ---- Broker interface -------------------------------------------------------------------
    def healthy(self) -> bool:
        return self._http.healthy()

    def submit(self, req: OrderRequest) -> BrokerOrder:
        body = self._order_body(req)
        resp = self._http.request("POST", "/v2/orders", body=json.dumps(body, separators=(",", ":")))
        if resp.status_code == 200:
            return self._to_order(resp.json())
        message = _message(resp)
        if resp.status_code == 403:
            raise BrokerRejected("INSUFFICIENT_FUNDS", message)
        if resp.status_code == 422 and "client_order_id" in message.lower():
            existing = self.get_order(req.client_order_id)  # duplicate id: the first submit landed
            if existing is not None:
                return existing
            raise BrokerRejected("DUPLICATE_CLIENT_ORDER_ID", message)
        raise BrokerRejected(
            "INVALID_ORDER" if resp.status_code == 422 else f"HTTP_{resp.status_code}", message
        )

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        parent_cid, suffix = _split_leg(client_order_id)
        raw = self._fetch_by_client_id(parent_cid)
        if raw is None:
            return None
        order = self._to_order(raw)
        if suffix is None:
            return order
        return next((c for c in order.children if c.client_order_id == client_order_id), None)

    def cancel(self, client_order_id: str) -> None:
        parent_cid, suffix = _split_leg(client_order_id)
        raw = self._fetch_by_client_id(parent_cid)
        if raw is None:
            return
        target = raw
        if suffix is not None:
            leg = next(
                (c for c in raw.get("legs") or [] if _leg_suffix(c) == suffix),
                None,
            )
            if leg is None:
                return
            target = leg
        resp = self._http.request("DELETE", f"/v2/orders/{target['id']}")
        if resp.status_code in (200, 204, 404, 422):  # 404/422: already gone or already terminal
            return
        raise BrokerUnavailable(f"cancel failed: HTTP {resp.status_code} {_message(resp)}")

    def get_positions(self) -> dict[str, Decimal]:
        resp = self._http.request("GET", "/v2/positions")
        _ok(resp)
        out: dict[str, Decimal] = {}
        for p in resp.json():
            qty = Decimal(str(p["qty"]))
            if qty != 0:
                out[p["symbol"]] = qty
        return out

    def get_cash(self) -> Decimal:
        resp = self._http.request("GET", "/v2/account")
        _ok(resp)
        return Decimal(str(resp.json()["cash"]))

    def list_open_orders(self) -> list[BrokerOrder]:
        resp = self._http.request(
            "GET",
            "/v2/orders",
            params={"status": "open", "nested": "true", "limit": 500, "direction": "desc"},
        )
        _ok(resp)
        return [self._to_order(o) for o in resp.json()]

    # ---- extras used by the executor --------------------------------------------------------
    def get_asset(self, symbol: str) -> dict[str, Any]:
        resp = self._http.request("GET", f"/v2/assets/{symbol}")
        if resp.status_code == 404:
            raise BrokerRejected("UNKNOWN_ASSET", symbol)
        _ok(resp)
        data: dict[str, Any] = resp.json()
        return data

    def market_clock(self) -> dict[str, Any]:
        resp = self._http.request("GET", "/v2/clock")
        _ok(resp)
        data: dict[str, Any] = resp.json()
        return data

    # ---- translation ------------------------------------------------------------------------
    def _order_body(self, req: OrderRequest) -> dict[str, Any]:
        if "-" in req.symbol or "/" in req.symbol:
            raise BrokerRejected("UNSUPPORTED_SYMBOL", req.symbol)
        if req.quantity != req.quantity.to_integral_value():
            raise BrokerRejected(
                "FRACTIONAL_NOT_SUPPORTED", "whole shares only (fractional cannot carry stops)"
            )
        if len(req.client_order_id) > 128:
            raise BrokerRejected("CLIENT_ID_TOO_LONG", "Alpaca allows at most 128 characters")
        kind = req.kind
        body: dict[str, Any] = {
            "symbol": req.symbol,
            "qty": format(req.quantity.to_integral_value(), "f"),
            "side": req.side.value.lower(),
            "type": kind.value.lower(),
            "time_in_force": "gtc",
            "client_order_id": req.client_order_id,
        }
        if kind is OrderType.LIMIT:
            assert req.limit_price is not None
            body["limit_price"] = _price(req.limit_price)
        elif kind is OrderType.STOP:
            assert req.stop_price is not None
            body["stop_price"] = _price(req.stop_price)
        if req.side is Side.BUY and (req.stop_loss_price or req.take_profit_price):
            body["order_class"] = "bracket" if req.stop_loss_price and req.take_profit_price else "oto"
            if req.take_profit_price:
                body["take_profit"] = {"limit_price": _price(req.take_profit_price)}
            if req.stop_loss_price:
                body["stop_loss"] = {"stop_price": _price(req.stop_loss_price)}
        return body

    def _fetch_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        resp = self._http.request(
            "GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id}
        )
        if resp.status_code == 404:
            return None
        _ok(resp)
        raw: dict[str, Any] = resp.json()
        if raw.get("order_class") in ("bracket", "oto") and raw.get("legs") is None:
            full = self._http.request("GET", f"/v2/orders/{raw['id']}", params={"nested": "true"})
            _ok(full)
            raw = full.json()
        return raw

    def _to_order(self, d: dict[str, Any]) -> BrokerOrder:
        status = _STATUS.get(str(d["status"]), BrokerStatus.ACCEPTED)
        filled = _dec(d.get("filled_qty")) or Decimal(0)
        if status is BrokerStatus.ACCEPTED and filled > 0:
            status = BrokerStatus.PARTIALLY_FILLED
        cid = d["client_order_id"]
        children = tuple(self._to_leg(cid, leg) for leg in d.get("legs") or [] if _leg_suffix(leg))
        return BrokerOrder(
            client_order_id=cid,
            broker_order_id=d["id"],
            symbol=d["symbol"],
            side=Side(str(d["side"]).upper()),
            status=status,
            quantity=Decimal(str(d["qty"])),
            filled_quantity=filled,
            avg_fill_price=_dec(d.get("filled_avg_price")),
            children=children,
            reject_reason="REJECTED" if status is BrokerStatus.REJECTED else None,
        )

    def _to_leg(self, parent_cid: str, leg: dict[str, Any]) -> BrokerOrder:
        suffix = _leg_suffix(leg)
        order = self._to_order({**leg, "legs": None})
        return order.model_copy(update={"client_order_id": f"{parent_cid}:{suffix}"})


def _leg_suffix(leg: dict[str, Any]) -> str | None:
    kind = str(leg.get("type") or leg.get("order_type") or "")
    if kind in ("stop", "stop_limit"):
        return "stop"
    if kind == "limit":
        return "tp"
    return None


def _split_leg(client_order_id: str) -> tuple[str, str | None]:
    for suffix in ("stop", "tp"):
        if client_order_id.endswith(f":{suffix}"):
            return client_order_id[: -len(suffix) - 1], suffix
    return client_order_id, None


def _message(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("message", resp.text))[:300]
    except ValueError:
        return resp.text[:300]


def _ok(resp: httpx.Response) -> None:
    """Reads that do not return 200 leave us unable to know the truth: treat as unavailable."""
    if resp.status_code != 200:
        raise BrokerUnavailable(f"HTTP {resp.status_code}: {_message(resp)}")
