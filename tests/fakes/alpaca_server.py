"""A fake Alpaca REST server (httpx.MockTransport) over the MockBroker matching engine.

It reproduces the documented API shapes and constraints (see docs/phase2-api-notes.md): cumulative fill
data only, bracket needs both legs, no fractional brackets, duplicate client_order_id -> 422, broker-generated
ids for bracket legs, 403 for insufficient buying power. It validates the ADAPTER's translation logic; the
opt-in live paper test validates the fake against the real API.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx

from engine.broker.base import BrokerRejected, BrokerStatus, OrderRequest, OrderType, ProtectionMode
from engine.broker.mock import MockBroker, SubmitFault, _Order
from engine.clock import Clock
from engine.domain.types import AssetClass, Side

BASE = "https://alpaca.test"
_STATUS = {
    BrokerStatus.ACCEPTED: "new",
    BrokerStatus.PARTIALLY_FILLED: "partially_filled",
    BrokerStatus.FILLED: "filled",
    BrokerStatus.CANCELLED: "canceled",
    BrokerStatus.EXPIRED: "expired",
    BrokerStatus.REJECTED: "rejected",
}


class FakeAlpaca:
    def __init__(self, clock: Clock, *, starting_cash: Decimal = Decimal(100_000)) -> None:
        self.mock = MockBroker(clock, starting_cash=starting_cash, protection=ProtectionMode.NATIVE_BRACKET)
        self.mock.prices["AAPL"] = Decimal(100)
        self.clock = clock
        self.outage = False
        self.market_open = True
        self.adv_usd: dict[str, Decimal] = {}  # per-symbol average daily dollar volume
        self.untradable: set[str] = set()
        self.no_quotes: set[str] = set()
        self.history: dict[
            str, list[Any]
        ] = {}  # canonical symbol -> Bars served by the historical-bars routes
        self.news: list[dict[str, Any]] = []  # raw Alpaca news rows served by /v1beta1/news
        self.bars_page = 3  # tiny pages so pagination is always exercised
        self._faults: deque[SubmitFault] = deque()
        self.requests: list[tuple[str, str]] = []
        self.posted: list[dict[str, Any]] = []  # JSON bodies of POST /v2/orders, as received
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    # ---- test controls (same surface as the other fakes) ------------------------------------
    def set_price(self, symbol: str, price: Decimal) -> None:
        self.mock.set_price(symbol, price)

    def fail_next_submit(self, kind: SubmitFault) -> None:
        self._faults.append(kind)

    # ---- transport --------------------------------------------------------------------------
    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if self.outage:
            raise httpx.ConnectError("connection refused", request=request)
        if request.headers.get("APCA-API-KEY-ID") in (None, ""):
            return httpx.Response(401, json={"message": "unauthorized."})
        path, method = request.url.path, request.method
        params = dict(request.url.params)
        if method == "POST" and path == "/v2/orders":
            return self._post_order(request)
        if method == "GET" and path == "/v2/orders:by_client_order_id":
            o = self.mock._orders.get(params["client_order_id"])
            if o is None or o.parent is not None:
                return httpx.Response(404, json={"code": 40410000, "message": "order not found"})
            return httpx.Response(200, json=self._order_json(o, nested=False))
        if method == "GET" and path.startswith("/v2/orders/"):
            oid = path.rsplit("/", 1)[1]
            o = next((x for x in self.mock._orders.values() if x.broker_order_id == oid), None)
            if o is None:
                return httpx.Response(404, json={"message": "order not found"})
            return httpx.Response(200, json=self._order_json(o, nested=params.get("nested") == "true"))
        if method == "GET" and path == "/v2/orders":
            rows = [
                self._order_json(self.mock._orders[v.client_order_id], nested=True)
                for v in self.mock.list_open_orders()
            ]
            return httpx.Response(200, json=rows)
        if method == "DELETE" and path.startswith("/v2/orders/"):
            oid = path.rsplit("/", 1)[1]
            o = next((x for x in self.mock._orders.values() if x.broker_order_id == oid), None)
            if o is None:
                return httpx.Response(404, json={"message": "order not found"})
            if not o.open:
                return httpx.Response(422, json={"message": "order is not cancelable"})
            self.mock.cancel(o.req.client_order_id)
            return httpx.Response(204)
        if method == "GET" and path == "/v2/positions":
            return httpx.Response(
                200, json=[{"symbol": s, "qty": str(q)} for s, q in self.mock.get_positions().items()]
            )
        if method == "GET" and path == "/v2/account":
            return httpx.Response(200, json={"cash": str(self.mock.cash), "status": "ACTIVE"})
        if method == "GET" and path == "/v2/stocks/quotes/latest":
            out = {}
            for sym in params["symbols"].split(","):
                if sym in self.mock.prices and sym not in self.no_quotes:
                    p = self.mock.prices[sym]
                    out[sym] = {
                        "ap": str(p * Decimal("1.0005")),
                        "bp": str(p * Decimal("0.9995")),
                        "t": self._ts(),
                    }
            return httpx.Response(200, json={"quotes": out})
        if method == "GET" and path == "/v2/stocks/trades/latest":
            out = {
                s: {"p": str(self.mock.prices[s]), "t": self._ts()}
                for s in params["symbols"].split(",")
                if s in self.mock.prices
            }
            return httpx.Response(200, json={"trades": out})
        if method == "GET" and path in ("/v2/stocks/bars", "/v1beta3/crypto/us/bars"):
            served = self._history_bars(path, params)
            if served is not None:
                return served
        if method == "GET" and path == "/v2/stocks/bars":
            bars = {}
            for sym in params["symbols"].split(","):
                if sym in self.mock.prices:
                    px = self.mock.prices[sym]
                    vol = self.adv_usd.get(sym, Decimal(50_000_000)) / px
                    bars[sym] = [{"c": str(px), "v": str(vol)} for _ in range(25)]
            return httpx.Response(200, json={"bars": bars})
        if method == "GET" and path.startswith("/v2/assets/"):
            sym = path.rsplit("/", 1)[1]
            if sym not in self.mock.prices:
                return httpx.Response(404, json={"message": "asset not found"})
            return httpx.Response(
                200,
                json={
                    "symbol": sym,
                    "tradable": sym not in self.untradable,
                    "status": "active",
                    "fractionable": True,
                },
            )
        if method == "GET" and path == "/v1beta1/news":
            return self._news(params)
        if method == "GET" and path == "/v2/clock":
            return httpx.Response(
                200, json={"is_open": self.market_open, "next_open": "x", "next_close": "y"}
            )
        return httpx.Response(404, json={"message": f"no route {method} {path}"})

    def _history_bars(self, path: str, params: dict[str, str]) -> httpx.Response | None:
        wire = params["symbols"].split(",")[0]
        bars = self.history.get(wire.replace("/", "-"))
        if not bars or "end" not in params:  # the volume lookup (no 'end') is served by the synthetic route
            return None
        if ("/" in wire) != ("crypto" in path):
            return httpx.Response(400, json={"message": "wrong endpoint for this symbol"})
        lo = datetime.fromisoformat(params["start"].replace("Z", "+00:00"))
        hi = datetime.fromisoformat(params["end"].replace("Z", "+00:00"))
        rows = [b for b in bars if lo <= b.ts <= hi]
        offset = int(params.get("page_token", "0"))
        page = rows[offset : offset + self.bars_page]
        more = offset + self.bars_page < len(rows)
        body = {
            "bars": {
                wire: [
                    {"t": b.ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": float(b.open), "h": float(b.high),
                     "l": float(b.low), "c": float(b.close), "v": float(b.volume)}
                    for b in page
                ]
            },
            "next_page_token": str(offset + self.bars_page) if more else None,
        }  # fmt: skip
        return httpx.Response(200, json=body)

    def _news(self, params: dict[str, str]) -> httpx.Response:
        wanted = set(params.get("symbols", "").split(","))
        rows = [
            n for n in self.news if wanted & set(n["symbols"]) and n["created_at"] >= params.get("start", "")
        ]
        limit = int(params.get("limit", "50"))
        offset = int(params.get("page_token") or 0)
        page = rows[offset : offset + limit]
        more = offset + limit < len(rows)
        return httpx.Response(
            200, json={"news": page, "next_page_token": str(offset + limit) if more else None}
        )

    def _ts(self) -> str:
        return self.clock.now().strftime("%Y-%m-%dT%H:%M:%S") + ".123456789Z"

    def _post_order(self, request: httpx.Request) -> httpx.Response:
        fault = self._faults.popleft() if self._faults else None
        if fault == "unavailable_before":
            raise httpx.ReadTimeout("timed out", request=request)
        if fault == "reject":
            return httpx.Response(422, json={"message": "injected rejection"})
        body: dict[str, Any] = json.loads(request.content)
        self.posted.append(body)
        cid = body["client_order_id"]
        if cid in self.mock._orders:
            return httpx.Response(422, json={"code": 40010001, "message": "client_order_id must be unique"})
        err = self._validate(body)
        if err:
            return httpx.Response(422, json={"message": err})
        try:
            self.mock.submit(self._to_request(body))
        except BrokerRejected as exc:
            if exc.code in ("INSUFFICIENT_FUNDS", "INSUFFICIENT_POSITION"):
                return httpx.Response(403, json={"message": "insufficient buying power or shares"})
            return httpx.Response(422, json={"message": f"{exc.code}: {exc.detail}"})
        if fault == "unavailable_after":
            raise httpx.ReadTimeout("timed out after placement", request=request)
        return httpx.Response(200, json=self._order_json(self.mock._orders[cid], nested=True))

    @staticmethod
    def _validate(body: dict[str, Any]) -> str | None:
        qty = str(body.get("qty", ""))
        klass = body.get("order_class", "simple")
        if "." in qty and (klass != "simple" or body.get("type") != "market"):
            return "fractional orders must be simple market orders"
        if klass == "bracket" and not (body.get("take_profit") and body.get("stop_loss")):
            return "bracket orders require both take_profit and stop_loss"
        if klass == "oto" and not (body.get("take_profit") or body.get("stop_loss")):
            return "oto orders require take_profit or stop_loss"
        if body.get("time_in_force") not in ("day", "gtc", "ioc", "fok", "opg", "cls"):
            return "invalid time_in_force"
        return None

    @staticmethod
    def _to_request(body: dict[str, Any]) -> OrderRequest:
        return OrderRequest(
            client_order_id=body["client_order_id"],
            symbol=body["symbol"],
            asset_class=AssetClass.STOCK,
            side=Side(body["side"].upper()),
            quantity=Decimal(body["qty"]),
            order_type=OrderType(body["type"].upper()),
            limit_price=Decimal(body["limit_price"]) if "limit_price" in body else None,
            stop_price=Decimal(body["stop_price"]) if "stop_price" in body else None,
            stop_loss_price=Decimal(body["stop_loss"]["stop_price"]) if body.get("stop_loss") else None,
            take_profit_price=Decimal(body["take_profit"]["limit_price"])
            if body.get("take_profit")
            else None,
        )

    def _order_json(self, o: _Order, *, nested: bool) -> dict[str, Any]:
        child = o.parent is not None
        kind = ("stop" if o.trigger == "stop" else "limit") if child else o.req.kind.value.lower()
        status = _STATUS[o.status]
        if child and not o.active and o.open:
            status = "held"
        avg = None
        if o.fills:
            avg = str(sum((f.quantity * f.price for f in o.fills), Decimal(0)) / o.filled)
        d: dict[str, Any] = {
            "id": o.broker_order_id,
            "client_order_id": f"leg-{o.broker_order_id}" if child else o.req.client_order_id,
            "symbol": o.req.symbol,
            "side": o.req.side.value.lower(),
            "qty": str(o.quantity),
            "filled_qty": str(o.filled),
            "filled_avg_price": avg,
            "status": status,
            "type": kind,
            "order_class": ("bracket" if len(o.children) == 2 else "oto" if o.children else "simple"),
        }
        if nested and o.children:
            d["legs"] = [self._order_json(self.mock._orders[c], nested=False) for c in o.children]
        return d
