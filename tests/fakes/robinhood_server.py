"""A fake Robinhood Crypto server (httpx.MockTransport) over the MockBroker matching engine.

It enforces the documented rules: Ed25519 request signatures (verified with the matching public key), a 30s
timestamp window, UUID client_order_id, repeat client_order_id returns the original order, stop_loss orders,
no executions ids/fees, cursor pagination, and no lookup by client id. It validates the ADAPTER's logic; the
opt-in live read-only test validates the fake against the real API.
"""

from __future__ import annotations

import base64
from datetime import timedelta
import json
import uuid
from collections import deque
from decimal import Decimal
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from engine.broker.base import BrokerRejected, OrderRequest, OrderType, ProtectionMode
from engine.broker.mock import MockBroker, SubmitFault, _Order
from engine.clock import Clock
from engine.domain.types import AssetClass, Side

BASE = "https://robinhood.test"
API_KEY = "rh-test-api-key"
_P = "/api/v1/crypto/trading"
PAGE = 2  # tiny pages so pagination is always exercised

STATE = {
    "ACCEPTED": "open",
    "PARTIALLY_FILLED": "partially_filled",
    "FILLED": "filled",
    "CANCELLED": "canceled",
}


def new_keypair() -> tuple[Ed25519PrivateKey, str]:
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes_raw()
    return key, base64.b64encode(seed).decode()


class FakeRobinhood:
    def __init__(
        self, clock: Clock, private_key: Ed25519PrivateKey, *, starting_cash: Decimal = Decimal(100_000)
    ) -> None:
        self.clock = clock
        self._public = private_key.public_key()
        self.mock = MockBroker(
            clock, starting_cash=starting_cash, protection=ProtectionMode.SEPARATE_ORDERS, fee_rate=Decimal(0)
        )
        self.mock.prices["BTC-USD"] = Decimal(100)
        self.outage = False
        self._faults: deque[SubmitFault] = deque()
        self.requests: list[tuple[str, str]] = []
        self.posted: list[dict[str, Any]] = []
        self.spread = Decimal("0.001")
        self.quote_style = "inclusive"  # or "bid_ask" / "garbage"
        self.quote_age = timedelta(0)  # how old the quotes it serves are
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    # ---- test controls ----------------------------------------------------------------------
    def set_price(self, symbol: str, price: Decimal) -> None:
        self.mock.set_price(symbol, price)

    def fail_next_submit(self, kind: SubmitFault) -> None:
        self._faults.append(kind)

    # ---- transport --------------------------------------------------------------------------
    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if self.outage:
            raise httpx.ConnectError("refused", request=request)
        if not self._authentic(request):
            return httpx.Response(401, json={"errors": [{"detail": "invalid signature"}]})
        path, method, params = request.url.path, request.method, request.url.params
        if method == "POST" and path == f"{_P}/orders/":
            return self._place(request)
        if method == "GET" and path == f"{_P}/orders/":
            return self._list_orders(params)
        if path.startswith(f"{_P}/orders/") and path.endswith("/cancel/") and method == "POST":
            o = self._by_id(path.split("/")[-3])
            if o is None:
                return httpx.Response(404, json={"errors": [{"detail": "not found"}]})
            if not o.open:
                return httpx.Response(400, json={"errors": [{"detail": "order is not cancelable"}]})
            self.mock.cancel(o.req.client_order_id)
            return httpx.Response(200, json={})
        if method == "GET" and path.startswith(f"{_P}/orders/"):
            o = self._by_id(path.rstrip("/").split("/")[-1])
            return (
                httpx.Response(404, json={}) if o is None else httpx.Response(200, json=self._order_json(o))
            )
        if method == "GET" and path == f"{_P}/holdings/":
            rows = [
                {
                    "asset_code": s.split("-")[0],
                    "total_quantity": str(q),
                    "quantity_available_for_trading": str(q),
                }
                for s, q in self.mock.get_positions().items()
            ]
            return httpx.Response(200, json={"next": None, "previous": None, "results": rows})
        if method == "GET" and path == f"{_P}/accounts/":
            return httpx.Response(
                200,
                json={
                    "account_number": "1",
                    "status": "active",
                    "buying_power": str(self.mock.cash),
                    "buying_power_currency": "USD",
                },
            )
        if method == "GET" and path == "/api/v1/crypto/marketdata/best_bid_ask/":
            return self._quotes(params.get_list("symbol"))
        return httpx.Response(404, json={"errors": [{"detail": f"no route {method} {path}"}]})

    # ---- auth -------------------------------------------------------------------------------
    def _authentic(self, request: httpx.Request) -> bool:
        key, ts, sig = (request.headers.get(h) for h in ("x-api-key", "x-timestamp", "x-signature"))
        if key != API_KEY or ts is None or sig is None:
            return False
        if abs(int(ts) - int(self.clock.now().timestamp())) > 30:
            return False
        target = request.url.raw_path.decode()
        message = f"{key}{ts}{target}{request.method}{request.content.decode()}".encode()
        try:
            self._public.verify(base64.b64decode(sig), message)
        except (InvalidSignature, ValueError):
            return False
        return True

    # ---- orders -----------------------------------------------------------------------------
    def _place(self, request: httpx.Request) -> httpx.Response:
        fault = self._faults.popleft() if self._faults else None
        if fault == "unavailable_before":
            raise httpx.ReadTimeout("timeout", request=request)
        if fault == "reject":
            return httpx.Response(400, json={"errors": [{"detail": "injected rejection"}]})
        body: dict[str, Any] = json.loads(request.content)
        self.posted.append(body)
        wire = body.get("client_order_id", "")
        try:
            uuid.UUID(wire)
        except ValueError:
            return httpx.Response(400, json={"errors": [{"detail": "client_order_id must be a UUID"}]})
        existing = self.mock._orders.get(wire)
        if existing is not None:  # idempotent: a repeat returns the original order
            return httpx.Response(201, json=self._order_json(existing))
        try:
            self.mock.submit(self._to_request(body))
        except BrokerRejected as exc:
            return httpx.Response(400, json={"errors": [{"detail": f"{exc.code}: {exc.detail}"}]})
        if fault == "unavailable_after":
            raise httpx.ReadTimeout("timeout after placement", request=request)
        return httpx.Response(201, json=self._order_json(self.mock._orders[wire]))

    @staticmethod
    def _to_request(body: dict[str, Any]) -> OrderRequest:
        kind = body["type"]
        cfg = body.get(f"{kind}_order_config", {})
        return OrderRequest(
            client_order_id=body["client_order_id"],
            symbol=body["symbol"],
            asset_class=AssetClass.CRYPTO,
            side=Side(body["side"].upper()),
            quantity=Decimal(cfg["asset_quantity"]),
            order_type={"market": OrderType.MARKET, "limit": OrderType.LIMIT, "stop_loss": OrderType.STOP}[
                kind
            ],
            limit_price=Decimal(cfg["limit_price"]) if "limit_price" in cfg else None,
            stop_price=Decimal(cfg["stop_price"]) if "stop_price" in cfg else None,
        )

    def _by_id(self, oid: str) -> _Order | None:
        return next((o for o in self.mock._orders.values() if o.broker_order_id == oid), None)

    def _order_json(self, o: _Order) -> dict[str, Any]:
        kind = {OrderType.MARKET: "market", OrderType.LIMIT: "limit", OrderType.STOP: "stop_loss"}[o.req.kind]
        avg = None
        if o.fills:
            avg = str(sum((f.quantity * f.price for f in o.fills), Decimal(0)) / o.filled)
        return {
            "id": o.broker_order_id,
            "account_number": "1",
            "symbol": o.req.symbol,
            "client_order_id": o.req.client_order_id,
            "side": o.req.side.value.lower(),
            "type": kind,
            "state": STATE.get(o.status.value, "failed"),
            "average_price": avg,
            "filled_asset_quantity": str(o.filled),
            "created_at": self.clock.now().isoformat().replace("+00:00", "Z"),
            "executions": [
                {
                    "effective_price": str(f.price),
                    "quantity": str(f.quantity),
                    "timestamp": f.ts.isoformat().replace("+00:00", "Z"),
                }
                for f in o.fills
            ],
        }  # note: no ordered quantity, no fill ids, no fees: exactly as in the published schema

    def _list_orders(self, params: httpx.QueryParams) -> httpx.Response:
        rows = [o for o in self.mock._orders.values() if o.parent is None or True]
        state = params.get("state")
        if state:
            rows = [o for o in rows if STATE.get(o.status.value) == state]
        start = int(params.get("cursor", "0"))
        page = rows[start : start + PAGE]
        nxt = f"{BASE}{_P}/orders/?cursor={start + PAGE}" if start + PAGE < len(rows) else None
        return httpx.Response(
            200, json={"next": nxt, "previous": None, "results": [self._order_json(o) for o in page]}
        )

    def _quotes(self, symbols: list[str]) -> httpx.Response:
        rows = []
        for s in symbols:
            if s not in self.mock.prices:
                continue
            mid = self.mock.prices[s]
            bid, ask = mid * (1 - self.spread / 2), mid * (1 + self.spread / 2)
            ts = (self.clock.now() - self.quote_age).isoformat().replace("+00:00", "Z")
            if self.quote_style == "inclusive":
                rows.append(
                    {
                        "symbol": s,
                        "bid_inclusive_of_sell_spread": str(bid),
                        "ask_inclusive_of_buy_spread": str(ask),
                        "timestamp": ts,
                    }
                )
            elif self.quote_style == "bid_ask":
                rows.append({"symbol": s, "bid_price": str(bid), "ask_price": str(ask), "timestamp": ts})
            else:
                rows.append({"symbol": s, "something_else": "1"})
        return httpx.Response(200, json={"results": rows})
