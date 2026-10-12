"""Alpaca news (Benzinga) over REST polling.

Polling every ~30s is plenty for swing-style strategies and is far simpler to make robust than a WebSocket:
no reconnect/gap logic, and a missed poll just means the next one fetches a larger window. A stream can be added
later behind the same `fetch` interface.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import httpx
from pydantic import SecretStr

from engine.broker.base import BrokerError
from engine.broker.http import BrokerHttp
from engine.marketdata.base import parse_rfc3339
from engine.news.feed import NewsFeedError
from engine.news.models import NewsItem, canonical_symbol

DATA_URL = "https://data.alpaca.markets"
MAX_PAGES = 5


class AlpacaNewsFeed:
    name = "alpaca"

    def __init__(
        self,
        key_id: SecretStr,
        secret: SecretStr,
        *,
        base_url: str = DATA_URL,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        page_size: int = 50,
    ) -> None:
        self._page_size = page_size
        self._http = BrokerHttp(
            base_url,
            name="alpaca-news",
            headers_for=lambda m, t, b: {
                "APCA-API-KEY-ID": key_id.get_secret_value(),
                "APCA-API-SECRET-KEY": secret.get_secret_value(),
            },
            client=client,
            sleeper=sleeper,
        )

    def fetch(self, start: datetime, symbols: Sequence[str], received_at: datetime) -> list[NewsItem]:
        out: list[NewsItem] = []
        token: str | None = None
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "symbols": ",".join(sorted(symbols)),
                "start": start.isoformat().replace("+00:00", "Z"),
                "limit": self._page_size,
                "sort": "asc",
                "include_content": "false",
            }
            if token:
                params["page_token"] = token
            try:
                resp = self._http.request("GET", "/v1beta1/news", params=params)
            except BrokerError as exc:
                raise NewsFeedError(str(exc)) from exc
            if resp.status_code != 200:
                raise NewsFeedError(f"alpaca-news: HTTP {resp.status_code}")
            try:
                body = resp.json()
                out.extend(self._item(raw, received_at) for raw in body.get("news") or [])
                token = body.get("next_page_token")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise NewsFeedError(f"alpaca-news: malformed response ({type(exc).__name__})") from exc
            if not token:
                break
        return out

    @staticmethod
    def _item(raw: dict[str, Any], received_at: datetime) -> NewsItem:
        return NewsItem(
            id=str(raw["id"]),
            source=str(raw.get("source") or "alpaca"),
            headline=str(raw.get("headline") or ""),
            summary=str(raw.get("summary") or ""),
            symbols=tuple(canonical_symbol(str(s)) for s in raw.get("symbols") or []),
            url=str(raw.get("url") or ""),
            published_at=parse_rfc3339(str(raw["created_at"])),
            received_at=received_at,
        )
