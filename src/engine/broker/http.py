"""Shared HTTP plumbing for broker adapters.

Failure semantics (these are what make the adapters safe):
* network error / timeout / 5xx / 429  -> BrokerUnavailable   (ambiguous: the request may have landed)
* 401                                  -> CredentialsError    (fatal: bad or expired keys)
* any other 4xx                        -> returned to the adapter, which decides (usually BrokerRejected)
* only idempotent GETs are retried; a POST is never blindly retried (the caller reconciles by client id)
Secrets are never logged: request headers and bodies are not part of any log line or exception message.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from engine.broker.base import BrokerUnavailable, CredentialsError

log = logging.getLogger("engine.broker.http")

HeadersFor = Callable[[str, str, str], Mapping[str, str]]  # (method, path_with_query, body) -> headers


class TokenBucket:
    """Client-side rate limiter so we stay under the broker's published limits."""

    def __init__(
        self,
        rate_per_s: float,
        burst: int,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate, self.capacity = rate_per_s, float(burst)
        self._tokens = float(burst)
        self._monotonic, self._sleep = monotonic, sleeper
        self._last = monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            self._sleep(wait)


class BrokerHttp:
    def __init__(
        self,
        base_url: str,
        *,
        name: str,
        headers_for: HeadersFor | None = None,
        client: httpx.Client | None = None,
        timeout_s: float = 10.0,
        max_get_retries: int = 2,
        bucket: TokenBucket | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        failure_threshold: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self._headers_for = headers_for
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=3.0))
        self._max_get_retries = max_get_retries
        self._bucket = bucket
        self._sleep = sleeper
        self._threshold = failure_threshold
        self.consecutive_failures = 0

    def healthy(self) -> bool:
        return self.consecutive_failures < self._threshold

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: str | None = None,
    ) -> httpx.Response:
        """`path` starts with '/'. `body` is the exact JSON string to send (and sign)."""
        attempts = 1 + (self._max_get_retries if method == "GET" else 0)
        last: Exception | None = None
        for attempt in range(attempts):
            if self._bucket is not None:
                self._bucket.acquire()
            try:
                resp = self._send(method, path, params, body)
            except httpx.HTTPError as exc:
                # Never include the request (headers!) in what we propagate or log.
                last = BrokerUnavailable(f"{self.name}: {type(exc).__name__}")
                self._failed(f"{method} {path}: {type(exc).__name__}")
            else:
                if resp.status_code == 401:
                    self.consecutive_failures = 0
                    raise CredentialsError(f"{self.name}: 401 unauthorized")
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = BrokerUnavailable(f"{self.name}: HTTP {resp.status_code}")
                    self._failed(f"{method} {path}: HTTP {resp.status_code}")
                    retry_after = _retry_after(resp)
                    if retry_after is not None and attempt + 1 < attempts:
                        self._sleep(min(retry_after, 5.0))
                        continue
                else:
                    self.consecutive_failures = 0
                    return resp
            if attempt + 1 < attempts:
                self._sleep(min(0.25 * (2**attempt) + random.random() * 0.1, 3.0))  # noqa: S311
        assert last is not None
        raise last

    def _send(
        self, method: str, path: str, params: Mapping[str, Any] | None, body: str | None
    ) -> httpx.Response:
        url = self.base_url + path
        req = self._client.build_request(method, url, params=params, content=body)
        if self._headers_for is not None:
            target = req.url.raw_path.decode()  # path plus encoded query, exactly as sent
            req.headers.update(self._headers_for(method, target, body or ""))
        if body is not None:
            req.headers["Content-Type"] = "application/json"
        return self._client.send(req)

    def _failed(self, what: str) -> None:
        self.consecutive_failures += 1
        log.warning("%s request failed (%d in a row): %s", self.name, self.consecutive_failures, what)


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None
