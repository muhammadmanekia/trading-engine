"""HTTP plumbing: failure semantics, retries, rate limiting, and secret hygiene."""

from __future__ import annotations

import httpx
import pytest

from engine.broker.base import BrokerUnavailable, CredentialsError
from engine.broker.http import BrokerHttp, TokenBucket


def make(handler, **kw):  # type: ignore[no-untyped-def]
    sleeps: list[float] = []
    http = BrokerHttp(
        "https://x.test",
        name="t",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=sleeps.append,
        **kw,
    )
    return http, sleeps


def test_get_is_retried_on_5xx_then_succeeds() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json={"ok": True})

    http, sleeps = make(handler)
    assert http.request("GET", "/a").json() == {"ok": True}
    assert len(calls) == 3 and len(sleeps) == 2
    assert http.consecutive_failures == 0 and http.healthy()


def test_get_gives_up_after_retries_with_unavailable() -> None:
    http, _ = make(lambda r: httpx.Response(500))
    with pytest.raises(BrokerUnavailable):
        http.request("GET", "/a")
    assert http.consecutive_failures == 3 and not http.healthy()


def test_post_is_never_retried() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    http, _ = make(handler)
    with pytest.raises(BrokerUnavailable):
        http.request("POST", "/orders", body="{}")
    assert len(calls) == 1


def test_timeout_on_post_is_ambiguous_unavailable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    http, _ = make(handler)
    with pytest.raises(BrokerUnavailable):
        http.request("POST", "/orders", body="{}")


def test_429_is_unavailable_and_honours_retry_after_for_gets() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "2"}) if len(calls) == 1 else httpx.Response(200)

    http, sleeps = make(handler)
    assert http.request("GET", "/a").status_code == 200
    assert sleeps == [2.0]
    http2, _ = make(lambda r: httpx.Response(429))
    with pytest.raises(BrokerUnavailable):
        http2.request("POST", "/o", body="{}")


def test_401_is_a_fatal_credentials_error_and_never_retried() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401)

    http, _ = make(handler)
    with pytest.raises(CredentialsError):
        http.request("GET", "/a")
    assert len(calls) == 1


def test_other_4xx_are_returned_for_the_adapter_to_interpret() -> None:
    http, _ = make(lambda r: httpx.Response(422, json={"message": "bad"}))
    assert http.request("POST", "/o", body="{}").status_code == 422
    assert http.healthy()  # a validation error is not a connectivity failure


def test_secrets_never_appear_in_errors_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=req)

    http = BrokerHttp(
        "https://x.test",
        name="t",
        headers_for=lambda m, p, b: {"APCA-API-SECRET-KEY": "TOP-SECRET-VALUE"},
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _s: None,
    )
    with caplog.at_level("DEBUG"), pytest.raises(BrokerUnavailable) as exc:
        http.request("POST", "/o", body='{"k":"TOP-SECRET-BODY"}')
    blob = str(exc.value) + caplog.text + repr(exc.value.__cause__)
    assert "TOP-SECRET" not in blob


def test_signing_headers_receive_the_exact_target_and_body() -> None:
    seen: list[tuple[str, str, str]] = []

    def headers_for(method: str, target: str, body: str) -> dict[str, str]:
        seen.append((method, target, body))
        return {"x-sig": "s"}

    http = BrokerHttp(
        "https://x.test",
        name="t",
        headers_for=headers_for,
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    http.request("GET", "/a/b/", params={"symbol": "BTC-USD"})
    http.request("POST", "/o/", body='{"a":1}')
    assert seen == [("GET", "/a/b/?symbol=BTC-USD", ""), ("POST", "/o/", '{"a":1}')]


def test_token_bucket_throttles_without_real_sleeping() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleeper(s: float) -> None:
        slept.append(s)
        now[0] += s

    b = TokenBucket(rate_per_s=2, burst=2, monotonic=lambda: now[0], sleeper=sleeper)
    for _ in range(2):
        b.acquire()
    assert slept == []  # burst available
    b.acquire()
    assert slept and abs(sum(slept) - 0.5) < 1e-9  # third token needs 0.5s at 2/s
