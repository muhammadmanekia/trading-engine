"""Alerts: event->alert rules and the Discord sink (dedupe, rate limit, never raises, never leaks the URL)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from pydantic import SecretStr

from engine.clock import FakeClock
from engine.domain.events import (
    AnomalyDetected,
    BreakerTrip,
    CircuitBreakerTripped,
    DriftDetected,
    KillSwitchSet,
    OrderFilled,
    OrderRejected,
    ReconciliationCompleted,
)
from engine.domain.types import Mode, Side
from engine.eventlog.base import make_event
from engine.eventlog.memory import InMemoryEventStore
from engine.domain.events import ModeSet
from engine.service.alerts import (
    Alert,
    DiscordWebhookSink,
    Level,
    MultiAlertSink,
    RecordingAlertSink,
    events_to_alerts,
    validate_discord_url,
)
from tests.helpers import ACCOUNT, D, T0

URL = "https://discord.com/api/webhooks/123456/SECRET-TOKEN-abc"


def stored(*models):  # type: ignore[no-untyped-def]
    s = InMemoryEventStore()
    return s.append([make_event(m, stream_id="x", ts=T0) for m in models])


def test_critical_events_become_critical_alerts() -> None:
    alerts = events_to_alerts(
        stored(
            CircuitBreakerTripped(
                account_id=ACCOUNT, trip=BreakerTrip(scope="global", code="DAILY_LOSS_CAP")
            ),
            KillSwitchSet(account_id=ACCOUNT, active=True, reason="drift"),
            AnomalyDetected(account_id=ACCOUNT, order_id="o", code="PROTECTION_FAILED", detail="d"),
            DriftDetected(account_id=ACCOUNT, kind="POSITION_MISMATCH", severity="critical", symbol="AAPL"),
        )
    )
    assert [a.level for a in alerts] == [Level.CRITICAL] * 4
    assert "DAILY_LOSS_CAP" in alerts[0].title and "drift" in alerts[1].body


def test_noise_is_filtered_and_fills_are_optional() -> None:
    fill = OrderFilled(
        account_id=ACCOUNT, order_id="o", strategy_id="s", symbol="AAPL", side=Side.BUY,
        quantity=D(1), price=D(100), fee=D(0), fill_id="f",
    )  # fmt: skip
    ok = ReconciliationCompleted(account_id=ACCOUNT, ok=True, drift_count=0)
    assert events_to_alerts(stored(ok)) == []
    assert [a.level for a in events_to_alerts(stored(fill))] == [Level.INFO]
    assert events_to_alerts(stored(fill), alert_on_fills=False) == []


def test_recovery_and_warnings() -> None:
    rec = ReconciliationCompleted(account_id=ACCOUNT, ok=True, drift_count=0, summary="recovered")
    rej = OrderRejected(account_id=ACCOUNT, order_id="o", code="NO_PROTECTION")
    warn = AnomalyDetected(account_id=ACCOUNT, code="SOMETHING_MINOR")
    levels = [
        a.level for a in events_to_alerts(stored(rec, rej, warn, ModeSet(account_id=ACCOUNT, mode=Mode.LIVE)))
    ]
    assert levels == [Level.INFO, Level.WARNING, Level.WARNING, Level.WARNING]


@pytest.mark.parametrize(
    "bad",
    [
        "http://discord.com/api/webhooks/1/x",  # not https
        "https://evil.example.com/api/webhooks/1/x",  # wrong host
        "https://discord.com.evil.example/api/webhooks/1/x",  # look-alike
        "https://discord.com/other/path",
    ],
)
def test_only_real_discord_webhook_urls_are_accepted(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_discord_url(bad)
    validate_discord_url(URL)


def sink(handler, **kw):  # type: ignore[no-untyped-def]
    clock = FakeClock()
    return (
        DiscordWebhookSink(SecretStr(URL), clock, client=httpx.Client(transport=httpx.MockTransport(handler)),
                           sleeper=lambda _s: None, **kw),
        clock,
    )  # fmt: skip


def test_posts_an_embed_with_level_and_body() -> None:
    sent: list[dict] = []  # type: ignore[type-arg]
    s, _ = sink(lambda r: (sent.append(json.loads(r.content)), httpx.Response(204))[1])
    s.send(Alert(Level.CRITICAL, "Kill switch ON", "acct: drift"))
    embed = sent[0]["embeds"][0]
    assert embed["title"] == "[CRITICAL] Kill switch ON" and embed["description"] == "acct: drift"


def test_duplicate_alerts_are_suppressed_within_the_window_then_allowed() -> None:
    sent: list[int] = []
    s, clock = sink(lambda r: (sent.append(1), httpx.Response(204))[1])
    a = Alert(Level.WARNING, "outage", dedupe_key="outage:acct")
    s.send(a)
    s.send(a)
    assert len(sent) == 1
    clock.advance(minutes=11)
    s.send(a)
    assert len(sent) == 2


def test_flood_sheds_info_but_never_critical() -> None:
    sent: list[int] = []
    s, _ = sink(lambda r: (sent.append(1), httpx.Response(204))[1], max_per_minute=3)
    for i in range(10):
        s.send(Alert(Level.INFO, f"fill {i}"))
    assert len(sent) == 3
    s.send(Alert(Level.CRITICAL, "halt"))
    assert len(sent) == 4


def test_429_is_retried_once_honouring_retry_after() -> None:
    calls: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"retry_after": 0.5}) if len(calls) == 1 else httpx.Response(204)

    s, _ = sink(handler)
    s.send(Alert(Level.CRITICAL, "x"))
    assert len(calls) == 2


def test_a_dead_webhook_never_raises_and_never_logs_the_url(caplog: pytest.LogCaptureFixture) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=r)

    s, _ = sink(handler)
    with caplog.at_level(logging.DEBUG):
        s.send(Alert(Level.CRITICAL, "x"))  # must not raise
        s.send(Alert(Level.CRITICAL, "y"))
    assert "SECRET-TOKEN" not in caplog.text and "webhooks" not in caplog.text
    resp_log = DiscordWebhookSink(SecretStr(URL), FakeClock(), client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))))  # fmt: skip
    with caplog.at_level(logging.DEBUG):
        resp_log.send(Alert(Level.CRITICAL, "z"))
    assert "SECRET-TOKEN" not in caplog.text


def test_multi_sink_isolates_a_failing_sink() -> None:
    class Boom:
        def send(self, alert: Alert) -> None:
            raise RuntimeError("boom")

    rec = RecordingAlertSink()
    MultiAlertSink([Boom(), rec]).send(Alert(Level.INFO, "hello"))
    assert [a.title for a in rec.alerts] == ["hello"]
