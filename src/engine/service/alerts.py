"""Alerts: turn important ledger events into notifications (Discord webhook now, push in Phase 7).

An alert sink must NEVER raise or block trading: a Discord outage cannot be allowed to stop the engine, and a
webhook URL is a credential, so it is never logged.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from engine.clock import Clock
from engine.domain.events import (
    AnomalyDetected,
    CircuitBreakerTripped,
    DriftDetected,
    KillSwitchSet,
    ModeSet,
    OrderFilled,
    OrderRejected,
    ReconciliationCompleted,
)
from engine.eventlog.base import StoredEvent, decode

log = logging.getLogger("engine.alerts")

# httpx logs every request URL at INFO. A webhook URL is a credential, so never let it reach a log handler.
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


class Level(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Alert:
    level: Level
    title: str
    body: str = ""
    dedupe_key: str | None = None


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None: ...


class LogAlertSink:
    def send(self, alert: Alert) -> None:
        lvl = {Level.INFO: logging.INFO, Level.WARNING: logging.WARNING, Level.CRITICAL: logging.CRITICAL}
        log.log(lvl[alert.level], "ALERT %s: %s %s", alert.level, alert.title, alert.body)


class RecordingAlertSink:
    """For tests."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)


class MultiAlertSink:
    def __init__(self, sinks: Sequence[AlertSink]) -> None:
        self.sinks = list(sinks)

    def send(self, alert: Alert) -> None:
        for s in self.sinks:
            try:
                s.send(alert)
            except Exception:
                log.exception("alert sink failed")


_DISCORD_HOSTS = {"discord.com", "discordapp.com", "ptb.discord.com", "canary.discord.com"}
_COLORS = {Level.INFO: 0x3498DB, Level.WARNING: 0xF1C40F, Level.CRITICAL: 0xE74C3C}


def validate_discord_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _DISCORD_HOSTS
        or not parsed.path.startswith("/api/webhooks/")
    ):
        raise ValueError("not a Discord webhook URL (expected https://discord.com/api/webhooks/...)")


class DiscordWebhookSink:
    def __init__(
        self,
        webhook_url: SecretStr,
        clock: Clock,
        *,
        client: httpx.Client | None = None,
        dedupe_window: timedelta = timedelta(minutes=10),
        max_per_minute: int = 20,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_discord_url(webhook_url.get_secret_value())
        self._url = webhook_url
        self._clock = clock
        self._client = client or httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0))
        self._window = dedupe_window
        self._max_per_minute = max_per_minute
        self._sleep = sleeper
        self._seen: dict[str, datetime] = {}
        self._sent: list[datetime] = []

    def send(self, alert: Alert) -> None:
        try:
            if self._suppressed(alert):
                return
            self._post(alert)
        except Exception as exc:  # never raise, never log the URL
            log.warning("discord alert not delivered: %s", type(exc).__name__)

    def _suppressed(self, alert: Alert) -> bool:
        now = self._clock.now()
        if alert.dedupe_key is not None:
            last = self._seen.get(alert.dedupe_key)
            if last is not None and now - last < self._window:
                return True
            self._seen[alert.dedupe_key] = now
        self._sent = [t for t in self._sent if now - t < timedelta(minutes=1)]
        if len(self._sent) >= self._max_per_minute and alert.level is not Level.CRITICAL:
            return True  # shed INFO/WARNING under a flood; CRITICAL always goes out
        self._sent.append(now)
        return False

    def _post(self, alert: Alert) -> None:
        payload = {
            "username": "trading-engine",
            "embeds": [
                {
                    "title": f"[{alert.level}] {alert.title}"[:250],
                    "description": alert.body[:1800],
                    "color": _COLORS[alert.level],
                    "timestamp": self._clock.now().isoformat(),
                }
            ],
        }
        for attempt in range(2):
            resp = self._client.post(self._url.get_secret_value(), json=payload)
            if resp.status_code == 429 and attempt == 0:
                try:
                    wait = float(resp.json().get("retry_after", 1))
                except (ValueError, AttributeError):
                    wait = 1.0
                self._sleep(min(wait, 5.0))
                continue
            if resp.status_code >= 400:
                log.warning("discord alert rejected: HTTP %s", resp.status_code)
            return


# ---- event -> alert rules ----------------------------------------------------------------------
_CRITICAL_ANOMALIES = {"PROTECTION_FAILED", "FLATTEN_FAILED", "OVERFILL", "ORDER_MISSING_AT_BROKER"}


def events_to_alerts(events: Sequence[StoredEvent], *, alert_on_fills: bool = True) -> list[Alert]:
    out: list[Alert] = []
    for stored in events:
        try:
            ev = decode(stored)
        except ValueError:
            continue
        acct = stored.account_id
        match ev:
            case CircuitBreakerTripped():
                t = ev.trip
                scope = "all trading" if t.scope == "global" else f"strategy {t.strategy_id}"
                out.append(
                    Alert(Level.CRITICAL, f"Circuit breaker: {t.code}", f"{acct}: halted {scope}. {t.detail}",
                          dedupe_key=f"trip:{acct}:{t.code}:{t.strategy_id}")
                )  # fmt: skip
            case KillSwitchSet():
                if ev.active:
                    out.append(
                        Alert(Level.CRITICAL, "Kill switch ON", f"{acct}: {ev.reason}", f"kill:{acct}")
                    )
                else:
                    out.append(Alert(Level.INFO, "Kill switch off", f"{acct}: trading may resume"))
            case ModeSet():
                out.append(Alert(Level.WARNING, f"Mode set to {ev.mode}", acct))
            case AnomalyDetected():
                lvl = Level.CRITICAL if ev.code in _CRITICAL_ANOMALIES else Level.WARNING
                out.append(Alert(lvl, f"Anomaly: {ev.code}", f"{acct} {ev.order_id or ''}: {ev.detail}",
                                 f"anomaly:{acct}:{ev.code}:{ev.order_id}"))  # fmt: skip
            case DriftDetected():
                lvl = Level.CRITICAL if ev.severity == "critical" else Level.WARNING
                vals = f"ledger={ev.ledger_value} broker={ev.broker_value}"
                detail = f"{acct} {ev.symbol or ''}: {ev.detail} ({vals})"
                out.append(Alert(lvl, f"Drift: {ev.kind}", detail, f"drift:{acct}:{ev.kind}:{ev.symbol}"))
            case ReconciliationCompleted():
                if ev.ok and ev.summary == "recovered":
                    out.append(
                        Alert(
                            Level.INFO, "Reconciliation recovered", f"{acct}: broker and ledger agree again"
                        )
                    )
            case OrderRejected():
                out.append(
                    Alert(Level.WARNING, f"Order rejected: {ev.code}", f"{acct} {ev.order_id}: {ev.detail}")
                )
            case OrderFilled():
                if alert_on_fills:
                    out.append(Alert(Level.INFO, f"Fill: {ev.side} {ev.quantity} {ev.symbol} @ {ev.price}",
                                     f"{acct} strategy {ev.strategy_id} order {ev.order_id}"))  # fmt: skip
            case _:
                pass
    return out
