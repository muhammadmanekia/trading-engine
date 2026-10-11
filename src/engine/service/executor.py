"""The execution service loop: the only process that holds broker credentials and places orders.

Each tick, per account: refresh paper quotes -> decide new intents -> sync open orders -> (periodically)
reconcile. Everything is fail-closed:
* a lost leader lock stops trading immediately (never two executors)
* bad credentials halt the account (kill switch + CRITICAL alert), they are never retried
* repeated unexpected errors halt the account rather than keep trading blind
* a broker/market-data outage just means 'no new decisions succeed' (the risk engine rejects on stale data)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from engine.broker.base import BrokerError, BrokerUnavailable, CredentialsError
from engine.clock import Clock
from engine.eventlog.base import EventStore
from engine.marketdata.base import MarketDataError
from engine.marketdata.recorder import QuoteSink
from engine.service.alerts import Alert, AlertSink, Level, events_to_alerts
from engine.service.inbox import IntentInbox, ingest
from engine.service.leader import LeaderLock
from engine.service.runtime import AccountRuntime

log = logging.getLogger("engine.executor")


@dataclass
class Intervals:
    sync: timedelta = timedelta(seconds=5)
    reconcile: timedelta = timedelta(seconds=60)


@dataclass
class ExecutorStatus:
    leader: bool = False
    started_at: float = field(default_factory=time.time)
    ticks: int = 0


class Executor:
    def __init__(
        self,
        store: EventStore,
        runtimes: list[AccountRuntime],
        inbox: IntentInbox,
        lock: LeaderLock,
        alerts: AlertSink,
        clock: Clock,
        *,
        intervals: Intervals | None = None,
        alert_on_fills: bool = True,
        max_consecutive_errors: int = 5,
        recorder: QuoteSink | None = None,
    ) -> None:
        self.store = store
        self.runtimes = runtimes
        self.inbox = inbox
        self.lock = lock
        self.alerts = alerts
        self.clock = clock
        self.intervals = intervals or Intervals()
        self.alert_on_fills = alert_on_fills
        self.max_errors = max_consecutive_errors
        self.recorder = recorder
        self.status = ExecutorStatus()
        self._alert_cursor = 0
        self._outage_alerted: set[str] = set()

    # ---- lifecycle --------------------------------------------------------------------------
    def start(self) -> bool:
        """Become the leader, rebuild state from the log, and finish whatever was in flight."""
        if not self.lock.acquire():
            log.error("another execution service holds the leader lock; refusing to start")
            return False
        self.status.leader = True
        last = self.store.read_all()
        self._alert_cursor = last[-1].seq if last else 0  # alert on what happens from now on
        for rt in self.runtimes:
            try:
                rt.bootstrap()
                rt.core.recover()
            except CredentialsError:
                self._fatal(rt, "credentials rejected during startup")
            except (BrokerError, MarketDataError) as exc:
                log.warning(
                    "%s: startup recovery incomplete (%s); will retry on the next sync", rt.cfg.id, exc
                )
        self.alerts.send(
            Alert(Level.INFO, "Execution service started", ", ".join(r.cfg.id for r in self.runtimes))
        )
        return True

    def stop(self) -> None:
        self.status.leader = False
        self.lock.release()

    def run_forever(self, stop: threading.Event, poll_s: float = 2.0) -> None:
        if not self.start():
            return
        try:
            while not stop.is_set():
                if not self.tick():
                    break
                stop.wait(poll_s)
        finally:
            self.stop()

    # ---- one iteration ----------------------------------------------------------------------
    def tick(self) -> bool:
        """Returns False when leadership was lost and the service must stop."""
        if not self.lock.still_held():
            self.status.leader = False
            log.critical("lost the leader lock; stopping")
            self.alerts.send(
                Alert(Level.CRITICAL, "Execution service lost leadership", "stopping immediately")
            )
            return False
        self.status.ticks += 1
        accepted_accounts = {rt.cfg.id for rt in self.runtimes}
        try:
            _, problems = ingest(self.inbox, self.store, self.clock, accepted_accounts)
            for p in problems:
                self.alerts.send(Alert(Level.WARNING, "Bad intent in inbox", p))
        except Exception:
            log.exception("inbox ingestion failed")
        for rt in self.runtimes:
            self._tick_runtime(rt)
        self._emit_alerts()
        return True

    def _tick_runtime(self, rt: AccountRuntime) -> None:
        if rt.fatal:
            return
        now = self.clock.now()
        try:
            self._refresh_paper(rt)
            if rt.runner is not None:
                rt.runner.tick()  # propose intents from newly closed bars
            rt.core.process_pending()
            reconcile_due = rt.last_reconcile is None or now - rt.last_reconcile >= self.intervals.reconcile
            sync_due = rt.last_sync is None or now - rt.last_sync >= self.intervals.sync
            if sync_due or reconcile_due:  # never compare against the broker with an out-of-date ledger
                rt.core.manager.sync()
                rt.last_sync = now
            if reconcile_due:
                rt.reconciler.run()
                rt.last_reconcile = now
            self._record_quotes(rt, now)
            rt.consecutive_errors = 0
            rt.last_tick = now
            self._outage_alerted.discard(rt.cfg.id)
        except CredentialsError:
            self._fatal(rt, "broker rejected our credentials")
        except (BrokerUnavailable, MarketDataError) as exc:
            rt.consecutive_errors += 1
            log.warning("%s: outage (%d): %s", rt.cfg.id, rt.consecutive_errors, exc)
            if rt.consecutive_errors >= 3 and rt.cfg.id not in self._outage_alerted:
                self._outage_alerted.add(rt.cfg.id)
                self.alerts.send(
                    Alert(
                        Level.WARNING,
                        f"{rt.cfg.id}: broker/data unreachable",
                        str(exc),
                        f"outage:{rt.cfg.id}",
                    )
                )
        except Exception as exc:
            rt.consecutive_errors += 1
            log.exception("%s: unexpected error (%d)", rt.cfg.id, rt.consecutive_errors)
            if rt.consecutive_errors >= self.max_errors:
                self._fatal(
                    rt, f"{rt.consecutive_errors} consecutive unexpected errors: {type(exc).__name__}"
                )

    def _record_quotes(self, rt: AccountRuntime, now: datetime) -> None:
        """Best effort: the quote history improves backtests but must never disturb trading."""
        if self.recorder is None or not rt.cfg.symbols or not self.recorder.due(rt.cfg.id, now):
            return
        try:
            self.recorder.store(rt.cfg.id, rt.market.snapshot(list(rt.cfg.symbols)), now)
        except CredentialsError:
            raise
        except Exception:
            log.warning("%s: quote recording skipped", rt.cfg.id, exc_info=True)

    def _refresh_paper(self, rt: AccountRuntime) -> None:
        if rt.paper is None:
            return
        symbols = rt.symbols()
        if not symbols:
            return
        snap = rt.market.snapshot(symbols)
        rt.paper.tick(snap.price_map())

    def _fatal(self, rt: AccountRuntime, reason: str) -> None:
        rt.fatal = True
        log.critical("%s: FATAL %s — halting the account", rt.cfg.id, reason)
        try:
            rt.core.activate_kill_switch(reason, cancel_open_entries=False)
        except Exception:
            log.exception("%s: could not record the kill switch", rt.cfg.id)
        self.alerts.send(Alert(Level.CRITICAL, f"{rt.cfg.id}: trading halted", reason, f"fatal:{rt.cfg.id}"))

    def _emit_alerts(self) -> None:
        try:
            new = self.store.read_all(after_seq=self._alert_cursor)
            if not new:
                return
            self._alert_cursor = new[-1].seq
            for alert in events_to_alerts(new, alert_on_fills=self.alert_on_fills):
                self.alerts.send(alert)
        except Exception:
            log.exception("alert scan failed")

    # ---- health -----------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        now = self.clock.now()
        accounts = {}
        ok = self.status.leader
        for rt in self.runtimes:
            stale = rt.last_tick is None or now - rt.last_tick > 5 * self.intervals.sync
            accounts[rt.cfg.id] = {
                "fatal": rt.fatal,
                "feeds_ok": rt.feeds_ok,
                "kill_switch": rt.core.projector.state.kill_switch,
                "consecutive_errors": rt.consecutive_errors,
                "last_tick": rt.last_tick.isoformat() if rt.last_tick else None,
                "stale": stale,
            }
            ok = ok and not rt.fatal and not stale
        return {"ok": ok, "leader": self.status.leader, "ticks": self.status.ticks, "accounts": accounts}
