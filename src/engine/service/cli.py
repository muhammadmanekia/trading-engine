"""Command line: `python -m engine.service <command>`.

Only `run` and `import-baseline` need broker credentials. Control-plane commands (kill, mode, adjust, status)
need just the database, so an operator can halt trading without ever touching a broker key.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from collections.abc import Mapping, Sequence
from decimal import Decimal

from engine.broker.paper import PostgresPaperState
from engine.clock import SystemClock
from engine.control import Control
from engine.domain.events import PositionDelta
from engine.domain.types import Mode
from engine.eventlog.postgres import PostgresEventStore, migrate
from engine.projections.account import AccountProjector
from engine.service.alerts import AlertSink, DiscordWebhookSink, LogAlertSink, MultiAlertSink
from engine.service.config import Secrets, ServiceConfig, load_config
from engine.service.executor import Executor, Intervals
from engine.service.health import HealthServer
from engine.service.inbox import PostgresInbox
from engine.service.leader import PostgresLeaderLock
from engine.service.logging_setup import configure_logging
from engine.service.runtime import AccountRuntime, build_runtime

log = logging.getLogger("engine.cli")


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    args = _parser().parse_args(argv)
    try:
        secrets = Secrets.from_env(env)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    configure_logging(secrets.values())
    dsn = secrets.database_url.get_secret_value()

    if args.command == "migrate":
        migrate(dsn)
        print("migrations applied")
        return 0

    cfg = load_config(args.config)
    if args.command == "run":
        return _run(cfg, secrets)
    store = PostgresEventStore(dsn)
    try:
        return _control_command(args, cfg, store, secrets)
    finally:
        store.close()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="engine.service")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply database migrations")
    for name, help_ in (
        ("run", "run the execution service"),
        ("status", "show ledger state per account"),
        ("import-baseline", "import a pre-existing broker account's balances"),
        ("kill", "turn an account's kill switch on (or off with --off)"),
        ("set-mode", "set an account's paper/live mode"),
        ("adjust", "record an operator-approved ledger correction"),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        if name in ("import-baseline", "kill", "set-mode", "adjust"):
            sp.add_argument("--account", required=True)
        if name == "kill":
            sp.add_argument("--off", action="store_true")
            sp.add_argument("--reason", default="operator")
        if name == "set-mode":
            sp.add_argument("--mode", choices=["PAPER", "LIVE"], required=True)
            sp.add_argument("--confirm", action="store_true", help="required for LIVE")
        if name == "adjust":
            sp.add_argument("--reason", required=True)
            sp.add_argument("--cash-delta", default="0")
            sp.add_argument("--position", action="append", default=[], metavar="STRATEGY:SYMBOL:QTY")
    return p


def _control_command(
    args: argparse.Namespace, cfg: ServiceConfig, store: PostgresEventStore, secrets: Secrets
) -> int:
    clock = SystemClock()
    if args.command == "status":
        for acct in cfg.accounts:
            st = AccountProjector(store, acct.id).refresh()
            held = {f"{s}/{y}": str(p.quantity) for (s, y), p in st.positions.items() if p.quantity}
            print(
                f"{acct.id}: mode={st.mode} cash={st.cash} kill_switch={st.kill_switch} "
                f"halted_global={st.halted_global} halted={sorted(st.halted_strategies)} "
                f"positions={held} "
                f"anomalies={len(st.anomalies)}"
            )
        return 0
    if args.account not in {a.id for a in cfg.accounts}:
        print(f"error: unknown account {args.account!r}", file=sys.stderr)
        return 2
    control = Control(store, clock, args.account)
    if args.command == "kill":
        control.set_kill_switch(not args.off, args.reason)
    elif args.command == "set-mode":
        mode = Mode(args.mode)
        control.set_mode(mode, confirmed=args.confirm)
    elif args.command == "adjust":
        deltas = []
        for spec in args.position:
            strategy, symbol, qty = spec.split(":")
            deltas.append(PositionDelta(strategy_id=strategy, symbol=symbol, quantity=Decimal(qty)))
        control.adjust(args.reason, Decimal(args.cash_delta), deltas)
    elif args.command == "import-baseline":
        acct_cfg = next(a for a in cfg.accounts if a.id == args.account)
        rt = build_runtime(acct_cfg, store, clock, secrets)
        cash, positions = rt.capture_baseline()
        done = control.import_baseline(cash, positions, note="imported by operator")
        print(
            f"baseline imported: cash={cash} positions={len(positions)}"
            if done
            else "baseline already imported"
        )
    print("ok")
    return 0


def _run(cfg: ServiceConfig, secrets: Secrets) -> int:
    dsn = secrets.database_url.get_secret_value()
    clock = SystemClock()
    store = PostgresEventStore(dsn)
    paper_state = PostgresPaperState(dsn)
    runtimes: list[AccountRuntime] = []
    for acct in cfg.accounts:
        if not acct.enabled:
            log.info("account %s is disabled", acct.id)
            continue
        runtimes.append(build_runtime(acct, store, clock, secrets, paper_state=paper_state))
    if not runtimes:
        print("error: no enabled accounts", file=sys.stderr)
        return 2

    sinks: list[AlertSink] = [LogAlertSink()]
    if secrets.discord_webhook is not None:
        sinks.append(DiscordWebhookSink(secrets.discord_webhook, clock))
    executor = Executor(
        store,
        runtimes,
        PostgresInbox(dsn),
        PostgresLeaderLock(dsn),
        MultiAlertSink(sinks),
        clock,
        intervals=Intervals(),
        alert_on_fills=cfg.alert_on_fills,
        max_consecutive_errors=cfg.max_consecutive_errors,
    )
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    health = HealthServer(cfg.health_port, executor.health) if cfg.health_port else None
    if health:
        health.start()
    try:
        executor.run_forever(stop, cfg.poll_interval_s)
    finally:
        if health:
            health.stop()
        store.close()
    return 0
