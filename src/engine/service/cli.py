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
import uuid
from collections.abc import Mapping, Sequence
from datetime import timedelta
from decimal import Decimal

from engine.broker.paper import PostgresPaperState
from engine.clock import SystemClock
from engine.control import Control
from engine.domain.events import PositionDelta
from engine.domain.intent import Intent
from engine.domain.market import MarketSnapshot
from engine.domain.types import AssetClass, Mode, Side
from engine.eventlog.postgres import PostgresEventStore, migrate
from engine.llm.evaluate import compare_models, format_report, score_calls, summarise
from engine.llm.recorder import PostgresLLMRecorder
from engine.marketdata.alpaca import AlpacaBars
from engine.marketdata.recorder import PostgresQuoteRecorder
from engine.projections.account import AccountProjector
from engine.risk.config import disposable_money_preset, replace
from engine.service.alerts import AlertSink, DiscordWebhookSink, LogAlertSink, MultiAlertSink
from engine.service.config import Secrets, ServiceConfig, load_config
from engine.service.dotenv import load_env_file
from engine.service.executor import Executor, Intervals
from engine.service.health import HealthServer
from engine.service.inbox import PostgresInbox, submit_to_inbox
from engine.service.leader import PostgresLeaderLock
from engine.service.llm_wiring import build_llm_deps
from engine.service.logging_setup import configure_logging
from engine.service.runtime import AccountRuntime, build_runtime
from engine.strategy.approvals import ApprovalError, PostgresProposalStore, approve, reject

log = logging.getLogger("engine.cli")


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    if env is None:
        load_env_file()  # secrets from .env, so they need setting only once
        env = os.environ
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
    if args.command == "panel":
        from engine.service.panel import run_panel

        return run_panel(args.config, args.port)
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
        ("submit-intent", "queue a manual trade intent (the running service decides and executes it)"),
        ("show", "show what happened to one intent: decision, orders, fills"),
        ("panel", "open a local web page with buttons for these commands"),
        ("llm-report", "score recorded model decisions on later prices and compare the models"),
        ("proposals", "list trade ideas from LLM strategies waiting for your approval"),
        ("approve", "approve a pending proposal: it is queued as an intent and still faces the risk engine"),
        ("reject", "reject a pending proposal"),
        ("quotes", "print the live quote, spread estimate and risk-relevant facts for an account's symbols"),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        if name in ("import-baseline", "kill", "set-mode", "adjust", "submit-intent", "show"):
            sp.add_argument("--account", required=True)
        if name == "kill":
            sp.add_argument("--off", action="store_true")
            sp.add_argument("--reason", default="operator")
        if name == "set-mode":
            sp.add_argument("--mode", choices=["PAPER", "LIVE"], required=True)
            sp.add_argument("--confirm", action="store_true", help="required for LIVE")
        if name == "submit-intent":
            sp.add_argument("--symbol", required=True)
            sp.add_argument("--asset-class", choices=["STOCK", "CRYPTO"], required=True)
            sp.add_argument("--side", choices=["BUY", "SELL"], required=True)
            sp.add_argument("--quantity", required=True)
            sp.add_argument("--stop-loss", default=None, help="stop price; required for every BUY")
            sp.add_argument("--limit-price", default=None)
            sp.add_argument("--strategy", default="manual")
            sp.add_argument(
                "--ttl-minutes", type=int, default=10, help="the intent expires if not decided by then"
            )
            sp.add_argument("--dry-run", action="store_true", help="print the intent without queueing it")
        if name == "panel":
            sp.add_argument("--port", type=int, default=8765)
        if name == "llm-report":
            sp.add_argument("--days", type=int, default=30)
            sp.add_argument(
                "--horizon-bars", type=int, default=24, help="bars to hold when scoring a decision"
            )
            sp.add_argument("--timeframe", default="1h")
        if name in ("approve", "reject"):
            sp.add_argument("--id", required=True, help="the proposal id (shown by `proposals`)")
        if name == "show":
            sp.add_argument("--intent-id", required=True)
        if name == "quotes":
            sp.add_argument("--account", required=True)
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
    if args.command == "llm-report":
        return _llm_report(args, cfg, secrets, clock)
    if args.command in ("proposals", "approve", "reject"):
        return _proposals_command(args, secrets, clock)
    if args.command != "quotes" and args.account not in {a.id for a in cfg.accounts}:
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
    elif args.command == "submit-intent":
        return _submit_intent(args, secrets, clock)
    elif args.command == "show":
        return _show(args, store)
    elif args.command == "quotes":
        acct_cfg = next((a for a in cfg.accounts if a.id == args.account), None)
        if acct_cfg is None:
            print(f"error: unknown account {args.account!r}", file=sys.stderr)
            return 2
        rt = build_runtime(
            acct_cfg, store, clock, secrets, base_risk=replace(disposable_money_preset(), cfg.risk)
        )
        limit = replace(disposable_money_preset(), cfg.risk).max_slippage_bps
        print(format_quotes(rt.market.snapshot(list(acct_cfg.symbols)), limit))
        return 0
    elif args.command == "import-baseline":
        acct_cfg = next(a for a in cfg.accounts if a.id == args.account)
        rt = build_runtime(
            acct_cfg, store, clock, secrets, base_risk=replace(disposable_money_preset(), cfg.risk)
        )
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
    base_risk = replace(disposable_money_preset(), cfg.risk)
    sinks: list[AlertSink] = [LogAlertSink()]
    if secrets.discord_webhook is not None:
        sinks.append(DiscordWebhookSink(secrets.discord_webhook, clock))
    alerts = MultiAlertSink(sinks)
    llm_deps = build_llm_deps(cfg.llm, secrets, store, clock, alerts, dsn=dsn) if cfg.llm else None
    runtimes: list[AccountRuntime] = []
    for acct in cfg.accounts:
        if not acct.enabled:
            log.info("account %s is disabled", acct.id)
            continue
        runtimes.append(
            build_runtime(
                acct,
                store,
                clock,
                secrets,
                paper_state=paper_state,
                base_risk=base_risk,
                llm=cfg.llm,
                llm_deps=llm_deps,
            )
        )
    if not runtimes:
        print("error: no enabled accounts", file=sys.stderr)
        return 2

    executor = Executor(
        store,
        runtimes,
        PostgresInbox(dsn),
        PostgresLeaderLock(dsn),
        alerts,
        clock,
        intervals=Intervals(),
        alert_on_fills=cfg.alert_on_fills,
        max_consecutive_errors=cfg.max_consecutive_errors,
        recorder=PostgresQuoteRecorder(dsn) if cfg.record_quotes else None,
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


def _llm_report(args: argparse.Namespace, cfg: ServiceConfig, secrets: Secrets, clock: SystemClock) -> int:
    creds = secrets.alpaca_paper or secrets.alpaca_live
    if creds is None:
        print("error: llm-report needs Alpaca credentials to fetch prices", file=sys.stderr)
        return 2
    recorder = PostgresLLMRecorder(secrets.database_url.get_secret_value())
    now = clock.now()
    calls = recorder.calls_since(now - timedelta(days=args.days))
    if not calls:
        print("no recorded model calls in that period")
        return 0
    history = AlpacaBars(creds[0], creds[1], clock)
    symbols = sorted({d["symbol"] for c in calls for d in c.decisions or []})
    start = min(c.ts for c in calls) - timedelta(days=2)
    bars = {s: history.bars(s, args.timeframe, start, now) for s in symbols}
    cutoffs = {
        n: p.knowledge_cutoff for n, p in (cfg.llm.providers.items() if cfg.llm else []) if p.knowledge_cutoff
    }
    scored, pending = score_calls(calls, bars, horizon_bars=args.horizon_bars, cutoffs=cutoffs)
    stats, agreement = compare_models(calls)
    print(format_report(summarise(scored, pending), stats, agreement, args.horizon_bars))
    return 0


def _proposals_command(args: argparse.Namespace, secrets: Secrets, clock: SystemClock) -> int:
    dsn = secrets.database_url.get_secret_value()
    proposals = PostgresProposalStore(dsn)
    try:
        now = clock.now()
        if args.command == "proposals":
            pending = proposals.pending(now)
            if not pending:
                print("no proposals waiting")
            for p in pending:
                i = p.intent
                mins = int((p.expires_at - now).total_seconds() // 60)
                print(f"{p.proposal_id}  {i.side} {i.quantity} {i.symbol} stop {i.stop_loss_price}  "
                      f"expires in {mins} min\n    why: {p.thesis}\n    news: {', '.join(p.cited)}")  # fmt: skip
            return 0
        try:
            if args.command == "approve":
                intent = approve(proposals, args.id, now, lambda i: submit_to_inbox(dsn, i))
                print(f"approved: {intent.side} {intent.quantity} {intent.symbol}. Queued; the risk engine "
                      f"still decides. Follow it:  show --account {intent.account_id} --intent-id {intent.intent_id}")  # fmt: skip
            else:
                reject(proposals, args.id, now)
                print("rejected")
        except ApprovalError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    finally:
        proposals.close()


def _submit_intent(args: argparse.Namespace, secrets: Secrets, clock: SystemClock) -> int:
    now = clock.now()
    intent = Intent(
        intent_id=f"manual-{uuid.uuid4().hex[:12]}",
        account_id=args.account,
        strategy_id=args.strategy,
        symbol=args.symbol,
        asset_class=AssetClass(args.asset_class),
        side=Side(args.side),
        quantity=Decimal(args.quantity),
        limit_price=Decimal(args.limit_price) if args.limit_price else None,
        stop_loss_price=Decimal(args.stop_loss) if args.stop_loss else None,
        confidence=Decimal(1),
        rationale="manual order via CLI",
        source="cli",
        created_at=now,
        expires_at=now + timedelta(minutes=args.ttl_minutes),
    )
    print(intent.model_dump_json(indent=2))
    if args.dry_run:
        print("dry run: nothing queued")
        return 0
    queued = submit_to_inbox(secrets.database_url.get_secret_value(), intent)
    print(f"queued: {queued}. The running service will decide it within a couple of seconds.")
    print(f"follow it with:  show --account {args.account} --intent-id {intent.intent_id}")
    return 0


def _show(args: argparse.Namespace, store: PostgresEventStore) -> int:
    state = AccountProjector(store, args.account).refresh()
    rec = state.intents.get(args.intent_id)
    if rec is None:
        print("not seen by the engine yet (is the service running, and is the intent in the inbox?)")
        return 1
    decided = [e for e in store.read_stream(f"intent:{args.intent_id}") if e.type == "RiskDecisionMade"]
    if decided:
        d = decided[-1].payload
        print(
            f"decision: {d['decision']} code={d['code']} approved_qty={d['approved_quantity']} {d['detail']}"
        )
        print(f"  checks passed: {', '.join(d['checks_passed'])}")
    else:
        print("decision: pending")
    for o in state.orders.values():
        if o.intent_id == args.intent_id:
            why = f" ({o.reject_code})" if o.reject_code else ""
            line = f"order {o.order_id}: {o.role} {o.side} {o.quantity} {o.symbol} -> {o.status}"
            print(f"{line} filled={o.filled_quantity}{why}")
    print(
        f"position now: {state.position_qty(rec.intent.strategy_id, rec.intent.symbol)} {rec.intent.symbol}"
    )
    return 0


def format_quotes(snapshot: MarketSnapshot, max_slippage_bps: Decimal) -> str:
    lines = [
        f"{'symbol':10} {'price':>14} {'est. spread/slippage':>22}  verdict (limit {max_slippage_bps} bps)"
    ]
    for sym, i in sorted(snapshot.instruments.items()):
        bps = i.est_slippage_bps
        if bps is None:
            lines.append(f"{sym:10} {i.price:>14}  {'n/a (stock)':>22}")
            continue
        verdict = "ok" if bps <= max_slippage_bps else "TOO WIDE: the engine will reject entries"
        lines.append(f"{sym:10} {i.price:>14.2f} {bps:>19.1f} bps  {verdict}")
    return "\n".join(lines)
