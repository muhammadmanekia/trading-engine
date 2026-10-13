"""`python -m engine.backtest <command>`: run, walk-forward, sensitivity and replay from the terminal.

Lives in the service layer because it reads credentials/environment (data keys, the database URL); the rest of
the backtest package never does.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from engine.backtest.compare import (
    ExamLedger,
    ExamStatus,
    Protocol,
    evaluate,
    exam_fingerprint,
    format_scoreboard,
    judge,
    scoreboard_json,
    stressed,
)
from engine.backtest.engine import BacktestConfig, BacktestEngine
from engine.backtest.metrics import compute_metrics
from engine.backtest.replay import replay_account, starting_cash
from engine.backtest.report import format_summary, write_report
from engine.backtest.venue import VenueProfile, get_venue
from engine.backtest.walkforward import parameter_grid, sensitivity, walk_forward
from engine.clock import SystemClock
from engine.eventlog.postgres import PostgresEventStore
from engine.marketdata.alpaca import AlpacaBars
from engine.marketdata.history import Bar, BarCache, CsvBars, HistoricalData
from engine.marketdata.recorder import recorded_spread_bps
from engine.risk.config import RiskConfig, disposable_money_preset, replace
from engine.service.config import load_config
from engine.service.dotenv import load_env_file
from engine.strategy.schema import StrategyDef, StrategyError, load_strategy
from engine.strategy.source import RuleStrategySource

ACCOUNT = "bt"
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # the trading-engine folder
DEFAULT_LEDGER = Path("runs/exam-ledger.json")  # under PROJECT_ROOT, wherever the command is run from


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    if env is None:
        load_env_file()  # same .env as the service, so keys need setting only once
        env = os.environ
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args, env)
    except (StrategyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="engine.backtest")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, strategies: bool = True) -> None:
        if strategies:
            sp.add_argument(
                "--strategy", action="append", required=True, help="strategy TOML file (repeatable)"
            )
        sp.add_argument("--venue", required=True, choices=["alpaca-stocks", "robinhood-crypto"])
        sp.add_argument("--data", required=True, help="csv:<directory> or alpaca")
        sp.add_argument("--start", help="YYYY-MM-DD (default: all available for csv, 1 year ago for alpaca)")
        sp.add_argument("--end", help="YYYY-MM-DD (default: now)")
        sp.add_argument("--cash", default="100000")
        sp.add_argument(
            "--config", help="service config whose [risk] layer to use (default: the built-in preset)"
        )
        sp.add_argument("--spread-bps", help="override the venue's full bid-ask spread")
        sp.add_argument(
            "--spread-from-db", action="store_true", help="use the median spread recorded by the service"
        )
        sp.add_argument("--slippage-bps")
        sp.add_argument("--fee-rate")
        sp.add_argument(
            "--adv-usd", help="fix the average daily dollar volume (otherwise computed from the bars)"
        )
        sp.add_argument(
            "--volume-scale", default="1", help="multiply bar volume (e.g. the free IEX feed undercounts)"
        )
        sp.add_argument("--out", help="directory for summary.json/txt, trades.csv, equity.csv")

    run = sub.add_parser("run", help="backtest one or more strategies")
    common(run)
    wf = sub.add_parser("walk-forward", help="walk-forward test with parameter selection on past data only")
    common(wf)
    wf.add_argument(
        "--grid", action="append", default=[], metavar="PATH=V1,V2,...", help="parameter values to search"
    )
    wf.add_argument("--train-bars", type=int, required=True)
    wf.add_argument("--test-bars", type=int, required=True)
    wf.add_argument("--objective", choices=["sharpe", "total_return", "avg_r"], default="sharpe")
    wf.add_argument("--min-trades", type=int, default=10)
    sens = sub.add_parser("sensitivity", help="nudge parameters +/-20% and see if the result survives")
    common(sens)
    sens.add_argument("--param", action="append", required=True, metavar="PATH")
    sens.add_argument("--objective", choices=["sharpe", "total_return", "avg_r"], default="sharpe")
    cmp_ = sub.add_parser(
        "compare", help="score strategies against buy-and-hold: rolling windows + a locked final exam"
    )
    common(cmp_)
    cmp_.add_argument("--train-bars", type=int, default=504)
    cmp_.add_argument("--test-bars", type=int, default=126)
    cmp_.add_argument(
        "--exam-start", default="2024-07-01", help="first bar of the final exam (YYYY-MM-DD[THH:MM:SS])"
    )
    cmp_.add_argument(
        "--final-exam", action="store_true", help="also run the final exam (recorded in the ledger)"
    )
    cmp_.add_argument("--ledger", help="exam ledger file (default: runs/exam-ledger.json in the project)")
    cmp_.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="PATH=V1,V2,...",
        help="parameters tuned on each training window",
    )
    cmp_.add_argument("--objective", choices=["sharpe", "total_return", "avg_r"], default="sharpe")
    rep = sub.add_parser("replay", help="re-simulate a real account's intents and report the tracking error")
    common(rep, strategies=False)
    rep.add_argument("--account", required=True)
    rep.add_argument("--timeframe", choices=["1h", "1d"], required=True)
    rep.add_argument("--symbols", required=True, help="comma-separated symbols to load bars for")
    return p


def _dispatch(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    venue = _venue(args, env)
    risk = _risk(args)
    cash = Decimal(args.cash)
    config = BacktestConfig(
        venue=venue,
        starting_cash=cash,
        risk=risk,
        account_id=ACCOUNT,
        volume_scale=Decimal(args.volume_scale),
    )

    if args.command == "replay":
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        store = PostgresEventStore(_dsn(env))
        try:
            events = store.read_all(account_id=args.account)
        finally:
            store.close()
        config = BacktestConfig(
            venue=venue, starting_cash=starting_cash(events, args.account), risk=risk, account_id=args.account
        )
        data = _load(args, env, symbols, args.timeframe)
        report = replay_account(events, args.account, data, config, args.timeframe)
        print(f"replayed {len(report.comparisons)} intents; mean fill error "
              f"{_fmt(report.mean_abs_error_bps)} bps, max {_fmt(report.max_abs_error_bps)} bps; "
              f"{len(report.mismatches)} filled in only one world")  # fmt: skip
        for c in report.comparisons:
            print(f"  {c.intent_id:50} {c.side:4} real {c.actual_qty}@{c.actual_price}  sim {c.simulated_qty}@{c.simulated_price}"
                  f"  err {_fmt(c.price_error_bps)} bps")  # fmt: skip
        for msg in report.warnings:
            print(f"  WARNING: {msg}")
        return 0

    strategies = [load_strategy(p) for p in args.strategy]
    timeframes = {s.timeframe for s in strategies}
    if len(timeframes) != 1:
        raise ValueError(f"strategies in one run must share a timeframe, got {sorted(timeframes)}")
    timeframe = timeframes.pop()
    if args.command == "compare" and args.start is None:
        args.start = "2020-01-01"  # the rolling windows need years of history, not the 1-year default
    symbols = sorted({sym for s in strategies for sym in s.universe})
    data = _load(args, env, symbols, timeframe)

    if args.command == "compare":
        return _compare(args, strategies, data, config, risk_note=args.config or "built-in preset")

    if args.command == "run":
        sources = [RuleStrategySource(s, ACCOUNT) for s in strategies]
        result = BacktestEngine(sources, data, config).run()
        metrics = compute_metrics(result)
        print(format_summary(result, metrics))
        if args.out:
            print(f"\nwritten to {write_report(result, metrics, args.out)}")
        return 0

    if len(strategies) != 1:
        raise ValueError(f"{args.command} takes exactly one strategy")
    strategy = strategies[0]
    if args.command == "walk-forward":
        grid = _parse_grid(args.grid)
        parameter_grid(grid)  # fail fast on a too-large search
        wfr = walk_forward(
            strategy, data, config, grid, train_bars=args.train_bars, test_bars=args.test_bars,
            objective=args.objective, min_trades=args.min_trades,
        )  # fmt: skip
        print(f"Walk-forward over {len(wfr.windows)} windows (objective: {wfr.objective})")
        for w in wfr.windows:
            print(f"  test {w.test[0]:%Y-%m-%d}..{w.test[1]:%Y-%m-%d}  params {dict(w.params)}  "
                  f"in-sample {_fmt(w.in_sample)}  out-of-sample {_fmt(w.out_of_sample)}  return {w.test_return:.2%}  trades {w.test_trades}")  # fmt: skip
        print(f"\n  out-of-sample return {wfr.oos_return:.2%} over {wfr.oos_trades} trades; "
              f"efficiency {_fmt(wfr.efficiency)}; parameter stability {wfr.stability:.0%}")  # fmt: skip
        for msg in wfr.warnings:
            print(f"  WARNING: {msg}")
        return 0

    sres = sensitivity(strategy, data, config, args.param, objective=args.objective)
    print(
        f"Base: return {sres.base.total_return:.2%}, trades {sres.base.trades}, {args.objective} {_fmt(sres.base.objective)}"
    )
    for p in sres.neighbours:
        print(
            f"  {p.path} = {p.value}: return {p.total_return:.2%}, trades {p.trades}, {args.objective} {_fmt(p.objective)}"
        )
    print(f"  {sres.profitable_share:.0%} of nearby settings are profitable")
    for msg in sres.warnings:
        print(f"  WARNING: {msg}")
    return 0


# ---- helpers ----------------------------------------------------------------------------------------
def _compare(
    args: argparse.Namespace,
    strategies: Sequence[StrategyDef],
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    *,
    risk_note: str,
) -> int:
    protocol = Protocol(train_bars=args.train_bars, test_bars=args.test_bars)
    exam_start = _parse_date(args.exam_start, datetime.now(UTC))
    ledger_path = Path(args.ledger) if args.ledger else PROJECT_ROOT / DEFAULT_LEDGER
    ledger = ExamLedger(ledger_path)
    grid = _parse_grid(args.grid)
    parameter_grid(grid)  # fail fast on a too-large search
    fingerprint = exam_fingerprint(
        grid=grid,
        objective=args.objective,
        protocol=dataclasses.asdict(protocol),
        exam_start=exam_start.isoformat(),
        risk=config.risk.model_dump(mode="json"),
        venue=dataclasses.asdict(config.venue),
        starting_cash=str(config.starting_cash),
        volume_scale=str(config.volume_scale),
        end=args.end,
    )
    # Read every status before anything is recorded, evaluate everything before recording any exam: a failure
    # partway through then uses up no strategy's exam.
    statuses: list[ExamStatus] = [
        ledger.status(s.id, s.hash, fingerprint) if args.final_exam else "not run" for s in strategies
    ]
    rows = []
    for strategy, status in zip(strategies, statuses, strict=True):
        base = evaluate(
            strategy, data, config, protocol, exam_start,
            run_exam=args.final_exam, grid=grid, objective=args.objective,
        )  # fmt: skip
        stress = evaluate(
            strategy, data, stressed(config, protocol.cost_multiplier), protocol, exam_start,
            run_exam=args.final_exam, grid=grid, objective=args.objective,
        )  # fmt: skip
        rows.append((base, judge(base, stress, protocol, status)))
    if args.final_exam:
        if not ledger_path.exists():
            print(
                f"note: creating a new exam ledger at {ledger_path}; if an older one was deleted, "
                "earlier exam results count as seen.",
                file=sys.stderr,
            )
        now = datetime.now(UTC)
        for strategy, (base, _) in zip(strategies, rows, strict=True):
            if base.exam is not None:
                ledger.record(
                    strategy.id,
                    strategy.hash,
                    fingerprint,
                    base.exam.strategy_return,
                    base.exam.benchmark_return,
                    now,
                )
    text = format_scoreboard(rows, risk_note=risk_note)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scoreboard.txt").write_text(text + "\n")
        (out / "scoreboard.json").write_text(json.dumps(scoreboard_json(rows), indent=2))
    return 0


def _fmt(x: Decimal | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _dsn(env: Mapping[str, str]) -> str:
    dsn = env.get("ENGINE_DATABASE_URL")
    if not dsn:
        raise ValueError("ENGINE_DATABASE_URL is required for this option")
    return dsn


def _risk(args: argparse.Namespace) -> RiskConfig:
    if args.config:
        return replace(disposable_money_preset(), load_config(args.config).risk)
    return disposable_money_preset()


def _venue(args: argparse.Namespace, env: Mapping[str, str]) -> VenueProfile:
    overrides: dict[str, Any] = {}
    for flag, field in (
        ("spread_bps", "spread_bps"),
        ("slippage_bps", "slippage_bps"),
        ("fee_rate", "fee_rate"),
        ("adv_usd", "adv_usd"),
    ):
        if getattr(args, flag, None) is not None:
            overrides[field] = Decimal(getattr(args, flag))
    venue = get_venue(args.venue, **overrides)
    if getattr(args, "spread_from_db", False) and "spread_bps" not in overrides:
        symbols = [s for s in (getattr(args, "symbols", "") or "").split(",") if s]
        for path in getattr(args, "strategy", None) or []:
            symbols += list(load_strategy(path).universe)
        measured = recorded_spread_bps(_dsn(env), symbols[0]) if symbols else None
        if measured is None:
            print("note: not enough recorded quotes yet; using the venue's default spread", file=sys.stderr)
        else:
            print(f"using the recorded median spread for {symbols[0]}: {measured:.1f} bps", file=sys.stderr)
            venue = venue.with_costs(spread_bps=measured)
    return venue


def _parse_date(text: str | None, default: datetime) -> datetime:
    return default if text is None else datetime.fromisoformat(text).replace(tzinfo=UTC)


def _load(
    args: argparse.Namespace, env: Mapping[str, str], symbols: Sequence[str], timeframe: str
) -> dict[str, list[Bar]]:
    now = datetime.now(UTC)
    start = _parse_date(
        args.start, now - timedelta(days=365) if args.data == "alpaca" else datetime(1970, 1, 1, tzinfo=UTC)
    )
    end = _parse_date(args.end, now + timedelta(days=1))
    source: HistoricalData
    if args.data.startswith("csv:"):
        source = CsvBars(Path(args.data[4:]))
    elif args.data == "alpaca":
        key = env.get("ALPACA_PAPER_KEY_ID") or env.get("ALPACA_LIVE_KEY_ID")
        secret = env.get("ALPACA_PAPER_SECRET") or env.get("ALPACA_LIVE_SECRET")
        if not (key and secret):
            raise ValueError(
                "--data alpaca needs ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET in the environment"
            )
        source = BarCache(AlpacaBars(SecretStr(key), SecretStr(secret), SystemClock()), Path(".cache/bars"))
    else:
        raise ValueError("--data must be 'csv:<directory>' or 'alpaca'")
    data = {sym: source.bars(sym, timeframe, start, end) for sym in symbols}
    empty = [s for s, b in data.items() if not b]
    if empty:
        raise ValueError(f"no bars returned for {empty} in that period")
    return data


def _parse_grid(specs: Sequence[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for spec in specs:
        path, _, values = spec.partition("=")
        if not values:
            raise ValueError(f"bad --grid {spec!r}; use PATH=V1,V2,V3")
        grid[path] = [int(v) if v.lstrip("-").isdigit() else v for v in values.split(",")]
    return grid


def strategies_summary(
    strategies: Sequence[StrategyDef],
) -> str:  # pragma: no cover - helper for docs/notebooks
    return ", ".join(f"{s.id}@{s.hash}" for s in strategies)
