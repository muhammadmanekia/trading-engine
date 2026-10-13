"""Human- and machine-readable output for a backtest: a terminal summary, JSON, and CSVs."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from engine.backtest.engine import BacktestResult
from engine.backtest.metrics import Metrics


def _pct(x: Decimal | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def _num(x: Decimal | None, places: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{places}f}"


def format_summary(r: BacktestResult, m: Metrics) -> str:
    c = r.config
    lines = [
        f"Backtest  {r.start:%Y-%m-%d} -> {r.end:%Y-%m-%d}   timeframe {r.timeframe}   venue {c.venue.name}",
        f"  spread {c.venue.spread_bps} bps (round trip costs ~{c.venue.round_trip_cost * 100:.2f}% of notional)",
        f"  strategies: {', '.join(f'{k}@{v}' for k, v in r.strategy_hashes.items()) or 'n/a'}",
        "",
        f"  Return        {_pct(m.total_return):>10}   (buy & hold of the first symbol only: {_pct(m.benchmark_return)}, excess {_pct(m.excess_return)})",
        f"  Equity        {_num(m.start_equity)} -> {_num(m.final_equity)}",
        f"  CAGR          {_pct(m.cagr):>10}",
        f"  Max drawdown  {_pct(m.max_drawdown):>10}   (longest underwater: {m.max_drawdown_bars} bars)",
        f"  Sharpe/Sortino {_num(m.sharpe)} / {_num(m.sortino)}",
        f"  Exposure      {_pct(m.exposure):>10}",
        "",
        f"  Trades {m.trades}   win rate {_pct(m.win_rate)}   profit factor {_num(m.profit_factor)}",
        f"  avg win {_num(m.avg_win)}  avg loss {_num(m.avg_loss)}  expectancy/trade {_num(m.expectancy)}  avg R {_num(m.avg_r)}",
        f"  avg holding {_num(m.avg_bars_held, 1)} bars",
        f"  Costs: fees {_num(m.fees)} + spread/slippage {_num(m.friction)} = {_pct(m.cost_pct_of_start)} of starting equity",
        f"  Risk engine: {dict(r.risk_decisions)}",
    ]
    if r.skipped:
        lines.append(f"  Skipped before the risk engine: {dict(r.skipped)}")
    if m.warnings:
        lines += ["", "  WARNINGS", *[f"   - {w}" for w in m.warnings]]
    return "\n".join(lines)


def to_dict(r: BacktestResult, m: Metrics) -> dict[str, Any]:
    c = r.config
    out: dict[str, Any] = json.loads(
        json.dumps(
            {
                "period": {"start": r.start, "end": r.end, "timeframe": r.timeframe},
                "venue": {
                    "name": c.venue.name,
                    "spread_bps": c.venue.spread_bps,
                    "slippage_bps": c.venue.slippage_bps,
                    "fee_rate": c.venue.fee_rate,
                },
                "starting_cash": c.starting_cash,
                "strategies": r.strategy_hashes,
                "metrics": m.__dict__,
                "risk_decisions": dict(r.risk_decisions),
                "skipped": dict(r.skipped),
                "signals": dict(r.signals),
                "open_positions": r.open_positions,
                "anomalies": r.anomalies,
                "trades": [t.__dict__ for t in r.trades],
            },
            default=str,  # Decimals/datetimes become exact strings
        )
    )
    return out


def write_report(r: BacktestResult, m: Metrics, directory: str | Path) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(to_dict(r, m), indent=2))
    (d / "summary.txt").write_text(format_summary(r, m) + "\n")
    with open(d / "trades.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "strategy",
                "symbol",
                "entry_ts",
                "exit_ts",
                "qty",
                "entry",
                "exit",
                "pnl",
                "fees",
                "r",
                "bars",
                "reason",
            ]
        )
        for t in r.trades:
            w.writerow([t.strategy_id, t.symbol, t.entry_ts.isoformat(), t.exit_ts.isoformat(), t.quantity,
                        t.entry_price, t.exit_price, t.pnl, t.fees, t.r_multiple, t.bars_held, t.exit_reason])  # fmt: skip
    with open(d / "equity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "equity"])
        for ts, e in r.equity_curve:
            w.writerow([ts.isoformat(), e])
    return d
