"""Performance statistics from a BacktestResult, in exact Decimal, plus plain-language warnings.

The warnings matter as much as the numbers: a Sharpe ratio from 9 trades is noise, and "no trades" usually means
the risk engine refused them, which the report says outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from engine.backtest.engine import BacktestResult

ZERO = Decimal(0)
ONE = Decimal(1)
SECONDS_PER_YEAR = Decimal("31557600")  # 365.25 days


@dataclass(frozen=True)
class Metrics:
    start_equity: Decimal
    final_equity: Decimal
    total_return: Decimal
    cagr: Decimal | None
    max_drawdown: Decimal  # as a positive fraction of the peak
    max_drawdown_bars: int
    sharpe: Decimal | None
    sortino: Decimal | None
    trades: int
    win_rate: Decimal | None
    avg_win: Decimal | None
    avg_loss: Decimal | None
    profit_factor: Decimal | None
    expectancy: Decimal | None  # average P&L per trade
    avg_r: Decimal | None  # average R-multiple (P&L in units of the risk taken)
    avg_bars_held: Decimal | None
    exposure: Decimal  # fraction of bars with an open position
    fees: Decimal
    friction: Decimal  # money lost to spread + slippage
    cost_pct_of_start: Decimal
    benchmark_return: Decimal | None
    excess_return: Decimal | None
    warnings: tuple[str, ...]


def _mean(xs: list[Decimal]) -> Decimal | None:
    return sum(xs, ZERO) / len(xs) if xs else None


def _stdev(xs: list[Decimal]) -> Decimal | None:
    if len(xs) < 2:
        return None
    m = sum(xs, ZERO) / len(xs)
    return (sum(((x - m) ** 2 for x in xs), ZERO) / (len(xs) - 1)).sqrt()


def _drawdown(equity: list[Decimal]) -> tuple[Decimal, int]:
    peak, worst, longest, since_peak = equity[0], ZERO, 0, 0
    for e in equity:
        if e >= peak:
            peak, since_peak = e, 0
        else:
            since_peak += 1
            longest = max(longest, since_peak)
            worst = max(worst, (peak - e) / peak)
    return worst, longest


def compute_metrics(r: BacktestResult) -> Metrics:
    cfg = r.config
    eq = [e for _, e in r.equity_curve]
    start, final = cfg.starting_cash, (eq[-1] if eq else cfg.starting_cash)
    total = final / start - ONE

    years = Decimal((r.end - r.start).total_seconds()) / SECONDS_PER_YEAR
    cagr = ((ONE + total) ** (ONE / years) - ONE) if years > 0 and final > 0 else None

    rets = [eq[i] / eq[i - 1] - ONE for i in range(1, len(eq)) if eq[i - 1] > 0]
    ppy = Decimal(cfg.venue.periods_per_year(r.timeframe))
    mean, sd = _mean(rets), _stdev(rets)
    sharpe = (mean / sd * ppy.sqrt()) if (mean is not None and sd) else None
    downside = [min(x, ZERO) ** 2 for x in rets]
    dd = (sum(downside, ZERO) / len(downside)).sqrt() if downside else None
    sortino = (mean / dd * ppy.sqrt()) if (mean is not None and dd) else None
    max_dd, dd_bars = _drawdown(eq) if eq else (ZERO, 0)

    pnls = [t.pnl for t in r.trades]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins, ZERO), -sum(losses, ZERO)
    rs = [t.r_multiple for t in r.trades if t.r_multiple is not None]
    fees = sum((c.fee for c in r.cost_log), ZERO)
    friction = sum((c.friction for c in r.cost_log), ZERO)
    exposure = Decimal(sum(r.exposure)) / len(r.exposure) if r.exposure else ZERO

    bench = r.benchmark_return
    m = Metrics(
        start_equity=start,
        final_equity=final,
        total_return=total,
        cagr=cagr,
        max_drawdown=max_dd,
        max_drawdown_bars=dd_bars,
        sharpe=sharpe,
        sortino=sortino,
        trades=len(r.trades),
        win_rate=Decimal(len(wins)) / len(pnls) if pnls else None,
        avg_win=_mean(wins),
        avg_loss=_mean(losses),
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else None,
        expectancy=_mean(pnls),
        avg_r=_mean(rs),
        avg_bars_held=_mean([Decimal(t.bars_held) for t in r.trades]),
        exposure=exposure,
        fees=fees,
        friction=friction,
        cost_pct_of_start=(fees + friction) / start,
        benchmark_return=bench,
        excess_return=(total - bench) if bench is not None else None,
        warnings=(),
    )
    return Metrics(**{**m.__dict__, "warnings": tuple(_warnings(r, m, years))})


def _warnings(r: BacktestResult, m: Metrics, years: Decimal) -> list[str]:
    out = list(r.warnings)
    if m.trades == 0:
        top = ", ".join(f"{k}={v}" for k, v in r.risk_decisions.most_common(4) if k != "APPROVED")
        out.append(
            "NO TRADES. " + (f"The risk engine rejected entries: {top}." if top else "No entry signal fired.")
        )
    elif m.trades < 30:
        out.append(f"only {m.trades} trades: these statistics are not statistically reliable (aim for 30+)")
    if years < 1:
        out.append(f"only {years:.2f} years of data: it likely covers a single market regime")
    slip = r.risk_decisions.get("SLIPPAGE_TOO_HIGH", 0)
    if slip:
        out.append(
            f"{slip} entries were refused as SLIPPAGE_TOO_HIGH: the venue spread exceeds max_slippage_bps. "
            "Loosen it only if you accept paying that spread on every trade."
        )
    if m.max_drawdown > Decimal("0.3"):
        out.append(f"max drawdown {m.max_drawdown:.0%}: far beyond what most people hold through")
    if r.anomalies:
        out.append(f"{len(r.anomalies)} ledger anomalies were recorded: investigate before trusting this run")
    if m.trades and m.cost_pct_of_start > abs(m.total_return) and m.total_return < 0:
        out.append("trading costs exceed the whole result: the strategy trades too often for this venue")
    if r.open_positions:
        out.append(
            f"{len(r.open_positions)} position(s) still open at the end (marked to market, not in the trade list)"
        )
    return out
