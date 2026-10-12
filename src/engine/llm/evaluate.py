"""Honest scoring of model decisions, and a like-for-like comparison of models.

Principles:
* FORWARD only. A decision is scored on prices that did not exist when it was made: enter at the open of the first
  bar after the call, exit `horizon_bars` bars later. Decisions without enough future bars are 'pending', not zero.
* Compare to a baseline. A stock that drifts up makes any buy look smart, so each decision is measured against the
  average return of every same-length window in the supplied bars (what buying at random would have earned).
* Report uncertainty. A bootstrap interval on the mean excess return, and no verdict below `MIN_SAMPLES`.
* Flag leakage. Models know the past: a decision made about a period before the model's knowledge cutoff cannot be
  trusted as evidence, so it is counted and excluded from the headline numbers.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from engine.llm.recorder import LLMCall
from engine.marketdata.history import Bar

ZERO = Decimal(0)
MIN_SAMPLES = 30
BOOTSTRAP_ROUNDS = 2000
BUCKETS = (
    (Decimal("0.5"), Decimal("0.7")),
    (Decimal("0.7"), Decimal("0.85")),
    (Decimal("0.85"), Decimal("1.01")),
)


@dataclass(frozen=True)
class ScoredDecision:
    call_id: str
    provider: str
    strategy_id: str
    symbol: str
    confidence: Decimal
    entry_price: Decimal
    exit_price: Decimal
    ret: Decimal
    baseline: Decimal
    contaminated: bool

    @property
    def excess(self) -> Decimal:
        return self.ret - self.baseline


def _window_returns(bars: Sequence[Bar], horizon: int) -> list[Decimal]:
    return [
        (bars[i + horizon - 1].close - bars[i].open) / bars[i].open
        for i in range(len(bars) - horizon + 1)
        if bars[i].open > 0
    ]


def score_calls(
    calls: Sequence[LLMCall],
    bars: Mapping[str, Sequence[Bar]],
    *,
    horizon_bars: int,
    cutoffs: Mapping[str, date] | None = None,
) -> tuple[list[ScoredDecision], int]:
    """(scored decisions, number still pending). Only valid calls' ENTER decisions are scored."""
    scored: list[ScoredDecision] = []
    pending = 0
    baselines = {
        s: (sum(r, ZERO) / len(r) if (r := _window_returns(b, horizon_bars)) else ZERO)
        for s, b in bars.items()
    }
    for call in calls:
        if not call.valid or not call.decisions:
            continue
        for d in call.decisions:
            if d.get("action") != "enter" or d["symbol"] not in bars:
                continue
            series = bars[d["symbol"]]
            idx = next((i for i, b in enumerate(series) if b.ts >= call.ts), None)
            if idx is None or idx + horizon_bars > len(series):
                pending += 1
                continue
            entry, exit_ = series[idx].open, series[idx + horizon_bars - 1].close
            if entry <= 0:
                continue
            cutoff = (cutoffs or {}).get(call.provider)
            scored.append(
                ScoredDecision(
                    call_id=call.call_id,
                    provider=call.provider,
                    strategy_id=call.strategy_id,
                    symbol=d["symbol"],
                    confidence=Decimal(str(d["confidence"])),
                    entry_price=entry,
                    exit_price=exit_,
                    ret=(exit_ - entry) / entry,
                    baseline=baselines[d["symbol"]],
                    contaminated=cutoff is not None and series[idx].ts.date() <= cutoff,
                )
            )
    return scored, pending


@dataclass(frozen=True)
class ProviderScore:
    provider: str
    n: int
    pending: int
    contaminated: int
    hit_rate: Decimal | None
    mean_return: Decimal | None
    mean_excess: Decimal | None
    excess_ci: tuple[Decimal, Decimal] | None
    calibration: tuple[tuple[str, int, Decimal | None], ...]  # (confidence bucket, n, hit rate)
    verdict: str


def _bootstrap_ci(values: Sequence[Decimal], seed: int = 0) -> tuple[Decimal, Decimal]:
    rng = random.Random(seed)  # noqa: S311  (statistics, not security)
    n = len(values)
    means = sorted(
        sum((values[rng.randrange(n)] for _ in range(n)), ZERO) / n for _ in range(BOOTSTRAP_ROUNDS)
    )
    return means[int(BOOTSTRAP_ROUNDS * 0.025)], means[int(BOOTSTRAP_ROUNDS * 0.975)]


def summarise(scored: Sequence[ScoredDecision], pending: int = 0) -> list[ProviderScore]:
    by_provider: dict[str, list[ScoredDecision]] = defaultdict(list)
    for s in scored:
        by_provider[s.provider].append(s)
    out: list[ProviderScore] = []
    for name, rows in sorted(by_provider.items()):
        clean = [r for r in rows if not r.contaminated]
        n = len(clean)
        if n == 0:
            out.append(
                ProviderScore(name, 0, pending, len(rows), None, None, None, None, (), "no clean decisions")
            )
            continue
        hit = Decimal(sum(1 for r in clean if r.ret > 0)) / n
        mean_ret = sum((r.ret for r in clean), ZERO) / n
        excess = [r.excess for r in clean]
        mean_ex = sum(excess, ZERO) / n
        ci = _bootstrap_ci(excess) if n >= 2 else None
        cal = []
        for lo, hi in BUCKETS:
            inside = [r for r in clean if lo <= r.confidence < hi]
            cal.append((f"{lo}-{min(hi, Decimal(1))}", len(inside),
                        Decimal(sum(1 for r in inside if r.ret > 0)) / len(inside) if inside else None))  # fmt: skip
        if n < MIN_SAMPLES:
            verdict = f"insufficient data: {n} decisions (need {MIN_SAMPLES}); treat as anecdote"
        elif ci is not None and ci[0] > 0:
            verdict = "beats the baseline (interval above zero); still not proof, keep forward testing"
        elif ci is not None and ci[1] < 0:
            verdict = "WORSE than the baseline"
        else:
            verdict = "indistinguishable from the baseline"
        out.append(
            ProviderScore(name, n, pending, len(rows) - n, hit, mean_ret, mean_ex, ci, tuple(cal), verdict)
        )
    return out


@dataclass(frozen=True)
class ModelStats:
    provider: str
    calls: int
    invalid: int
    enters: int
    total_cost_usd: Decimal
    mean_latency_ms: int


def compare_models(
    calls: Sequence[LLMCall],
) -> tuple[list[ModelStats], dict[tuple[str, str], tuple[int, int]]]:
    """Per-model cost/latency/validity, and pairwise agreement: for each trigger two models both answered, did
    they reach the same set of (symbol, action) conclusions? Returns {(a, b): (agreed, compared)}."""
    stats: list[ModelStats] = []
    for name in sorted({c.provider for c in calls}):
        mine = [c for c in calls if c.provider == name]
        stats.append(
            ModelStats(
                provider=name,
                calls=len(mine),
                invalid=sum(1 for c in mine if not c.valid),
                enters=sum(1 for c in mine for d in c.decisions or [] if d.get("action") == "enter"),
                total_cost_usd=sum((c.cost_usd for c in mine), ZERO),
                mean_latency_ms=sum(c.latency_ms for c in mine) // len(mine),
            )
        )
    by_trigger: dict[tuple[str, str], dict[str, frozenset[tuple[str, str]]]] = defaultdict(dict)
    for c in calls:
        if c.valid and c.decisions is not None and c.trigger:
            by_trigger[(c.strategy_id, c.trigger)][c.provider] = frozenset(
                (d["symbol"], d["action"]) for d in c.decisions if d.get("action") != "hold"
            )
    agreement: dict[tuple[str, str], tuple[int, int]] = {}
    for answers in by_trigger.values():
        names = sorted(answers)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                agreed, total = agreement.get((a, b), (0, 0))
                agreement[(a, b)] = (agreed + (answers[a] == answers[b]), total + 1)
    return stats, agreement


def format_report(
    scores: Sequence[ProviderScore],
    stats: Sequence[ModelStats],
    agreement: Mapping[tuple[str, str], tuple[int, int]],
    horizon_bars: int,
) -> str:
    lines = [f"Forward results ({horizon_bars}-bar horizon, entry at the next open, vs same-symbol baseline)"]
    for s in scores:
        if s.n == 0:
            lines.append(
                f"  {s.provider}: {s.verdict} ({s.contaminated} flagged as possibly leaked, {s.pending} pending)"
            )
            continue
        assert s.hit_rate is not None and s.mean_return is not None and s.mean_excess is not None
        ci = f" CI [{s.excess_ci[0]:.2%}, {s.excess_ci[1]:.2%}]" if s.excess_ci else ""
        lines.append(f"  {s.provider}: n={s.n} hit={s.hit_rate:.0%} mean={s.mean_return:.2%} "
                     f"excess={s.mean_excess:.2%}{ci}  pending={s.pending} leaked={s.contaminated}")  # fmt: skip
        lines.append(f"      verdict: {s.verdict}")
        for label, n, hit in s.calibration:
            lines.append(f"      confidence {label}: n={n} hit={'-' if hit is None else f'{hit:.0%}'}")
    lines.append("Model comparison")
    for m in stats:
        lines.append(f"  {m.provider}: calls={m.calls} invalid={m.invalid} enters={m.enters} "
                     f"cost=${m.total_cost_usd:.4f} latency={m.mean_latency_ms}ms")  # fmt: skip
    for (a, b), (agreed, total) in sorted(agreement.items()):
        lines.append(f"  {a} vs {b}: agreed on {agreed}/{total} shared triggers")
    return "\n".join(lines)
