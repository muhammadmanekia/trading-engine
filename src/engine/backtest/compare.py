"""Compare strategies the same way, against buy-and-hold, without fooling ourselves.

Protocol (see docs/superpowers/specs/2026-10-13-strategy-evaluation-design.md):
* rolling windows BEFORE the exam start: choose parameters on the preceding `train_bars`, score on the next
  `test_bars`, then step forward by `test_bars`;
* one final exam from the exam start to the end of the data, run only on request and recorded in a ledger so a
  strategy edited after seeing its exam is flagged;
* a strategy "beats buy-and-hold" only if it wins the exam, wins at least two thirds of the windows, trades enough
  to be more than luck, and still wins with spread, slippage and fees doubled.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

ZERO = Decimal(0)


@dataclass(frozen=True)
class Protocol:
    train_bars: int = 504  # about two years of daily bars
    test_bars: int = 126  # about six months
    min_trades: int = 30  # across all scored periods, or the sample is too thin to trust
    tune_min_trades: int = 10  # a parameter set must trade this much on its training window to be eligible
    drawdown_flag: Decimal = Decimal("0.40")  # shown, never a pass/fail condition
    cost_multiplier: Decimal = Decimal(2)  # the cost-stress rerun


@dataclass(frozen=True)
class WindowSpan:
    train_lo: int  # indexes into the strategy's bar timeline
    test_lo: int
    test_hi: int  # exclusive


def required_wins(windows: int) -> int:
    """The smallest number of winning windows that is at least two thirds of `windows`."""
    return (2 * windows + 2) // 3


def rolling_spans(bars_before_exam: int, p: Protocol) -> list[WindowSpan]:
    spans: list[WindowSpan] = []
    test_lo = p.train_bars
    while test_lo + p.test_bars <= bars_before_exam:
        spans.append(WindowSpan(test_lo - p.train_bars, test_lo, test_lo + p.test_bars))
        test_lo += p.test_bars
    return spans
