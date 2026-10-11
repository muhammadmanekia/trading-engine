"""Technical indicators over a window of closed bars, in exact Decimal arithmetic.

Every function is pure and returns a list aligned with the input where early values (the warm-up) are None.
Recursive indicators (EMA, RSI, ATR) are seeded from the first `period` values of the window, so the result
depends on where the window starts. That is deliberate: the evaluator always recomputes over the same
fixed-length trailing window, which makes live and backtest values identical and lookahead impossible.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from engine.domain.types import Frozen, PosDec
from engine.marketdata.history import Bar

ZERO = Decimal(0)
ONE = Decimal(1)
HUNDRED = Decimal(100)
Series = list[Decimal | None]

Source = Literal["open", "high", "low", "close", "volume", "hl2", "hlc3", "ohlc4"]
IndicatorType = Literal[
    "sma", "ema", "rsi", "atr", "stdev", "highest", "lowest", "roc", "bb_upper", "bb_lower"
]


class IndicatorSpec(Frozen):
    type: IndicatorType
    period: int = Field(ge=1, le=5000)
    source: Source = "close"
    k: PosDec = Decimal(2)  # Bollinger band width in standard deviations

    @model_validator(mode="after")
    def _check(self) -> IndicatorSpec:
        if self.type in ("rsi", "atr", "roc", "bb_upper", "bb_lower", "stdev") and self.period < 2:
            raise ValueError(f"{self.type} needs period >= 2")
        if self.type == "atr" and self.source != "close":
            raise ValueError("atr is computed from high/low/close; do not set a source")
        return self

    @property
    def warmup(self) -> int:
        """Bars needed before the first defined value."""
        return self.period + 1 if self.type in ("rsi", "atr", "roc") else self.period


def _price(b: Bar, source: Source) -> Decimal:
    match source:
        case "open":
            return b.open
        case "high":
            return b.high
        case "low":
            return b.low
        case "close":
            return b.close
        case "volume":
            return b.volume
        case "hl2":
            return (b.high + b.low) / 2
        case "hlc3":
            return (b.high + b.low + b.close) / 3
        case "ohlc4":
            return (b.open + b.high + b.low + b.close) / 4


def source_values(bars: Sequence[Bar], source: Source) -> list[Decimal]:
    return [_price(b, source) for b in bars]


def sma(values: Sequence[Decimal], n: int) -> Series:
    out: Series = [None] * len(values)
    if len(values) < n:
        return out
    window = sum(values[:n], ZERO)
    out[n - 1] = window / n
    for i in range(n, len(values)):
        window += values[i] - values[i - n]
        out[i] = window / n
    return out


def ema(values: Sequence[Decimal], n: int) -> Series:
    out: Series = [None] * len(values)
    if len(values) < n:
        return out
    alpha = Decimal(2) / (n + 1)
    prev = sum(values[:n], ZERO) / n
    out[n - 1] = prev
    for i in range(n, len(values)):
        prev = alpha * values[i] + (ONE - alpha) * prev
        out[i] = prev
    return out


def rsi(values: Sequence[Decimal], n: int) -> Series:
    """Wilder's RSI. A window with no movement at all is neutral (50); no losses is 100."""
    out: Series = [None] * len(values)
    if len(values) < n + 1:
        return out
    gains = [max(values[i] - values[i - 1], ZERO) for i in range(1, len(values))]
    losses = [max(values[i - 1] - values[i], ZERO) for i in range(1, len(values))]
    avg_gain = sum(gains[:n], ZERO) / n
    avg_loss = sum(losses[:n], ZERO) / n

    def value() -> Decimal:
        if avg_loss == 0:
            return HUNDRED if avg_gain > 0 else Decimal(50)
        return HUNDRED - HUNDRED / (ONE + avg_gain / avg_loss)

    out[n] = value()
    for i in range(n + 1, len(values)):
        avg_gain = (avg_gain * (n - 1) + gains[i - 1]) / n
        avg_loss = (avg_loss * (n - 1) + losses[i - 1]) / n
        out[i] = value()
    return out


def atr(bars: Sequence[Bar], n: int) -> Series:
    out: Series = [None] * len(bars)
    if len(bars) < n + 1:
        return out
    tr = [
        max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        for i in range(1, len(bars))
    ]
    prev = sum(tr[:n], ZERO) / n
    out[n] = prev
    for i in range(n + 1, len(bars)):
        prev = (prev * (n - 1) + tr[i - 1]) / n
        out[i] = prev
    return out


def stdev(values: Sequence[Decimal], n: int) -> Series:
    """Population standard deviation (what Bollinger Bands use)."""
    out: Series = [None] * len(values)
    for i in range(n - 1, len(values)):
        w = values[i - n + 1 : i + 1]
        mean = sum(w, ZERO) / n
        out[i] = (sum(((x - mean) ** 2 for x in w), ZERO) / n).sqrt()
    return out


def highest(values: Sequence[Decimal], n: int) -> Series:
    return [None if i < n - 1 else max(values[i - n + 1 : i + 1]) for i in range(len(values))]


def lowest(values: Sequence[Decimal], n: int) -> Series:
    return [None if i < n - 1 else min(values[i - n + 1 : i + 1]) for i in range(len(values))]


def roc(values: Sequence[Decimal], n: int) -> Series:
    """Rate of change over n bars, as a fraction (0.05 = +5%)."""
    return [None if i < n else values[i] / values[i - n] - ONE for i in range(len(values))]


def compute(spec: IndicatorSpec, bars: Sequence[Bar]) -> Series:
    if spec.type == "atr":
        return atr(bars, spec.period)
    values = source_values(bars, spec.source)
    match spec.type:
        case "sma":
            return sma(values, spec.period)
        case "ema":
            return ema(values, spec.period)
        case "rsi":
            return rsi(values, spec.period)
        case "stdev":
            return stdev(values, spec.period)
        case "highest":
            return highest(values, spec.period)
        case "lowest":
            return lowest(values, spec.period)
        case "roc":
            return roc(values, spec.period)
        case "bb_upper" | "bb_lower":
            mid, sd = sma(values, spec.period), stdev(values, spec.period)
            sign = ONE if spec.type == "bb_upper" else -ONE
            return [
                None if m is None or s is None else m + sign * spec.k * s
                for m, s in zip(mid, sd, strict=True)
            ]
    raise ValueError(f"unknown indicator {spec.type}")  # pragma: no cover
