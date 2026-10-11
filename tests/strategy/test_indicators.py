"""Indicators: hand-computed values, an independent reference implementation, warm-up, and prefix stability."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from engine.strategy import indicators as ind
from tests.helpers import D, bars_from_closes


def dec(*xs: float | int | str) -> list:  # type: ignore[type-arg]
    return [D(str(x)) for x in xs]


def test_sma_and_warmup() -> None:
    assert ind.sma(dec(1, 2, 3, 4, 5), 3) == [None, None, D(2), D(3), D(4)]
    assert ind.sma(dec(1, 2), 3) == [None, None]


def test_ema_is_seeded_with_the_sma_then_recursive() -> None:
    # n=3 -> alpha 0.5; seed = mean(1,2,3) = 2; then 0.5*4 + 0.5*2 = 3; 0.5*5 + 0.5*3 = 4
    assert ind.ema(dec(1, 2, 3, 4, 5), 3) == [None, None, D(2), D(3), D(4)]


def test_stdev_population_classic_example() -> None:
    out = ind.stdev(dec(2, 4, 4, 4, 5, 5, 7, 9), 8)
    assert out[:-1] == [None] * 7 and out[-1] == D(2)


def test_highest_lowest_roc() -> None:
    v = dec(5, 3, 8, 6, 2)
    assert ind.highest(v, 3) == [None, None, D(8), D(8), D(8)]
    assert ind.lowest(v, 3) == [None, None, D(3), D(3), D(2)]
    assert ind.roc(dec(100, 110, 121), 1) == [None, D("0.1"), D("0.1")]


def test_atr_of_constant_range_is_that_range() -> None:
    bars = bars_from_closes([10] * 20, spread="1")  # high 11, low 9, close 10 -> TR = 2 every bar
    out = ind.atr(bars, 5)
    assert out[:5] == [None] * 5 and all(x == D(2) for x in out[5:])


def test_atr_accounts_for_gaps() -> None:
    bars = bars_from_closes([10, 10, 10, 30], spread="0")  # a +20 gap on the last bar: TR = 20
    assert ind.atr(bars, 2)[3] == (D(0) * 1 + D(20)) / 2  # Wilder smoothing of TR [0,0,20]: (0*(2-1)+20)/2


def reference_rsi(closes: list[float], n: int) -> list[float | None]:
    """An independent float implementation of Wilder's RSI to check the Decimal one against."""
    out: list[float | None] = [None] * len(closes)
    gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, len(closes))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    for i in range(n, len(closes)):
        if i > n:
            ag = (ag * (n - 1) + gains[i - 1]) / n
            al = (al * (n - 1) + losses[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


STOCKCHARTS = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
]  # fmt: skip


def test_rsi_matches_an_independent_implementation_and_the_textbook_range() -> None:
    mine = ind.rsi(dec(*STOCKCHARTS), 14)
    ref = reference_rsi(STOCKCHARTS, 14)
    assert mine[:14] == [None] * 14
    for a, b in zip(mine[14:], ref[14:], strict=True):
        assert a is not None and b is not None and math.isclose(float(a), b, rel_tol=1e-9)
    assert 70 < float(mine[14]) < 71  # type: ignore[arg-type]  # the well-known first value, about 70.5


def test_rsi_edge_cases() -> None:
    assert ind.rsi(dec(*range(1, 20)), 5)[-1] == D(100)  # only gains
    assert ind.rsi(dec(*range(20, 1, -1)), 5)[-1] == D(0)  # only losses
    assert ind.rsi(dec(*([7] * 20)), 5)[-1] == D(50)  # flat is neutral, not a divide-by-zero


def test_bollinger_bands_bracket_the_mean_and_collapse_on_flat_prices() -> None:
    bars = bars_from_closes([10, 12, 11, 13, 12, 14, 13, 15])
    up = ind.compute(ind.IndicatorSpec(type="bb_upper", period=4, k=D(2)), bars)
    lo = ind.compute(ind.IndicatorSpec(type="bb_lower", period=4, k=D(2)), bars)
    mid = ind.sma([b.close for b in bars], 4)
    for u, m, low in zip(up[3:], mid[3:], lo[3:], strict=True):
        assert low < m < u and abs((u - m) - (m - low)) < D("1e-20")  # type: ignore[operator]
    flat = bars_from_closes([5] * 6)
    assert ind.compute(ind.IndicatorSpec(type="bb_upper", period=3), flat)[-1] == D(5)


@pytest.mark.parametrize(
    ("spec", "first_defined"),
    [
        (dict(type="sma", period=5), 4),
        (dict(type="ema", period=5), 4),
        (dict(type="rsi", period=5), 5),
        (dict(type="atr", period=5), 5),
        (dict(type="stdev", period=5), 4),
        (dict(type="highest", period=5), 4),
        (dict(type="roc", period=5), 5),
        (dict(type="bb_upper", period=5), 4),
    ],
)
def test_warmup_property_matches_the_first_defined_index(spec: dict[str, object], first_defined: int) -> None:
    s = ind.IndicatorSpec.model_validate(spec)
    out = ind.compute(s, bars_from_closes(range(10, 40)))
    assert s.warmup == first_defined + 1
    assert out[first_defined - 1 if first_defined else 0] is None or first_defined == 0
    assert out[first_defined] is not None and all(x is None for x in out[:first_defined])


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(min_value=5, max_value=500), min_size=30, max_size=80),
    st.sampled_from(
        ["sma", "ema", "rsi", "atr", "stdev", "highest", "lowest", "roc", "bb_upper", "bb_lower"]
    ),
    st.integers(min_value=2, max_value=12),
    st.integers(min_value=15, max_value=29),
)
def test_appending_future_bars_never_changes_past_values(closes, kind, period, cut) -> None:  # type: ignore[no-untyped-def]
    """The no-lookahead guarantee at the indicator level: values up to bar `cut` are identical whether or not
    later bars exist."""
    bars = bars_from_closes(closes)
    spec = ind.IndicatorSpec(type=kind, period=period)
    full = ind.compute(spec, bars)
    prefix = ind.compute(spec, bars[:cut])
    assert prefix == full[:cut]


def test_invalid_specs_are_rejected() -> None:
    for bad in (
        dict(type="sma", period=0),
        dict(type="rsi", period=1),
        dict(type="nonsense", period=5),
        dict(type="atr", period=5, source="open"),
        dict(type="sma", period=5, source="vwap"),
        dict(type="sma", period=5, surprise=1),
    ):
        with pytest.raises(ValidationError):
            ind.IndicatorSpec.model_validate(bad)
