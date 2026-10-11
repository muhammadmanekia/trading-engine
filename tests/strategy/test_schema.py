"""Strategy definitions: parsing, validation, venue rules, hashing, overrides."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from engine.broker.base import ProtectionMode
from engine.domain.types import AssetClass
from engine.strategy.schema import (
    All,
    Compare,
    Cross,
    Num,
    Ref,
    StrategyError,
    load_strategy,
    parse_condition,
    parse_strategy,
    validate_for_venue,
    with_overrides,
)

BASE: dict[str, Any] = {
    "id": "ema-cross",
    "universe": ["BTC-USD"],
    "timeframe": "1h",
    "indicators": {
        "fast": {"type": "ema", "period": 12},
        "slow": {"type": "ema", "period": 48},
        "atr": {"type": "atr", "period": 14},
    },
    "entry": {"all": [{"cross_above": ["fast", "slow"]}, {"gt": ["close", {"mul": ["slow", 0.98]}]}]},
    "exit": {"cross_below": ["fast", "slow"]},
    "stop": {"type": "atr", "multiple": 2.5},
    "sizing": {"type": "risk_fraction", "risk_pct": 0.01},
}


def make(**changes: Any) -> dict[str, Any]:
    return {**BASE, **changes}


def test_a_valid_strategy_parses_with_exact_decimals() -> None:
    s = parse_strategy(BASE)
    assert s.stop.multiple == Decimal("2.5") and s.sizing.risk_pct == Decimal("0.01")
    assert s.asset_class is AssetClass.CRYPTO and s.timeframe == "1h"
    entry = s.entry_condition
    assert (
        isinstance(entry, All) and isinstance(entry.items[0], Cross) and isinstance(entry.items[1], Compare)
    )
    assert entry.items[1].right.right == Num(Decimal("0.98"))  # type: ignore[union-attr]  # not 0.9800000000000000266


def test_warmup_covers_the_slowest_indicator_plus_cross_lookback() -> None:
    s = parse_strategy(BASE)
    assert (
        s.warmup == 48 + 1 and s.window >= 300
    )  # ema48 warmup, +1 bar because a cross compares to the prior bar
    assert parse_strategy(make(lookback=120)).window == 120


def test_lagged_references_extend_the_warmup() -> None:
    entry = {"gt": ["close", {"ref": "slow", "lag": 5}]}
    assert parse_strategy(make(entry=entry)).warmup == 48 + 5


def test_hash_is_stable_and_changes_with_any_parameter() -> None:
    a, b = parse_strategy(BASE), parse_strategy(make())
    assert a.hash == b.hash and len(a.hash) == 16
    assert parse_strategy(make(stop={"type": "atr", "multiple": 3})).hash != a.hash


@pytest.mark.parametrize(
    "bad",
    [
        make(id="Has Spaces"),
        make(universe=[]),
        make(universe=["BTC-USD", "BTC-USD"]),
        make(universe=["AAPL", "BTC-USD"]),  # mixed asset classes
        make(timeframe="5m"),
        make(entry={"gt": ["close", "ghost"]}),  # undefined series
        make(exit={"gt": ["close", "ghost"]}),
        make(indicators={"close": {"type": "sma", "period": 5}}),  # shadows a price field
        make(indicators={"fast": {"type": "ema", "period": 12}}),  # atr stop but no atr indicator
        make(stop={"type": "atr"}),  # missing multiple
        make(stop={"type": "percent", "pct": 1.5}),
        make(sizing={"type": "risk_fraction"}),
        make(sizing={"type": "risk_fraction", "risk_pct": 5}),
        make(surprise=1),
    ],
)
def test_invalid_strategies_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(StrategyError):
        parse_strategy(bad)


@pytest.mark.parametrize(
    "raw",
    [
        {"gt": ["close"]},  # wrong arity
        {"gt": ["close", 1], "lt": ["close", 2]},  # two keys
        {"between": ["close", 1, 2]},  # unknown operator
        {"all": []},  # empty
        {"gt": ["Close", 1]},  # bad name
        {"gt": [True, 1]},  # booleans are not numbers
        {"gt": [{"ref": "close", "lag": -1}, 1]},
        {"gt": [{"div": ["close"]}, 1]},
        "close > 1",  # no expression strings, ever
    ],
)
def test_malformed_conditions_are_rejected(raw: Any) -> None:
    with pytest.raises(StrategyError):
        parse_condition(raw)


def test_expressions_are_bounded_in_depth_and_size() -> None:
    deep: Any = {"gt": ["close", 1]}
    for _ in range(12):
        deep = {"not": deep}
    with pytest.raises(StrategyError, match="deeper"):
        parse_condition(deep)
    wide = {"all": [{"gt": ["close", i]} for i in range(60)]}
    with pytest.raises(StrategyError, match="too large"):
        parse_condition(wide)


def test_numbers_may_be_written_as_ints_floats_or_numeric_strings() -> None:
    c = parse_condition({"gt": ["rsi", 70]})
    assert isinstance(c, Compare) and c.right == Num(Decimal(70)) and c.left == Ref("rsi")
    assert parse_condition({"gt": ["rsi", "0.5"]}).right == Num(Decimal("0.5"))  # type: ignore[union-attr]


def test_venue_rules_fail_at_load_time_not_at_order_time() -> None:
    with_tp = parse_strategy(make(take_profit={"type": "r_multiple", "multiple": 2}))
    validate_for_venue(
        with_tp, ProtectionMode.NATIVE_BRACKET, AssetClass.CRYPTO
    )  # a bracket venue can hold both
    with pytest.raises(StrategyError, match="take_profit"):
        validate_for_venue(with_tp, ProtectionMode.SEPARATE_ORDERS, AssetClass.CRYPTO)  # Robinhood cannot
    with pytest.raises(StrategyError, match="trades"):
        validate_for_venue(parse_strategy(BASE), ProtectionMode.SEPARATE_ORDERS, AssetClass.STOCK)


def test_overrides_are_revalidated_and_can_reach_into_conditions() -> None:
    s = parse_strategy(BASE)
    t = with_overrides(s, {"indicators.fast.period": 8, "stop.multiple": 3, "entry.all.1.gt.1.mul.1": "0.95"})
    assert t.indicators["fast"].period == 8 and t.stop.multiple == 3 and t.hash != s.hash
    assert s.indicators["fast"].period == 12  # the original is untouched
    with pytest.raises(StrategyError, match="does not exist"):
        with_overrides(s, {"indicators.nope.period": 3})
    with pytest.raises(StrategyError):
        with_overrides(s, {"stop.multiple": -1})  # an override can never produce an invalid strategy


def test_toml_files_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "s.toml"
    f.write_text(
        'id = "rsi-dip"\nuniverse = ["AAPL"]\ntimeframe = "1d"\n'
        '[indicators]\nrsi = { type = "rsi", period = 14 }\natr = { type = "atr", period = 14 }\n'
        '[entry]\nlt = ["rsi", 30.5]\n[exit]\ngt = ["rsi", 55]\n'
        '[stop]\ntype = "percent"\npct = 0.05\n[sizing]\ntype = "fixed_notional"\nnotional = 1000\n'
    )
    s = load_strategy(f)
    assert s.asset_class is AssetClass.STOCK and s.entry_condition.right == Num(Decimal("30.5"))  # type: ignore[union-attr]
