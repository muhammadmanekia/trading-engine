from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.risk.config import DIRECTIONS, RiskConfig, RiskOverride, disposable_money_preset, merge

D = Decimal


def test_override_fields_mirror_config_fields() -> None:
    assert set(DIRECTIONS) == set(RiskOverride.model_fields)
    assert set(DIRECTIONS) <= set(RiskConfig.model_fields)


def test_stricter_cap_wins_and_looser_is_ignored() -> None:
    base = disposable_money_preset()
    tighter = merge(base, RiskOverride(max_position_pct=D("0.10")))
    assert tighter.max_position_pct == D("0.10")
    looser = merge(base, RiskOverride(max_position_pct=D("0.99")))
    assert looser.max_position_pct == base.max_position_pct


def test_floor_direction_is_inverted() -> None:
    base = disposable_money_preset()
    assert merge(base, RiskOverride(cash_reserve_pct=D("0.5"))).cash_reserve_pct == D("0.5")
    assert merge(base, RiskOverride(cash_reserve_pct=D("0.0"))).cash_reserve_pct == base.cash_reserve_pct


def test_stop_loss_requirement_cannot_be_disabled() -> None:
    base = disposable_money_preset()
    assert merge(base, RiskOverride(require_stop_loss=False)).require_stop_loss is True


def test_strategy_budget_none_is_loosest() -> None:
    base = disposable_money_preset()
    assert base.strategy_budget is None
    assert merge(base, RiskOverride(strategy_budget=D(500))).strategy_budget == 500
    again = merge(merge(base, RiskOverride(strategy_budget=D(500))), RiskOverride(strategy_budget=D(900)))
    assert again.strategy_budget == 500


def test_layers_apply_in_order_global_account_strategy() -> None:
    base = disposable_money_preset()
    out = merge(base, RiskOverride(max_position_pct=D("0.2")), RiskOverride(max_position_pct=D("0.1")))
    assert out.max_position_pct == D("0.1")
    assert merge(base, None, None) == base


def test_config_is_frozen_and_rejects_floats() -> None:
    with pytest.raises(Exception):  # noqa: B017
        RiskOverride(max_position_pct=0.1)  # type: ignore[arg-type]


def test_preset_keeps_structural_protections_on() -> None:
    p = disposable_money_preset()
    assert p.require_stop_loss and p.cash_reserve_pct > 0 and p.daily_loss_cap_pct < 1


_dec = st.decimals(min_value=D(0), max_value=D(1000), places=2)
_int = st.integers(1, 10_000)


@settings(max_examples=100, deadline=None)
@given(
    pos=_dec,
    reserve=_dec,
    age=_int,
    flag=st.booleans(),
    budget=st.one_of(st.none(), _dec),
)
def test_property_inner_layer_never_loosens(pos, reserve, age, flag, budget) -> None:  # type: ignore[no-untyped-def]
    base = disposable_money_preset().model_copy(update={"strategy_budget": D(300)})
    merged = merge(
        base,
        RiskOverride(
            max_position_pct=pos,
            cash_reserve_pct=reserve,
            max_data_age_s=age,
            require_stop_loss=flag,
            strategy_budget=budget,
        ),
    )
    assert merged.max_position_pct <= base.max_position_pct
    assert merged.cash_reserve_pct >= base.cash_reserve_pct
    assert merged.max_data_age_s <= base.max_data_age_s
    assert merged.require_stop_loss >= base.require_stop_loss
    assert merged.strategy_budget is not None and merged.strategy_budget <= D(300)
