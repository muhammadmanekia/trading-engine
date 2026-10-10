"""Risk configuration, layered global -> account -> strategy. The stricter value always wins.

Every field has a direction: LOWER limits are stricter (caps), HIGHER floors are stricter (minimums),
and booleans are stricter when True. An inner layer can therefore never loosen an outer one.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import Field

from engine.domain.types import Dec, Frozen, NonNegDec

# field -> "cap" (smaller is stricter) | "floor" (larger is stricter) | "flag" (True is stricter)
DIRECTIONS: dict[str, str] = {
    "max_data_age_s": "cap",
    "max_orders_per_minute": "cap",
    "daily_loss_cap_pct": "cap",
    "strategy_drawdown_limit_pct": "cap",
    "max_consecutive_losses": "cap",
    "max_position_pct": "cap",
    "max_total_exposure_pct": "cap",
    "max_slippage_bps": "cap",
    "max_follower_price_drift_pct": "cap",
    "strategy_budget": "cap",
    "cash_reserve_pct": "floor",
    "min_price": "floor",
    "min_avg_daily_volume_usd": "floor",
    "min_crypto_liquidity_usd": "floor",
    "min_crypto_market_cap_usd": "floor",
    "min_order_notional": "floor",
    "follower_haircut_pct_per_day": "floor",
    "require_stop_loss": "flag",
}


class RiskConfig(Frozen):
    """Fully-specified limits. Percentages are fractions (0.10 == 10%)."""

    max_data_age_s: int = Field(gt=0)
    max_orders_per_minute: int = Field(gt=0)
    daily_loss_cap_pct: NonNegDec
    strategy_drawdown_limit_pct: NonNegDec
    max_consecutive_losses: int = Field(gt=0)
    max_position_pct: NonNegDec
    max_total_exposure_pct: NonNegDec
    max_slippage_bps: NonNegDec
    max_follower_price_drift_pct: NonNegDec
    strategy_budget: NonNegDec | None = None  # None == no explicit budget (loosest)
    cash_reserve_pct: NonNegDec
    min_price: NonNegDec
    min_avg_daily_volume_usd: NonNegDec
    min_crypto_liquidity_usd: NonNegDec
    min_crypto_market_cap_usd: NonNegDec
    min_order_notional: NonNegDec
    follower_haircut_pct_per_day: NonNegDec
    require_stop_loss: bool = True


class RiskOverride(Frozen):
    """A partial layer (account or strategy). Unset fields inherit."""

    max_data_age_s: int | None = None
    max_orders_per_minute: int | None = None
    daily_loss_cap_pct: Dec | None = None
    strategy_drawdown_limit_pct: Dec | None = None
    max_consecutive_losses: int | None = None
    max_position_pct: Dec | None = None
    max_total_exposure_pct: Dec | None = None
    max_slippage_bps: Dec | None = None
    max_follower_price_drift_pct: Dec | None = None
    strategy_budget: Dec | None = None
    cash_reserve_pct: Dec | None = None
    min_price: Dec | None = None
    min_avg_daily_volume_usd: Dec | None = None
    min_crypto_liquidity_usd: Dec | None = None
    min_crypto_market_cap_usd: Dec | None = None
    min_order_notional: Dec | None = None
    follower_haircut_pct_per_day: Dec | None = None
    require_stop_loss: bool | None = None


def _stricter(direction: str, a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    if direction == "cap":
        return min(a, b)
    if direction == "floor":
        return max(a, b)
    return bool(a) or bool(b)


def merge(base: RiskConfig, *layers: RiskOverride | None) -> RiskConfig:
    """Apply override layers in order; for each field the stricter value wins."""
    values: dict[str, Any] = base.model_dump()
    for layer in layers:
        if layer is None:
            continue
        for name, direction in DIRECTIONS.items():
            values[name] = _stricter(direction, values[name], getattr(layer, name))
    return RiskConfig.model_validate(values)


def replace(base: RiskConfig, override: RiskOverride | None) -> RiskConfig:
    """The GLOBAL layer: set fields outright (loosening or tightening the preset). Account and strategy layers
    then go through `merge`, which can only tighten."""
    if override is None:
        return base
    changes = {k: v for k, v in override.model_dump().items() if v is not None}
    return RiskConfig.model_validate({**base.model_dump(), **changes})


def disposable_money_preset() -> RiskConfig:
    """Loose limits for money the owner is willing to lose. Structural protections stay on."""
    return RiskConfig(
        max_data_age_s=120,
        max_orders_per_minute=20,
        daily_loss_cap_pct=Decimal("0.25"),
        strategy_drawdown_limit_pct=Decimal("0.40"),
        max_consecutive_losses=8,
        max_position_pct=Decimal("0.30"),
        max_total_exposure_pct=Decimal("0.90"),
        max_slippage_bps=Decimal("150"),
        max_follower_price_drift_pct=Decimal("0.05"),
        strategy_budget=None,
        cash_reserve_pct=Decimal("0.05"),
        min_price=Decimal("1"),
        min_avg_daily_volume_usd=Decimal("1000000"),
        min_crypto_liquidity_usd=Decimal("250000"),
        min_crypto_market_cap_usd=Decimal("10000000"),
        min_order_notional=Decimal("1"),
        follower_haircut_pct_per_day=Decimal("0.02"),
        require_stop_loss=True,
    )
