"""An Intent is what a signal source wants to do. It carries no authority; the risk engine decides."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from engine.domain.types import AssetClass, Dec, Frozen, PosDec, Side, UtcDatetime


class CopyInfo(Frozen):
    """Present when the intent mirrors a followed wallet/filing."""

    source: str
    source_price: PosDec
    source_trade_at: UtcDatetime


class Intent(Frozen):
    intent_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    asset_class: AssetClass
    side: Side
    quantity: PosDec
    limit_price: PosDec | None = None
    stop_loss_price: PosDec | None = None
    take_profit_price: PosDec | None = None
    confidence: Dec = Field(ge=0, le=1)
    rationale: str = ""
    source: str = Field(min_length=1)
    created_at: UtcDatetime
    expires_at: UtcDatetime
    copy_info: CopyInfo | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self
