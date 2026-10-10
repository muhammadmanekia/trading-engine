"""Core value types. Money and quantities are Decimal only; floats are rejected at the boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class AssetClass(StrEnum):
    STOCK = "STOCK"
    CRYPTO = "CRYPTO"


class Mode(StrEnum):
    PAPER = "PAPER"
    LIVE = "LIVE"


def _to_decimal(v: Any) -> Decimal:
    if isinstance(v, bool | float):
        raise ValueError("floats/bools are not allowed for money or quantity; use Decimal or str")
    if isinstance(v, Decimal):
        d = v
    elif isinstance(v, int | str):
        try:
            d = Decimal(v)
        except InvalidOperation as exc:
            raise ValueError(f"not a decimal: {v!r}") from exc
    else:
        raise ValueError(f"unsupported type for decimal: {type(v).__name__}")
    if not d.is_finite():
        raise ValueError("decimal must be finite")
    return d


def _to_utc(v: Any) -> datetime:
    if isinstance(v, str):
        v = datetime.fromisoformat(v)
    if not isinstance(v, datetime):
        raise ValueError("expected datetime")
    if v.tzinfo is None:
        raise ValueError("naive datetimes are not allowed; use timezone-aware UTC")
    return v.astimezone(UTC)


Dec = Annotated[Decimal, BeforeValidator(_to_decimal)]
PosDec = Annotated[Decimal, BeforeValidator(_to_decimal), Field(gt=0)]
NonNegDec = Annotated[Decimal, BeforeValidator(_to_decimal), Field(ge=0)]
UtcDatetime = Annotated[datetime, BeforeValidator(_to_utc)]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
