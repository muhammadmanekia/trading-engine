"""The risk engine's output. Rejections always carry a stable reason code."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from engine.domain.events import BreakerTrip

Kind = Literal["approve", "resize", "reject"]


@dataclass(frozen=True)
class Decision:
    kind: Kind
    intent_id: str
    code: str
    detail: str = ""
    quantity: Decimal | None = None  # approved quantity (approve/resize)
    checks_passed: tuple[str, ...] = ()
    trips: tuple[BreakerTrip, ...] = ()
    reference_price: Decimal | None = None

    @property
    def allowed(self) -> bool:
        return self.kind != "reject"

    @staticmethod
    def reject(intent_id: str, code: str, detail: str = "", **kw: object) -> Decision:
        return Decision(kind="reject", intent_id=intent_id, code=code, detail=detail, **kw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Reject:
    code: str
    detail: str = ""


@dataclass(frozen=True)
class Resize:
    quantity: Decimal
    code: str
    detail: str = ""


Outcome = Reject | Resize | None


@dataclass
class Trace:
    """Mutable scratch collected while checks run."""

    passed: list[str] = field(default_factory=list)
    trips: list[BreakerTrip] = field(default_factory=list)
