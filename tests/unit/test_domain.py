from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from engine.domain.events import OrderFilled, RiskDecisionMade
from engine.domain.types import Side
from engine.eventlog.base import decode, make_event
from tests.helpers import T0, make_intent


def test_floats_are_rejected_for_money() -> None:
    with pytest.raises(ValidationError):
        make_intent(quantity=1.5)
    with pytest.raises(ValidationError):
        make_intent(stop_loss_price=90.0)


def test_bool_is_not_a_quantity() -> None:
    with pytest.raises(ValidationError):
        make_intent(quantity=True)


def test_decimal_strings_and_ints_are_accepted() -> None:
    i = make_intent(quantity="2.5", limit_price=101)
    assert i.quantity == Decimal("2.5") and i.limit_price == Decimal(101)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "abc"])
def test_non_finite_or_garbage_decimals_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        make_intent(quantity=bad)


def test_nonpositive_quantity_rejected() -> None:
    with pytest.raises(ValidationError):
        make_intent(quantity=0)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        make_intent(created_at=datetime(2026, 1, 5, 12, 0), expires_at=datetime(2026, 1, 5, 13, 0))


def test_expiry_must_follow_creation() -> None:
    with pytest.raises(ValidationError):
        make_intent(expires_at=T0 - timedelta(seconds=1))


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        make_intent(confidence="1.5")


def test_unknown_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        make_intent(surprise="x")


def test_models_are_frozen() -> None:
    i = make_intent()
    with pytest.raises(ValidationError):
        i.quantity = Decimal(5)  # type: ignore[misc]


def test_event_roundtrip_preserves_decimals_and_nested_models() -> None:
    from engine.domain.events import IntentReceived

    i = make_intent(quantity="0.12345678")
    ev = IntentReceived(account_id="acct", intent=i)
    new = make_event(ev, stream_id="intent:x", ts=T0)
    # payload is JSON-safe (no Decimal objects)
    import json

    json.dumps(new.payload)
    from engine.eventlog.memory import InMemoryEventStore

    stored = InMemoryEventStore().append([new])[0]
    back = decode(stored)
    assert isinstance(back, IntentReceived) and back.intent == i


def test_event_models_require_account_id_and_registration() -> None:
    with pytest.raises(ValidationError):
        OrderFilled.model_validate({"order_id": "o"})
    with pytest.raises(ValueError, match="unregistered"):
        make_event(make_intent(), stream_id="x", ts=T0)  # an Intent is not an event


def test_decision_event_defaults() -> None:
    d = RiskDecisionMade(account_id="a", intent_id="i", decision="reject", code="X")
    assert d.trips == () and d.approved_quantity is None
    assert Side.BUY.value == "BUY"
