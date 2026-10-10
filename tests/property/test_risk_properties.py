"""Property tests: no generated input may ever produce an approval that breaches a hard limit."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from engine.domain.types import Side
from engine.projections.account import AccountState, Position
from engine.risk.config import RiskConfig, disposable_money_preset
from engine.risk.engine import evaluate
from tests.helpers import ACCOUNT, T0, D, make_intent, market, stock, system

money = st.decimals(min_value=D(0), max_value=D(200_000), places=2)
price = st.decimals(min_value=D("0.5"), max_value=D(500), places=2)
qty = st.decimals(min_value=D("0.01"), max_value=D(1000), places=2)
pct = st.decimals(min_value=D("0.01"), max_value=D(1), places=2)


@st.composite
def scenario(draw):  # type: ignore[no-untyped-def]
    mkt_price = draw(price)
    cash = draw(money)
    held_qty = draw(st.decimals(min_value=D(0), max_value=D(500), places=2))
    avg = draw(price)
    other_qty = draw(st.decimals(min_value=D(0), max_value=D(300), places=2))
    state = AccountState(account_id=ACCOUNT, cash=cash)
    if held_qty:
        state.positions[("s1", "AAPL")] = Position(held_qty, avg)
    if other_qty:
        state.positions[("s2", "MSFT")] = Position(other_qty, D(100))
    state.kill_switch = draw(st.booleans())
    state.consecutive_losses["s1"] = draw(st.integers(0, 12))
    state.realized_by_day[T0.date().isoformat()] = -draw(
        st.decimals(min_value=D(0), max_value=D(50_000), places=2)
    )
    side = draw(st.sampled_from([Side.BUY, Side.SELL]))
    stop = draw(st.one_of(st.none(), st.decimals(min_value=D("0.01"), max_value=D(600), places=2)))
    limit = draw(st.one_of(st.none(), price))
    cfg = disposable_money_preset().model_copy(
        update={
            "max_position_pct": draw(pct),
            "max_total_exposure_pct": draw(pct),
            "cash_reserve_pct": draw(st.decimals(min_value=D(0), max_value=D("0.5"), places=2)),
            "strategy_budget": draw(st.one_of(st.none(), money)),
        }
    )
    stale = draw(st.booleans())
    inst = stock(price=str(mkt_price), at=T0 - timedelta(minutes=30 if stale else 0))
    intent = make_intent(
        side=side,
        quantity=draw(qty),
        stop_loss_price=stop,
        limit_price=limit,
        symbol="AAPL",
    )
    mkt = market(inst, stock("MSFT", 100))
    return intent, state, mkt, cfg, stale


@settings(max_examples=400, deadline=None)
@given(scenario())
def test_evaluate_never_raises_and_never_breaches_limits(sc) -> None:  # type: ignore[no-untyped-def]
    intent, state, mkt, cfg, stale = sc
    d = evaluate(intent, state, mkt, system(), cfg)

    assert d.kind in ("approve", "resize", "reject")
    if d.kind == "reject":
        assert d.code and d.quantity is None
        return

    assert d.quantity is not None and 0 < d.quantity <= intent.quantity

    if intent.side is Side.SELL:
        assert d.quantity <= state.position_qty("s1", "AAPL")
        return

    # --- entries: every hard limit must hold after the fill ---
    assert not state.kill_switch
    assert not stale
    assert intent.stop_loss_price is not None
    assert state.consecutive_losses["s1"] < cfg.max_consecutive_losses
    prices = mkt.price_map()
    equity: Decimal = state.equity(prices)
    ref = d.reference_price
    assert ref is not None
    notional = d.quantity * ref
    assert state.symbol_exposure("AAPL", prices) + notional <= cfg.max_position_pct * equity
    assert state.exposure(prices) + notional <= cfg.max_total_exposure_pct * equity
    assert state.cash - notional >= cfg.cash_reserve_pct * equity
    if cfg.strategy_budget is not None:
        assert state.strategy_exposure("s1", prices) + notional <= cfg.strategy_budget
    assert notional >= cfg.min_order_notional


@settings(max_examples=200, deadline=None)
@given(scenario())
def test_kill_switch_always_blocks_entries(sc) -> None:  # type: ignore[no-untyped-def]
    intent, state, mkt, cfg, _ = sc
    state.kill_switch = True
    d = evaluate(intent, state, mkt, system(), cfg)
    if intent.side is Side.BUY:
        assert d.kind == "reject"


@settings(max_examples=200, deadline=None)
@given(scenario())
def test_decision_is_a_pure_function_of_its_inputs(sc) -> None:  # type: ignore[no-untyped-def]
    intent, state, mkt, cfg, _ = sc
    assert evaluate(intent, state, mkt, system(), cfg) == evaluate(intent, state, mkt, system(), cfg)


@settings(max_examples=150, deadline=None)
@given(scenario(), st.sampled_from(["market", "state", "system", "config", "intent"]))
def test_missing_inputs_never_approve(sc, missing: str) -> None:  # type: ignore[no-untyped-def]
    intent, state, mkt, cfg, _ = sc
    args: dict[str, object] = {
        "intent": intent,
        "state": state,
        "market": mkt,
        "system": system(),
        "config": cfg,
    }
    args[missing] = None
    d = evaluate(**args)  # type: ignore[arg-type]
    assert d.kind == "reject"


def test_preset_config_type() -> None:
    assert isinstance(disposable_money_preset(), RiskConfig)
