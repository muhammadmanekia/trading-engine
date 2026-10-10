"""One focused test per risk rule, in the spec's order. Every rejection must carry a stable code."""

from __future__ import annotations

from datetime import timedelta

import pytest

from engine.domain.types import AssetClass, Mode, Side
from engine.projections.account import AccountState, IntentRecord, Position
from engine.risk.config import RiskOverride, disposable_money_preset, merge
from engine.risk.decision import Decision
from engine.risk.engine import evaluate
from tests.helpers import (
    D,
    T0,
    crypto,
    default_config,
    fresh_state,
    make_intent,
    market,
    stock,
    system,
)


def run(intent=None, state=None, mkt=None, sysx=None, cfg=None) -> Decision:  # type: ignore[no-untyped-def]
    return evaluate(
        intent or make_intent(),
        state or fresh_state(),
        mkt or market(),
        sysx or system(),
        cfg or default_config(),
    )


def held(
    state: AccountState, qty: int, avg: int = 100, strategy: str = "s1", symbol: str = "AAPL"
) -> AccountState:
    state.positions[(strategy, symbol)] = Position(D(qty), D(avg))
    state.cash -= D(qty * avg)
    return state


def test_clean_entry_is_approved_unchanged() -> None:
    d = run()
    assert d.kind == "approve" and d.code == "OK" and d.quantity == 10
    assert d.checks_passed == (
        "system_state",
        "idempotency_rate",
        "breakers",
        "sizing",
        "instrument_filters",
        "protective",
        "follower",
    )


# ---- 1. system state ----------------------------------------------------------------------------
def test_kill_switch_blocks_entries() -> None:
    s = fresh_state()
    s.kill_switch = True
    assert run(state=s).code == "KILL_SWITCH"


def test_kill_switch_does_not_block_exits() -> None:
    s = held(fresh_state(), 10)
    s.kill_switch = True
    d = run(make_intent(side=Side.SELL, stop_loss_price=None), s)
    assert d.kind == "approve"


def test_mode_mismatch_rejected() -> None:
    assert run(sysx=system(broker_mode=Mode.LIVE)).code == "MODE_MISMATCH"
    s = fresh_state()
    s.mode = Mode.LIVE
    assert run(state=s).code == "MODE_MISMATCH"  # account LIVE but broker PAPER


def test_broker_down_rejects_even_exits() -> None:
    s = held(fresh_state(), 10)
    assert (
        run(make_intent(side=Side.SELL, stop_loss_price=None), s, sysx=system(broker_healthy=False)).code
        == "BROKER_DOWN"
    )


def test_feed_down_rejects_entries() -> None:
    assert run(sysx=system(feeds_ok=False)).code == "FEED_DOWN"


def test_stale_quote_rejects_entries_but_exits_may_proceed() -> None:
    old = T0 - timedelta(minutes=10)
    assert run(mkt=market(stock(at=old))).code == "STALE_DATA"
    s = held(fresh_state(), 10)
    d = run(make_intent(side=Side.SELL, stop_loss_price=None), s, mkt=market(stock(at=old)))
    assert d.kind == "approve"


def test_future_dated_quote_is_rejected() -> None:
    assert run(mkt=market(stock(at=T0 + timedelta(minutes=5)))).code == "BAD_TIMESTAMP"


def test_missing_instrument_fails_closed() -> None:
    assert run(make_intent(symbol="ZZZZ"), mkt=market(stock())).code == "NO_MARKET_DATA"


def test_asset_class_mismatch() -> None:
    assert run(make_intent(asset_class=AssetClass.CRYPTO)).code == "ASSET_CLASS_MISMATCH"


def test_account_mismatch() -> None:
    assert run(make_intent(account_id="someone-else")).code == "ACCOUNT_MISMATCH"


# ---- 2. idempotency and rate limits ------------------------------------------------------------
def test_duplicate_intent_rejected_only_once_decided() -> None:
    i = make_intent()
    s = fresh_state()
    s.intents[i.intent_id] = IntentRecord(intent=i)  # received, not yet decided: still evaluable
    assert run(i, s).kind == "approve"
    s.intents[i.intent_id].decision = "approve"
    assert run(i, s).code == "DUPLICATE_INTENT"


def test_expired_intent_rejected() -> None:
    assert (
        run(sysx=system(now=T0 + timedelta(hours=2)), mkt=market(stock(at=T0 + timedelta(hours=2)))).code
        == "INTENT_EXPIRED"
    )


def test_rate_limit() -> None:
    s = fresh_state()
    s.order_times = [T0 - timedelta(seconds=5)] * 20
    assert run(state=s).code == "RATE_LIMIT"
    s.order_times = [T0 - timedelta(seconds=120)] * 20  # outside the window
    assert run(state=s).kind == "approve"


# ---- 3. circuit breakers -----------------------------------------------------------------------
def test_daily_loss_cap_trips_global_breaker() -> None:
    s = fresh_state(7_000)
    s.realized_by_day[T0.date().isoformat()] = D(-3_000)  # start-of-day ≈ 10k, cap 25% = 2.5k
    d = run(state=s)
    assert d.code == "DAILY_LOSS_CAP"
    assert [(t.scope, t.code) for t in d.trips] == [("global", "DAILY_LOSS_CAP")]


def test_loss_below_cap_is_fine() -> None:
    s = fresh_state(9_000)
    s.realized_by_day[T0.date().isoformat()] = D(-1_000)
    assert run(state=s).kind == "approve"


def test_strategy_drawdown_trips_only_that_strategy() -> None:
    s = fresh_state(10_000)
    s.strategy_peak["s1"] = D(0)
    s.strategy_realized["s1"] = D(-4_500)  # 45% of equity > 40% limit
    d = run(state=s)
    assert d.code == "STRATEGY_DRAWDOWN" and d.trips[0].strategy_id == "s1"
    assert run(make_intent(strategy_id="s2"), s).kind == "approve"


def test_consecutive_loss_pause() -> None:
    s = fresh_state()
    s.consecutive_losses["s1"] = 8
    d = run(state=s)
    assert d.code == "CONSECUTIVE_LOSSES" and d.trips[0].scope == "strategy"


def test_already_halted_rejects_without_new_trip() -> None:
    s = fresh_state()
    s.halted_global = True
    d = run(state=s)
    assert d.code == "HALTED_GLOBAL" and d.trips == ()
    s2 = fresh_state()
    s2.halted_strategies.add("s1")
    assert run(state=s2).code == "HALTED_STRATEGY"


def test_breakers_do_not_block_exits() -> None:
    s = held(fresh_state(), 10)
    s.halted_global = True
    assert run(make_intent(side=Side.SELL, stop_loss_price=None), s).kind == "approve"


def test_zero_equity_rejected() -> None:
    assert run(state=fresh_state(0)).code == "NO_EQUITY"


# ---- 4. sizing ---------------------------------------------------------------------------------
def test_oversized_intent_is_resized_not_rejected() -> None:
    d = run(make_intent(quantity=100))  # $10k order vs 30% position cap ($3k)
    assert d.kind == "resize" and d.quantity == 30 and d.code == "RESIZED_MAX_POSITION"


def test_total_exposure_cap_binds() -> None:
    s = held(fresh_state(), 25, 100)  # AAPL exposure 2.5k of 10k equity
    cfg = merge(default_config(), RiskOverride(max_total_exposure_pct=D("0.30")))
    d = run(make_intent(symbol="MSFT", quantity=50), s, market(stock("AAPL"), stock("MSFT")), cfg=cfg)
    assert d.kind == "resize" and d.quantity == 5 and d.code == "RESIZED_MAX_EXPOSURE"


def test_cash_reserve_is_untouchable() -> None:
    s = fresh_state(1_000)  # equity 1k, reserve 5% = 50 -> only 950 spendable
    cfg = default_config().model_copy(update={"max_position_pct": D(1), "max_total_exposure_pct": D(1)})
    d = run(make_intent(quantity=10), s, cfg=cfg)
    assert d.kind == "resize" and d.quantity == D("9.5") and d.code == "RESIZED_CASH_RESERVE"


def test_strategy_budget_limits_size() -> None:
    cfg = merge(default_config(), RiskOverride(strategy_budget=D(500)))
    d = run(cfg=cfg)
    assert d.kind == "resize" and d.quantity == 5 and d.code == "RESIZED_STRATEGY_BUDGET"


def test_no_capacity_rejects() -> None:
    s = held(fresh_state(), 30, 100)  # position cap fully used
    assert run(state=s).code == "NO_CAPACITY"


def test_below_min_notional_rejected() -> None:
    cfg = merge(default_config(), RiskOverride(min_order_notional=D(5_000)))
    assert run(cfg=cfg).code == "BELOW_MIN_NOTIONAL"


def test_quantity_rounded_down_to_step() -> None:
    d = run(make_intent(quantity="1.55"), mkt=market(stock(qty_step="1")))
    assert d.kind == "resize" and d.quantity == 1 and d.code == "ROUNDED_TO_STEP"
    assert run(make_intent(quantity="0.4"), mkt=market(stock(qty_step="1"))).code == "NO_CAPACITY"


def test_limit_price_above_market_sizes_conservatively() -> None:
    d = run(make_intent(quantity=100, limit_price=150, stop_loss_price=90))
    assert d.reference_price == 150 and d.quantity == 20  # 3000 / 150


# ---- 5. instrument filters ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("inst", "code"),
    [
        (dict(tradable=False), "NOT_TRADABLE"),
        (dict(halted=True), "HALTED"),
        (dict(price="0.5"), "PRICE_TOO_LOW"),
        (dict(avg_daily_volume_usd=D(10)), "LOW_VOLUME"),
        (dict(avg_daily_volume_usd=None), "LOW_VOLUME"),
    ],
)
def test_stock_filters(inst: dict[str, object], code: str) -> None:
    i = make_intent(stop_loss_price="0.1")
    assert run(i, mkt=market(stock(**inst))).code == code


@pytest.mark.parametrize(
    ("inst", "code"),
    [
        (dict(robinhood_listed=False), "NOT_LISTED"),
        (dict(robinhood_listed=None), "NOT_LISTED"),
        (dict(avg_daily_volume_usd=D(1_000)), "LOW_LIQUIDITY"),
        (dict(market_cap_usd=D(1_000)), "LOW_MARKET_CAP"),
        (dict(market_cap_usd=None), "LOW_MARKET_CAP"),
        (dict(est_slippage_bps=D(900)), "SLIPPAGE_TOO_HIGH"),
        (dict(est_slippage_bps=None), "SLIPPAGE_TOO_HIGH"),
    ],
)
def test_crypto_filters_fail_closed(inst: dict[str, object], code: str) -> None:
    i = make_intent(symbol="BTC", asset_class=AssetClass.CRYPTO, quantity="0.01", stop_loss_price=50_000)
    assert run(i, mkt=market(crypto(**inst))).code == code


def test_healthy_crypto_passes() -> None:
    i = make_intent(symbol="BTC", asset_class=AssetClass.CRYPTO, quantity="0.01", stop_loss_price=50_000)
    assert run(i, mkt=market(crypto())).kind == "approve"


# ---- 6. protective orders ----------------------------------------------------------------------
def test_entry_without_stop_is_rejected() -> None:
    assert run(make_intent(stop_loss_price=None)).code == "NO_STOP_LOSS"


@pytest.mark.parametrize("stop", [100, 120])
def test_stop_must_be_below_price(stop: int) -> None:
    assert run(make_intent(stop_loss_price=stop)).code == "INVALID_STOP"


def test_take_profit_must_be_above_price() -> None:
    assert run(make_intent(take_profit_price=99)).code == "INVALID_TAKE_PROFIT"
    assert run(make_intent(take_profit_price=120)).kind == "approve"


# ---- exits -------------------------------------------------------------------------------------
def test_spot_only_no_shorting() -> None:
    assert run(make_intent(side=Side.SELL, stop_loss_price=None)).code == "NO_POSITION"


def test_exit_larger_than_position_is_clamped() -> None:
    s = held(fresh_state(), 4)
    d = run(make_intent(side=Side.SELL, quantity=10, stop_loss_price=None), s)
    assert d.kind == "resize" and d.quantity == 4 and d.code == "EXIT_CLAMPED_TO_POSITION"


def test_exit_is_per_strategy() -> None:
    s = held(fresh_state(), 10, strategy="other")
    assert run(make_intent(side=Side.SELL, stop_loss_price=None), s).code == "NO_POSITION"


# ---- 7. follower-specific ----------------------------------------------------------------------
def copy(age_days: float, price: int = 100):  # type: ignore[no-untyped-def]
    return {
        "source": "wallet:0xabc",
        "source_price": D(price),
        "source_trade_at": T0 - timedelta(days=age_days),
    }


def test_follower_price_drift_rejected() -> None:
    assert run(make_intent(copy_info=copy(0, price=80))).code == "PRICE_DRIFT"


def test_follower_haircut_scales_down_stale_copies() -> None:
    d = run(make_intent(copy_info=copy(5), quantity=10))  # 5 days * 2%/day = 10% haircut
    assert d.kind == "resize" and d.quantity == 9 and d.code == "FOLLOWER_HAIRCUT"


def test_follower_fresh_copy_untouched() -> None:
    assert run(make_intent(copy_info=copy(0))).kind == "approve"


def test_follower_too_old_is_rejected() -> None:
    assert run(make_intent(copy_info=copy(60))).code == "STALE_COPY"


# ---- fail-closed -------------------------------------------------------------------------------
def test_engine_error_becomes_rejection() -> None:
    class Boom:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    d = evaluate(make_intent(), Boom(), market(), system(), default_config())  # type: ignore[arg-type]
    assert d.kind == "reject" and d.code == "ENGINE_ERROR"


def test_garbage_input_becomes_rejection_not_exception() -> None:
    d = evaluate(None, None, None, None, None)  # type: ignore[arg-type]
    assert d.kind == "reject" and d.code == "ENGINE_ERROR" and d.intent_id == "unknown"


def test_decisions_are_deterministic() -> None:
    i, s = make_intent(quantity=100), fresh_state()
    assert run(i, s) == run(i, s)
    assert disposable_money_preset() == disposable_money_preset()


def test_exit_with_missing_config_still_fails_closed() -> None:
    s = held(fresh_state(), 10)
    d = evaluate(make_intent(side=Side.SELL, stop_loss_price=None), s, market(), system(), None)  # type: ignore[arg-type]
    assert d.kind == "reject" and d.code == "ENGINE_ERROR"
