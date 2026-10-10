"""Individual risk checks. Each is pure: (Ctx) -> Reject | Resize | None (pass).

Entries (BUY) run every check. Exits (SELL) are reduce-only and run a reduced set, so a kill switch or a
tripped breaker can never strand an open position.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_FLOOR, Decimal

from engine.domain.events import BreakerTrip
from engine.domain.intent import Intent
from engine.domain.market import InstrumentInfo, MarketSnapshot, SystemState
from engine.domain.types import AssetClass, Side
from engine.projections.account import AccountState
from engine.risk.config import RiskConfig
from engine.risk.decision import Outcome, Reject, Resize, Trace

ZERO = Decimal(0)
FUTURE_QUOTE_TOLERANCE = timedelta(seconds=5)


@dataclass
class Ctx:
    intent: Intent
    state: AccountState
    market: MarketSnapshot
    system: SystemState
    config: RiskConfig
    instrument: InstrumentInfo | None
    qty: Decimal
    ref_price: Decimal | None
    prices: dict[str, Decimal]
    trace: Trace

    @property
    def is_entry(self) -> bool:
        return self.intent.side is Side.BUY

    def inst(self) -> InstrumentInfo:
        if self.instrument is None:  # unreachable once check_system_state has passed
            raise RuntimeError("instrument required")
        return self.instrument

    def ref(self) -> Decimal:
        if self.ref_price is None:
            raise RuntimeError("reference price required")
        return self.ref_price


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


# ---- 1. system state ---------------------------------------------------------------------------
def check_system_state(c: Ctx) -> Outcome:
    if c.intent.account_id != c.state.account_id:
        return Reject("ACCOUNT_MISMATCH")
    if c.system.broker_mode != c.state.mode:
        return Reject("MODE_MISMATCH", f"account={c.state.mode} broker={c.system.broker_mode}")
    if not c.system.broker_healthy:
        return Reject("BROKER_DOWN")
    if c.is_entry:
        if c.state.kill_switch:
            return Reject("KILL_SWITCH")
        if not c.system.feeds_ok:
            return Reject("FEED_DOWN")
    inst = c.instrument
    if inst is None:
        return Reject("NO_MARKET_DATA", c.intent.symbol)
    if c.is_entry:
        age = c.system.now - inst.quote_ts
        if age > timedelta(seconds=c.config.max_data_age_s):
            return Reject("STALE_DATA", f"quote age {age.total_seconds():.0f}s")
        if age < -FUTURE_QUOTE_TOLERANCE:
            return Reject("BAD_TIMESTAMP", "quote timestamp is in the future")
    if inst.asset_class is not c.intent.asset_class:
        return Reject("ASSET_CLASS_MISMATCH")
    if c.is_entry and not inst.market_open:
        return Reject("MARKET_CLOSED", "entries are only placed during the regular session")
    return None


# ---- 2. idempotency and rate limits ------------------------------------------------------------
def check_idempotency_and_rate(c: Ctx) -> Outcome:
    rec = c.state.intents.get(c.intent.intent_id)
    if rec is not None and rec.decision is not None:
        return Reject("DUPLICATE_INTENT")
    if c.system.now >= c.intent.expires_at:
        return Reject("INTENT_EXPIRED")
    if c.is_entry:
        recent = c.state.orders_since(c.system.now - timedelta(seconds=60))
        if recent >= c.config.max_orders_per_minute:
            return Reject("RATE_LIMIT", f"{recent} orders in the last minute")
    return None


# ---- 3. loss circuit breakers (entries only) ---------------------------------------------------
def check_breakers(c: Ctx) -> Outcome:
    s = c.intent.strategy_id
    if c.state.halted_global:
        return Reject("HALTED_GLOBAL")
    if s in c.state.halted_strategies:
        return Reject("HALTED_STRATEGY", s)
    equity = c.state.equity(c.prices)
    if equity <= 0:
        return Reject("NO_EQUITY")

    today = c.system.now.date().isoformat()
    realized_today = c.state.realized_by_day.get(today, ZERO)
    start_equity = max(equity - realized_today, ZERO)
    if realized_today < 0 and -realized_today >= c.config.daily_loss_cap_pct * start_equity:
        trip = BreakerTrip(scope="global", code="DAILY_LOSS_CAP", detail=f"realized today {realized_today}")
        c.trace.trips.append(trip)
        return Reject("DAILY_LOSS_CAP", trip.detail)

    budget = c.config.strategy_budget
    dd = c.state.drawdown(s)
    dd_base = budget if budget is not None else equity
    if dd > 0 and dd >= c.config.strategy_drawdown_limit_pct * dd_base:
        trip = BreakerTrip(scope="strategy", strategy_id=s, code="STRATEGY_DRAWDOWN", detail=f"drawdown {dd}")
        c.trace.trips.append(trip)
        return Reject("STRATEGY_DRAWDOWN", trip.detail)

    losses = c.state.consecutive_losses.get(s, 0)
    if losses >= c.config.max_consecutive_losses:
        trip = BreakerTrip(
            scope="strategy", strategy_id=s, code="CONSECUTIVE_LOSSES", detail=f"{losses} losses in a row"
        )
        c.trace.trips.append(trip)
        return Reject("CONSECUTIVE_LOSSES", trip.detail)
    return None


# ---- exits: reduce-only (spot only; no shorting) -----------------------------------------------
def check_exit(c: Ctx) -> Outcome:
    held = c.state.position_qty(c.intent.strategy_id, c.intent.symbol)
    if held <= 0:
        return Reject("NO_POSITION", "spot only: cannot sell what the strategy does not hold")
    if c.qty > held:
        return Resize(held, "EXIT_CLAMPED_TO_POSITION")
    return None


# ---- 4. sizing (entries) -----------------------------------------------------------------------
def check_sizing(c: Ctx) -> Outcome:
    cfg, st = c.config, c.state
    inst, ref = c.inst(), c.ref()
    equity = st.equity(c.prices)
    caps: list[tuple[str, Decimal]] = [
        ("MAX_POSITION", cfg.max_position_pct * equity - st.symbol_exposure(c.intent.symbol, c.prices)),
        ("MAX_EXPOSURE", cfg.max_total_exposure_pct * equity - st.exposure(c.prices)),
        ("CASH_RESERVE", st.cash - cfg.cash_reserve_pct * equity),
    ]
    if cfg.strategy_budget is not None:
        caps.append(
            ("STRATEGY_BUDGET", cfg.strategy_budget - st.strategy_exposure(c.intent.strategy_id, c.prices))
        )
    binding, allowed = min(caps, key=lambda kv: kv[1])
    if allowed <= 0:
        return Reject("NO_CAPACITY", binding)

    qty = floor_to_step(c.qty, inst.qty_step)
    code = "ROUNDED_TO_STEP"
    if qty * ref > allowed:
        qty = floor_to_step(allowed / ref, inst.qty_step)
        code = f"RESIZED_{binding}"
    if qty <= 0:
        return Reject("NO_CAPACITY", f"{binding}: below one step")
    if qty * ref < cfg.min_order_notional:
        return Reject("BELOW_MIN_NOTIONAL", f"{qty * ref}")
    return Resize(qty, code) if qty != c.qty else None


# ---- 5. instrument filters (entries) -----------------------------------------------------------
def check_instrument_filters(c: Ctx) -> Outcome:
    i, cfg = c.inst(), c.config
    if not i.tradable:
        return Reject("NOT_TRADABLE")
    if i.halted:
        return Reject("HALTED")
    if i.asset_class is AssetClass.STOCK:
        if i.price < cfg.min_price:
            return Reject("PRICE_TOO_LOW")
        if i.avg_daily_volume_usd is None or i.avg_daily_volume_usd < cfg.min_avg_daily_volume_usd:
            return Reject("LOW_VOLUME")
        return None
    # crypto: unknown data fails closed
    if i.robinhood_listed is not True:
        return Reject("NOT_LISTED")
    if i.avg_daily_volume_usd is None or i.avg_daily_volume_usd < cfg.min_crypto_liquidity_usd:
        return Reject("LOW_LIQUIDITY")
    if i.market_cap_usd is None or i.market_cap_usd < cfg.min_crypto_market_cap_usd:
        return Reject("LOW_MARKET_CAP")
    if i.est_slippage_bps is None or i.est_slippage_bps > cfg.max_slippage_bps:
        return Reject("SLIPPAGE_TOO_HIGH")
    return None


# ---- 6. protective orders (entries) ------------------------------------------------------------
def check_protective(c: Ctx) -> Outcome:
    it, price = c.intent, c.inst().price
    if c.config.require_stop_loss and it.stop_loss_price is None:
        return Reject("NO_STOP_LOSS")
    stop = it.stop_loss_price
    if stop is not None and (stop >= price or (it.limit_price is not None and stop >= it.limit_price)):
        return Reject("INVALID_STOP", "stop must be below the entry price")
    if it.take_profit_price is not None and it.take_profit_price <= price:
        return Reject("INVALID_TAKE_PROFIT", "take-profit must be above the entry price")
    return None


# ---- 7. follower-specific (entries) ------------------------------------------------------------
def check_follower(c: Ctx) -> Outcome:
    info = c.intent.copy_info
    if info is None:
        return None
    cfg, price = c.config, c.inst().price
    drift = abs(price - info.source_price) / info.source_price
    if drift > cfg.max_follower_price_drift_pct:
        return Reject("PRICE_DRIFT", f"{drift:.4f} since the followed trade")
    age = max(c.system.now - info.source_trade_at, timedelta(0))
    age_days = Decimal(age.days) + Decimal(age.seconds) / Decimal(86400)  # exact, no floats
    haircut = min(age_days * cfg.follower_haircut_pct_per_day, Decimal(1))
    if haircut >= 1:
        return Reject("STALE_COPY", f"{age_days:.1f} days old")
    if haircut > 0:
        qty = floor_to_step(c.qty * (1 - haircut), c.inst().qty_step)
        if qty <= 0 or qty * c.ref() < cfg.min_order_notional:
            return Reject("BELOW_MIN_NOTIONAL", "after staleness haircut")
        return Resize(qty, "FOLLOWER_HAIRCUT")
    return None


CheckFn = Callable[[Ctx], Outcome]

ENTRY_CHECKS: tuple[tuple[str, CheckFn], ...] = (
    ("system_state", check_system_state),
    ("idempotency_rate", check_idempotency_and_rate),
    ("breakers", check_breakers),
    ("sizing", check_sizing),
    ("instrument_filters", check_instrument_filters),
    ("protective", check_protective),
    ("follower", check_follower),
)

EXIT_CHECKS: tuple[tuple[str, CheckFn], ...] = (
    ("system_state", check_system_state),
    ("idempotency_rate", check_idempotency_and_rate),
    ("exit_reduce_only", check_exit),
)
