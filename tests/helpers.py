"""Test factories. Defaults describe a healthy paper account holding $10,000 cash and AAPL at $100."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from decimal import Decimal

from engine.broker.mock import MockBroker
from engine.clock import FakeClock
from engine.core import TradingCore
from engine.domain.intent import Intent
from engine.domain.market import InstrumentInfo, MarketSnapshot, SystemState
from engine.domain.types import AssetClass, Mode, Side
from engine.eventlog.base import EventStore
from engine.projections.account import AccountState
from engine.risk.config import RiskConfig, disposable_money_preset

D = Decimal
ACCOUNT = "acct"
T0 = FakeClock().now()
_ids = itertools.count(1)


def make_intent(**kw: object) -> Intent:
    base: dict[str, object] = {
        "intent_id": f"i{next(_ids)}",
        "account_id": ACCOUNT,
        "strategy_id": "s1",
        "symbol": "AAPL",
        "asset_class": AssetClass.STOCK,
        "side": Side.BUY,
        "quantity": D(10),
        "stop_loss_price": D(90),
        "confidence": D("0.7"),
        "source": "test",
        "created_at": T0,
        "expires_at": T0 + timedelta(hours=1),
    }
    base.update(kw)
    return Intent.model_validate(base)


def stock(symbol: str = "AAPL", price: str | int = 100, at: datetime = T0, **kw: object) -> InstrumentInfo:
    base: dict[str, object] = {
        "symbol": symbol,
        "asset_class": AssetClass.STOCK,
        "price": D(price),
        "quote_ts": at,
        "avg_daily_volume_usd": D(50_000_000),
    }
    base.update(kw)
    return InstrumentInfo.model_validate(base)


def crypto(symbol: str = "BTC", price: str | int = 60000, at: datetime = T0, **kw: object) -> InstrumentInfo:
    base: dict[str, object] = {
        "symbol": symbol,
        "asset_class": AssetClass.CRYPTO,
        "price": D(price),
        "quote_ts": at,
        "avg_daily_volume_usd": D(5_000_000_000),
        "market_cap_usd": D(1_000_000_000_000),
        "robinhood_listed": True,
        "est_slippage_bps": D(5),
    }
    base.update(kw)
    return InstrumentInfo.model_validate(base)


def market(*instruments: InstrumentInfo, at: datetime = T0) -> MarketSnapshot:
    items = instruments or (stock(),)
    return MarketSnapshot(as_of=at, instruments={i.symbol: i for i in items})


def system(**kw: object) -> SystemState:
    base: dict[str, object] = {"now": T0, "broker_mode": Mode.PAPER, "broker_healthy": True, "feeds_ok": True}
    base.update(kw)
    return SystemState.model_validate(base)


def default_config() -> RiskConfig:
    return disposable_money_preset()


def fresh_state(cash: int | str = 10_000) -> AccountState:
    return AccountState(account_id=ACCOUNT, cash=D(cash))


def build_core(
    store: EventStore,
    clock: FakeClock | None = None,
    broker: MockBroker | None = None,
    *,
    cash: int = 10_000,
    prices: dict[str, str] | None = None,
    config: RiskConfig | None = None,
    hook=None,  # type: ignore[no-untyped-def]
    fund: bool = True,
) -> tuple[TradingCore, MockBroker, FakeClock]:
    clock = clock or FakeClock()
    broker = broker or MockBroker(clock)
    for sym, p in (prices or {"AAPL": "100"}).items():
        broker.prices.setdefault(sym, D(p))
    cfg = config or default_config()

    def market_provider() -> MarketSnapshot:
        now = clock.now()
        return MarketSnapshot(
            as_of=now, instruments={s: stock(s, p, at=now) for s, p in broker.prices.items()}
        )

    def system_provider() -> SystemState:
        return SystemState(
            now=clock.now(), broker_mode=broker.mode, broker_healthy=broker.healthy(), feeds_ok=True
        )

    core = TradingCore(
        store,
        broker,
        clock,
        ACCOUNT,
        market=market_provider,
        system=system_provider,
        config=lambda _intent: cfg,
        hook=hook,
    )
    if fund:
        core.control.fund(D(cash))
        core.projector.refresh()
    return core, broker, clock


def intent_now(clock: FakeClock, **kw: object) -> Intent:
    now = clock.now()
    return make_intent(created_at=now, expires_at=now + timedelta(hours=1), **kw)


class SimulatedCrash(BaseException):
    """Raised by crash hooks. BaseException so no `except Exception` in the engine can swallow it."""


class Crasher:
    """Hook that 'kills the process' the first time it reaches the named point."""

    def __init__(self, point: str) -> None:
        self.point = point
        self.fired = False

    def __call__(self, name: str) -> None:
        if name == self.point and not self.fired:
            self.fired = True
            raise SimulatedCrash(name)
