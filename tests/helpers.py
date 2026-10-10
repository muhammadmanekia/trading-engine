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


# ---- Phase 2: an Alpaca-backed account runtime over the fake server ------------------------------
def alpaca_runtime(  # type: ignore[no-untyped-def]
    store: EventStore,
    clock: FakeClock | None = None,
    *,
    fake=None,
    key: str = "key",
    symbols: tuple[str, ...] = ("AAPL",),
    account_id: str = ACCOUNT,
):
    """(runtime, fake server, clock) for a paper Alpaca account whose HTTP goes to the in-process fake."""
    from pydantic import SecretStr

    from engine.broker.alpaca import AlpacaBroker
    from engine.marketdata.alpaca import AlpacaMarketData
    from engine.service.config import AccountConfig
    from engine.service.runtime import assemble_runtime
    from tests.fakes.alpaca_server import BASE, FakeAlpaca

    clock = clock or FakeClock()
    fake = fake or FakeAlpaca(clock)
    broker = AlpacaBroker(
        SecretStr(key),
        SecretStr("secret"),
        Mode.PAPER,
        base_url=BASE,
        client=fake.client,
        sleeper=lambda _s: None,
    )
    md = AlpacaMarketData(
        SecretStr(key),
        SecretStr("secret"),
        broker,
        clock,
        base_url=BASE,
        client=fake.client,
        sleeper=lambda _s: None,
    )
    cfg = AccountConfig(id=account_id, broker="alpaca", mode=Mode.PAPER, symbols=symbols)
    return assemble_runtime(cfg, store, clock, broker, md), fake, clock


def intent_payload(clock: FakeClock, **kw: object) -> dict[str, object]:
    return intent_now(clock, **kw).model_dump(mode="json")


def paper_crypto_runtime(store: EventStore, clock: FakeClock | None = None, *, fake=None, paper_state=None):  # type: ignore[no-untyped-def]
    """(runtime, fake Robinhood, clock): crypto paper trading on a simulator fed by (fake) live Robinhood quotes."""
    from pydantic import SecretStr

    from engine.broker.paper import InMemoryPaperState, PaperBroker
    from engine.broker.robinhood_crypto import PairInfo, RobinhoodApi
    from engine.marketdata.robinhood import RobinhoodMarketData
    from engine.service.config import AccountConfig
    from engine.service.runtime import assemble_runtime
    from tests.fakes.robinhood_server import API_KEY, BASE, FakeRobinhood, new_keypair

    clock = clock or FakeClock()
    if fake is None:
        key, seed = new_keypair()
        fake = FakeRobinhood(clock, key)
        fake._seed = seed
    api = RobinhoodApi(
        SecretStr(API_KEY), SecretStr(fake._seed), base_url=BASE, client=fake.client,
        time_source=lambda: clock.now().timestamp(), sleeper=lambda _s: None,
    )  # fmt: skip
    pairs = {
        "BTC-USD": PairInfo(
            asset_increment=D("0.00001"), price_increment=D("0.01"), min_order_size=D("0.0001"), max_order_size=D(10),
            market_cap_usd=D(1_200_000_000_000), avg_daily_volume_usd=D(30_000_000_000),
        )
    }  # fmt: skip
    paper_state = paper_state or InMemoryPaperState()
    paper = PaperBroker(clock, ACCOUNT, paper_state, starting_cash=D(10_000), pairs=pairs)
    cfg = AccountConfig(
        id=ACCOUNT, broker="paper-crypto", mode=Mode.PAPER, starting_cash=D(10_000), symbols=("BTC-USD",)
    )
    rt = assemble_runtime(cfg, store, clock, paper, RobinhoodMarketData(api, pairs, clock), paper=paper)
    return rt, fake, clock, paper_state
