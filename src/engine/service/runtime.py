"""An AccountRuntime is everything needed to trade one ledger account: broker, market data, core, reconciler.

`build_runtime` is the single place broker kinds are wired up. Live accounts additionally require the explicit
ENGINE_ALLOW_LIVE=1 environment flag; the ledger's account mode must also be LIVE (set via the CLI with
confirmation), otherwise the risk engine rejects with MODE_MISMATCH.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from engine.broker.alpaca import AlpacaBroker
from engine.broker.base import Broker
from engine.broker.paper import PaperBroker, PaperStateStore
from engine.clock import Clock
from engine.core import TradingCore
from engine.domain.events import BaselinePosition
from engine.domain.intent import Intent
from engine.domain.market import MarketSnapshot, SystemState
from engine.domain.types import Mode
from engine.eventlog.base import EventStore
from engine.marketdata.alpaca import AlpacaMarketData
from engine.marketdata.base import MarketDataError, MarketDataProvider
from engine.reconciler import Reconciler
from engine.risk.config import RiskConfig, disposable_money_preset, merge
from engine.service.config import AccountConfig, Secrets


class LiveNotAllowed(Exception):
    pass


@dataclass
class AccountRuntime:
    cfg: AccountConfig
    core: TradingCore
    broker: Broker
    market: MarketDataProvider
    reconciler: Reconciler
    paper: PaperBroker | None = None
    feeds_ok: bool = True
    fatal: bool = False
    consecutive_errors: int = 0
    last_sync: datetime | None = None
    last_reconcile: datetime | None = None
    last_tick: datetime | None = None

    def symbols(self) -> list[str]:
        """Everything we must have a quote for: the watchlist, holdings, working orders, pending intents."""
        state = self.core.projector.state
        syms = set(self.cfg.symbols)
        syms |= {s for (_st, s), p in state.positions.items() if p.quantity > 0}
        syms |= {o.symbol for o in state.open_orders()}
        syms |= {r.intent.symbol for r in state.intents.values() if r.decision is None and not r.abandoned}
        return sorted(syms)

    def capture_baseline(self) -> tuple[Decimal, list[BaselinePosition]]:
        """Broker cash and holdings at current prices (true cost basis is unknown, so price = cost)."""
        positions = self.broker.get_positions()
        snap = self.market.snapshot(sorted(positions)) if positions else None
        out = []
        for sym, qty in positions.items():
            assert snap is not None
            out.append(BaselinePosition(symbol=sym, quantity=qty, avg_cost=snap.instruments[sym].price))
        return self.broker.get_cash(), out

    def bootstrap(self) -> None:
        """First start only: give the ledger its opening balances. Idempotent."""
        state = self.core.projector.refresh()
        if state.funded:
            return
        if self.paper is not None:
            assert self.cfg.starting_cash is not None
            self.core.control.fund_once(self.cfg.starting_cash, note="paper starting cash")
        else:
            cash, positions = self.capture_baseline()
            self.core.control.import_baseline(cash, positions, note="imported from broker at first start")
        self.core.projector.refresh()


class _CachedMarket:
    """Intents arrive in bursts; one quote fetch can serve several decisions made within the same second."""

    def __init__(
        self, rt_symbols: Callable[[], list[str]], provider: MarketDataProvider, clock: Clock
    ) -> None:
        self._symbols, self._provider, self._clock = rt_symbols, provider, clock
        self._snap: MarketSnapshot | None = None
        self._at: datetime | None = None

    def __call__(self) -> MarketSnapshot:
        now, wanted = self._clock.now(), self._symbols()
        if (
            self._snap is not None
            and self._at is not None
            and now - self._at < timedelta(seconds=1)
            and all(s in self._snap.instruments for s in wanted)
        ):
            return self._snap
        self._snap, self._at = self._provider.snapshot(wanted), now
        return self._snap


def build_runtime(
    cfg: AccountConfig,
    store: EventStore,
    clock: Clock,
    secrets: Secrets,
    *,
    paper_state: PaperStateStore | None = None,
    volume_scale: Decimal = Decimal(1),
) -> AccountRuntime:
    """Create the real broker + market data for an account from configuration and secrets."""
    if cfg.mode is Mode.LIVE and not secrets.allow_live:
        raise LiveNotAllowed(f"{cfg.id}: LIVE accounts need ENGINE_ALLOW_LIVE=1")
    if cfg.broker == "alpaca":
        creds = secrets.alpaca_live if cfg.mode is Mode.LIVE else secrets.alpaca_paper
        if creds is None:
            raise ValueError(f"{cfg.id}: missing Alpaca {cfg.mode.value.lower()} credentials")
        alpaca = AlpacaBroker(creds[0], creds[1], cfg.mode)
        market = AlpacaMarketData(creds[0], creds[1], alpaca, clock, volume_scale=volume_scale)
        return assemble_runtime(cfg, store, clock, alpaca, market)
    raise NotImplementedError(f"broker {cfg.broker!r} is wired up in sub-phase 2c")


def assemble_runtime(
    cfg: AccountConfig,
    store: EventStore,
    clock: Clock,
    broker: Broker,
    market: MarketDataProvider,
    *,
    paper: PaperBroker | None = None,
) -> AccountRuntime:
    """Wire a broker and market-data provider into a TradingCore. Pure wiring, no I/O (also used by tests)."""
    holder: dict[str, AccountRuntime] = {}
    cached = _CachedMarket(lambda: holder["rt"].symbols(), market, clock)

    def market_provider() -> MarketSnapshot:
        rt = holder["rt"]
        try:
            snap = cached()
        except MarketDataError:
            rt.feeds_ok = False
            raise
        rt.feeds_ok = True
        return snap

    def system_provider() -> SystemState:
        return SystemState(
            now=clock.now(),
            broker_mode=broker.mode,
            broker_healthy=broker.healthy(),
            feeds_ok=holder["rt"].feeds_ok,
        )

    def config_provider(intent: Intent) -> RiskConfig:
        return merge(disposable_money_preset(), cfg.risk, cfg.strategies.get(intent.strategy_id))

    core = TradingCore(
        store, broker, clock, cfg.id, market=market_provider, system=system_provider, config=config_provider
    )
    rt = AccountRuntime(
        cfg=cfg,
        core=core,
        broker=broker,
        market=market,
        reconciler=Reconciler(core, broker, clock),
        paper=paper,
    )
    holder["rt"] = rt
    return rt
