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
from engine.broker.paper import InMemoryPaperState, PaperBroker, PaperStateStore
from engine.broker.robinhood_crypto import RobinhoodApi, RobinhoodCryptoBroker, load_pairs
from engine.clock import Clock
from engine.core import TradingCore
from engine.domain.events import BaselinePosition
from engine.domain.intent import Intent
from engine.domain.market import MarketSnapshot, SystemState
from engine.domain.types import AssetClass, Mode
from engine.eventlog.base import EventStore
from engine.marketdata.alpaca import AlpacaBars, AlpacaMarketData
from engine.marketdata.alpaca_news import AlpacaNewsFeed
from engine.marketdata.base import MarketDataError, MarketDataProvider
from engine.marketdata.history import HistoricalData
from engine.marketdata.robinhood import RobinhoodMarketData
from engine.news.listener import NewsListener
from engine.projections.account import AccountState
from engine.reconciler import Reconciler
from engine.risk.config import RiskConfig, disposable_money_preset, merge
from engine.service.config import AccountConfig, LLMConfig, Secrets
from engine.strategy.llm_schema import LLMStrategyDef, load_any_strategy
from engine.strategy.llm_source import LLMDeps, LLMStrategySource
from engine.strategy.runner import StrategyRunner
from engine.strategy.schema import StrategyDef, validate_for_venue
from engine.strategy.source import RuleStrategySource, SignalSource


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
    runner: StrategyRunner | None = None

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
        """Teach the adapter the ledger's order ids, then (first start only) give the ledger its opening
        balances. Idempotent."""
        state = self.core.projector.refresh()
        register = getattr(self.broker, "register_client_ids", None)
        if register is not None:
            register(list(state.orders))
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


def _build_base(
    cfg: AccountConfig,
    store: EventStore,
    clock: Clock,
    secrets: Secrets,
    *,
    paper_state: PaperStateStore | None = None,
    base_risk: RiskConfig | None = None,
) -> AccountRuntime:
    """Create the real broker + market data for an account from configuration and secrets."""
    if cfg.mode is Mode.LIVE and not secrets.allow_live:
        raise LiveNotAllowed(f"{cfg.id}: LIVE accounts need ENGINE_ALLOW_LIVE=1")
    if cfg.broker == "alpaca":
        creds = secrets.alpaca_live if cfg.mode is Mode.LIVE else secrets.alpaca_paper
        if creds is None:
            raise ValueError(f"{cfg.id}: missing Alpaca {cfg.mode.value.lower()} credentials")
        alpaca = AlpacaBroker(creds[0], creds[1], cfg.mode)
        market = AlpacaMarketData(creds[0], creds[1], alpaca, clock, volume_scale=cfg.volume_scale)
        return assemble_runtime(cfg, store, clock, alpaca, market, base_risk=base_risk)
    # Robinhood Crypto (live) and the crypto paper simulator both need Robinhood quotes
    if secrets.robinhood is None:
        raise ValueError(
            f"{cfg.id}: missing Robinhood API credentials (needed for quotes even in paper mode)"
        )
    if not cfg.crypto_metadata:
        raise ValueError(f"{cfg.id}: crypto_metadata (the tradable-coin allowlist file) is required")
    pairs = load_pairs(cfg.crypto_metadata)
    api = RobinhoodApi(secrets.robinhood[0], secrets.robinhood[1])
    rh_market = RobinhoodMarketData(api, pairs, clock)
    if cfg.broker == "paper-crypto":
        assert cfg.starting_cash is not None
        paper = PaperBroker(
            clock, cfg.id, paper_state or InMemoryPaperState(), starting_cash=cfg.starting_cash, pairs=pairs
        )
        return assemble_runtime(cfg, store, clock, paper, rh_market, paper=paper, base_risk=base_risk)
    return assemble_runtime(
        cfg, store, clock, RobinhoodCryptoBroker(api, pairs), rh_market, base_risk=base_risk
    )


def assemble_runtime(
    cfg: AccountConfig,
    store: EventStore,
    clock: Clock,
    broker: Broker,
    market: MarketDataProvider,
    *,
    paper: PaperBroker | None = None,
    base_risk: RiskConfig | None = None,
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
        return merge(base_risk or disposable_money_preset(), cfg.risk, cfg.strategies.get(intent.strategy_id))

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


def build_runtime(
    cfg: AccountConfig,
    store: EventStore,
    clock: Clock,
    secrets: Secrets,
    *,
    paper_state: PaperStateStore | None = None,
    base_risk: RiskConfig | None = None,
    llm: LLMConfig | None = None,
    llm_deps: LLMDeps | None = None,
) -> AccountRuntime:
    """The real broker + market data for an account, plus its strategies (if any) wired to a StrategyRunner."""
    rt = _build_base(cfg, store, clock, secrets, paper_state=paper_state, base_risk=base_risk)
    if cfg.rules or cfg.llm_strategies:
        creds = secrets.alpaca_live if cfg.mode is Mode.LIVE else secrets.alpaca_paper
        creds = creds or secrets.alpaca_paper or secrets.alpaca_live
        if creds is None:
            raise ValueError(f"{cfg.id}: strategies need Alpaca data credentials for historical bars")
        history = AlpacaBars(creds[0], creds[1], clock)
        rules: list[StrategyDef] = []
        llms: list[LLMStrategyDef] = []
        for path in (*cfg.rules, *cfg.llm_strategies):
            loaded = load_any_strategy(path)
            (llms if isinstance(loaded, LLMStrategyDef) else rules).append(loaded)  # type: ignore[arg-type]
        listener = None
        if llms:
            if llm is None or llm_deps is None:
                raise ValueError(f"{cfg.id}: llm_strategies need an [llm] section in the service config")
            feed = AlpacaNewsFeed(creds[0], creds[1])
            listener = NewsListener(
                feed,
                llm_deps.archive,
                clock,
                sorted({s for st in llms for s in st.universe}),
                notify=llm_deps.notify,
                max_age=timedelta(minutes=llm.news_max_age_minutes),
            )
        attach_strategies(
            rt, rules, history, clock, base_risk=base_risk, llms=llms, llm_deps=llm_deps, listener=listener,
            news_interval=timedelta(seconds=llm.news_poll_s) if llm else timedelta(seconds=30),
            news_max_age=timedelta(minutes=llm.news_max_age_minutes) if llm else timedelta(hours=2),
        )  # fmt: skip
    return rt


def attach_rules(
    rt: AccountRuntime,
    strategies: list[StrategyDef],
    history: HistoricalData,
    clock: Clock,
    *,
    base_risk: RiskConfig | None = None,
    min_interval: timedelta = timedelta(seconds=30),
) -> StrategyRunner:
    return attach_strategies(rt, strategies, history, clock, base_risk=base_risk, min_interval=min_interval)


def attach_strategies(
    rt: AccountRuntime,
    strategies: list[StrategyDef],
    history: HistoricalData,
    clock: Clock,
    *,
    base_risk: RiskConfig | None = None,
    min_interval: timedelta = timedelta(seconds=30),
    llms: list[LLMStrategyDef] | None = None,
    llm_deps: LLMDeps | None = None,
    listener: NewsListener | None = None,
    news_interval: timedelta = timedelta(seconds=30),
    news_max_age: timedelta = timedelta(hours=2),
) -> StrategyRunner:
    """Validate strategies against this account's venue and attach one runner for all of them."""
    venue = _venue_class(rt)
    for strategy in strategies:
        validate_for_venue(strategy, rt.broker.protection, venue)
    sources: list[SignalSource] = [RuleStrategySource(s, rt.cfg.id) for s in strategies]
    for ls in llms or []:
        if ls.asset_class is not venue:
            raise ValueError(f"{ls.id}: trades {ls.asset_class} but account {rt.cfg.id} trades {venue}")
        assert llm_deps is not None
        sources.append(LLMStrategySource(ls, rt.cfg.id, llm_deps))

    def budget(name: str, state: AccountState, snapshot: MarketSnapshot) -> Decimal:
        risk = merge(base_risk or disposable_money_preset(), rt.cfg.risk, rt.cfg.strategies.get(name))
        return (
            risk.strategy_budget if risk.strategy_budget is not None else state.equity(snapshot.price_map())
        )

    rt.runner = StrategyRunner(
        sources, history, rt.core, rt.market, clock, budget, min_interval=min_interval,
        news=listener, news_interval=news_interval, news_max_age=news_max_age,
    )  # fmt: skip
    return rt.runner


def _venue_class(rt: AccountRuntime) -> AssetClass:
    return AssetClass.STOCK if rt.cfg.broker == "alpaca" else AssetClass.CRYPTO
