"""BacktestEngine: replay historical bars through the REAL trading core.

Nothing about trading is re-implemented here. Intents go through the same risk engine, order manager,
event log and projections as live trading; only the broker (bar-driven, pessimistic) and the clock are
simulated. That is what makes a backtest result mean something about the live system.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from engine.backtest.broker import BacktestBroker, CostRecord
from engine.backtest.venue import VenueProfile
from engine.core import TradingCore
from engine.domain.events import OrderCreated, OrderFilled, RiskDecisionMade
from engine.domain.market import InstrumentInfo, MarketSnapshot, SystemState
from engine.domain.types import AssetClass, Mode, Side
from engine.eventlog.base import StoredEvent, decode
from engine.eventlog.memory import InMemoryEventStore
from engine.marketdata.history import Bar, close_time, timeframe_delta, validate_bars
from engine.projections.account import AccountState
from engine.risk.config import RiskConfig
from engine.strategy.schema import validate_for_venue
from engine.strategy.source import Decision, SignalSource, SourceContext

ZERO = Decimal(0)


class SimClock:
    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def set(self, when: datetime) -> None:
        self._now = when


@dataclass(frozen=True)
class BacktestConfig:
    venue: VenueProfile
    starting_cash: Decimal
    risk: RiskConfig
    account_id: str = "bt"
    volume_scale: Decimal = Decimal(1)  # multiply bar volume (e.g. IEX shows only a slice of real volume)


@dataclass(frozen=True)
class Trade:
    strategy_id: str
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal  # after fees and spread/slippage (those are inside the fill prices)
    fees: Decimal
    r_multiple: Decimal | None
    bars_held: int
    exit_reason: str


@dataclass
class BacktestResult:
    config: BacktestConfig
    timeframe: str
    start: datetime
    end: datetime
    equity_curve: list[tuple[datetime, Decimal]]
    exposure: list[bool]
    trades: list[Trade]
    open_positions: dict[str, Decimal]
    cost_log: list[CostRecord]
    risk_decisions: Counter[str]
    skipped: Counter[str]
    signals: Counter[str]
    anomalies: list[str]
    warnings: list[str]
    strategy_hashes: dict[str, str]
    events: list[StoredEvent] = field(repr=False, default_factory=list)
    benchmark_return: Decimal | None = None  # buy & hold of the first symbol, net of one round trip of costs

    @property
    def final_equity(self) -> Decimal:
        return self.equity_curve[-1][1] if self.equity_curve else self.config.starting_cash


class BacktestEngine:
    def __init__(
        self,
        sources: Sequence[SignalSource],
        data: Mapping[str, Sequence[Bar]],
        config: BacktestConfig,
    ) -> None:
        timeframes = {s.timeframe for s in sources}
        if len(timeframes) != 1:
            raise ValueError(f"all sources in one run must share a timeframe, got {sorted(timeframes)}")
        self.timeframe = timeframes.pop()
        self.sources = list(sources)
        self.config = config
        self.data = {sym: list(bars) for sym, bars in data.items()}
        self.warnings: list[str] = []
        for sym, bars in self.data.items():
            self.warnings += [f"{sym}: {w}" for w in validate_bars(bars, self.timeframe)]
        for src in sources:  # fail at load time what the venue cannot do, not as a rejected order mid-run
            strategy = getattr(src, "strategy", None)
            if strategy is not None:
                validate_for_venue(strategy, config.venue.protection, config.venue.asset_class)
        missing = {sym for s in sources for sym in s.symbols} - set(self.data)
        if missing:
            raise ValueError(f"no data for {sorted(missing)}")
        if not any(self.data.values()):
            raise ValueError("no bars to test on")

    # ------------------------------------------------------------------------------------------
    def run(self, trade_from: datetime | None = None) -> BacktestResult:
        """`trade_from`: bars before it are warm-up only (indicators see them, nothing trades and no equity is
        recorded). Walk-forward uses this so each test window starts with realistic indicator history."""
        cfg, venue = self.config, self.config.venue
        delta = timeframe_delta(self.timeframe)
        timeline = sorted({b.ts for bars in self.data.values() for b in bars})
        by_ts = {sym: {b.ts: b for b in bars} for sym, bars in self.data.items()}
        history: dict[str, list[Bar]] = {sym: [] for sym in self.data}
        needed = max(s.history_needed() for s in self.sources)

        store = InMemoryEventStore()
        clock = SimClock(timeline[0])
        broker = BacktestBroker(clock, venue, starting_cash=cfg.starting_cash)
        snapshot_holder: dict[str, MarketSnapshot] = {}

        def market() -> MarketSnapshot:
            return snapshot_holder["s"]

        def system() -> SystemState:
            return SystemState(now=clock.now(), broker_mode=Mode.PAPER, broker_healthy=True, feeds_ok=True)

        core = TradingCore(
            store, broker, clock, cfg.account_id, market=market, system=system, config=lambda _i: cfg.risk
        )
        core.control.fund(cfg.starting_cash)
        core.projector.refresh()

        equity_curve: list[tuple[datetime, Decimal]] = []
        exposure: list[bool] = []
        skipped: Counter[str] = Counter()
        signals: Counter[str] = Counter()

        for ts in timeline:
            clock.set(ts)
            bars_now = {sym: m[ts] for sym, m in by_ts.items() if ts in m}
            broker.begin_bar(bars_now)
            broker.run_phase("open")  # market orders fill at the open
            core.manager.sync()  # book fills; the OrderManager places protection
            broker.run_phase("range")  # then resting stops/targets are tested against the bar's range
            core.manager.sync()
            broker.end_bar()
            for sym, bar in bars_now.items():
                history[sym].append(bar)

            now = ts + delta  # the bar is closed: this is the earliest moment anything can act on it
            clock.set(now)
            snapshot_holder["s"] = self._snapshot(history, now)
            state = core.projector.refresh()
            if trade_from is not None and now <= trade_from:
                continue  # warm-up only
            equity = state.equity(broker.prices)
            equity_curve.append((now, equity))
            exposure.append(any(p.quantity > 0 for p in state.positions.values()))

            for src in self.sources:
                ctx = SourceContext(
                    now=now,
                    timeframe=self.timeframe,
                    bars={sym: history[sym][-needed:] for sym in src.symbols if history[sym]},
                    state=state,
                    snapshot=snapshot_holder["s"],
                    budget=cfg.risk.strategy_budget if cfg.risk.strategy_budget is not None else equity,
                )
                for intent in src.on_bar_close(ctx):
                    core.submit_intent(intent)
                for d in getattr(src, "last_decisions", []):
                    self._tally(d, signals, skipped)

        events = store.read_all()
        state = core.projector.refresh()
        first = self.data[self.sources[0].symbols[0]]
        benchmark = first[-1].close / first[0].close - 1 - venue.round_trip_cost if len(first) > 1 else None
        return BacktestResult(
            config=cfg,
            timeframe=self.timeframe,
            start=equity_curve[0][0] if equity_curve else timeline[0],
            end=timeline[-1] + delta,
            equity_curve=equity_curve,
            exposure=exposure,
            trades=build_trades(events, self.timeframe),
            open_positions={
                f"{s}/{sym}": p.quantity for (s, sym), p in state.positions.items() if p.quantity > 0
            },
            cost_log=broker.cost_log,
            risk_decisions=Counter(
                (e.code if e.decision == "reject" else "APPROVED")
                for e in (decode(x) for x in events if x.type == "RiskDecisionMade")
                if isinstance(e, RiskDecisionMade)
            ),
            skipped=skipped,
            signals=signals,
            anomalies=list(state.anomalies),
            warnings=self.warnings,
            strategy_hashes={
                getattr(s, "name", "?"): getattr(getattr(s, "strategy", None), "hash", "")
                for s in self.sources
            },
            events=events,
            benchmark_return=benchmark,
        )

    # ------------------------------------------------------------------------------------------
    def _snapshot(self, history: Mapping[str, Sequence[Bar]], now: datetime) -> MarketSnapshot:
        venue = self.config.venue
        per_day = max(
            1,
            round(
                venue.periods_per_year(self.timeframe)
                / (365 if venue.asset_class is AssetClass.CRYPTO else 252)
            ),
        )
        out: dict[str, InstrumentInfo] = {}
        for sym, bars in history.items():
            if not bars:
                continue
            last = bars[-1]
            if venue.adv_usd is not None:
                adv = venue.adv_usd
            else:
                recent = bars[-20 * per_day :]
                adv = (
                    sum((b.close * b.volume for b in recent), ZERO)
                    / len(recent)
                    * per_day
                    * self.config.volume_scale
                )
            crypto = venue.asset_class is AssetClass.CRYPTO
            out[sym] = InstrumentInfo(
                symbol=sym,
                asset_class=venue.asset_class,
                price=last.close,
                quote_ts=now,
                qty_step=venue.qty_step,
                avg_daily_volume_usd=adv,
                market_cap_usd=venue.market_cap_usd if crypto else None,
                robinhood_listed=True if crypto else None,
                est_slippage_bps=venue.spread_bps if crypto else None,
            )
        return MarketSnapshot(as_of=now, instruments=out)

    @staticmethod
    def _tally(d: Decision, signals: Counter[str], skipped: Counter[str]) -> None:
        signals[d.signal.kind] += 1
        if d.skipped:
            skipped[d.skipped] += 1


@dataclass
class _RoundTrip:
    entry_ts: datetime
    stop: Decimal | None = None
    qty: Decimal = ZERO  # currently held
    bought: Decimal = ZERO
    buy_cost: Decimal = ZERO
    sold: Decimal = ZERO
    sell_value: Decimal = ZERO
    flow: Decimal = ZERO  # cash in/out including fees; its final value is the trade's P&L
    fees: Decimal = ZERO
    reason: str = "other"


def build_trades(events: Sequence[StoredEvent], timeframe: str) -> list[Trade]:
    """Round trips (flat -> flat) reconstructed from the ledger's fills, per (strategy, symbol)."""
    delta = timeframe_delta(timeframe)
    created: dict[str, OrderCreated] = {}
    open_: dict[tuple[str, str], _RoundTrip] = {}
    trades: list[Trade] = []
    for stored in events:
        ev = decode(stored)
        if isinstance(ev, OrderCreated):
            created[ev.order_id] = ev
            continue
        if not isinstance(ev, OrderFilled):
            continue
        key = (ev.strategy_id, ev.symbol)
        order = created.get(ev.order_id)
        rt = open_.get(key)
        if rt is None:
            rt = open_[key] = _RoundTrip(entry_ts=stored.ts)
        if ev.side is Side.BUY:
            if rt.qty == 0:
                rt.entry_ts = stored.ts
                rt.stop = order.stop_loss_price if order is not None else None
            rt.qty += ev.quantity
            rt.bought += ev.quantity
            rt.buy_cost += ev.quantity * ev.price
            rt.flow -= ev.quantity * ev.price + ev.fee
        else:
            role = order.role if order is not None else ""
            rt.reason = "flatten" if (order is not None and order.managed and role == "exit") else {
                "stop": "stop", "take_profit": "take_profit", "exit": "rule"
            }.get(role, "other")  # fmt: skip
            rt.qty -= ev.quantity
            rt.sold += ev.quantity
            rt.sell_value += ev.quantity * ev.price
            rt.flow += ev.quantity * ev.price - ev.fee
        rt.fees += ev.fee
        if rt.qty == 0 and rt.bought > 0:
            entry_price, exit_price = rt.buy_cost / rt.bought, rt.sell_value / rt.sold
            risk = rt.bought * (entry_price - rt.stop) if rt.stop is not None else ZERO
            trades.append(
                Trade(
                    strategy_id=key[0], symbol=key[1], entry_ts=rt.entry_ts, exit_ts=stored.ts,
                    quantity=rt.bought, entry_price=entry_price, exit_price=exit_price, pnl=rt.flow,
                    fees=rt.fees, r_multiple=(rt.flow / risk) if risk > 0 else None,
                    bars_held=int((stored.ts - rt.entry_ts) / delta), exit_reason=rt.reason,
                )
            )  # fmt: skip
            del open_[key]
    return trades


def final_state(result: BacktestResult) -> AccountState:  # pragma: no cover - convenience for notebooks/tests
    from engine.projections.account import replay

    return replay(result.config.account_id, result.events)


def close_of(bar: Bar, timeframe: str) -> datetime:  # pragma: no cover
    return close_time(bar, timeframe)
