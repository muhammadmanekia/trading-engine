"""StrategyRunner: run SignalSources on live (paper or real) data inside the execution service.

It does exactly what the backtester does, with real time instead of a replay: when a new bar has CLOSED, hand
the closed history to each source, and submit whatever Intents come back through TradingCore (risk engine,
order manager, ledger). Because the source's output depends only on the closed bars, and intent ids are derived
from the bar, evaluating the same bar twice (a restart, a retry) is harmless.

Stocks are only evaluated while the market is open (the engine refuses entries otherwise), which makes the
live fill land near the next session's open, exactly what the backtest assumes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from engine.broker.base import BrokerError, CredentialsError
from engine.clock import Clock
from engine.core import TradingCore
from engine.domain.intent import Intent
from engine.domain.market import MarketSnapshot
from engine.domain.types import AssetClass
from engine.marketdata.base import MarketDataError, MarketDataProvider
from engine.marketdata.history import Bar, HistoricalData, timeframe_delta
from engine.news.listener import NewsListener
from engine.news.models import NewsItem, canonical_symbol
from engine.projections.account import AccountState
from engine.strategy.source import SignalSource, SourceContext

log = logging.getLogger("engine.strategy.runner")

BudgetFn = Callable[[str, AccountState, MarketSnapshot], Decimal]  # (source name, state, snapshot) -> capital


class StrategyRunner:
    def __init__(
        self,
        sources: Sequence[SignalSource],
        history: HistoricalData,
        core: TradingCore,
        market: MarketDataProvider,
        clock: Clock,
        budget: BudgetFn,
        *,
        min_interval: timedelta = timedelta(seconds=30),
        news: NewsListener | None = None,
        news_interval: timedelta = timedelta(seconds=30),
        news_max_age: timedelta = timedelta(hours=2),
    ) -> None:
        self.sources = list(sources)
        self.history = history
        self.core = core
        self.market = market
        self.clock = clock
        self.budget = budget
        self.min_interval = min_interval
        self._evaluated: dict[
            str, tuple[datetime, ...]
        ] = {}  # source -> last bar ts per symbol, as evaluated
        self._last_poll: dict[str, datetime] = {}
        self.errors = 0
        self.news = news
        self.news_interval = news_interval
        self.news_max_age = news_max_age
        self._last_news_poll: datetime | None = None
        self._waiting_news: dict[str, list[NewsItem]] = {}  # source -> items held until the market opens

    def tick(self) -> list[Intent]:
        """Evaluate every source whose symbols have a newly closed bar. Returns the intents submitted."""
        submitted: list[Intent] = []
        for src in self.sources:
            try:
                submitted += self._tick_source(src)
            except CredentialsError:
                raise  # the executor halts the account
            except (BrokerError, MarketDataError) as exc:
                self.errors += 1
                log.warning("strategy %s: data unavailable, will retry: %s", src.name, exc)
        submitted += self._news_tick()
        return submitted

    # ---- news -------------------------------------------------------------------------------
    def _news_tick(self) -> list[Intent]:
        """Poll the news feed and hand fresh items to the sources that listen for them. Items that arrive while
        the market is closed wait (briefly) for the open; anything older than `news_max_age` is dropped, because
        trading a stale headline is how people lose money."""
        if self.news is None:
            return []
        listeners = [s for s in self.sources if hasattr(s, "on_news")]
        if not listeners:
            return []
        now = self.clock.now()
        if self._last_news_poll is None or now - self._last_news_poll >= self.news_interval:
            self._last_news_poll = now
            fresh = self.news.poll()
            for src in listeners:
                mine = [i for i in fresh if self._mentions(src, i)]
                if mine:
                    self._waiting_news.setdefault(src.name, []).extend(mine)
        submitted: list[Intent] = []
        for src in listeners:
            waiting = [
                i for i in self._waiting_news.get(src.name, []) if now - i.published_at <= self.news_max_age
            ]
            self._waiting_news[src.name] = waiting[-20:]
            if not waiting:
                continue
            try:
                intents = self._deliver_news(src, waiting[-20:], now)
            except CredentialsError:
                raise
            except (BrokerError, MarketDataError) as exc:
                self.errors += 1
                log.warning("strategy %s: news handling deferred, data unavailable: %s", src.name, exc)
                continue
            if intents is None:
                continue  # market closed: keep waiting
            self._waiting_news[src.name] = []
            submitted += intents
        return submitted

    @staticmethod
    def _mentions(src: SignalSource, item: NewsItem) -> bool:
        return bool({canonical_symbol(s) for s in item.symbols} & {canonical_symbol(s) for s in src.symbols})

    def _deliver_news(self, src: SignalSource, items: list[NewsItem], now: datetime) -> list[Intent] | None:
        needed = src.history_needed()
        delta = timeframe_delta(src.timeframe)
        slack = 3 if self._is_stock(src) else 1
        start = now - delta * (needed * slack + 10)
        bars: dict[str, Sequence[Bar]] = {}
        for symbol in src.symbols:
            fetched = self.history.bars(symbol, src.timeframe, start, now)
            if fetched:
                bars[symbol] = fetched[-needed:]
        if not bars:
            return []
        snapshot = self.market.snapshot(list(src.symbols))
        if any(not snapshot.instruments[s].market_open for s in bars if s in snapshot.instruments):
            return None
        state = self.core.projector.refresh()
        ctx = SourceContext(
            now=now, timeframe=src.timeframe, bars=bars, state=state, snapshot=snapshot,
            budget=self.budget(src.name, state, snapshot),
        )  # fmt: skip
        intents: list[Intent] = src.on_news(ctx, items)  # type: ignore[attr-defined]
        for intent in intents:
            log.info("strategy %s (news) -> %s %s %s", src.name, intent.side, intent.quantity, intent.symbol)
            self.core.submit_intent(intent)
        return intents

    def _tick_source(self, src: SignalSource) -> list[Intent]:
        now = self.clock.now()
        last = self._last_poll.get(src.name)
        if last is not None and now - last < self.min_interval:
            return []
        self._last_poll[src.name] = now

        delta = timeframe_delta(src.timeframe)
        needed = src.history_needed()
        slack = 3 if self._is_stock(src) else 1  # weekends and holidays have no bars
        start = now - delta * (needed * slack + 10)
        bars: dict[str, Sequence[Bar]] = {}
        for symbol in src.symbols:
            fetched = self.history.bars(symbol, src.timeframe, start, now)
            if fetched:
                bars[symbol] = fetched[-needed:]
        if not bars:
            return []
        stamp = tuple(bars[s][-1].ts for s in sorted(bars))
        if self._evaluated.get(src.name) == stamp:
            return []  # nothing new has closed since the last evaluation

        snapshot = self.market.snapshot(list(src.symbols))
        if any(not snapshot.instruments[s].market_open for s in bars if s in snapshot.instruments):
            return []  # market closed: wait for the open, then act on the last closed bar
        self._evaluated[src.name] = stamp

        state = self.core.projector.refresh()
        ctx = SourceContext(
            now=now,
            timeframe=src.timeframe,
            bars=bars,
            state=state,
            snapshot=snapshot,
            budget=self.budget(src.name, state, snapshot),
        )
        intents = src.on_bar_close(ctx)
        for intent in intents:
            log.info("strategy %s -> %s %s %s", src.name, intent.side, intent.quantity, intent.symbol)
            self.core.submit_intent(intent)
        return intents

    @staticmethod
    def _is_stock(src: SignalSource) -> bool:
        strategy = getattr(src, "strategy", None)
        return strategy is None or strategy.asset_class is AssetClass.STOCK
