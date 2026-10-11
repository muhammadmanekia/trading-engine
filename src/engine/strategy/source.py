"""SignalSource: anything that looks at the world once per bar and proposes Intents.

Rule strategies are the first implementation. A news listener or an LLM strategy is just another
SignalSource: it proposes intents, and the same risk engine, order manager and reconciler judge and execute
them. Sources hold no broker credentials and cannot place orders; the worst a faulty source can do is
propose a bad trade, which the risk engine then refuses.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from engine.domain.intent import Intent
from engine.domain.market import MarketSnapshot
from engine.domain.types import Side
from engine.marketdata.history import Bar, timeframe_delta
from engine.projections.account import AccountState
from engine.strategy.evaluator import Signal, evaluate
from engine.strategy.schema import StrategyDef
from engine.strategy.signals import Skip, build_entry_intent, build_exit_intent

log = logging.getLogger("engine.strategy")


@dataclass(frozen=True)
class SourceContext:
    now: datetime
    timeframe: str
    bars: Mapping[str, Sequence[Bar]]  # closed bars per symbol, oldest first; the last one has just closed
    state: AccountState
    snapshot: MarketSnapshot  # instrument facts (lot size, ...) at decision time
    budget: Decimal  # capital to size against (strategy budget, else account equity)


@dataclass(frozen=True)
class Decision:
    """What a source concluded for one symbol, whether or not it produced an intent (for reports/logs)."""

    symbol: str
    signal: Signal
    intent: Intent | None = None
    skipped: str | None = None


class SignalSource(Protocol):
    name: str
    timeframe: str
    symbols: tuple[str, ...]

    def history_needed(self) -> int:
        """How many closed bars per symbol the source wants in its context."""
        ...

    def on_bar_close(self, ctx: SourceContext) -> list[Intent]: ...


class RuleStrategySource:
    def __init__(self, strategy: StrategyDef, account_id: str) -> None:
        self.strategy = strategy
        self.account_id = account_id
        self.name = strategy.id
        self.timeframe: str = strategy.timeframe
        self.symbols = strategy.universe
        self.last_decisions: list[Decision] = []

    def history_needed(self) -> int:
        return self.strategy.window

    def on_bar_close(self, ctx: SourceContext) -> list[Intent]:
        s = self.strategy
        self.last_decisions = []
        intents: list[Intent] = []
        held = {sym for sym in s.universe if ctx.state.position_qty(s.id, sym) > 0}
        pending = {sym for sym in s.universe if self._pending_entry(ctx.state, sym)}
        for symbol in s.universe:
            bars = ctx.bars.get(symbol)
            if not bars:
                continue
            position = ctx.state.position_qty(s.id, symbol)
            signal = evaluate(
                s,
                bars,
                in_position=position > 0,
                bars_since_exit=self._bars_since_exit(ctx, symbol),
            )
            decision = Decision(symbol, signal)
            if signal.kind == "enter":
                decision = self._enter(ctx, symbol, signal, bars, held | pending)
                if decision.intent is not None:
                    held.add(symbol)
            elif signal.kind == "exit" and not self._pending_exit(ctx.state, symbol):
                intent = build_exit_intent(
                    s, symbol, signal, position=position, account_id=self.account_id, now=ctx.now
                )
                decision = Decision(symbol, signal, intent)
            self.last_decisions.append(decision)
            if decision.intent is not None:
                intents.append(decision.intent)
        return intents

    # ---- helpers ----
    def _enter(
        self, ctx: SourceContext, symbol: str, signal: Signal, bars: Sequence[Bar], occupied: set[str]
    ) -> Decision:
        s = self.strategy
        if symbol in occupied:
            return Decision(symbol, signal, skipped="already holding or entering")
        if len(occupied) >= s.max_positions:
            return Decision(symbol, signal, skipped=f"max_positions={s.max_positions} reached")
        inst = ctx.snapshot.instruments.get(symbol)
        if inst is None:
            return Decision(symbol, signal, skipped="no instrument data")
        built = build_entry_intent(
            s,
            symbol,
            signal,
            close=bars[-1].close,
            budget=ctx.budget,
            qty_step=inst.qty_step,
            account_id=self.account_id,
            now=ctx.now,
        )
        if isinstance(built, Skip):
            return Decision(symbol, signal, skipped=built.reason)
        return Decision(symbol, signal, built)

    def _pending_entry(self, state: AccountState, symbol: str) -> bool:
        sid = self.strategy.id
        if any(
            o.strategy_id == sid and o.symbol == symbol and o.role == "entry" and not o.is_terminal
            for o in state.orders.values()
        ):
            return True
        return any(  # received/approved but no order yet
            r.intent.strategy_id == sid
            and r.intent.symbol == symbol
            and r.intent.side is Side.BUY
            and not r.abandoned
            and (r.decision is None or (r.decision in ("approve", "resize") and r.order_id is None))
            for r in state.intents.values()
        )

    def _pending_exit(self, state: AccountState, symbol: str) -> bool:
        sid = self.strategy.id
        return any(
            o.strategy_id == sid and o.symbol == symbol and o.role == "exit" and not o.is_terminal
            for o in state.orders.values()
        )

    def _bars_since_exit(self, ctx: SourceContext, symbol: str) -> int | None:
        if self.strategy.cooldown_bars == 0:
            return None
        sid = self.strategy.id
        exits = [
            r.intent.created_at
            for r in ctx.state.intents.values()
            if r.intent.strategy_id == sid
            and r.intent.symbol == symbol
            and r.intent.side is Side.SELL
            and r.decision in ("approve", "resize")
        ]
        if not exits:
            return None
        return int((ctx.now - max(exits)) / timeframe_delta(self.strategy.timeframe))
