"""Turn a Signal into an Intent: stop and take-profit prices, position size, and a deterministic id.

The intent id is derived from (strategy, symbol, bar close time, side), so evaluating the same bar twice (a
restarted service, a re-run backtest) yields the same id and the engine's idempotency drops the duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal

from engine.domain.intent import Intent
from engine.domain.types import AssetClass, Side
from engine.marketdata.history import timeframe_delta
from engine.strategy.evaluator import Signal
from engine.strategy.schema import StrategyDef

PRICE_QUANTUM = Decimal("0.00000001")


@dataclass(frozen=True)
class Skip:
    """No intent, with the reason (surfaced in backtest reports and logs)."""

    reason: str


def intent_id_for(strategy_id: str, symbol: str, bar_close: datetime, side: Side) -> str:
    return f"{strategy_id}.{symbol}.{bar_close:%Y%m%dT%H%M%SZ}.{side.value}"


def floor_to(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def stop_price_for(strategy: StrategyDef, close: Decimal, signal: Signal) -> Decimal | Skip:
    spec = strategy.stop
    if spec.type == "percent":
        assert spec.pct is not None
        stop = close * (1 - spec.pct)
    else:
        assert spec.multiple is not None
        atr = signal.values.get(spec.indicator)
        if atr is None:
            return Skip(f"stop indicator {spec.indicator!r} not ready")
        stop = close - spec.multiple * atr
    stop = stop.quantize(PRICE_QUANTUM, rounding=ROUND_DOWN)
    if stop <= 0 or stop >= close:
        return Skip(f"stop {stop} is not below the price {close}")
    return stop


def build_entry_intent(
    strategy: StrategyDef,
    symbol: str,
    signal: Signal,
    *,
    close: Decimal,
    budget: Decimal,
    qty_step: Decimal,
    account_id: str,
    now: datetime,
) -> Intent | Skip:
    stop = stop_price_for(strategy, close, signal)
    if isinstance(stop, Skip):
        return stop
    sizing = strategy.sizing
    if sizing.type == "risk_fraction":
        assert sizing.risk_pct is not None
        raw_qty = (sizing.risk_pct * budget) / (close - stop)
    elif sizing.type == "fixed_notional":
        assert sizing.notional is not None
        raw_qty = sizing.notional / close
    else:
        assert sizing.quantity is not None
        raw_qty = sizing.quantity
    qty = floor_to(raw_qty, qty_step)
    if qty <= 0:
        return Skip("position size rounds to zero")

    take_profit = None
    if strategy.take_profit is not None:
        tp = strategy.take_profit
        if tp.type == "r_multiple":
            assert tp.multiple is not None
            take_profit = close + tp.multiple * (close - stop)
        else:
            assert tp.pct is not None
            take_profit = close * (1 + tp.pct)
        take_profit = take_profit.quantize(PRICE_QUANTUM, rounding=ROUND_DOWN)

    shown = ", ".join(f"{k}={v:.6g}" for k, v in sorted(signal.values.items()))
    return Intent(
        intent_id=intent_id_for(strategy.id, symbol, signal.bar_close, Side.BUY),
        account_id=account_id,
        strategy_id=strategy.id,
        symbol=symbol,
        asset_class=strategy.asset_class,
        side=Side.BUY,
        quantity=qty,
        stop_loss_price=stop,
        take_profit_price=take_profit,
        confidence=Decimal(1),
        rationale=f"{signal.reason}; close={close}; {shown}"[:500],
        source=f"rule:{strategy.id}@{strategy.hash}",
        created_at=now,
        expires_at=now + timeframe_delta(strategy.timeframe),
    )


def build_exit_intent(
    strategy: StrategyDef,
    symbol: str,
    signal: Signal,
    *,
    position: Decimal,
    account_id: str,
    now: datetime,
) -> Intent:
    return Intent(
        intent_id=intent_id_for(strategy.id, symbol, signal.bar_close, Side.SELL),
        account_id=account_id,
        strategy_id=strategy.id,
        symbol=symbol,
        asset_class=strategy.asset_class
        if isinstance(strategy.asset_class, AssetClass)
        else AssetClass.STOCK,
        side=Side.SELL,
        quantity=position,
        confidence=Decimal(1),
        rationale=signal.reason,
        source=f"rule:{strategy.id}@{strategy.hash}",
        created_at=now,
        expires_at=now + timeframe_delta(strategy.timeframe),
    )
