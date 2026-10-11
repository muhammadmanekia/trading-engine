"""Replay a real (paper/live) run through the simulator and measure how far reality drifted from the model.

Take the intents the account actually received, feed the same intents to the BacktestEngine at the bars they
arrived in, and compare simulated fills with the real ones. This is the evidence for the rollout rule "raise
budgets only when live results match paper within tolerance": if the simulator tracks reality closely, its
verdicts on a new strategy can be trusted; if not, the cost model needs fixing before believing any backtest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from engine.backtest.engine import BacktestConfig, BacktestEngine
from engine.domain.events import AccountFunded, BaselineImported, IntentReceived, OrderFilled
from engine.domain.intent import Intent
from engine.eventlog.base import StoredEvent, decode
from engine.marketdata.history import Bar, timeframe_delta
from engine.strategy.source import SourceContext

ZERO = Decimal(0)
BPS = Decimal(10_000)


class ReplaySource:
    """A SignalSource that emits previously recorded intents at the bar close that just preceded them."""

    def __init__(self, intents: Sequence[Intent], timeframe: str) -> None:
        self.name = "replay"
        self.timeframe = timeframe
        self.symbols = tuple(sorted({i.symbol for i in intents}))
        self._intents = sorted(intents, key=lambda i: i.created_at)
        self._delta = timeframe_delta(timeframe)

    def history_needed(self) -> int:
        return 1

    def on_bar_close(self, ctx: SourceContext) -> list[Intent]:
        due = [i for i in self._intents if ctx.now <= i.created_at < ctx.now + self._delta]
        # same id, but timestamps re-expressed in simulated time so expiry checks behave
        return [
            i.model_copy(update={"created_at": ctx.now, "expires_at": ctx.now + self._delta}) for i in due
        ]


@dataclass(frozen=True)
class FillComparison:
    intent_id: str
    symbol: str
    side: str
    actual_qty: Decimal
    actual_price: Decimal | None
    simulated_qty: Decimal
    simulated_price: Decimal | None

    @property
    def price_error_bps(self) -> Decimal | None:
        """Simulated minus actual, in basis points of the actual price (positive = the model was pessimistic for buys)."""
        if self.actual_price is None or self.simulated_price is None:
            return None
        return (self.simulated_price - self.actual_price) / self.actual_price * BPS

    @property
    def matched(self) -> bool:
        return (self.actual_qty > 0) == (self.simulated_qty > 0)


@dataclass
class TrackingReport:
    comparisons: list[FillComparison]
    mean_abs_error_bps: Decimal | None
    max_abs_error_bps: Decimal | None
    mismatches: list[FillComparison]
    warnings: list[str] = field(default_factory=list)


def intents_from_events(events: Sequence[StoredEvent], account_id: str) -> list[Intent]:
    out = []
    for e in events:
        if e.type == "IntentReceived" and e.account_id == account_id:
            ev = decode(e)
            assert isinstance(ev, IntentReceived)
            out.append(ev.intent)
    return out


def actual_fills(events: Sequence[StoredEvent], account_id: str) -> dict[str, tuple[Decimal, Decimal | None]]:
    """order_id -> (filled quantity, average price). Intents map to orders as 'ord-<intent_id>'."""
    qty: dict[str, Decimal] = {}
    value: dict[str, Decimal] = {}
    for e in events:
        if e.type == "OrderFilled" and e.account_id == account_id:
            ev = decode(e)
            assert isinstance(ev, OrderFilled)
            qty[ev.order_id] = qty.get(ev.order_id, ZERO) + ev.quantity
            value[ev.order_id] = value.get(ev.order_id, ZERO) + ev.quantity * ev.price
    return {k: (q, value[k] / q if q else None) for k, q in qty.items()}


def starting_cash(events: Sequence[StoredEvent], account_id: str) -> Decimal:
    for e in events:
        if e.account_id != account_id:
            continue
        ev = decode(e)
        if isinstance(ev, AccountFunded):
            return ev.amount
        if isinstance(ev, BaselineImported):
            return ev.cash
    raise ValueError(f"account {account_id!r} has no opening balance in the log")


def replay_account(
    events: Sequence[StoredEvent],
    account_id: str,
    data: Mapping[str, Sequence[Bar]],
    config: BacktestConfig,
    timeframe: str,
    *,
    since: datetime | None = None,
) -> TrackingReport:
    intents = [i for i in intents_from_events(events, account_id) if since is None or i.created_at >= since]
    if not intents:
        return TrackingReport([], None, None, [], ["the account has no intents to replay"])
    result = BacktestEngine([ReplaySource(intents, timeframe)], data, config).run()
    simulated = actual_fills(result.events, config.account_id)
    real = actual_fills(events, account_id)

    comparisons = []
    for i in intents:
        order_id = f"ord-{i.intent_id}"
        aq, ap = real.get(order_id, (ZERO, None))
        sq, sp = simulated.get(order_id, (ZERO, None))
        comparisons.append(FillComparison(i.intent_id, i.symbol, i.side.value, aq, ap, sq, sp))
    errors = [abs(c.price_error_bps) for c in comparisons if c.price_error_bps is not None]
    mismatches = [c for c in comparisons if not c.matched]
    report = TrackingReport(
        comparisons=comparisons,
        mean_abs_error_bps=sum(errors, ZERO) / len(errors) if errors else None,
        max_abs_error_bps=max(errors) if errors else None,
        mismatches=mismatches,
    )
    if mismatches:
        report.warnings.append(
            f"{len(mismatches)} of {len(comparisons)} intents were filled in only one of reality and the simulation"
        )
    if report.mean_abs_error_bps is not None and report.mean_abs_error_bps > Decimal(25):
        report.warnings.append(
            f"mean fill-price error is {report.mean_abs_error_bps:.1f} bps: the cost model does not match reality"
        )
    return report
