"""The risk engine: a pure, deterministic, fail-closed function. No I/O, no LLM, no clock reads."""

from __future__ import annotations

from decimal import Decimal

from engine.domain.intent import Intent
from engine.domain.market import MarketSnapshot, SystemState
from engine.domain.types import Side
from engine.projections.account import AccountState
from engine.risk.checks import ENTRY_CHECKS, EXIT_CHECKS, Ctx
from engine.risk.config import RiskConfig
from engine.risk.decision import Decision, Reject, Resize, Trace


def evaluate(
    intent: Intent,
    state: AccountState,
    market: MarketSnapshot,
    system: SystemState,
    config: RiskConfig,
) -> Decision:
    """Never raises. Any internal error or missing input becomes a rejection."""
    try:
        return _evaluate(intent, state, market, system, config)
    except Exception as exc:  # fail closed, by design
        return Decision.reject(
            getattr(intent, "intent_id", "unknown"), "ENGINE_ERROR", f"{type(exc).__name__}: {exc}"
        )


def _evaluate(
    intent: Intent,
    state: AccountState,
    market: MarketSnapshot,
    system: SystemState,
    config: RiskConfig,
) -> Decision:
    # Fail closed on any missing/mistyped input, even for code paths that would not read it.
    for name, value, kind in (
        ("intent", intent, Intent),
        ("state", state, AccountState),
        ("market", market, MarketSnapshot),
        ("system", system, SystemState),
        ("config", config, RiskConfig),
    ):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be {kind.__name__}, got {type(value).__name__}")
    instrument = market.instruments.get(intent.symbol)
    ref: Decimal | None = None
    if instrument is not None:
        ref = instrument.price
        if intent.side is Side.BUY and intent.limit_price is not None:
            ref = max(ref, intent.limit_price)  # size against the worse of limit and market
    ctx = Ctx(
        intent=intent,
        state=state,
        market=market,
        system=system,
        config=config,
        instrument=instrument,
        qty=intent.quantity,
        ref_price=ref,
        prices=market.price_map(),
        trace=Trace(),
    )
    checks = ENTRY_CHECKS if intent.side is Side.BUY else EXIT_CHECKS
    resize_code: str | None = None
    for name, fn in checks:
        out = fn(ctx)
        if isinstance(out, Reject):
            return Decision.reject(
                intent.intent_id,
                out.code,
                out.detail,
                checks_passed=tuple(ctx.trace.passed),
                trips=tuple(ctx.trace.trips),
                reference_price=ref,
            )
        if isinstance(out, Resize):
            ctx.qty = out.quantity
            resize_code = resize_code or out.code
        ctx.trace.passed.append(name)

    if ctx.qty <= 0 or ctx.qty > intent.quantity:  # invariant: never approve more than was asked
        return Decision.reject(intent.intent_id, "ENGINE_INVARIANT", f"qty {ctx.qty}")
    resized = ctx.qty != intent.quantity
    return Decision(
        kind="resize" if resized else "approve",
        intent_id=intent.intent_id,
        code=(resize_code or "RESIZED") if resized else "OK",
        quantity=ctx.qty,
        checks_passed=tuple(ctx.trace.passed),
        trips=tuple(ctx.trace.trips),
        reference_price=ref,
    )
