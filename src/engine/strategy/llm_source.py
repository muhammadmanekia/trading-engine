"""LLMStrategySource: a SignalSource whose idea generator is a language model.

What the model gets: a compact, sanitised data packet (prices, indicators, its current position, recent screened
news) and a system prompt that tells it news is untrusted data. What it may return: a tiny strict schema. What
happens next is entirely deterministic code: validate (the security boundary), size by confidence tier, compute the
stop, and either propose it to a human, hand it to the risk engine, or just record it (shadow). The model holds no
credentials, sets no prices or quantities, and every call is recorded and budgeted.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from functools import partial

from engine.domain.events import LlmDecisionMade
from engine.domain.intent import Intent
from engine.domain.types import Side
from engine.llm import prompts as prompt_mod
from engine.llm.budget import UNCONFIGURED_PRICING, BudgetTracker, Pricing, cost_usd, worst_case_cost
from engine.llm.provider import (
    LLMBudgetExceeded,
    LLMCredentialsError,
    LLMError,
    LLMInvalidResponse,
    LLMProvider,
    LLMRequest,
    LLMUnavailable,
    input_hash,
    sha256_text,
)
from engine.llm.recorder import LLMCall, LLMRecorder
from engine.marketdata.history import Bar
from engine.news.archive import NewsArchive
from engine.news.filters import screen
from engine.news.models import NewsItem, canonical_symbol
from engine.projections.account import AccountState
from engine.strategy import indicators as ind
from engine.strategy.approvals import Proposal, ProposalStore
from engine.strategy.llm_schema import (
    DECISIONS_JSON_SCHEMA,
    SCHEMA_NAME,
    Checked,
    LLMDecision,
    LLMStrategyDef,
    Rejection,
    parse_decisions,
    quantity_for,
    validate_decisions,
)
from engine.strategy.source import SourceContext

log = logging.getLogger("engine.strategy.llm")
ZERO = Decimal(0)
PRICE_QUANTUM = Decimal("0.00000001")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,40}$")
Notify = Callable[[str, str, str, str | None], None]  # (level, title, body, dedupe_key)


@dataclass
class LLMDeps:
    providers: Mapping[str, LLMProvider]
    pricing: Mapping[str, Pricing]
    recorder: LLMRecorder
    budget: BudgetTracker
    prompts: prompt_mod.PromptRegistry
    archive: NewsArchive
    audit: Callable[[LlmDecisionMade], None]
    notify: Notify
    proposals: ProposalStore | None = None


@dataclass
class _Outcome:
    provider: str
    model: str
    call_id: str
    input_hash: str
    response_hash: str | None
    cost: Decimal
    decisions: list[LLMDecision]
    ok: bool


@dataclass(frozen=True)
class _CallMeta:
    call_id: str
    ts: datetime
    mode: str
    provider: str
    model_requested: str
    template: prompt_mod.PromptTemplate
    input_hash: str
    request: LLMRequest
    trigger: str


@dataclass
class RunReport:
    """What one evaluation did, for tests and diagnostics."""

    trigger: str
    outcomes: list[_Outcome] = field(default_factory=list)
    intents: list[Intent] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


class LLMStrategySource:
    def __init__(self, strategy: LLMStrategyDef, account_id: str, deps: LLMDeps) -> None:
        self.strategy = strategy
        self.account_id = account_id
        self.deps = deps
        self.name = strategy.id
        self.timeframe: str = strategy.timeframe
        self.symbols = strategy.universe
        self.consecutive_invalid = 0
        self.paused_reason: str | None = None
        self.last_report: RunReport | None = None
        self._disabled: set[str] = set()

    def history_needed(self) -> int:
        return self.strategy.context_bars

    # ---- SignalSource ----------------------------------------------------------------------------
    def on_bar_close(self, ctx: SourceContext) -> list[Intent]:
        if not self.strategy.review_on_bar_close:
            return []
        return self._run(ctx, f"review:{ctx.now:%Y%m%dT%H%M}", [])

    def on_news(self, ctx: SourceContext, items: Sequence[NewsItem]) -> list[Intent]:
        if not self.strategy.news.enabled:
            return []
        universe = set(self.strategy.universe)
        fresh = [i for i in items if universe & {canonical_symbol(s) for s in i.symbols}]
        if not fresh:
            return []
        return self._run(ctx, "news:" + ",".join(sorted(i.id for i in fresh)), fresh)

    # ---- one evaluation --------------------------------------------------------------------------
    def _run(self, ctx: SourceContext, trigger: str, new_items: Sequence[NewsItem]) -> list[Intent]:
        s, d = self.strategy, self.deps
        report = self.last_report = RunReport(trigger)
        if self.paused_reason:
            return []
        if s.models.primary in self._disabled or s.models.primary not in d.providers:
            self._once("no-primary", "CRITICAL", f"{s.id}: primary model unavailable",
                       f"provider {s.models.primary!r} is not configured or was disabled")  # fmt: skip
            return []

        day_start = ctx.now.replace(hour=0, minute=0, second=0, microsecond=0)
        calls_today = len(d.recorder.calls_since(day_start, s.id))
        if calls_today >= s.limits.max_calls_per_day:
            self._once(
                "max-calls",
                "WARNING",
                f"{s.id}: daily model-call limit reached",
                str(s.limits.max_calls_per_day),
            )
            return []

        template = d.prompts.get(s.prompt, s.prompt_version)
        system = template.render(instructions=_clean_instructions(s.instructions))
        items = self._gather_news(ctx.now, new_items)
        supplied = {i.id for i in items}
        user = build_packet(s, ctx, items, {i.id for i in new_items}, trigger)
        request = LLMRequest(
            system=system, user=user, schema_name=SCHEMA_NAME, json_schema=DECISIONS_JSON_SCHEMA
        )

        names = [
            s.models.primary,
            *[n for n in s.models.shadow if n in d.providers and n not in self._disabled],
        ]
        primary_decisions: list[LLMDecision] | None = None
        per_provider: dict[str, list[LLMDecision]] = {}
        for idx, name in enumerate(names):
            mode = "primary" if idx == 0 else "shadow"
            outcome = self._call(d.providers[name], mode, request, template, trigger, ctx.now)
            if outcome is None:
                continue
            report.outcomes.append(outcome)
            per_provider[name] = outcome.decisions if outcome.ok else []
            if idx == 0:
                primary_decisions = outcome.decisions if outcome.ok else None
        if primary_decisions is None:
            return []

        entries_today = self._entries_today(ctx.state, day_start)
        checked = {
            name: validate_decisions(
                decs, s, supplied_ids=supplied,
                position_qty=lambda sym: ctx.state.position_qty(s.id, sym),
                pending_entry=lambda sym: _pending(ctx.state, s.id, sym, "entry"),
                pending_exit=lambda sym: _pending(ctx.state, s.id, sym, "exit"),
                entries_today=entries_today, news_triggered=bool(new_items),
            )
            for name, decs in per_provider.items()
        }  # fmt: skip
        primary = checked[s.models.primary]
        report.rejected = list(primary.rejected)
        agreed = self._consensus(primary.accepted, checked) if s.models.consensus else primary.accepted

        outcome_by_provider = {o.provider: o for o in report.outcomes}
        self._audit_all(template, checked, outcome_by_provider, agreed)
        if s.models.mode == "shadow_only":
            return []
        return self._act(ctx, agreed, outcome_by_provider[s.models.primary], trigger, report)

    # ---- model call (budget, cache, record) ---------------------------------------------------------
    def _call(
        self,
        provider: LLMProvider,
        mode: str,
        request: LLMRequest,
        template: prompt_mod.PromptTemplate,
        trigger: str,
        now: datetime,
    ) -> _Outcome | None:
        s, d = self.strategy, self.deps
        ih = input_hash(
            provider.name, provider.model, template.hash, request.system, request.user, request.schema_name
        )
        cached = d.recorder.find_by_input_hash(ih)
        if cached is not None and cached.decisions is not None:  # identical question already answered: free
            decs = [LLMDecision.model_validate(x) for x in cached.decisions]
            return _Outcome(
                provider.name,
                cached.model_reported or provider.model,
                cached.call_id,
                ih,
                None,
                ZERO,
                decs,
                True,
            )

        pricing = d.pricing.get(provider.name, UNCONFIGURED_PRICING)
        try:
            d.budget.check(
                worst_case_cost(pricing, request.system, request.user, request.max_output_tokens),
                strategy_id=s.id,
                strategy_daily_usd=s.limits.daily_usd,
            )
        except LLMBudgetExceeded as exc:
            log.warning("%s: %s", s.id, exc)
            return None

        meta = _CallMeta(
            call_id=sha256_text(f"{ih}|{now.isoformat()}|{mode}")[:24], ts=now, mode=mode, provider=provider.name,
            model_requested=provider.model, template=template, input_hash=ih, request=request, trigger=trigger,
        )  # fmt: skip
        started = time.monotonic()

        def took() -> int:
            return int((time.monotonic() - started) * 1000)

        try:
            resp = provider.complete(request)
            decisions = parse_decisions(resp.parsed)
        except LLMCredentialsError as exc:
            self._disabled.add(provider.name)
            self._record(meta, error=f"credentials: {exc}")
            d.notify(
                "CRITICAL",
                f"LLM provider {provider.name} rejected its API key",
                "disabled until restart",
                f"llm-key:{provider.name}",
            )
            return None
        except (LLMUnavailable, LLMInvalidResponse, LLMError, ValueError) as exc:
            self._record(meta, error=f"{type(exc).__name__}: {exc}"[:300], latency=took())
            if mode == "primary":
                self._note_failure(isinstance(exc, (LLMInvalidResponse, ValueError)), str(exc))
            return None

        cost = cost_usd(pricing, resp.input_tokens, resp.output_tokens)
        self._record(
            meta, response=resp.text, reported=resp.model_reported, decisions=[x.model_dump(mode="json") for x in decisions],
            tokens=(resp.input_tokens, resp.output_tokens), cost=cost, latency=resp.latency_ms,
        )  # fmt: skip
        if mode == "primary":
            self.consecutive_invalid = 0
        return _Outcome(
            provider.name,
            resp.model_reported or provider.model,
            meta.call_id,
            ih,
            sha256_text(resp.text),
            cost,
            decisions,
            True,
        )

    def _record(
        self,
        m: _CallMeta,
        *,
        response: str | None = None,
        reported: str | None = None,
        error: str | None = None,
        decisions: list[dict[str, object]] | None = None,
        tokens: tuple[int, int] = (0, 0),
        cost: Decimal = ZERO,
        latency: int = 0,
    ) -> None:
        self.deps.recorder.record(
            LLMCall(
                call_id=m.call_id,
                ts=m.ts,
                account_id=self.account_id,
                strategy_id=self.strategy.id,
                mode=m.mode,
                provider=m.provider,
                model_requested=m.model_requested,
                model_reported=reported,
                prompt_name=m.template.name,
                prompt_version=m.template.version,
                prompt_hash=m.template.hash,
                input_hash=m.input_hash,
                system_text=m.request.system,
                user_text=m.request.user,
                response_text=response,
                valid=error is None,
                error=error,
                decisions=decisions,
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                cost_usd=cost,
                latency_ms=latency,
                trigger=m.trigger,
            )  # fmt: skip
        )

    def _note_failure(self, invalid: bool, why: str) -> None:
        if not invalid:
            return  # an outage is not the model misbehaving
        self.consecutive_invalid += 1
        if self.consecutive_invalid >= self.strategy.limits.max_consecutive_invalid:
            self.paused_reason = f"{self.consecutive_invalid} invalid model replies in a row"
            self.deps.notify(
                "CRITICAL",
                f"{self.strategy.id}: paused",
                f"{self.paused_reason}. Last: {why[:200]}",
                f"llm-paused:{self.strategy.id}",
            )

    # ---- acting on validated decisions ------------------------------------------------------------------
    def _consensus(self, primary: list[LLMDecision], checked: Mapping[str, Checked]) -> list[LLMDecision]:
        shadows = [n for n in self.strategy.models.shadow if n in checked]
        if len(shadows) < len(self.strategy.models.shadow):
            return []  # a required voter is missing: fail closed
        agreed: list[LLMDecision] = []
        for d in primary:
            votes = []
            for name in shadows:
                match = [x for x in checked[name].accepted if x.symbol == d.symbol and x.action == d.action]
                if not match:
                    break
                votes.append(match[0])
            else:
                weakest = min([d, *votes], key=lambda x: x.confidence)
                agreed.append(d.model_copy(update={"confidence": weakest.confidence}))
        return agreed

    def _act(
        self,
        ctx: SourceContext,
        decisions: list[LLMDecision],
        outcome: _Outcome,
        trigger: str,
        report: RunReport,
    ) -> list[Intent]:
        s = self.strategy
        ttl = timedelta(minutes=s.limits.proposal_ttl_minutes)
        intents: list[Intent] = []
        for dec in decisions:
            inst = ctx.snapshot.instruments.get(dec.symbol)
            bars = ctx.bars.get(dec.symbol)
            if inst is None or not bars:
                report.rejected.append(Rejection(dec.symbol, "no market data for this symbol"))
                continue
            close = bars[-1].close
            intent = (
                self._entry(dec, close, inst.qty_step, ctx.now, ttl, trigger, outcome)
                if dec.action == "enter"
                else self._exit(dec, ctx, ttl, trigger, outcome)
            )
            if intent is None:
                report.rejected.append(Rejection(dec.symbol, "position size rounds to zero"))
                continue
            intents.append(intent)

        if s.models.mode == "auto":
            report.intents = intents
            return intents
        store = self.deps.proposals
        for intent in intents:  # propose_only
            matching = next(d for d in decisions if d.symbol == intent.symbol)
            proposal = Proposal(
                proposal_id=intent.intent_id, account_id=self.account_id, strategy_id=s.id, intent=intent,
                thesis=matching.thesis, cited=matching.cited_ids, created_at=ctx.now, expires_at=ctx.now + ttl,
                call_id=outcome.call_id,
            )  # fmt: skip
            if store is not None and store.add(proposal):
                report.proposals.append(proposal)
                self.deps.notify("INFO", f"Trade proposal: {intent.side} {intent.quantity} {intent.symbol}",
                                 _proposal_text(proposal, matching), f"proposal:{proposal.proposal_id}")  # fmt: skip
        return []

    def _entry(
        self,
        dec: LLMDecision,
        close: Decimal,
        step: Decimal,
        now: datetime,
        ttl: timedelta,
        trigger: str,
        o: _Outcome,
    ) -> Intent | None:
        assert dec.stop_loss_pct is not None
        qty = quantity_for(dec, self.strategy.sizing, close, step)
        if qty <= 0:
            return None
        stop = (close * (1 - dec.stop_loss_pct)).quantize(PRICE_QUANTUM, rounding=ROUND_DOWN)
        return Intent(
            intent_id=self._intent_id(dec.symbol, trigger, Side.BUY), account_id=self.account_id,
            strategy_id=self.strategy.id, symbol=dec.symbol, asset_class=self.strategy.asset_class, side=Side.BUY,
            quantity=qty, stop_loss_price=stop, confidence=dec.confidence, rationale=_rationale(dec, o),
            source=f"llm:{self.strategy.id}@{self.strategy.hash}", created_at=now, expires_at=now + ttl,
        )  # fmt: skip

    def _exit(
        self, dec: LLMDecision, ctx: SourceContext, ttl: timedelta, trigger: str, o: _Outcome
    ) -> Intent | None:
        qty = ctx.state.position_qty(self.strategy.id, dec.symbol)
        if qty <= 0:
            return None
        return Intent(
            intent_id=self._intent_id(dec.symbol, trigger, Side.SELL), account_id=self.account_id,
            strategy_id=self.strategy.id, symbol=dec.symbol, asset_class=self.strategy.asset_class, side=Side.SELL,
            quantity=qty, confidence=dec.confidence, rationale=_rationale(dec, o),
            source=f"llm:{self.strategy.id}@{self.strategy.hash}", created_at=ctx.now, expires_at=ctx.now + ttl,
        )  # fmt: skip

    def _intent_id(self, symbol: str, trigger: str, side: Side) -> str:
        return f"{self.strategy.id}.{symbol}.{sha256_text(trigger)[:10]}.{side.value}"

    # ---- audit ---------------------------------------------------------------------------------------------
    def _audit_all(
        self,
        template: prompt_mod.PromptTemplate,
        checked: Mapping[str, Checked],
        outcomes: Mapping[str, _Outcome],
        agreed: list[LLMDecision],
    ) -> None:
        s = self.strategy
        acting = s.models.mode != "shadow_only"
        acted = {(d.symbol, d.action) for d in agreed} if acting else set()
        for name, chk in checked.items():
            o = outcomes[name]
            primary = name == s.models.primary

            emit = partial(self._emit, template, name, o, primary)

            for d in chk.accepted:
                if not primary:
                    emit(d.symbol, d.action, d.confidence, "shadow")
                elif (d.symbol, d.action) in acted:
                    emit(
                        d.symbol, d.action, d.confidence, "intent" if s.models.mode == "auto" else "proposal"
                    )
                elif not acting:
                    emit(d.symbol, d.action, d.confidence, "shadow")
                else:
                    emit(d.symbol, d.action, d.confidence, "rejected", "no consensus among the listed models")
            for d in chk.holds:
                emit(d.symbol, "hold", d.confidence, "hold")
            for r in chk.rejected:
                emit(r.symbol, None, None, "rejected", r.reason)
            if not (chk.accepted or chk.holds or chk.rejected):
                emit(None, None, None, "hold", "empty reply")

    def _emit(
        self,
        template: prompt_mod.PromptTemplate,
        provider: str,
        o: _Outcome,
        primary: bool,
        symbol: str | None,
        action: str | None,
        confidence: Decimal | None,
        result: str,
        reason: str = "",
    ) -> None:
        self.deps.audit(
            LlmDecisionMade(
                account_id=self.account_id,
                strategy_id=self.strategy.id,
                call_id=o.call_id,
                provider=provider,
                model=o.model,
                mode="primary" if primary else "shadow",
                prompt_name=template.name,
                prompt_version=template.version,
                prompt_hash=template.hash,
                input_hash=o.input_hash,
                response_hash=o.response_hash,
                symbol=symbol,
                action=action,
                confidence=confidence,
                outcome=result,
                reason=reason[:300],
                cost_usd=o.cost,
            )  # fmt: skip
        )

    # ---- context ------------------------------------------------------------------------------------------------
    def _gather_news(self, now: datetime, new_items: Sequence[NewsItem]) -> list[NewsItem]:
        n = self.strategy.news
        pool: dict[str, NewsItem] = {
            i.key: i
            for i in self.deps.archive.since(
                now - timedelta(hours=n.lookback_hours), list(self.strategy.universe)
            )
        }
        for item in new_items:
            pool[item.key] = item
        clean: list[NewsItem] = []
        for item in sorted(pool.values(), key=lambda i: i.published_at, reverse=True):
            if not _SAFE_ID.match(item.id) or len(item.symbols) > n.max_symbols_per_item:
                continue
            s = screen(item)  # screen again here: the archive holds raw text, and this is the last gate
            if s.item is not None:
                clean.append(s.item)
            if len(clean) >= n.max_items:
                break
        return clean

    def _entries_today(self, state: AccountState, day_start: datetime) -> int:
        n = sum(
            1 for r in state.intents.values()
            if r.intent.strategy_id == self.strategy.id and r.intent.side is Side.BUY
            and r.intent.created_at >= day_start and r.decision != "reject" and not r.abandoned
        )  # fmt: skip
        pending = self.deps.proposals.count_since(day_start, self.strategy.id) if self.deps.proposals else 0
        return n + pending

    def _once(self, key: str, level: str, title: str, body: str) -> None:
        self.deps.notify(level, title, body, f"llm:{self.strategy.id}:{key}")


# ---- helpers ------------------------------------------------------------------------------------------------------
def _clean_instructions(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def _pending(state: AccountState, strategy_id: str, symbol: str, role: str) -> bool:
    if any(
        o.strategy_id == strategy_id and o.symbol == symbol and o.role == role and not o.is_terminal
        for o in state.orders.values()
    ):
        return True
    side = Side.BUY if role == "entry" else Side.SELL
    return any(
        r.intent.strategy_id == strategy_id and r.intent.symbol == symbol and r.intent.side is side and not r.abandoned
        and (r.decision is None or (r.decision in ("approve", "resize") and r.order_id is None))
        for r in state.intents.values()
    )  # fmt: skip


def _rationale(d: LLMDecision, o: _Outcome) -> str:
    return f"[{o.provider}/{o.model}] {d.thesis} | cited: {','.join(d.cited_ids)} | call {o.call_id}"[:500]


def _proposal_text(p: Proposal, d: LLMDecision) -> str:
    return (
        f"{d.thesis}\nconfidence {d.confidence}, stop {p.intent.stop_loss_price}, expires {p.expires_at:%H:%M} UTC"
        f"\ncited news: {', '.join(p.cited) or 'none'}\napprove:  python -m engine.service approve --config <cfg> --id {p.proposal_id}"
        f"\nreject:   python -m engine.service reject --config <cfg> --id {p.proposal_id}"
    )


def _fmt(x: Decimal | None, places: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def _symbol_line(strategy: LLMStrategyDef, state: AccountState, symbol: str, bars: Sequence[Bar]) -> str:
    if not bars:
        return f"{symbol}: no price data"
    closes = [b.close for b in bars]
    last = closes[-1]

    def ret(n: int) -> Decimal | None:
        return (last / closes[-1 - n] - 1) * 100 if len(closes) > n else None

    atr = ind.atr(list(bars), 14)[-1]
    rsi = ind.rsi(closes, 14)[-1]
    qty = state.position_qty(strategy.id, symbol)
    pos = state.positions.get((strategy.id, symbol))
    if qty > 0 and pos is not None and pos.avg_cost > 0:
        position = f"LONG {qty} @ {_fmt(pos.avg_cost)} ({_fmt((last / pos.avg_cost - 1) * 100)}% since entry)"
    else:
        position = "flat"
    return (
        f"{symbol}: last={_fmt(last, 4)} ret_1bar={_fmt(ret(1))}% ret_5bar={_fmt(ret(5))}% ret_20bar={_fmt(ret(20))}% "
        f"atr14={_fmt(atr / last * 100 if atr else None)}% rsi14={_fmt(rsi)} position={position}"
    )


def build_packet(
    strategy: LLMStrategyDef, ctx: SourceContext, items: Sequence[NewsItem], new_ids: set[str], trigger: str
) -> str:
    """The user message: facts, then the untrusted news inside explicit tags. Everything dynamic is sanitised."""
    lines = [
        "DATA PACKET",
        f"now: {ctx.now:%Y-%m-%d %H:%M} UTC",
        f"trigger: {'new news' if trigger.startswith('news:') else 'scheduled review'}",
        f"symbols: {', '.join(strategy.universe)}",
        f"bar timeframe: {strategy.timeframe}",
        "",
        "MARKET STATE (computed by us from closed bars)",
    ]
    for sym in strategy.universe:
        lines.append(_symbol_line(strategy, ctx.state, sym, ctx.bars.get(sym, ())[-strategy.context_bars :]))
    lines += ["", "NEWS (untrusted data, newest first; items marked new=yes just arrived)"]
    if not items:
        lines.append("(no recent news)")
    for i in items:
        flag = ' new="yes"' if i.id in new_ids else ""
        syms = ",".join(canonical_symbol(s) for s in i.symbols)
        lines.append(
            f'<news_item id="{i.id}" published="{i.published_at:%Y-%m-%d %H:%M}Z" source="{i.source}" symbols="{syms}"{flag}>'
            f"{i.headline}{' — ' + i.summary if i.summary else ''}</news_item>"
        )
    lines += ["", "LEARNING DIGEST", "(none yet)"]
    return "\n".join(lines)
