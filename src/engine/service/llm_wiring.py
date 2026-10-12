"""Builds the language-model side of the service from configuration and secrets.

The only code that turns an API key into a provider. Everything downstream (the strategy source, the budget, the
recorder) sees providers through the LLMProvider interface and holds no keys.
"""

from __future__ import annotations

from collections.abc import Callable

from engine.clock import Clock
from engine.domain.events import LlmDecisionMade
from engine.eventlog.base import EventStore, make_event
from engine.llm.anthropic import AnthropicProvider
from engine.llm.budget import BudgetPolicy, BudgetTracker, Pricing
from engine.llm.prompts import PromptRegistry
from engine.llm.provider import LLMProvider
from engine.llm.recorder import InMemoryLLMRecorder, LLMRecorder, PostgresLLMRecorder
from engine.llm.xai import XAIProvider
from engine.news.archive import InMemoryNewsArchive, NewsArchive, PostgresNewsArchive
from engine.service.alerts import Alert, AlertSink, Level
from engine.service.config import LLMConfig, Secrets
from engine.strategy.approvals import InMemoryProposalStore, PostgresProposalStore, ProposalStore
from engine.strategy.llm_source import LLMDeps


def build_providers(cfg: LLMConfig, secrets: Secrets) -> tuple[dict[str, LLMProvider], dict[str, Pricing]]:
    """One provider per configured name. A provider whose key is missing is simply left out (and a strategy that
    needs it as primary fails closed with a clear alert), rather than crashing the whole service."""
    providers: dict[str, LLMProvider] = {}
    pricing: dict[str, Pricing] = {}
    for name, pc in cfg.providers.items():
        key = secrets.anthropic_key if pc.kind == "anthropic" else secrets.xai_key
        if key is None:
            continue
        if pc.kind == "anthropic":
            providers[name] = AnthropicProvider(
                key, pc.model, knowledge_cutoff=pc.knowledge_cutoff, name=name
            )
        else:
            providers[name] = XAIProvider(key, pc.model, knowledge_cutoff=pc.knowledge_cutoff, name=name)
        if pc.input_per_mtok is not None and pc.output_per_mtok is not None:
            pricing[name] = Pricing(input_per_mtok=pc.input_per_mtok, output_per_mtok=pc.output_per_mtok)
    return providers, pricing


def audit_sink(store: EventStore, clock: Clock) -> Callable[[LlmDecisionMade], None]:
    def write(ev: LlmDecisionMade) -> None:
        store.append([make_event(ev, stream_id=f"account:{ev.account_id}", ts=clock.now())])

    return write


def alert_notify(alerts: AlertSink) -> Callable[[str, str, str, str | None], None]:
    def notify(level: str, title: str, body: str, key: str | None = None) -> None:
        alerts.send(Alert(Level(level), title, body, key))

    return notify


def build_llm_deps(
    cfg: LLMConfig,
    secrets: Secrets,
    store: EventStore,
    clock: Clock,
    alerts: AlertSink,
    *,
    dsn: str | None = None,
    providers: dict[str, LLMProvider] | None = None,
    pricing: dict[str, Pricing] | None = None,
) -> LLMDeps:
    """`dsn` selects Postgres-backed recorder/archive/proposals; without it everything is in memory (tests)."""
    built, prices = build_providers(cfg, secrets)
    recorder: LLMRecorder = PostgresLLMRecorder(dsn) if dsn else InMemoryLLMRecorder()
    archive: NewsArchive = PostgresNewsArchive(dsn) if dsn else InMemoryNewsArchive()
    proposals: ProposalStore = PostgresProposalStore(dsn) if dsn else InMemoryProposalStore()
    notify = alert_notify(alerts)
    budget = BudgetTracker(
        BudgetPolicy(daily_usd=cfg.daily_usd, monthly_usd=cfg.monthly_usd),
        recorder,
        clock,
        alert=lambda level, title, body: notify(level, title, body, f"llm-budget:{title}"),
    )
    return LLMDeps(
        providers=providers if providers is not None else built,
        pricing=pricing if pricing is not None else prices,
        recorder=recorder,
        budget=budget,
        prompts=PromptRegistry(),
        archive=archive,
        audit=audit_sink(store, clock),
        notify=notify,
        proposals=proposals,
    )
