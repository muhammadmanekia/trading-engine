"""Spending caps for model calls. Fails closed: if a call could push spend past a cap, it is not made.

Token counts for a call are only known afterwards, so before calling we price the WORST case (estimated input
tokens plus the full output allowance) at the provider's price. Prices come from configuration; if a provider has
none configured we assume a deliberately high default so a missing price can only make us more cautious.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from engine.clock import Clock
from engine.domain.types import Frozen, PosDec
from engine.llm.provider import LLMBudgetExceeded
from engine.llm.recorder import LLMRecorder

MILLION = Decimal(1_000_000)


class Pricing(Frozen):
    input_per_mtok: PosDec
    output_per_mtok: PosDec


# Used when a provider has no configured price. Intentionally pessimistic.
UNCONFIGURED_PRICING = Pricing(input_per_mtok=Decimal(15), output_per_mtok=Decimal(75))


class BudgetPolicy(Frozen):
    daily_usd: PosDec = Decimal("2")
    monthly_usd: PosDec = Decimal("30")
    warn_fraction: Decimal = Decimal("0.8")


def cost_usd(pricing: Pricing, input_tokens: int, output_tokens: int) -> Decimal:
    return (pricing.input_per_mtok * input_tokens + pricing.output_per_mtok * output_tokens) / MILLION


def estimate_input_tokens(*texts: str) -> int:
    """Conservative: roughly one token per 3 characters (real English averages about 4)."""
    return math.ceil(sum(len(t) for t in texts) / 3)


def worst_case_cost(pricing: Pricing, system: str, user: str, max_output_tokens: int) -> Decimal:
    return cost_usd(pricing, estimate_input_tokens(system, user), max_output_tokens)


class BudgetTracker:
    def __init__(
        self,
        policy: BudgetPolicy,
        recorder: LLMRecorder,
        clock: Clock,
        alert: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.policy = policy
        self._recorder = recorder
        self._clock = clock
        self._alert = alert  # (level, title, body)
        self._warned: set[tuple[str, str]] = set()

    def _day_start(self) -> datetime:
        n = self._clock.now()
        return n.replace(hour=0, minute=0, second=0, microsecond=0)

    def _month_start(self) -> datetime:
        return self._day_start().replace(day=1)

    def spent_today(self, strategy_id: str | None = None) -> Decimal:
        return self._recorder.cost_since(self._day_start(), strategy_id)

    def spent_this_month(self) -> Decimal:
        return self._recorder.cost_since(self._month_start())

    def check(
        self, projected: Decimal, *, strategy_id: str, strategy_daily_usd: Decimal | None = None
    ) -> None:
        """Raise LLMBudgetExceeded if a call costing up to `projected` would cross any cap."""
        checks = [
            ("daily", self.spent_today() + projected, self.policy.daily_usd),
            ("monthly", self.spent_this_month() + projected, self.policy.monthly_usd),
        ]
        if strategy_daily_usd is not None:
            checks.append(
                (
                    f"strategy {strategy_id} daily",
                    self.spent_today(strategy_id) + projected,
                    strategy_daily_usd,
                )
            )
        for label, would_be, cap in checks:
            if would_be > cap:
                self._notify(
                    "CRITICAL",
                    f"LLM {label} budget reached",
                    f"{would_be:.4f} would exceed the ${cap} cap",
                    label,
                )
                raise LLMBudgetExceeded(f"{label} budget: ${would_be:.4f} would exceed ${cap}")
            if would_be > cap * self.policy.warn_fraction:
                self._notify(
                    "WARNING",
                    f"LLM {label} budget at {would_be / cap:.0%}",
                    f"${would_be:.4f} of ${cap}",
                    label,
                )

    def _notify(self, level: str, title: str, body: str, label: str) -> None:
        key = (label, f"{self._clock.now():%Y-%m-%d}:{level}")
        if self._alert is not None and key not in self._warned:
            self._warned.add(key)
            self._alert(level, title, body)
