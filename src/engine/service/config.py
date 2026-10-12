"""Service configuration: a TOML file for behaviour, environment variables for secrets.

This module (and the rest of engine.service) is the ONLY place that reads the environment. Secrets are
SecretStr so they never show up in repr(), logs or error messages.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator

from engine.domain.types import Frozen, Mode, PosDec
from engine.risk.config import RiskOverride

BrokerKind = Literal["alpaca", "paper-crypto", "robinhood"]


class AccountConfig(Frozen):
    id: str = Field(min_length=1)
    broker: BrokerKind
    mode: Mode = Mode.PAPER
    enabled: bool = True
    symbols: tuple[str, ...] = ()  # watchlist: always quoted (plus anything held or requested)
    starting_cash: PosDec | None = None  # paper-crypto only
    crypto_metadata: str | None = None  # path to the tradable-crypto allowlist (robinhood / paper-crypto)
    rules: tuple[str, ...] = ()  # strategy files (TOML) to run on this account
    llm_strategies: tuple[str, ...] = ()  # LLM strategy files (TOML); needs the [llm] section
    volume_scale: PosDec = Decimal(1)  # multiply reported volume (the free IEX feed shows only a slice of it)
    risk: RiskOverride | None = None
    strategies: dict[str, RiskOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> AccountConfig:
        if self.broker == "paper-crypto" and (self.mode is not Mode.PAPER or self.starting_cash is None):
            raise ValueError("paper-crypto accounts are PAPER and need starting_cash")
        if self.broker == "robinhood" and self.mode is not Mode.LIVE:
            raise ValueError("robinhood has no sandbox: use broker='paper-crypto' for crypto paper trading")
        return self


class LLMProviderConfig(Frozen):
    kind: Literal["anthropic", "xai"]
    model: str = Field(min_length=1)
    knowledge_cutoff: date | None = None  # for flagging leakage when scoring against older news
    input_per_mtok: PosDec | None = (
        None  # USD per million tokens; unset -> the budget assumes a pessimistic price
    )
    output_per_mtok: PosDec | None = None


class LLMConfig(Frozen):
    daily_usd: PosDec = Decimal(2)
    monthly_usd: PosDec = Decimal(30)
    news_poll_s: float = Field(default=30.0, gt=0)
    news_max_age_minutes: int = Field(default=120, gt=0)
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)


class ServiceConfig(Frozen):
    poll_interval_s: float = Field(default=2.0, gt=0)
    sync_interval_s: float = Field(default=5.0, gt=0)
    reconcile_interval_s: float = Field(default=60.0, gt=0)
    health_port: int | None = 8081
    alert_on_fills: bool = True
    record_quotes: bool = True  # keep price/spread history in Postgres for honest backtests
    max_consecutive_errors: int = Field(default=5, gt=0)
    llm: LLMConfig | None = None
    risk: RiskOverride | None = (
        None  # GLOBAL layer: replaces preset values; accounts/strategies can only tighten
    )
    accounts: tuple[AccountConfig, ...]

    @model_validator(mode="after")
    def _unique(self) -> ServiceConfig:
        ids = [a.id for a in self.accounts]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate account ids")
        return self


@dataclass(frozen=True)
class Secrets:
    database_url: SecretStr
    alpaca_paper: tuple[SecretStr, SecretStr] | None = None
    alpaca_live: tuple[SecretStr, SecretStr] | None = None
    robinhood: tuple[SecretStr, SecretStr] | None = None  # (api key, base64 private key)
    anthropic_key: SecretStr | None = None
    xai_key: SecretStr | None = None
    discord_webhook: SecretStr | None = None
    allow_live: bool = False
    _all: tuple[str, ...] = field(default=(), repr=False)

    @staticmethod
    def from_env(env: Mapping[str, str]) -> Secrets:
        def pair(a: str, b: str) -> tuple[SecretStr, SecretStr] | None:
            return (SecretStr(env[a]), SecretStr(env[b])) if env.get(a) and env.get(b) else None

        def one(k: str) -> SecretStr | None:
            return SecretStr(env[k]) if env.get(k) else None

        db = env.get("ENGINE_DATABASE_URL")
        if not db:
            raise ValueError("ENGINE_DATABASE_URL is required")
        s = Secrets(
            database_url=SecretStr(db),
            alpaca_paper=pair("ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET"),
            alpaca_live=pair("ALPACA_LIVE_KEY_ID", "ALPACA_LIVE_SECRET"),
            robinhood=pair("ROBINHOOD_API_KEY", "ROBINHOOD_PRIVATE_KEY"),
            discord_webhook=one("DISCORD_WEBHOOK_URL"),
            anthropic_key=one("ANTHROPIC_API_KEY"),
            xai_key=one("XAI_API_KEY"),
            allow_live=env.get("ENGINE_ALLOW_LIVE") == "1",
        )
        values = [db]
        for p in (s.alpaca_paper, s.alpaca_live, s.robinhood):
            if p:
                values += [x.get_secret_value() for x in p]
        for k in (s.discord_webhook, s.anthropic_key, s.xai_key):
            if k:
                values.append(k.get_secret_value())
        object.__setattr__(s, "_all", tuple(v for v in values if v))
        return s

    def values(self) -> tuple[str, ...]:
        """Every secret string, for log redaction."""
        return self._all


def load_config(path: str | Path) -> ServiceConfig:
    data = tomllib.loads(Path(path).read_text())
    service = data.get("service", {})
    accounts = [_account(a) for a in data.get("accounts", [])]
    extra: dict[str, object] = {}
    if isinstance(data.get("risk"), dict):
        extra["risk"] = {k: _num(v) for k, v in data["risk"].items()}
    if isinstance(data.get("llm"), dict):
        extra["llm"] = _floats_to_str(data["llm"])
    return ServiceConfig.model_validate({**service, **extra, "accounts": accounts})


def _account(raw: dict[str, object]) -> dict[str, object]:
    out = dict(raw)
    for key in ("risk",):
        if key in out and isinstance(out[key], dict):
            out[key] = {k: _num(v) for k, v in out[key].items()}  # type: ignore[attr-defined]
    if isinstance(out.get("strategies"), dict):
        out["strategies"] = {
            sid: {k: _num(v) for k, v in over.items()}
            for sid, over in out["strategies"].items()  # type: ignore[attr-defined]
        }
    for key in ("starting_cash", "volume_scale"):
        if isinstance(out.get(key), (int, float)):
            out[key] = str(out[key])
    return out


def _num(v: object) -> object:
    """TOML floats become strings so they reach pydantic as exact decimals, never as binary floats."""
    return str(v) if isinstance(v, float) else v


def _floats_to_str(v: object) -> object:
    if isinstance(v, dict):
        return {k: _floats_to_str(x) for k, x in v.items()}
    return _num(v)
