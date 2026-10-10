"""Service configuration and secrets."""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from engine.domain.types import Mode
from engine.service.config import Secrets, load_config
from engine.service.logging_setup import RedactingFilter

TOML = """
[service]
poll_interval_s = 1.5
reconcile_interval_s = 30
alert_on_fills = false

[[accounts]]
id = "alpaca-paper"
broker = "alpaca"
mode = "PAPER"
symbols = ["AAPL", "MSFT"]

[accounts.risk]
max_position_pct = 0.1
daily_loss_cap_pct = 0.05

[accounts.strategies.momentum]
strategy_budget = 2500

[[accounts]]
id = "crypto-paper"
broker = "paper-crypto"
mode = "PAPER"
starting_cash = 1000.5
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def test_load_config_keeps_decimals_exact(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, TOML))
    a, c = cfg.accounts
    assert cfg.poll_interval_s == 1.5 and cfg.reconcile_interval_s == 30 and not cfg.alert_on_fills
    assert a.symbols == ("AAPL", "MSFT") and a.risk is not None
    assert a.risk.max_position_pct == Decimal("0.1") and a.risk.daily_loss_cap_pct == Decimal("0.05")
    assert a.strategies["momentum"].strategy_budget == 2500
    assert c.starting_cash == Decimal("1000.5") and c.mode is Mode.PAPER


@pytest.mark.parametrize(
    "bad",
    [
        '[[accounts]]\nid="a"\nbroker="robinhood"\nmode="PAPER"\n',  # RH has no sandbox
        '[[accounts]]\nid="a"\nbroker="paper-crypto"\nmode="PAPER"\n',  # needs starting_cash
        '[[accounts]]\nid="a"\nbroker="paper-crypto"\nmode="LIVE"\nstarting_cash=1\n',
        '[[accounts]]\nid="a"\nbroker="alpaca"\n[[accounts]]\nid="a"\nbroker="alpaca"\n',  # duplicate ids
        '[[accounts]]\nid="a"\nbroker="alpaca"\nsurprise=1\n',  # unknown key
        '[[accounts]]\nid="a"\nbroker="binance"\n',
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, bad))


ENV = {
    "ENGINE_DATABASE_URL": "postgresql://u:pw-SECRET@db/engine",
    "ALPACA_PAPER_KEY_ID": "PKTESTKEYID123",
    "ALPACA_PAPER_SECRET": "alpaca-secret-value-xyz",
    "ROBINHOOD_API_KEY": "rh-api-key-abc",
    "ROBINHOOD_PRIVATE_KEY": "cmgtcHJpdmF0ZS1rZXk=",
    "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/1/tok",
}


def test_secrets_load_from_env_and_never_appear_in_repr() -> None:
    s = Secrets.from_env(ENV)
    assert s.alpaca_paper is not None and s.alpaca_live is None and s.robinhood is not None
    blob = repr(s) + str(s)
    for v in ENV.values():
        assert v not in blob
    assert s.allow_live is False
    assert Secrets.from_env({**ENV, "ENGINE_ALLOW_LIVE": "1"}).allow_live is True
    assert Secrets.from_env({**ENV, "ENGINE_ALLOW_LIVE": "true"}).allow_live is False  # exactly "1"


def test_database_url_is_required() -> None:
    with pytest.raises(ValueError, match="ENGINE_DATABASE_URL"):
        Secrets.from_env({})


def test_half_a_credential_pair_counts_as_missing() -> None:
    s = Secrets.from_env({"ENGINE_DATABASE_URL": "x", "ALPACA_PAPER_KEY_ID": "only-id"})
    assert s.alpaca_paper is None


def test_log_redaction_removes_every_secret_value(caplog: pytest.LogCaptureFixture) -> None:
    s = Secrets.from_env(ENV)
    f = RedactingFilter(s.values())
    rec = logging.LogRecord(
        "t", logging.INFO, "", 0, "using key alpaca-secret-value-xyz and rh-api-key-abc", None, None
    )
    f.filter(rec)
    assert "alpaca-secret-value-xyz" not in rec.getMessage() and "rh-api-key-abc" not in rec.getMessage()
    assert "***" in rec.getMessage()
