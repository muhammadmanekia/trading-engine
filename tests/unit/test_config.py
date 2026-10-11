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


def test_global_risk_layer_can_loosen_but_account_layers_can_only_tighten(tmp_path: Path) -> None:
    from engine.risk.config import disposable_money_preset, merge, replace

    cfg = load_config(
        write(
            tmp_path,
            '[risk]\nmax_position_pct = 0.6\n\n[[accounts]]\nid = "a"\nbroker = "alpaca"\n'
            "volume_scale = 25\n[accounts.risk]\nmax_position_pct = 0.9\ncash_reserve_pct = 0.2\n",
        )
    )
    base = replace(disposable_money_preset(), cfg.risk)
    assert (
        base.max_position_pct == Decimal("0.6") > disposable_money_preset().max_position_pct
    )  # global loosened
    acct = merge(base, cfg.accounts[0].risk)
    assert acct.max_position_pct == Decimal("0.6")  # the account asked for 0.9: ignored (cannot loosen)
    assert acct.cash_reserve_pct == Decimal("0.2")  # but it may tighten
    assert cfg.accounts[0].volume_scale == 25


# ---- CLI (control plane needs only the database, never a broker key) --------------------------------
CLI_TOML = """
[[accounts]]
id = "alpaca-paper"
broker = "alpaca"
mode = "PAPER"

[[accounts]]
id = "crypto-live"
broker = "robinhood"
mode = "LIVE"
enabled = false
"""


def cli(pg_dsn: str, tmp_path: Path, *argv: str, extra_env: dict[str, str] | None = None) -> int:
    from engine.service.cli import main

    cfg = write(tmp_path, CLI_TOML)
    args = [argv[0], *(["--config", str(cfg)] if argv[0] != "migrate" else []), *argv[1:]]
    return main(args, {"ENGINE_DATABASE_URL": pg_dsn, **(extra_env or {})})


def test_cli_kill_switch_status_and_adjust(
    pg_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from engine.eventlog.postgres import PostgresEventStore
    from engine.projections.account import AccountProjector

    assert cli(pg_dsn, tmp_path, "migrate") == 0
    assert cli(pg_dsn, tmp_path, "kill", "--account", "alpaca-paper", "--reason", "drill") == 0
    store = PostgresEventStore(pg_dsn)
    assert AccountProjector(store, "alpaca-paper").refresh().kill_switch is True
    assert cli(pg_dsn, tmp_path, "kill", "--account", "alpaca-paper", "--off") == 0
    assert AccountProjector(store, "alpaca-paper").refresh().kill_switch is False
    assert cli(pg_dsn, tmp_path, "adjust", "--account", "alpaca-paper", "--reason", "deposit",
               "--cash-delta", "250.5", "--position", "s1:AAPL:5") == 0  # fmt: skip
    st = AccountProjector(store, "alpaca-paper").refresh()
    assert st.cash == Decimal("250.5") and st.positions[("s1", "AAPL")].quantity == 5
    capsys.readouterr()
    assert cli(pg_dsn, tmp_path, "status") == 0
    out = capsys.readouterr().out
    assert "alpaca-paper: mode=PAPER cash=250.5" in out and "crypto-live" in out
    store.close()


def test_cli_live_mode_needs_explicit_confirmation(pg_dsn: str, tmp_path: Path) -> None:
    from engine.eventlog.postgres import PostgresEventStore
    from engine.projections.account import AccountProjector

    with pytest.raises(PermissionError):
        cli(pg_dsn, tmp_path, "set-mode", "--account", "crypto-live", "--mode", "LIVE")
    assert cli(pg_dsn, tmp_path, "set-mode", "--account", "crypto-live", "--mode", "LIVE", "--confirm") == 0
    store = PostgresEventStore(pg_dsn)
    assert AccountProjector(store, "crypto-live").refresh().mode is Mode.LIVE
    store.close()


def test_cli_rejects_unknown_accounts_and_missing_database(pg_dsn: str, tmp_path: Path) -> None:
    from engine.service.cli import main

    assert cli(pg_dsn, tmp_path, "kill", "--account", "nope") == 2
    assert main(["status", "--config", str(write(tmp_path, CLI_TOML))], {}) == 2


# ---- runtime construction guards --------------------------------------------------------------------
def test_live_accounts_cannot_start_without_the_explicit_environment_flag() -> None:
    from engine.clock import SystemClock
    from engine.eventlog.memory import InMemoryEventStore
    from engine.service.config import AccountConfig
    from engine.service.runtime import LiveNotAllowed, build_runtime

    secrets = Secrets.from_env({**ENV, "ROBINHOOD_API_KEY": "k"})
    acct = AccountConfig(id="live", broker="robinhood", mode=Mode.LIVE)
    with pytest.raises(LiveNotAllowed):
        build_runtime(acct, InMemoryEventStore(), SystemClock(), secrets)


def test_missing_credentials_or_allowlist_are_clear_startup_errors() -> None:
    from engine.clock import SystemClock
    from engine.eventlog.memory import InMemoryEventStore
    from engine.service.config import AccountConfig
    from engine.service.runtime import build_runtime

    store, clock = InMemoryEventStore(), SystemClock()
    no_alpaca = Secrets.from_env({"ENGINE_DATABASE_URL": "x"})
    with pytest.raises(ValueError, match="Alpaca"):
        build_runtime(AccountConfig(id="a", broker="alpaca"), store, clock, no_alpaca)
    with pytest.raises(ValueError, match="Robinhood"):
        build_runtime(
            AccountConfig(id="c", broker="paper-crypto", starting_cash=Decimal(1)), store, clock, no_alpaca
        )
    with_rh = Secrets.from_env({**ENV, "ROBINHOOD_PRIVATE_KEY": ENV["ROBINHOOD_PRIVATE_KEY"]})
    cfg = AccountConfig(id="c", broker="paper-crypto", starting_cash=Decimal(1))
    with pytest.raises(ValueError, match="crypto_metadata"):
        build_runtime(cfg, store, clock, with_rh)


# ---- health endpoint and JSON logging ---------------------------------------------------------------
def test_health_endpoint_reports_status_codes() -> None:
    import json
    import urllib.error
    import urllib.request

    from engine.service.health import HealthServer

    state = {"ok": True, "leader": True}
    server = HealthServer(0, lambda: state)
    server.start()
    try:
        url = f"http://127.0.0.1:{server.port}/healthz"
        with urllib.request.urlopen(url, timeout=3) as r:  # noqa: S310
            assert r.status == 200 and json.loads(r.read())["leader"] is True
        state["ok"] = False
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(url, timeout=3)  # noqa: S310
        assert exc.value.code == 503
        with pytest.raises(urllib.error.HTTPError) as nf:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/secrets", timeout=3)  # noqa: S310
        assert nf.value.code == 404
    finally:
        server.stop()


def test_json_logging_redacts_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from engine.service.logging_setup import configure_logging

    configure_logging(["super-secret-value"])
    logging.getLogger("t").warning("token is %s", "super-secret-value")
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["level"] == "WARNING" and "super-secret-value" not in line["msg"] and "***" in line["msg"]


def test_cli_manual_order_flow_end_to_end(
    pg_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """submit-intent -> (service decides) -> show. Here the 'service' is TradingCore over the same database."""
    import re

    from engine.clock import SystemClock
    from engine.eventlog.postgres import PostgresEventStore
    from engine.service.inbox import PostgresInbox, ingest

    assert cli(pg_dsn, tmp_path, "submit-intent", "--account", "alpaca-paper", "--symbol", "AAPL",
               "--asset-class", "STOCK", "--side", "BUY", "--quantity", "3", "--stop-loss", "90", "--dry-run") == 0  # fmt: skip
    assert "dry run" in capsys.readouterr().out
    assert cli(pg_dsn, tmp_path, "submit-intent", "--account", "alpaca-paper", "--symbol", "AAPL",
               "--asset-class", "STOCK", "--side", "BUY", "--quantity", "3", "--stop-loss", "90") == 0  # fmt: skip
    iid = re.search(r"--intent-id (manual-\w+)", capsys.readouterr().out).group(1)  # type: ignore[union-attr]

    store, clock = PostgresEventStore(pg_dsn), SystemClock()
    assert (
        cli(pg_dsn, tmp_path, "show", "--account", "alpaca-paper", "--intent-id", iid) == 1
    )  # not ingested yet
    ingest(PostgresInbox(pg_dsn), store, clock, {"alpaca-paper"})
    capsys.readouterr()
    assert cli(pg_dsn, tmp_path, "show", "--account", "alpaca-paper", "--intent-id", iid) == 0
    assert "decision: pending" in capsys.readouterr().out
    store.close()
