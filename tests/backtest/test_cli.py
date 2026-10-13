"""Backtest CLI and the shipped example strategies."""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import pytest

from engine.broker.base import ProtectionMode
from engine.domain.types import AssetClass
from engine.marketdata.history import to_csv
from engine.service import backtest_cli
from engine.service.backtest_cli import main
from engine.strategy.schema import StrategyError, load_strategy, validate_for_venue
from tests.helpers import T0, bars_from_closes

ROOT = Path(__file__).resolve().parents[2]
STRATEGY = """
id = "wave"
universe = ["BTC-USD"]
timeframe = "1h"
lookback = 60
[indicators]
fast = { type = "sma", period = 3 }
slow = { type = "sma", period = 9 }
atr  = { type = "atr", period = 5 }
[entry]
cross_above = ["fast", "slow"]
[exit]
cross_below = ["fast", "slow"]
[stop]
type = "atr"
multiple = 3
[sizing]
type = "fixed_quantity"
quantity = 5
"""


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    closes = [100 + round(12 * math.sin(i / 7)) for i in range(220)]
    to_csv(bars_from_closes(closes, spread="1", volume="500000000"), tmp_path / "BTC-USD_1h.csv")
    (tmp_path / "wave.toml").write_text(STRATEGY)
    return tmp_path


def args(w: Path, command: str, *extra: str) -> list[str]:
    return [command, "--strategy", str(w / "wave.toml"), "--venue", "robinhood-crypto", "--data", f"csv:{w}",
            "--spread-bps", "20", *extra]  # fmt: skip


def test_run_prints_a_summary_and_writes_the_reports(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(workdir, "run", "--out", str(workdir / "out")), {}) == 0
    out = capsys.readouterr().out
    assert "Backtest" in out and "robinhood-crypto" in out and "Trades" in out and "Risk engine" in out
    assert {p.name for p in (workdir / "out").iterdir()} == {
        "summary.json",
        "summary.txt",
        "trades.csv",
        "equity.csv",
    }


def test_the_default_risk_limit_blocks_a_wide_spread_and_the_report_says_so(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wide = [a for a in args(workdir, "run") if a not in ("--spread-bps", "20")] + ["--spread-bps", "400"]
    assert main(wide, {}) == 0
    out = capsys.readouterr().out
    assert "SLIPPAGE_TOO_HIGH" in out and "NO TRADES" in out


def test_walk_forward_and_sensitivity_commands(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(args(workdir, "walk-forward", "--train-bars", "80", "--test-bars", "40", "--min-trades", "1",
                     "--grid", "indicators.fast.period=2,3", "--grid", "stop.multiple=2,3"), {}) == 0  # fmt: skip
    wf = capsys.readouterr().out
    assert "Walk-forward over" in wf and "out-of-sample return" in wf and "parameter stability" in wf
    assert (
        main(
            args(workdir, "sensitivity", "--param", "indicators.fast.period", "--param", "stop.multiple"), {}
        )
        == 0
    )
    assert "nearby settings are profitable" in capsys.readouterr().out


def test_bad_inputs_exit_with_a_clear_error(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(args(workdir, "run")[:-2] + ["--data", "alpaca"], {}) == 2  # alpaca data needs keys
    assert "ALPACA_PAPER_KEY_ID" in capsys.readouterr().err
    assert (
        main(
            [
                "run",
                "--strategy",
                str(workdir / "missing.toml"),
                "--venue",
                "robinhood-crypto",
                "--data",
                f"csv:{workdir}",
            ],
            {},
        )
        == 2
    )
    assert main(args(workdir, "walk-forward", "--train-bars", "80", "--test-bars", "40",
                     "--grid", "a=" + ",".join(map(str, range(30))), "--grid", "b=" + ",".join(map(str, range(30)))), {}) == 2  # fmt: skip
    assert "overfits" in capsys.readouterr().err


def test_the_shipped_examples_are_valid_and_respect_their_venues() -> None:
    crypto = load_strategy(ROOT / "strategies/examples/ema-cross-btc.toml")
    stocks = load_strategy(ROOT / "strategies/examples/rsi-dip-stocks.toml")
    validate_for_venue(crypto, ProtectionMode.SEPARATE_ORDERS, AssetClass.CRYPTO)  # Robinhood: stop only
    validate_for_venue(stocks, ProtectionMode.NATIVE_BRACKET, AssetClass.STOCK)  # Alpaca: bracket allowed
    with pytest.raises(StrategyError, match="take_profit"):
        validate_for_venue(
            stocks.model_copy(update={"universe": ("AAPL",)}),
            ProtectionMode.SEPARATE_ORDERS,
            AssetClass.STOCK,
        )
    assert crypto.warmup == 49 and stocks.warmup == 100 and crypto.hash != stocks.hash


def test_replay_command_reports_zero_error_for_a_perfectly_modelled_run(
    pg_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from engine.eventlog.postgres import PostgresEventStore
    from engine.service.inbox import InMemoryInbox  # noqa: F401  (the live harness builds its own)
    from tests.backtest.test_parity import live, wavy_bars

    bars = wavy_bars()
    store = PostgresEventStore(pg_dsn)
    live(bars, store=store)
    store.close()
    to_csv(bars, tmp_path / "AAPL_1h.csv")
    code = main(
        ["replay", "--account", "acct", "--venue", "alpaca-stocks", "--data", f"csv:{tmp_path}",
         "--timeframe", "1h", "--symbols", "AAPL", "--spread-bps", "0", "--slippage-bps", "0",
         "--fee-rate", "0", "--adv-usd", "1000000000"],
        {"ENGINE_DATABASE_URL": pg_dsn},
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "mean fill error 0.00 bps" in out and "0 filled in only one world" in out
    assert "WARNING" not in out


def test_compare_prints_a_scoreboard_and_locks_the_exam_until_asked(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exam = (T0 + timedelta(hours=150)).strftime("%Y-%m-%dT%H:%M:%S")
    ledger = workdir / "ledger.json"
    common = args(workdir, "compare", "--train-bars", "60", "--test-bars", "30", "--exam-start", exam,
                  "--ledger", str(ledger), "--out", str(workdir / "cmp"))  # fmt: skip
    assert main(common, {}) == 0
    out = capsys.readouterr().out
    assert "Strategy comparison" in out and "wave@" in out and "EXAM NOT RUN" in out and "window 3" in out
    assert not ledger.exists()  # looking at the rolling windows does not use up the exam

    assert main([*common, "--final-exam"], {}) == 0
    out = capsys.readouterr().out
    assert "final exam" in out and "EXAM NOT RUN" not in out and ledger.exists()
    assert {p.name for p in (workdir / "cmp").iterdir()} == {"scoreboard.txt", "scoreboard.json"}

    assert main([*common, "--final-exam"], {}) == 0  # the same file under the same settings stays clean
    assert "EXAM ALREADY SEEN" not in capsys.readouterr().out

    (workdir / "wave.toml").write_text(
        (workdir / "wave.toml").read_text().replace("quantity = 5", "quantity = 6")
    )
    assert main([*common, "--final-exam"], {}) == 0
    assert "EXAM ALREADY SEEN" in capsys.readouterr().out  # edited after its exam: never a clean pass


def test_compare_refuses_an_exam_start_with_too_little_history_before_it(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exam = (T0 + timedelta(hours=40)).strftime("%Y-%m-%dT%H:%M:%S")
    assert (
        main(args(workdir, "compare", "--train-bars", "60", "--test-bars", "30", "--exam-start", exam), {})
        == 2
    )
    assert "not enough history before the exam" in capsys.readouterr().err


def test_rerunning_the_exam_under_different_settings_is_marked_as_seen(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exam = (T0 + timedelta(hours=150)).strftime("%Y-%m-%dT%H:%M:%S")
    later = (T0 + timedelta(hours=160)).strftime("%Y-%m-%dT%H:%M:%S")
    ledger = workdir / "ledger.json"

    def run(*extra: str) -> str:
        cmd = args(workdir, "compare", "--train-bars", "60", "--test-bars", "30", "--ledger", str(ledger),
                   "--final-exam", *extra)  # fmt: skip
        assert main(cmd, {}) == 0
        return capsys.readouterr().out

    assert "EXAM ALREADY SEEN" not in run("--exam-start", exam)
    assert "EXAM ALREADY SEEN" not in run("--exam-start", exam)
    assert "EXAM ALREADY SEEN" in run("--exam-start", exam, "--grid", "indicators.fast.period=2,3")
    assert "EXAM ALREADY SEEN" in run("--exam-start", later)
    assert "EXAM ALREADY SEEN" in run("--exam-start", exam, "--cash", "50000")


def test_a_failure_partway_through_compare_records_no_exam(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    closes = [100 + round(12 * math.sin(i / 5)) for i in range(100)]  # ETH-USD stops after 100 bars
    to_csv(bars_from_closes(closes, spread="1", volume="500000000"), workdir / "ETH-USD_1h.csv")
    (workdir / "pair.toml").write_text(
        STRATEGY.replace('id = "wave"', 'id = "pair"').replace('["BTC-USD"]', '["BTC-USD", "ETH-USD"]')
    )
    exam = (T0 + timedelta(hours=150)).strftime("%Y-%m-%dT%H:%M:%S")
    ledger = workdir / "ledger.json"
    cmd = args(workdir, "compare", "--strategy", str(workdir / "pair.toml"), "--train-bars", "60",
               "--test-bars", "30", "--exam-start", exam, "--ledger", str(ledger), "--final-exam")  # fmt: skip
    assert main(cmd, {}) == 2
    assert "no bars for ETH-USD" in capsys.readouterr().err
    assert not ledger.exists()  # "wave" was fine, but its exam is not used up by a run that failed


def test_the_default_ledger_lives_in_the_project_not_the_current_directory(
    workdir: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    assert backtest_cli.PROJECT_ROOT == ROOT  # the trading-engine folder
    project, elsewhere = tmp_path_factory.mktemp("project"), tmp_path_factory.mktemp("elsewhere")
    monkeypatch.setattr(backtest_cli, "PROJECT_ROOT", project)
    monkeypatch.chdir(elsewhere)
    exam = (T0 + timedelta(hours=150)).strftime("%Y-%m-%dT%H:%M:%S")
    cmd = args(workdir, "compare", "--train-bars", "60", "--test-bars", "30", "--exam-start", exam,
               "--final-exam")  # fmt: skip
    assert main(cmd, {}) == 0
    ledger = project / "runs" / "exam-ledger.json"
    assert ledger.exists() and not (elsewhere / "runs").exists()
    assert f"note: creating a new exam ledger at {ledger}" in capsys.readouterr().err
    assert main(cmd, {}) == 0
    assert "creating a new exam ledger" not in capsys.readouterr().err  # only when it did not exist
