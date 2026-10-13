"""The control panel: argument allowlists, request security, and that buttons run the real commands."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from engine.service.config import load_config
from engine.service.panel import BadRequest, Panel, build_argv

TOML = """
[service]
health_port = 1
[[accounts]]
id = "alpaca-paper"
broker = "alpaca"
mode = "PAPER"
[[accounts]]
id = "crypto-paper"
broker = "paper-crypto"
mode = "PAPER"
starting_cash = 1000
crypto_metadata = "x.json"
"""


@pytest.fixture
def cfg_path(tmp_path: Path) -> str:
    p = tmp_path / "c.toml"
    p.write_text(TOML)
    return str(p)


def argv(cfg_path: str, action: str, **params: Any) -> list[str]:
    return build_argv(action, params, cfg_path, load_config(cfg_path))


def test_buttons_become_the_same_cli_commands(cfg_path: str) -> None:
    a = argv(
        cfg_path, "order", account="alpaca-paper", side="BUY", symbol="AAPL", quantity="2", stop_loss="95.5"
    )
    assert (
        a[2:5] == ["engine.service", "submit-intent", "--config"]
        and a[a.index("--asset-class") + 1] == "STOCK"
    )
    assert a[a.index("--stop-loss") + 1] == "95.5" and a[a.index("--quantity") + 1] == "2"
    c = argv(cfg_path, "order", account="crypto-paper", side="SELL", symbol="BTC-USD", quantity="0.001")
    assert c[c.index("--asset-class") + 1] == "CRYPTO" and "--stop-loss" not in c
    assert "--off" in argv(cfg_path, "kill", account="alpaca-paper", off=True)
    assert argv(cfg_path, "approve", id="news.AAPL.7fe4.BUY")[-2:] == ["--id", "news.AAPL.7fe4.BUY"]
    assert "live_paper" in argv(cfg_path, "check_alpaca")


@pytest.mark.parametrize(
    ("action", "params"),
    [
        ("order", {"account": "nope", "side": "BUY", "symbol": "AAPL", "quantity": "1", "stop_loss": "9"}),
        (
            "order",
            {"account": "alpaca-paper", "side": "BUY", "symbol": "AAPL", "quantity": "1"},
        ),  # BUY needs a stop
        (
            "order",
            {
                "account": "alpaca-paper",
                "side": "BUY",
                "symbol": "AAPL; rm -rf /",
                "quantity": "1",
                "stop_loss": "9",
            },
        ),
        ("order", {"account": "alpaca-paper", "side": "HOLD", "symbol": "AAPL", "quantity": "1"}),
        ("order", {"account": "alpaca-paper", "side": "SELL", "symbol": "AAPL", "quantity": "-1"}),
        ("approve", {"id": "x y"}),
        ("approve", {"id": "../../etc"}),
        ("report", {"days": "9999", "horizon": "5"}),
        ("set-mode", {}),  # LIVE cannot be switched from the panel
        ("run", {}),
    ],
)
def test_bad_arguments_and_unlisted_actions_are_refused(
    cfg_path: str, action: str, params: dict[str, Any]
) -> None:
    with pytest.raises(BadRequest):
        argv(cfg_path, action, **params)


def call(panel: Panel, path: str, *, body: Any = None, token: str | None = None, host: str | None = None,
         ctype: str = "application/json") -> tuple[int, Any]:  # fmt: skip
    headers = {"Host": host or f"127.0.0.1:{panel.port}", "Content-Type": ctype}
    if token:
        headers["X-Panel-Token"] = token
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{panel.port}{path}", data=data, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
            raw = r.read()
            return r.status, (
                json.loads(raw) if r.headers["Content-Type"] == "application/json" else raw.decode()
            )
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


@pytest.fixture
def panel(cfg_path: str):  # type: ignore[no-untyped-def]
    import threading

    p = Panel(cfg_path, port=0)
    t = threading.Thread(target=p._server.serve_forever, daemon=True)
    t.start()
    yield p
    p._server.shutdown()
    p._server.server_close()


def test_the_page_carries_the_token_and_requests_need_it(panel: Panel) -> None:
    status, page = call(panel, "/")
    assert status == 200 and panel.token in page and "Trading Engine Panel" in page
    assert call(panel, "/api/state")[0] == 403  # no token
    assert call(panel, "/api/state", token="wrong")[0] == 403  # noqa: S106
    assert call(panel, "/api/run", body={"action": "status"})[0] == 403
    assert call(panel, "/api/run", body={"action": "status"}, token="wrong")[0] == 403  # noqa: S106


def test_a_foreign_host_header_is_rejected_even_with_the_token(panel: Panel) -> None:
    assert call(panel, "/", host="evil.example.com")[0] == 403  # DNS rebinding
    assert (
        call(panel, "/api/run", body={"action": "status"}, token=panel.token, host="evil.example.com:80")[0]
        == 403
    )


def test_posts_must_be_json_and_bad_requests_run_nothing(panel: Panel) -> None:
    assert call(panel, "/api/run", body={"action": "status"}, token=panel.token, ctype="text/plain")[0] == 415
    status, res = call(
        panel, "/api/run", body={"action": "order", "params": {"account": "x"}}, token=panel.token
    )
    assert status == 400 and "refused" in res["output"]


def test_state_lists_accounts_and_never_exposes_secrets(
    panel: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENGINE_DATABASE_URL", "postgresql://user:hunter2-secret@localhost:1/none")
    status, state = call(panel, "/api/state", token=panel.token)
    assert status == 200 and [a["id"] for a in state["accounts"]] == ["alpaca-paper", "crypto-paper"]
    assert state["service"]["running"] is False and "hunter2" not in json.dumps(state)


def test_a_button_runs_the_real_command_and_redacts_secrets(
    panel: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an explicit (unreachable) URL beats any real .env, so the test never touches a real database
    monkeypatch.setenv("ENGINE_DATABASE_URL", "postgresql://u:hunter2secret@127.0.0.1:1/none")
    status, res = call(panel, "/api/run", body={"action": "status"}, token=panel.token)
    assert status == 200 and res["ok"] is False and res["code"] != 0
    assert "hunter2secret" not in res["output"]
    from engine.service.panel import _redact

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecretvalue")
    assert _redact("oops sk-ant-supersecretvalue leaked") == "oops *** leaked"
