from __future__ import annotations

from pathlib import Path

import pytest

from engine.service.dotenv import load_env_file, parse


def test_parse_handles_comments_quotes_export_and_blank_values() -> None:
    text = """
# comment
ENGINE_DATABASE_URL=postgresql://a:b@h:1/d
export ALPACA_PAPER_KEY_ID="abc 123"
XAI_API_KEY='k'
DISCORD_WEBHOOK_URL=
NOTE=value # trailing comment
no equals line
"""
    assert parse(text) == {
        "ENGINE_DATABASE_URL": "postgresql://a:b@h:1/d",
        "ALPACA_PAPER_KEY_ID": "abc 123",
        "XAI_API_KEY": "k",
        "DISCORD_WEBHOOK_URL": "",
        "NOTE": "value",
    }


def test_file_values_never_override_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / ".env"
    f.write_text("A_KEY=from-file\nB_KEY=from-file\nEMPTY=\n")
    monkeypatch.setenv("A_KEY", "from-shell")
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    assert load_env_file((tmp_path / "missing", f)) == ["B_KEY"]
    import os

    assert (
        os.environ["A_KEY"] == "from-shell"
        and os.environ["B_KEY"] == "from-file"
        and "EMPTY" not in os.environ
    )
    monkeypatch.delenv("B_KEY")
