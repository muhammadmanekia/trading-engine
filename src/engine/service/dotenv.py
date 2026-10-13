"""Load KEY=VALUE pairs from a `.env` file into the environment, so secrets need exporting only once.

Real environment variables always win over the file. The file is git-ignored; never commit it.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the trading-engine folder


def parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key.strip():
            out[key.strip()] = value
    return out


def load_env_file(paths: tuple[Path, ...] = (Path(".env"), ROOT / ".env")) -> list[str]:
    """Set variables from the first existing file that is not already set. Returns the names that were set."""
    for path in paths:
        if path.is_file():
            applied = [k for k, v in parse(path.read_text()).items() if k not in os.environ and v]
            for k, v in parse(path.read_text()).items():
                if k in applied:
                    os.environ[k] = v
            return applied
    return []
