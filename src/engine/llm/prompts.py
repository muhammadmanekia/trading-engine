"""Versioned prompts: files under `prompts/<name>/v<N>.md`, identified by name, version and content hash.

A prompt is never edited in place. A change is a new version, so every recorded decision names the exact text
that produced it (the hash proves it) and results of different prompt versions can be compared fairly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from engine.llm.provider import sha256_text

_VERSION = re.compile(r"^v(\d+)\.md$")
_SLOT = re.compile(r"\{\{(\w+)\}\}")
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "prompts"


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str  # "v1"
    text: str

    @property
    def hash(self) -> str:
        return sha256_text(self.text)

    def render(self, **values: str) -> str:
        slots = set(_SLOT.findall(self.text))
        missing, extra = slots - set(values), set(values) - slots
        if missing or extra:
            raise PromptError(
                f"{self.name}/{self.version}: missing {sorted(missing)}, unexpected {sorted(extra)}"
            )
        return _SLOT.sub(lambda m: values[m.group(1)], self.text)


def _numbered(folder: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in folder.glob("v*.md"):
        match = _VERSION.match(path.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    return sorted(found)


class PromptRegistry:
    def __init__(self, directory: str | Path = DEFAULT_DIR) -> None:
        self._dir = Path(directory)

    def get(self, name: str, version: str = "latest") -> PromptTemplate:
        folder = self._dir / name
        versions = _numbered(folder)
        if not versions:
            raise PromptError(f"no prompt named {name!r} in {self._dir}")
        if version == "latest":
            n, path = versions[-1]
            return PromptTemplate(name, f"v{n}", path.read_text())
        path = folder / f"{version}.md"
        if not path.exists():
            have = [f"v{n}" for n, _ in versions]
            raise PromptError(f"prompt {name}/{version} does not exist (have: {have})")
        return PromptTemplate(name, version, path.read_text())

    def versions(self, name: str) -> list[str]:
        return [f"v{n}" for n, _ in _numbered(self._dir / name)]


def values_to_slots(mapping: Mapping[str, str]) -> dict[str, str]:
    return dict(mapping)
