"""Everything between the news feed and a language model: relevance, de-duplication, sanitisation and an
injection screen.

News text is attacker-controlled. This module is defence in depth, NOT the security boundary: a determined
attacker can phrase around any pattern list. The boundary is structural (the model can only emit a tiny validated
schema, never sets prices or sizes, and the risk engine has the final say). What this module does is remove the
cheap attacks, keep delimiters intact and tell you when someone is trying.
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from engine.news.models import NewsItem, canonical_symbol

HEADLINE_MAX = 300
SUMMARY_MAX = 600
_ZERO_WIDTH_OR_BIDI = re.compile("[​‌‍⁠﻿‪-‮⁦-⁩]")
_WORD = re.compile(r"[a-z0-9$%.]+")

# name -> pattern. Any match BLOCKS the item from reaching a model.
_BLOCK: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for name, pattern in {
        "ignore-instructions": r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instruction|prompt|rule|direction|guideline|constraint|limit)s?\b",
        "role-play": r"\b(you are now|act as|pretend (to be|you)|new instructions?|from now on you|your (new )?task is)\b",
        "chat-markers": r"(^|\n)\s*(system|assistant|developer|user|human)\s*:",
        "special-tokens": r"<\|.{0,30}\|>|\[/?(inst|sys)\]|###\s*(instruction|system)",
        "tool-syntax": r"\b(function_call|tool_use|tool_call|tool_result)\b",
        "addresses-the-model": r"\b(you (must|should|need to|will|have to)|please)\b.{0,50}\b(buy|sell|short|purchase|liquidate|enter|exit|trade)\b",
        "prompt-exfiltration": r"\b(reveal|print|show|repeat|output)\b.{0,40}\b(system prompt|instructions|api[ _-]?key|secret|credentials?)\b",
        "our-delimiters": r"</?\s*(news|item|data|context|instructions?|system)\b|BEGIN_(NEWS|DATA)|END_(NEWS|DATA)",
        "encoded-payload": r"[A-Za-z0-9+/]{80,}={0,2}",
    }.items()
}


@dataclass(frozen=True)
class Screened:
    item: NewsItem | None  # the sanitised item, or None if blocked
    blocked: tuple[str, ...]  # why, for the log and the alert


def sanitize(text: str, max_len: int) -> str:
    """Normalise to NFKC, drop invisible/control characters, neutralise angle brackets so the text can never
    close one of our delimiters, collapse whitespace, and truncate."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_OR_BIDI.sub("", text)
    text = "".join(" " if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") else c for c in text)
    text = text.replace("<", "‹").replace(">", "›")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def scan(text: str) -> list[str]:
    """Names of every blocking pattern the RAW text trips (also flags hidden characters)."""
    folded = unicodedata.normalize(
        "NFKC", text
    )  # full-width / compatibility forms must not evade the patterns
    hits = [name for name, rx in _BLOCK.items() if rx.search(folded) or rx.search(text)]
    if _ZERO_WIDTH_OR_BIDI.search(text):
        hits.append("hidden-characters")
    return hits


def screen(item: NewsItem) -> Screened:
    reasons = scan(f"{item.headline}\n{item.summary}")
    if reasons:
        return Screened(None, tuple(reasons))
    clean = item.model_copy(
        update={
            "headline": sanitize(item.headline, HEADLINE_MAX),
            "summary": sanitize(item.summary, SUMMARY_MAX),
        }
    )
    return Screened(clean, ())


def relevant_symbols(item: NewsItem, universe: Iterable[str], *, max_symbols: int = 5) -> list[str]:
    """Universe symbols the item is about. Items tagged with many tickers are market-wide noise, not signal."""
    tagged = {canonical_symbol(s) for s in item.symbols}
    if not tagged or len(tagged) > max_symbols:
        return []
    return sorted(tagged & {canonical_symbol(u) for u in universe})


class NoveltyFilter:
    """Drops near-duplicates: the same story re-published by a feed, or re-worded by an aggregator."""

    def __init__(self, window: timedelta = timedelta(hours=6), threshold: float = 0.7) -> None:
        self._window = window
        self._threshold = threshold
        self._seen: deque[tuple[datetime, frozenset[str], frozenset[str]]] = deque()

    @staticmethod
    def _tokens(text: str) -> frozenset[str]:
        return frozenset(w for w in _WORD.findall(text.lower()) if len(w) > 2)

    def is_new(self, item: NewsItem, symbols: Sequence[str]) -> bool:
        now = item.received_at
        while self._seen and now - self._seen[0][0] > self._window:
            self._seen.popleft()
        toks, syms = self._tokens(item.headline), frozenset(symbols)
        for _, other_toks, other_syms in self._seen:
            if other_syms & syms and toks and other_toks:
                overlap = len(toks & other_toks) / len(toks | other_toks)
                if overlap >= self._threshold:
                    return False
        self._seen.append((now, toks, syms))
        return True
