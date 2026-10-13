"""The comparison protocol: windows never leak, the rule is exact at its boundaries, the exam cannot be peeked."""

from __future__ import annotations

import pytest

from engine.backtest.compare import Protocol, WindowSpan, required_wins, rolling_spans


def test_four_windows_fit_between_mid_2020_and_the_exam() -> None:
    spans = rolling_spans(1008, Protocol(train_bars=504, test_bars=126))
    assert spans == [
        WindowSpan(0, 504, 630),
        WindowSpan(126, 630, 756),
        WindowSpan(252, 756, 882),
        WindowSpan(378, 882, 1008),
    ]


def test_windows_never_overlap_never_reach_the_exam_and_train_before_they_test() -> None:
    for exam_idx in (700, 1000, 1008, 1300):
        spans = rolling_spans(exam_idx, Protocol(train_bars=504, test_bars=126))
        for a, b in zip(spans, spans[1:], strict=False):
            assert a.test_hi <= b.test_lo  # scored periods are disjoint
        assert all(s.test_hi <= exam_idx and s.train_lo + 504 == s.test_lo for s in spans)


def test_no_windows_when_history_before_the_exam_is_too_short() -> None:
    assert rolling_spans(600, Protocol(train_bars=504, test_bars=126)) == []


@pytest.mark.parametrize(("windows", "needed"), [(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (6, 4), (9, 6)])
def test_two_thirds_rounds_up(windows: int, needed: int) -> None:
    assert required_wins(windows) == needed
