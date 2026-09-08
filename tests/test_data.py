"""Unit tests for shared.data (chunking logic only; nothing here hits Kite)."""

from __future__ import annotations

from datetime import date, timedelta

from shared.data import _chunk_date_range


def test_chunk_date_range_returns_single_chunk_when_within_limit() -> None:
    chunks = _chunk_date_range("2023-01-01", "2023-12-31", max_days=1900)

    assert chunks == [("2023-01-01", "2023-12-31")]


def test_chunk_date_range_splits_a_long_span_without_gaps_or_overlap() -> None:
    chunks = _chunk_date_range("2015-01-01", "2026-09-08", max_days=1900)

    assert len(chunks) > 1
    assert chunks[0][0] == "2015-01-01"
    assert chunks[-1][1] == "2026-09-08"
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert date.fromisoformat(next_start) == date.fromisoformat(prev_end) + timedelta(days=1)
