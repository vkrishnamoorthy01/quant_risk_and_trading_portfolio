"""Unit tests for shared.data. Uses a fake Kite client where needed; nothing
here hits live Kite.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import shared.data as data_module
from shared.data import _chunk_date_range, _fetch_chunk, fetch_price_data


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


class _FakeKite:
    """historical_data returns queued responses in order; everything else is unused here."""

    def __init__(self, responses: list[list[dict]]) -> None:
        self._responses = iter(responses)

    def historical_data(self, token: int, start: str, end: str, interval: str) -> list[dict]:
        return next(self._responses)


def test_fetch_chunk_retries_with_backoff_until_a_non_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module.time, "sleep", lambda _: None)
    # Empty on the first 3 attempts (within _MAX_FETCH_ATTEMPTS), succeeds on the 4th.
    kite = _FakeKite(responses=[[], [], [], [{"date": "2020-01-01", "close": 100.0}]])

    result = _fetch_chunk(kite, token=1, chunk_start="2020-01-01", chunk_end="2020-01-01", interval="day")

    assert not result.empty
    assert result.iloc[0]["close"] == 100.0


def test_fetch_chunk_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module.time, "sleep", lambda _: None)
    kite = _FakeKite(responses=[[], [], [], []])

    result = _fetch_chunk(kite, token=1, chunk_start="2020-01-01", chunk_end="2020-01-01", interval="day")

    assert result.empty


def test_fetch_price_data_warns_when_a_ticker_is_missing_most_of_its_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(data_module, "get_kite_client", lambda: object())
    monkeypatch.setattr(data_module, "_get_instrument_token", lambda kite, ticker, exchange: {"GOOD": 1, "BAD": 2}[ticker])

    dates = pd.bdate_range("2020-01-01", periods=20)
    full_history = [{"date": d.isoformat(), "close": 100.0} for d in dates]
    mostly_missing = [{"date": d.isoformat(), "close": 100.0} for d in dates[:3]]

    def _fake_historical_data(token: int, start: str, end: str, interval: str) -> list[dict]:
        return full_history if token == 1 else mostly_missing

    monkeypatch.setattr(
        data_module,
        "_fetch_chunk",
        lambda kite, token, chunk_start, chunk_end, interval: pd.DataFrame(
            _fake_historical_data(token, chunk_start, chunk_end, interval)
        ),
    )

    with pytest.warns(UserWarning, match="BAD"):
        prices = fetch_price_data(["GOOD", "BAD"], "2020-01-01", "2020-01-31")

    assert "BAD" in prices.columns
    assert prices["BAD"].isna().mean() > 0.5


def test_fetch_price_data_normalizes_day_candle_timestamps_for_cross_ticker_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reproduces the real bug: Kite's older "day" candles come back
    # timestamped at market-open (09:15 IST) rather than midnight. Two
    # tickers on the same trading days, one at each convention -- without
    # normalizing away the time-of-day, these look like 4 distinct dates
    # instead of 2 shared ones, and each ticker shows spurious gaps on the
    # other's timestamps.
    monkeypatch.setattr(data_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(data_module, "get_kite_client", lambda: object())
    monkeypatch.setattr(data_module, "_get_instrument_token", lambda kite, ticker, exchange: {"OLD": 1, "NEW": 2}[ticker])

    old_vintage = [
        {"date": "2015-02-28T09:15:00+0530", "close": 100.0},
        {"date": "2015-03-02T09:15:00+0530", "close": 101.0},
    ]
    new_vintage = [
        {"date": "2015-02-28T00:00:00+0530", "close": 200.0},
        {"date": "2015-03-02T00:00:00+0530", "close": 201.0},
    ]
    monkeypatch.setattr(
        data_module,
        "_fetch_chunk",
        lambda kite, token, chunk_start, chunk_end, interval: pd.DataFrame(
            old_vintage if token == 1 else new_vintage
        ),
    )

    prices = fetch_price_data(["OLD", "NEW"], "2015-01-01", "2015-12-31")

    assert len(prices) == 2  # both trading days align into 2 rows, not 4
    assert prices["OLD"].isna().sum() == 0
    assert prices["NEW"].isna().sum() == 0


def test_fetch_price_data_does_not_normalize_minute_interval_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Normalizing intraday candles the same way would collapse distinct
    # minute bars onto the same timestamp -- must only apply to "day".
    monkeypatch.setattr(data_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(data_module, "get_kite_client", lambda: object())
    monkeypatch.setattr(data_module, "_get_instrument_token", lambda kite, ticker, exchange: 1)

    intraday = [
        {"date": "2015-02-28T09:15:00+0530", "close": 100.0},
        {"date": "2015-02-28T09:16:00+0530", "close": 101.0},
    ]
    monkeypatch.setattr(
        data_module, "_fetch_chunk", lambda kite, token, chunk_start, chunk_end, interval: pd.DataFrame(intraday)
    )

    prices = fetch_price_data(["TCS"], "2015-01-01", "2015-12-31", interval="minute")

    assert len(prices) == 2
