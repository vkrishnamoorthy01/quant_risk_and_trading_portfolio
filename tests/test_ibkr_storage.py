"""Unit tests for data_providers.ibkr_storage. Builds HistoricalData directly or
through a fake fetch function; writes only to pytest's tmp_path and never
needs TWS running.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pandas import DataFrame, Timestamp, bdate_range, read_csv

import data_providers.ibkr_storage as storage_module
from data_providers.ibkr_provider import EXCHANGE_TIMEZONE, HistoricalData, IBKRNoDataError, RequestMetadata
from data_providers.ibkr_storage import (
    IBKRStorageConflictError,
    IBKRStorageError,
    dataset_path,
    export_csv,
    fetch_and_store,
    load_bars,
    load_dataset,
    missing_ranges,
    save_bars,
)

# A Monday evening after the session is final, so nothing in these tests counts as "today, still trading".
_NOW = datetime(2026, 10, 26, 17, 0, tzinfo=EXCHANGE_TIMEZONE)


def _result(
    start: date,
    end: date,
    close: float = 100.0,
    con_id: int = 1,
    skip: set[date] = frozenset(),
    intraday: bool = False,
    symbol: str = "TATASTEEL",
) -> HistoricalData:
    """A HistoricalData for every weekday in [start, end] (minus `skip`), as the provider would return it."""
    days = [ts.date() for ts in bdate_range(start, end) if ts.date() not in skip]
    if intraday:
        times = [Timestamp(datetime(d.year, d.month, d.day, 9, 15), tz=EXCHANGE_TIMEZONE) + timedelta(minutes=5 * k)
                 for d in days for k in range(3)]
        time_column, bar_size = "timestamp", "5 mins"
    else:
        times, time_column, bar_size = [Timestamp(d) for d in days], "date", "1 day"
    bars = DataFrame({
        time_column: times, "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 1000.0, "vwap": close, "bar_count": 10,
    }).assign(symbol=symbol, exchange="NSE", currency="INR", con_id=con_id)
    metadata = RequestMetadata(
        symbol=symbol, exchange="NSE", currency="INR", con_id=con_id, environment="paper", bar_size=bar_size,
        what_to_show="TRADES", use_rth=True, requested_start=start, requested_end=end,
        first_date=days[0], last_date=days[-1], n_bars=len(bars), n_requests=1, duplicates_dropped=0,
        placeholder_bars_dropped=[], missing_weekdays=sorted(skip), empty_chunks=[],
        downloaded_at=datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc),
    )
    return HistoricalData(bars=bars, metadata=metadata)


# --- save and load --------------------------------------------------------------


def test_save_then_load_round_trips_bars_and_metadata(tmp_path: Path) -> None:
    saved = save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=_NOW)

    stored = load_dataset("TATASTEEL", "paper", root=tmp_path)

    assert stored.path == tmp_path / "paper" / "daily" / "NSE_TATASTEEL.parquet"
    assert stored.bars.equals(saved.bars)
    assert len(stored.bars) == 5
    assert set(stored.bars["source"]) == {"IBKR"}
    assert str(stored.bars["downloaded_at"].dt.tz) == "UTC"
    assert stored.info["con_id"] == 1 and stored.info["what_to_show"] == "TRADES" and stored.info["use_rth"] is True
    assert stored.info["coverage"] == [["2026-10-19", "2026-10-23"]]
    assert len(stored.info["downloads"]) == 1
    assert stored.info["downloads"][0]["requested_start"] == "2026-10-19"


def test_intraday_round_trip_keeps_ist_timestamps(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 22), date(2026, 10, 23), intraday=True), root=tmp_path, now=_NOW)

    bars = load_bars("TATASTEEL", "paper", bar_size="5 mins", root=tmp_path)

    assert dataset_path("TATASTEEL", "paper", "5 mins", root=tmp_path).parent.name == "5min"
    assert str(bars["timestamp"].dt.tz) == "Asia/Kolkata"
    assert bars["timestamp"].iloc[0].isoformat() == "2026-10-22T09:15:00+05:30"


def test_saving_identical_data_again_adds_no_rows_and_logs_the_download(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=_NOW)
    stored = save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=_NOW)

    assert len(stored.bars) == 5
    assert len(stored.info["downloads"]) == 2


def test_saving_a_later_range_appends_and_merges_coverage(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 12), date(2026, 10, 16)), root=tmp_path, now=_NOW)
    stored = save_bars(_result(date(2026, 10, 17), date(2026, 10, 23)), root=tmp_path, now=_NOW)

    assert len(stored.bars) == 10
    assert stored.bars["date"].is_monotonic_increasing
    assert stored.info["coverage"] == [["2026-10-12", "2026-10-23"]]


def test_a_changed_price_raises_and_leaves_the_file_untouched(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=_NOW)
    path = dataset_path("TATASTEEL", "paper", root=tmp_path)
    before = path.read_bytes()

    with pytest.raises(IBKRStorageConflictError, match="split or bonus"):
        save_bars(_result(date(2026, 10, 19), date(2026, 10, 23), close=50.0), root=tmp_path, now=_NOW)

    assert path.read_bytes() == before


def test_a_stored_bar_missing_from_the_new_download_is_a_conflict(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=_NOW)

    with pytest.raises(IBKRStorageConflictError, match="missing from new data"):
        save_bars(_result(date(2026, 10, 19), date(2026, 10, 23), skip={date(2026, 10, 21)}), root=tmp_path, now=_NOW)


def test_on_conflict_replace_lets_the_new_download_win(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 12), date(2026, 10, 23)), root=tmp_path, now=_NOW)

    stored = save_bars(_result(date(2026, 10, 19), date(2026, 10, 23), close=50.0), root=tmp_path,
                       on_conflict="replace", now=_NOW)

    closes = stored.bars.set_index("date")["close"]
    assert closes[Timestamp("2026-10-16")] == 100.0  # outside the new range: kept
    assert closes[Timestamp("2026-10-19")] == 50.0  # inside: replaced
    assert stored.info["downloads"][-1]["on_conflict"] == "replace"


def test_a_different_contract_id_is_refused(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=_NOW)

    with pytest.raises(IBKRStorageError, match="con_id"):
        save_bars(_result(date(2026, 10, 26), date(2026, 10, 26), con_id=2), root=tmp_path, now=_NOW)


def test_todays_session_is_not_stored_before_it_is_final(tmp_path: Path) -> None:
    during_session = datetime(2026, 10, 23, 14, 0, tzinfo=EXCHANGE_TIMEZONE)

    stored = save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=during_session)

    assert stored.bars["date"].max() == Timestamp("2026-10-22")
    assert stored.info["coverage"] == [["2026-10-19", "2026-10-22"]]


def test_todays_session_is_stored_after_1600_ist(tmp_path: Path) -> None:
    after_close = datetime(2026, 10, 23, 16, 5, tzinfo=EXCHANGE_TIMEZONE)

    stored = save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=after_close)

    assert stored.bars["date"].max() == Timestamp("2026-10-23")


def test_a_failed_read_back_leaves_the_existing_file_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=_NOW)
    path = dataset_path("TATASTEEL", "paper", root=tmp_path)
    before = path.read_bytes()
    real_read_table = storage_module.read_table

    def _corrupt_read_back(source: Any) -> Any:
        table = real_read_table(source)
        return table.slice(0, 1) if str(source).endswith(".tmp") else table

    monkeypatch.setattr(storage_module, "read_table", _corrupt_read_back)

    with pytest.raises(IBKRStorageError, match="Read-back"):
        save_bars(_result(date(2026, 10, 26), date(2026, 10, 26)), root=tmp_path, now=_NOW)

    assert path.read_bytes() == before
    assert not path.with_suffix(".parquet.tmp").exists()


def test_load_bars_filters_by_session_date(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 12), date(2026, 10, 23)), root=tmp_path, now=_NOW)

    bars = load_bars("TATASTEEL", "paper", start=date(2026, 10, 19), end=date(2026, 10, 21), root=tmp_path)

    assert list(bars["date"].dt.date) == [date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)]


def test_load_bars_raises_when_nothing_is_stored(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bars("TATASTEEL", "paper", root=tmp_path)


def test_export_csv_writes_the_stored_bars(tmp_path: Path) -> None:
    save_bars(_result(date(2026, 10, 19), date(2026, 10, 23)), root=tmp_path, now=_NOW)

    csv_path = export_csv("TATASTEEL", "paper", tmp_path / "out.csv", root=tmp_path)

    assert len(read_csv(csv_path)) == 5


# --- avoiding repeat downloads ----------------------------------------------------


def test_missing_ranges_with_no_coverage_is_the_whole_range_trimmed_to_weekdays() -> None:
    # 2026-10-17 is a Saturday, 2026-10-25 a Sunday.
    assert missing_ranges([], date(2026, 10, 17), date(2026, 10, 25)) == [(date(2026, 10, 19), date(2026, 10, 23))]


def test_missing_ranges_finds_gaps_either_side_of_coverage() -> None:
    coverage = [(date(2026, 10, 13), date(2026, 10, 15))]

    gaps = missing_ranges(coverage, date(2026, 10, 12), date(2026, 10, 23))

    assert gaps == [(date(2026, 10, 12), date(2026, 10, 12)), (date(2026, 10, 16), date(2026, 10, 23))]


def test_missing_ranges_skips_weekend_only_gaps_and_full_coverage() -> None:
    covered_to_friday = [(date(2026, 10, 12), date(2026, 10, 23))]

    assert missing_ranges(covered_to_friday, date(2026, 10, 12), date(2026, 10, 25)) == []


class _FakeFetch:
    """Stands in for get_historical_data: records each requested range and returns synthetic bars."""

    def __init__(self, no_data: set[tuple[date, date]] = frozenset()) -> None:
        self.calls: list[tuple[date, date]] = []
        self._no_data = no_data

    def __call__(self, ib: Any, symbol: str, start: date, end: date, environment: str, **kwargs: Any) -> HistoricalData:
        self.calls.append((start, end))
        if (start, end) in self._no_data:
            raise IBKRNoDataError("no bars")
        return _result(start, end, symbol=symbol)


def test_fetch_and_store_downloads_only_what_is_not_stored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeFetch()
    monkeypatch.setattr(storage_module, "get_historical_data", fake)

    fetch_and_store(object(), "TATASTEEL", date(2026, 10, 12), date(2026, 10, 16), "paper", root=tmp_path, now=_NOW)
    fetch_and_store(object(), "TATASTEEL", date(2026, 10, 12), date(2026, 10, 16), "paper", root=tmp_path, now=_NOW)
    stored = fetch_and_store(object(), "TATASTEEL", date(2026, 10, 12), date(2026, 10, 23), "paper",
                             root=tmp_path, now=_NOW)

    assert fake.calls == [(date(2026, 10, 12), date(2026, 10, 16)), (date(2026, 10, 19), date(2026, 10, 23))]
    assert len(stored.bars) == 10


def test_fetch_and_store_never_requests_beyond_the_last_final_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeFetch()
    monkeypatch.setattr(storage_module, "get_historical_data", fake)
    during_session = datetime(2026, 10, 23, 11, 0, tzinfo=EXCHANGE_TIMEZONE)

    fetch_and_store(object(), "TATASTEEL", date(2026, 10, 19), date(2026, 10, 30), "paper",
                    root=tmp_path, now=during_session)

    assert fake.calls == [(date(2026, 10, 19), date(2026, 10, 22))]


def test_fetch_and_store_replace_downloads_the_whole_range_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeFetch()
    monkeypatch.setattr(storage_module, "get_historical_data", fake)

    fetch_and_store(object(), "TATASTEEL", date(2026, 10, 19), date(2026, 10, 23), "paper", root=tmp_path, now=_NOW)
    fetch_and_store(object(), "TATASTEEL", date(2026, 10, 19), date(2026, 10, 23), "paper", root=tmp_path,
                    on_conflict="replace", now=_NOW)

    assert fake.calls == [(date(2026, 10, 19), date(2026, 10, 23))] * 2


def test_fetch_and_store_tolerates_a_holiday_only_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holiday = (date(2026, 10, 26), date(2026, 10, 26))
    fake = _FakeFetch(no_data={holiday})
    monkeypatch.setattr(storage_module, "get_historical_data", fake)
    later = datetime(2026, 10, 27, 9, 0, tzinfo=EXCHANGE_TIMEZONE)  # last final session: Mon 10-26

    fetch_and_store(object(), "TATASTEEL", date(2026, 10, 19), date(2026, 10, 23), "paper", root=tmp_path, now=later)
    stored = fetch_and_store(object(), "TATASTEEL", date(2026, 10, 19), date(2026, 10, 26), "paper",
                             root=tmp_path, now=later)

    assert fake.calls[-1] == holiday
    assert len(stored.bars) == 5
    assert stored.info["coverage"] == [["2026-10-19", "2026-10-23"]]  # left open, retried next time


def test_fetch_and_store_uses_the_intraday_fetcher_for_5_minute_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def _fake_intraday(ib: Any, symbol: str, start: date, end: date, environment: str, **kwargs: Any) -> HistoricalData:
        calls.append(kwargs["bar_size"])
        return _result(start, end, intraday=True)

    monkeypatch.setattr(storage_module, "get_intraday_data", _fake_intraday)

    stored = fetch_and_store(object(), "TATASTEEL", date(2026, 10, 22), date(2026, 10, 23), "paper",
                             bar_size="5 mins", root=tmp_path, now=_NOW)

    assert calls == ["5 mins"]
    assert stored.path.parent.name == "5min"
