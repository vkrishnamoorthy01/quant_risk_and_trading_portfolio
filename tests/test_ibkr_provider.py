"""Unit tests for data_providers.ibkr_provider. Uses a fake IB client; nothing
here needs TWS running.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

import pytest
from pandas import DataFrame, Series, Timestamp, bdate_range

from data_providers.ibkr_provider import (
    IBKRContractError,
    IBKRNoDataError,
    IBKRPermissionError,
    IBKRRequestError,
    IBKRValidationError,
    chunk_date_range,
    drop_flat_index_bars,
    clean_bars,
    find_missing_weekdays,
    check_intraday_sessions,
    expected_session_slots,
    session_close,
    get_historical_data,
    get_intraday_data,
    qualify_contract,
    qualify_stock,
)


@dataclass
class _Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    average: float = 100.0
    barCount: int = 10


def _bar(day: date, close: float = 100.0) -> _Bar:
    return _Bar(day, open=close, high=close + 1, low=close - 1, close=close, volume=1000.0)


@dataclass
class _Contract:
    symbol: str
    conId: int
    exchange: str = "NSE"
    primaryExchange: str = "NSE"
    localSymbol: str = ""


@dataclass
class _Details:
    contract: _Contract


class _Event:
    """Minimal stand-in for an eventkit Event: supports += / -= and emit."""

    def __init__(self) -> None:
        self.handlers: list[Callable[..., None]] = []

    def __iadd__(self, handler: Callable[..., None]) -> _Event:
        self.handlers.append(handler)
        return self

    def __isub__(self, handler: Callable[..., None]) -> _Event:
        self.handlers.remove(handler)
        return self

    def emit(self, *args: Any) -> None:
        for handler in list(self.handlers):
            handler(*args)


class _FakeIB:
    """Each reqHistoricalData call pops one (bars, errors) response and emits the errors."""

    def __init__(self, responses: list[tuple[list[_Bar], list[tuple[int, str]]]], n_contracts: int = 1) -> None:
        self._responses = iter(responses)
        self._n_contracts = n_contracts
        self.errorEvent = _Event()
        self.history_calls: list[dict] = []
        self.sleeps: list[float] = []

    def reqContractDetails(self, contract: Any) -> list[_Details]:
        return [_Details(_Contract(contract.symbol, conId=1000 + i)) for i in range(self._n_contracts)]

    def reqHistoricalData(self, contract: Any, **kwargs: Any) -> list[_Bar]:
        self.history_calls.append(kwargs)
        bars, errors = next(self._responses)
        for code, message in errors:
            self.errorEvent.emit(1, code, message, contract)
        return bars

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _weekday_bars(start: date, end: date) -> list[_Bar]:
    return [_bar(ts.date()) for ts in bdate_range(start, end)]


def _frame(rows: list[dict]) -> DataFrame:
    base = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "vwap": 100.0, "bar_count": 10}
    return DataFrame([{**base, **row} for row in rows])


# --- contract qualification ---------------------------------------------------


def test_qualify_stock_returns_the_single_match() -> None:
    contract = qualify_stock(_FakeIB([], n_contracts=1), "RELIANCE")

    assert contract.conId == 1000


def test_qualify_stock_rejects_an_unknown_symbol() -> None:
    with pytest.raises(IBKRContractError, match="No IB STK contract"):
        qualify_stock(_FakeIB([], n_contracts=0), "NOTASTOCK")


def test_qualify_stock_rejects_an_ambiguous_symbol_and_lists_the_matches() -> None:
    with pytest.raises(IBKRContractError, match="ambiguous.*conId=1000.*conId=1001"):
        qualify_stock(_FakeIB([], n_contracts=2), "RELIANCE")


# --- date chunking ------------------------------------------------------------


def test_chunk_date_range_covers_three_years_without_gaps_or_overlap() -> None:
    chunks = chunk_date_range(date(2023, 10, 24), date(2026, 10, 24), max_days=365)

    assert chunks[0][0] == date(2023, 10, 24)
    assert chunks[-1][1] == date(2026, 10, 24)
    assert all((end - start).days + 1 <= 365 for start, end in chunks)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start == prev_end + timedelta(days=1)


def test_chunk_date_range_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError):
        chunk_date_range(date(2026, 1, 2), date(2026, 1, 1))


# --- normalization and validation ---------------------------------------------


def test_clean_bars_sorts_and_drops_exact_duplicates() -> None:
    frame = _frame([
        {"date": Timestamp("2026-10-23")},
        {"date": Timestamp("2026-10-21")},
        {"date": Timestamp("2026-10-23")},
    ])

    cleaned, dropped = clean_bars(frame, "TEST")

    assert dropped == 1
    assert list(cleaned["date"]) == [Timestamp("2026-10-21"), Timestamp("2026-10-23")]


def test_clean_bars_rejects_the_same_date_with_different_prices() -> None:
    frame = _frame([{"date": Timestamp("2026-10-23"), "close": 100.0}, {"date": Timestamp("2026-10-23"), "close": 100.5}])

    with pytest.raises(IBKRValidationError, match="2026-10-23"):
        clean_bars(frame, "TEST")


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"close": 0.0, "low": 0.0}, "non-positive"),
        ({"high": 99.5}, "high/low"),  # high below open/close
        ({"low": 100.5}, "high/low"),  # low above open/close
        ({"volume": -1.0}, "negative volume"),
        ({"close": float("nan")}, "missing"),
    ],
)
def test_clean_bars_rejects_impossible_values(override: dict, problem: str) -> None:
    frame = _frame([{"date": Timestamp("2026-10-23"), **override}])

    with pytest.raises(IBKRValidationError, match=problem):
        clean_bars(frame, "TEST")


def test_find_missing_weekdays_ignores_weekends_and_reports_holidays() -> None:
    # Fri 2026-10-16, Mon 10-19, Wed 10-21: the weekend is fine, Tue 10-20 is reported.
    frame = _frame([{"date": Timestamp(d)} for d in ("2026-10-16", "2026-10-19", "2026-10-21")])

    assert find_missing_weekdays(frame) == [date(2026, 10, 20)]


# --- get_historical_data ------------------------------------------------------


def test_get_historical_data_chunks_a_long_range_and_returns_sorted_unique_bars() -> None:
    start, end = date(2024, 1, 1), date(2025, 6, 30)
    chunks = chunk_date_range(start, end)
    responses = [(_weekday_bars(s, e), []) for s, e in chunks]
    ib = _FakeIB(responses)

    result = get_historical_data(ib, "RELIANCE", start, end, environment="paper")

    assert len(ib.history_calls) == len(chunks) == 2
    assert result.bars["date"].is_monotonic_increasing
    assert not result.bars["date"].duplicated().any()
    assert result.metadata.n_requests == 2
    assert result.metadata.first_date == date(2024, 1, 1)
    assert result.metadata.last_date == date(2025, 6, 30)
    assert result.metadata.what_to_show == "TRADES" and result.metadata.use_rth is True
    assert set(result.bars["symbol"]) == {"RELIANCE"}
    assert ib.sleeps == [2.0]  # one pause between the two requests


def test_get_historical_data_requests_each_chunk_up_to_end_of_session_ist() -> None:
    ib = _FakeIB([(_weekday_bars(date(2026, 10, 1), date(2026, 10, 23)), [])])

    get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")

    end = ib.history_calls[0]["endDateTime"]
    assert isinstance(end, datetime)
    assert (end.date(), end.hour, str(end.tzinfo)) == (date(2026, 10, 23), 23, "Asia/Kolkata")
    assert ib.history_calls[0]["durationStr"] == "23 D"


def test_get_historical_data_trims_bars_outside_the_requested_range() -> None:
    bars = _weekday_bars(date(2026, 9, 25), date(2026, 10, 23))  # IB returned a few extra early sessions
    ib = _FakeIB([(bars, [])])

    result = get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")

    assert result.bars["date"].min() == Timestamp("2026-10-01")


def test_get_historical_data_reports_holidays_without_failing() -> None:
    bars = [b for b in _weekday_bars(date(2026, 10, 1), date(2026, 10, 23)) if b.date != date(2026, 10, 20)]
    ib = _FakeIB([(bars, [])])

    result = get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")

    assert result.metadata.missing_weekdays == [date(2026, 10, 20)]
    assert result.metadata.warnings == []


def test_get_historical_data_drops_zero_volume_zero_trade_placeholder_bars() -> None:
    # Mirrors IB's real 2024-11-20 bar: an NSE holiday returned with volume 0 and barCount 0.
    bars = _weekday_bars(date(2024, 11, 18), date(2024, 11, 22))
    bars[2] = _Bar(date(2024, 11, 20), 100.0, 100.0, 100.0, 100.0, volume=0.0, barCount=0)
    ib = _FakeIB([(bars, [])])

    result = get_historical_data(ib, "RELIANCE", date(2024, 11, 18), date(2024, 11, 22), environment="paper")

    assert Timestamp("2024-11-20") not in set(result.bars["date"])
    assert result.metadata.placeholder_bars_dropped == [Timestamp("2024-11-20")]
    assert result.metadata.missing_weekdays == [date(2024, 11, 20)]


def test_get_historical_data_keeps_a_real_low_volume_session() -> None:
    # A Muhurat-style short session: low volume but real trades, so it stays.
    bars = _weekday_bars(date(2024, 10, 28), date(2024, 11, 1))
    bars[-1] = _Bar(date(2024, 11, 1), 100.0, 101.0, 99.0, 100.0, volume=42729.0, barCount=3494)
    ib = _FakeIB([(bars, [])])

    result = get_historical_data(ib, "RELIANCE", date(2024, 10, 28), date(2024, 11, 1), environment="paper")

    assert result.metadata.placeholder_bars_dropped == []
    assert result.metadata.n_bars == 5


def test_get_historical_data_warns_when_many_weekdays_are_missing() -> None:
    bars = _weekday_bars(date(2026, 10, 1), date(2026, 10, 23))[::2]
    ib = _FakeIB([(bars, [])])

    with pytest.warns(UserWarning, match="weekdays have no bar"):
        get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")


def test_get_historical_data_treats_a_no_data_chunk_as_empty_not_fatal() -> None:
    # Older chunk predates the listing (IB error 162); newer chunk has data.
    start, end = date(2024, 1, 1), date(2025, 6, 30)
    first, second = chunk_date_range(start, end)
    ib = _FakeIB([([], [(162, "Historical Market Data Service error message:HMDS query returned no data")]),
                  (_weekday_bars(*second), [])])

    with pytest.warns(UserWarning, match="later than the requested"):
        result = get_historical_data(ib, "NEWLISTING", start, end, environment="paper")

    assert result.metadata.empty_chunks == [first]
    assert result.metadata.first_date == second[0] or result.metadata.first_date > second[0]


def test_get_historical_data_raises_when_every_chunk_is_empty() -> None:
    ib = _FakeIB([([], [(162, "HMDS query returned no data")])])

    with pytest.raises(IBKRNoDataError):
        get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")


def test_get_historical_data_raises_a_permission_error() -> None:
    ib = _FakeIB([([], [(162, "Historical Market Data Service error message:No market data permissions for NSE STK")])])

    with pytest.raises(IBKRPermissionError, match="permission"):
        get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")


def test_get_historical_data_retries_once_after_a_timeout_then_succeeds() -> None:
    bars = _weekday_bars(date(2026, 10, 1), date(2026, 10, 23))
    ib = _FakeIB([([], []), (bars, [])])  # empty with no error = timed out

    result = get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")

    assert len(ib.history_calls) == 2
    assert result.metadata.n_bars == len(bars)


def test_get_historical_data_gives_up_after_repeated_timeouts() -> None:
    ib = _FakeIB([([], []), ([], [])])

    with pytest.raises(IBKRRequestError, match="after 2 attempts"):
        get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")


def test_get_historical_data_waits_longer_after_a_pacing_violation() -> None:
    bars = _weekday_bars(date(2026, 10, 1), date(2026, 10, 23))
    ib = _FakeIB([([], [(162, "Historical Market Data Service error message:Pacing violation")]), (bars, [])])

    get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")

    assert ib.sleeps == [15.0]


def test_get_historical_data_raises_on_other_ib_errors() -> None:
    ib = _FakeIB([([], [(321, "Error validating request")])])

    with pytest.raises(IBKRRequestError, match="321"):
        get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")


def test_get_historical_data_ignores_data_farm_status_notices() -> None:
    bars = _weekday_bars(date(2026, 10, 1), date(2026, 10, 23))
    ib = _FakeIB([(bars, [(2104, "Market data farm connection is OK")])])

    result = get_historical_data(ib, "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), environment="paper")

    assert result.metadata.n_bars == len(bars)


def test_get_historical_data_rejects_unsupported_bar_sizes() -> None:
    with pytest.raises(ValueError, match="1 day"):
        get_historical_data(_FakeIB([]), "RELIANCE", date(2026, 10, 1), date(2026, 10, 23), "paper", bar_size="5 mins")


# --- intraday -----------------------------------------------------------------

_UTC = timezone.utc


def _session_bars(day: date, skip_slots: set[int] = frozenset(), n_slots: int = 75) -> list[_Bar]:
    """5-minute bars for one NSE session as ib_async returns them with formatDate=2: UTC-aware bar starts.

    09:15 IST is 03:45 UTC; slot k starts 5*k minutes later.
    """
    open_utc = datetime(day.year, day.month, day.day, 3, 45, tzinfo=_UTC)
    return [_bar(open_utc + timedelta(minutes=5 * k)) for k in range(n_slots) if k not in skip_slots]


def test_expected_session_slots_covers_0915_to_1525_ist() -> None:
    slots = expected_session_slots(date(2026, 7, 24), bar_minutes=5)

    assert len(slots) == 75
    assert slots[0].isoformat() == "2026-07-24T09:15:00+05:30"
    assert slots[-1].isoformat() == "2026-07-24T15:25:00+05:30"


def test_get_intraday_data_returns_ist_timestamps_for_full_sessions() -> None:
    bars = _session_bars(date(2026, 7, 23)) + _session_bars(date(2026, 7, 24))
    ib = _FakeIB([(bars, [])])

    result = get_intraday_data(ib, "TATASTEEL", date(2026, 7, 23), date(2026, 7, 24), environment="paper")

    stamps = result.bars["timestamp"]
    assert str(stamps.dt.tz) == "Asia/Kolkata"
    assert stamps.iloc[0].isoformat() == "2026-07-23T09:15:00+05:30"
    assert stamps.iloc[-1].isoformat() == "2026-07-24T15:25:00+05:30"
    assert result.metadata.n_bars == 150
    assert result.metadata.incomplete_sessions == {}
    assert result.metadata.bars_outside_session == 0
    assert ib.history_calls[0]["barSizeSetting"] == "5 mins"
    assert ib.history_calls[0]["formatDate"] == 2


def test_get_intraday_data_chunks_by_week() -> None:
    start, end = date(2026, 7, 2), date(2026, 7, 24)
    chunks = chunk_date_range(start, end, max_days=7)
    responses = [([b for ts in bdate_range(s, e) for b in _session_bars(ts.date())], []) for s, e in chunks]
    ib = _FakeIB(responses)

    result = get_intraday_data(ib, "TATASTEEL", start, end, environment="paper")

    assert result.metadata.n_requests == len(chunks) == 4
    assert all(call["durationStr"] in {"7 D", "2 D"} for call in ib.history_calls)
    assert result.bars["timestamp"].is_monotonic_increasing


def test_get_intraday_data_reports_missing_bars_without_filling_them() -> None:
    bars = _session_bars(date(2026, 7, 24), skip_slots={10, 11, 12})
    ib = _FakeIB([(bars, [])])

    with pytest.warns(UserWarning, match="missing bars"):
        result = get_intraday_data(ib, "TATASTEEL", date(2026, 7, 24), date(2026, 7, 24), environment="paper")

    assert result.metadata.n_bars == 72
    assert result.metadata.incomplete_sessions == {date(2026, 7, 24): 3}


def test_get_intraday_data_reports_a_short_session_as_incomplete() -> None:
    # A Muhurat-style one-hour session: 12 real bars, kept as they are.
    bars = _session_bars(date(2024, 11, 1), n_slots=12)
    ib = _FakeIB([(bars, [])])

    with pytest.warns(UserWarning, match="missing bars"):
        result = get_intraday_data(ib, "TATASTEEL", date(2024, 11, 1), date(2024, 11, 1), environment="paper")

    assert result.metadata.n_bars == 12
    assert result.metadata.incomplete_sessions == {date(2024, 11, 1): 63}


def test_get_intraday_data_counts_bars_outside_the_session_grid() -> None:
    bars = _session_bars(date(2026, 7, 24)) + [_bar(datetime(2026, 7, 24, 10, 0, tzinfo=_UTC))]  # 15:30 IST
    ib = _FakeIB([(bars, [])])

    with pytest.warns(UserWarning, match="outside"):
        result = get_intraday_data(ib, "TATASTEEL", date(2026, 7, 24), date(2026, 7, 24), environment="paper")

    assert result.metadata.bars_outside_session == 1
    assert result.metadata.n_bars == 76  # reported, not dropped


def test_get_intraday_data_drops_duplicate_timestamps_and_rejects_conflicting_ones() -> None:
    bars = _session_bars(date(2026, 7, 24))
    ib = _FakeIB([(bars + bars[:5], [])])

    result = get_intraday_data(ib, "TATASTEEL", date(2026, 7, 24), date(2026, 7, 24), environment="paper")
    assert result.metadata.duplicates_dropped == 5

    conflicting = _bar(bars[0].date, close=100.5)
    ib = _FakeIB([(bars + [conflicting], [])])
    with pytest.raises(IBKRValidationError, match="timestamp"):
        get_intraday_data(ib, "TATASTEEL", date(2026, 7, 24), date(2026, 7, 24), environment="paper")


def test_get_intraday_data_skips_a_holiday_session_without_inventing_bars() -> None:
    bars = _session_bars(date(2026, 7, 20)) + _session_bars(date(2026, 7, 22))  # 07-21 closed
    ib = _FakeIB([(bars, [])])

    result = get_intraday_data(ib, "TATASTEEL", date(2026, 7, 20), date(2026, 7, 22), environment="paper")

    assert result.metadata.missing_weekdays == [date(2026, 7, 21)]
    assert result.metadata.n_bars == 150


def test_get_intraday_data_raises_a_permission_error() -> None:
    ib = _FakeIB([([], [(354, "Requested market data is not subscribed.")])])

    with pytest.raises(IBKRPermissionError):
        get_intraday_data(ib, "TATASTEEL", date(2026, 7, 24), date(2026, 7, 24), environment="paper")


def test_get_intraday_data_rejects_unsupported_bar_sizes() -> None:
    with pytest.raises(ValueError, match="5 mins"):
        get_intraday_data(_FakeIB([]), "TATASTEEL", date(2026, 7, 24), date(2026, 7, 24), "paper", bar_size="1 min")


def test_check_intraday_sessions_on_an_empty_frame_reports_nothing() -> None:
    empty = DataFrame({"timestamp": Series([], dtype="datetime64[ns, Asia/Kolkata]")})

    assert check_intraday_sessions(empty, bar_minutes=5) == ({}, 0)


# --- closing auction (from 2026-08-03) ----------------------------------------


def test_session_close_moves_to_1515_when_the_closing_auction_starts() -> None:
    assert session_close(date(2026, 7, 31)) == time(15, 30)
    assert session_close(date(2026, 8, 3)) == time(15, 15)
    assert session_close(date(2026, 10, 23)) == time(15, 15)


def test_expected_session_slots_end_at_1510_with_the_closing_auction() -> None:
    slots = expected_session_slots(date(2026, 8, 3), bar_minutes=5)

    assert len(slots) == 72
    assert slots[-1].isoformat() == "2026-08-03T15:10:00+05:30"


def test_a_post_auction_session_ending_at_1510_is_complete() -> None:
    # What IB returns since 2026-08-24: 72 real bars, then zero-trade placeholders at 15:20 and 15:25.
    day = date(2026, 10, 23)
    placeholders = [_Bar(datetime(2026, 10, 23, 9 + h, m, tzinfo=_UTC), 166.95, 166.95, 166.95, 166.95, 0.0, 0.0, 0)
                    for h, m in ((0, 50), (0, 55))]  # 15:20 and 15:25 IST
    ib = _FakeIB([(_session_bars(day, n_slots=72) + placeholders, [])])

    result = get_intraday_data(ib, "TATASTEEL", day, day, environment="paper")

    assert result.metadata.n_bars == 72
    assert result.metadata.incomplete_sessions == {}
    assert result.metadata.bars_outside_session == 0
    assert [ts.strftime("%H:%M") for ts in result.metadata.placeholder_bars_dropped] == ["15:20", "15:25"]


def test_a_1515_bar_after_the_auction_start_is_reported_not_dropped() -> None:
    # Some August 2026 sessions have a real bar starting 15:15, outside the 09:15-15:15 grid.
    day = date(2026, 8, 6)
    ib = _FakeIB([(_session_bars(day, n_slots=73), [])])

    with pytest.warns(UserWarning, match="outside the session grid"):
        result = get_intraday_data(ib, "TATASTEEL", day, day, environment="paper")

    assert result.metadata.n_bars == 73
    assert result.metadata.bars_outside_session == 1
    assert result.metadata.incomplete_sessions == {}


def test_a_range_spanning_the_auction_start_uses_each_sessions_own_grid() -> None:
    bars = _session_bars(date(2026, 7, 31)) + _session_bars(date(2026, 8, 3), n_slots=72)
    ib = _FakeIB([(bars, [])])

    result = get_intraday_data(ib, "TATASTEEL", date(2026, 7, 31), date(2026, 8, 3), environment="paper")

    assert result.metadata.n_bars == 75 + 72
    assert result.metadata.incomplete_sessions == {}
    assert result.metadata.bars_outside_session == 0


# --- index contracts ------------------------------------------------------------


def _index_bar(day: date, close: float = 22600.0) -> _Bar:
    """An index bar as IB returns it: real OHLC, but no volume and no trade count."""
    return _Bar(day, open=close - 50, high=close + 40, low=close - 80, close=close, volume=0.0, average=0.0, barCount=0)


def test_qualify_contract_asks_ib_for_an_index_when_told_to() -> None:
    requested = []

    class _RecordingIB(_FakeIB):
        def reqContractDetails(self, contract: Any) -> list[_Details]:
            requested.append(contract.secType)
            return super().reqContractDetails(contract)

    contract = qualify_contract(_RecordingIB([]), "NIFTY50", sec_type="IND")

    assert requested == ["IND"]
    assert contract.symbol == "NIFTY50"


def test_qualify_contract_rejects_other_security_types() -> None:
    with pytest.raises(ValueError, match="STK"):
        qualify_contract(_FakeIB([]), "NIFTY", sec_type="FUT")


def test_drop_flat_index_bars_removes_only_bars_that_never_moved() -> None:
    frame = _frame([
        {"date": Timestamp("2024-11-19"), "open": 23500.0, "high": 23600.0, "low": 23450.0, "close": 23520.0},
        {"date": Timestamp("2024-11-20"), "open": 23520.0, "high": 23520.0, "low": 23520.0, "close": 23520.0},
    ])

    kept, dropped = drop_flat_index_bars(frame)

    assert list(kept["date"]) == [Timestamp("2024-11-19")]
    assert dropped == [Timestamp("2024-11-20")]


def test_index_bars_with_no_volume_are_kept_and_flat_holiday_bars_dropped() -> None:
    days = [ts.date() for ts in bdate_range(date(2024, 11, 18), date(2024, 11, 22))]
    bars = [_index_bar(d) for d in days]
    bars[2] = _Bar(days[2], 22600.0, 22600.0, 22600.0, 22600.0, volume=0.0, average=0.0, barCount=0)  # 2024-11-20
    ib = _FakeIB([(bars, [])])

    result = get_historical_data(ib, "NIFTY50", days[0], days[-1], environment="live", sec_type="IND")

    assert result.metadata.n_bars == 4  # zero volume alone does not make an index bar a placeholder
    assert result.metadata.placeholder_bars_dropped == [Timestamp("2024-11-20")]
    assert result.metadata.sec_type == "IND"


def test_clean_bars_rejects_infinite_prices() -> None:
    frame = _frame([{"date": Timestamp("2026-10-23"), "high": float("inf")}])

    with pytest.raises(IBKRValidationError, match="non-finite"):
        clean_bars(frame, "TEST")


def test_stock_requests_still_record_stk() -> None:
    ib = _FakeIB([(_weekday_bars(date(2026, 10, 19), date(2026, 10, 23)), [])])

    result = get_historical_data(ib, "RELIANCE", date(2026, 10, 19), date(2026, 10, 23), environment="paper")

    assert result.metadata.sec_type == "STK"
