"""Unit tests for data_providers.nse_calendar (offline, no data downloads)."""

from __future__ import annotations

from datetime import date

from data_providers.nse_calendar import (
    CALENDAR_VERIFIED_THROUGH,
    expected_sessions,
    is_known_closure,
    provider_omitted_sessions,
)


def test_known_holidays_are_closures_and_ordinary_days_are_sessions() -> None:
    assert is_known_closure(date(2026, 1, 26))  # Republic Day
    assert is_known_closure(date(2024, 11, 20))  # Maharashtra election
    assert not is_known_closure(date(2026, 10, 14))  # an ordinary Wednesday


def test_expected_sessions_skip_closures_and_include_special_weekend_sessions() -> None:
    sessions = expected_sessions(date(2025, 1, 30), date(2025, 2, 3))

    assert sessions == [date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 3)]


def test_ib_omissions_are_recorded_separately_from_exchange_closures() -> None:
    # NSE traded on these days, but IB has no data for them.
    omitted = provider_omitted_sessions("IBKR")

    assert date(2016, 1, 1) in omitted and date(2018, 11, 7) in omitted
    assert not is_known_closure(date(2016, 1, 1))
    assert date(2016, 1, 1) in expected_sessions(date(2016, 1, 1), date(2016, 1, 1))
    assert date(2016, 1, 1) not in expected_sessions(date(2016, 1, 1), date(2016, 1, 1), provider="IBKR")


def test_the_calendar_states_how_far_it_is_verified() -> None:
    assert CALENDAR_VERIFIED_THROUGH == date(2026, 10, 23)
