"""NSE equity session calendar, used to tell real data gaps from expected closures.

A date missing from a download is only acceptable if the exchange was
closed or the provider is known not to carry it. The calendar data lives
in nse_sessions.csv, one row per exceptional date:

    closed           a weekday the exchange was shut (holidays, election days)
    weekend_session  a special Saturday/Sunday session (Budget day, Muhurat
                     trading, disaster-recovery drills)
    omitted:IBKR     a session NSE held that IB has no data for (New Year's
                     Day in 2015-2016 and 2018-2021, the 2018-11-07 Muhurat
                     session, and seven weekend sessions)

Every other weekday is an ordinary session, every other weekend day is not.

The closures were derived from IBKR live data (weekdays with no bar for any
of the 16 portfolio stocks or NIFTY 50) and spot-checked against known
holidays; they have not been reconciled line by line with NSE's circulars.
They are only known through CALENDAR_VERIFIED_THROUGH. After that date
every weekday counts as an expected session, so a later holiday shows up
as an unresolved gap until the calendar is extended, rather than being
silently accepted.
"""

from __future__ import annotations

from csv import DictReader
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

CALENDAR_FILE = Path(__file__).resolve().parent / "nse_sessions.csv"
CALENDAR_START = date(2015, 1, 1)
CALENDAR_VERIFIED_THROUGH = date(2026, 10, 23)


@lru_cache(maxsize=1)
def _calendar() -> dict[str, frozenset[date]]:
    """Dates grouped by kind, read once from CALENDAR_FILE."""
    groups: dict[str, set[date]] = {}
    with CALENDAR_FILE.open(newline="") as f:
        rows = DictReader(line for line in f if not line.startswith("#"))
        for row in rows:
            groups.setdefault(row["kind"], set()).add(date.fromisoformat(row["date"]))
    return {kind: frozenset(days) for kind, days in groups.items()}


def is_known_closure(day: date) -> bool:
    """Whether the calendar records `day` as a weekday exchange closure."""
    return day in _calendar().get("closed", frozenset())


def weekend_sessions() -> frozenset[date]:
    """Special Saturday/Sunday sessions."""
    return _calendar().get("weekend_session", frozenset())


def provider_omitted_sessions(provider: str) -> frozenset[date]:
    """Sessions NSE held that `provider` (e.g. "IBKR") is known not to carry."""
    return _calendar().get(f"omitted:{provider}", frozenset())


def is_expected_session(day: date, provider: str | None = None) -> bool:
    """Whether NSE traded on `day` (and, if `provider` is given, the provider should have it)."""
    if provider is not None and day in provider_omitted_sessions(provider):
        return False
    if day.weekday() >= 5:
        return day in weekend_sessions()
    return not is_known_closure(day)


def expected_sessions(start: date, end: date, provider: str | None = None) -> list[date]:
    """Expected session dates in [start, end], in order."""
    days = []
    day = start
    while day <= end:
        if is_expected_session(day, provider):
            days.append(day)
        day += timedelta(days=1)
    return days
