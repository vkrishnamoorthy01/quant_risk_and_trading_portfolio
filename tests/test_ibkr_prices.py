"""Unit tests for data_providers.ibkr_prices. Stores synthetic bars in
pytest's tmp_path; never needs TWS running.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pandas import DataFrame, Timestamp, bdate_range

import data_providers.ibkr_prices as prices_module
from data_providers.ibkr_prices import (
    IBKRInsufficientDataError,
    build_close_prices,
    fetch_ibkr_price_data,
    ib_symbol,
)
from data_providers.ibkr_provider import EXCHANGE_TIMEZONE, HistoricalData, RequestMetadata
from data_providers.ibkr_storage import save_bars
from risk_project.var_models import portfolio_returns
from shared.data import compute_returns

_NOW = datetime(2026, 10, 26, 17, 0, tzinfo=EXCHANGE_TIMEZONE)
_START, _END = date(2026, 9, 1), date(2026, 10, 23)


def _store(root: Path, symbol: str, start: date = _START, end: date = _END,
           skip: set[date] = frozenset(), close: float = 100.0) -> None:
    """Save synthetic daily bars for every weekday in [start, end] minus `skip`, closes rising by 1 a day."""
    days = [ts for ts in bdate_range(start, end) if ts.date() not in skip]
    closes = [close + i for i in range(len(days))]
    bars = DataFrame({
        "date": days, "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "close": closes, "volume": 1000.0, "vwap": closes, "bar_count": 10,
    }).assign(symbol=symbol, exchange="NSE", currency="INR", con_id=sum(map(ord, symbol)))
    metadata = RequestMetadata(
        symbol=symbol, exchange="NSE", currency="INR", con_id=sum(map(ord, symbol)), environment="paper",
        bar_size="1 day", what_to_show="TRADES", use_rth=True, requested_start=start, requested_end=end,
        first_date=days[0].date(), last_date=days[-1].date(), n_bars=len(days), n_requests=1,
        duplicates_dropped=0, placeholder_bars_dropped=[], missing_weekdays=[], empty_chunks=[],
        downloaded_at=datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc),
    )
    save_bars(HistoricalData(bars=bars, metadata=metadata), root=root, now=_NOW)


def _build(root: Path, tickers: list[str], **kwargs: Any) -> DataFrame:
    kwargs.setdefault("min_observations", 10)
    return build_close_prices(tickers, _START.isoformat(), _END.isoformat(), root=root, **kwargs)


def test_ib_symbol_maps_the_nse_symbols_ib_spells_differently() -> None:
    assert ib_symbol("M&M") == "MM"
    assert ib_symbol("HINDUNILVR") == "HINDUNILV"
    assert ib_symbol("ULTRACEMCO") == "ULTRACEMC"
    assert ib_symbol("RELIANCE") == "RELIANCE"


def test_build_close_prices_matches_fetch_price_data_shape(tmp_path: Path) -> None:
    for symbol in ("RELIANCE", "MM"):
        _store(tmp_path, symbol)

    prices = _build(tmp_path, ["RELIANCE", "M&M"])

    assert list(prices.columns) == ["RELIANCE", "M&M"]  # NSE names, in the order asked for
    assert prices.index.name == "date"
    assert prices.index.tz is None
    assert prices.index.is_monotonic_increasing
    assert prices.index[0] == Timestamp("2026-09-01")
    assert prices.notna().all().all()
    assert prices.attrs["source"] == "IBKR"


def test_build_close_prices_output_feeds_the_risk_engine_without_gaps(tmp_path: Path) -> None:
    for symbol in ("RELIANCE", "TATASTEEL", "HDFCBANK"):
        _store(tmp_path, symbol)

    prices = _build(tmp_path, ["RELIANCE", "TATASTEEL", "HDFCBANK"])
    returns = compute_returns(prices, method="simple")
    weights = DataFrame({"w": [1 / 3] * 3}, index=["RELIANCE", "TATASTEEL", "HDFCBANK"])["w"]
    port = portfolio_returns(returns, weights)

    assert len(port) == len(prices) - 1
    assert port.notna().all()


def test_a_ticker_with_nothing_stored_is_named_in_the_error(tmp_path: Path) -> None:
    _store(tmp_path, "RELIANCE")

    with pytest.raises(IBKRInsufficientDataError, match="TATASTEEL: nothing stored"):
        _build(tmp_path, ["RELIANCE", "TATASTEEL"])


def test_a_short_history_is_refused(tmp_path: Path) -> None:
    _store(tmp_path, "RELIANCE")
    _store(tmp_path, "DRREDDY", start=date(2026, 10, 1))

    with pytest.raises(IBKRInsufficientDataError, match="DRREDDY: history starts 2026-10-01"):
        _build(tmp_path, ["RELIANCE", "DRREDDY"])


def test_stale_data_is_refused(tmp_path: Path) -> None:
    _store(tmp_path, "RELIANCE")
    _store(tmp_path, "TATASTEEL", end=date(2026, 10, 9))

    with pytest.raises(IBKRInsufficientDataError, match="TATASTEEL: last close 2026-10-09"):
        _build(tmp_path, ["RELIANCE", "TATASTEEL"])


def test_a_session_one_ticker_lacks_raises_by_default(tmp_path: Path) -> None:
    _store(tmp_path, "RELIANCE")
    _store(tmp_path, "TATASTEEL", skip={date(2026, 10, 7)})

    with pytest.raises(IBKRInsufficientDataError, match=r"TATASTEEL missing 1 session\(s\), e.g. \['2026-10-07'\]"):
        _build(tmp_path, ["RELIANCE", "TATASTEEL"])


def test_allowed_gaps_stay_nan_and_warn(tmp_path: Path) -> None:
    _store(tmp_path, "RELIANCE")
    _store(tmp_path, "TATASTEEL", skip={date(2026, 10, 7)})

    with pytest.warns(UserWarning, match="TATASTEEL is missing"):
        prices = _build(tmp_path, ["RELIANCE", "TATASTEEL"], max_missing_fraction=0.05)

    assert prices["TATASTEEL"].isna().sum() == 1  # not forward-filled
    assert prices.loc[Timestamp("2026-10-07"), "RELIANCE"] > 0


def test_a_holiday_no_ticker_traded_is_not_a_gap(tmp_path: Path) -> None:
    holiday = {date(2026, 10, 2)}
    _store(tmp_path, "RELIANCE", skip=holiday)
    _store(tmp_path, "TATASTEEL", skip=holiday)

    prices = _build(tmp_path, ["RELIANCE", "TATASTEEL"])

    assert Timestamp("2026-10-02") not in prices.index
    assert prices.notna().all().all()


def test_too_few_sessions_is_refused(tmp_path: Path) -> None:
    _store(tmp_path, "RELIANCE")

    with pytest.raises(IBKRInsufficientDataError, match="need at least 250"):
        _build(tmp_path, ["RELIANCE"], min_observations=250)


def test_fetch_ibkr_price_data_downloads_with_ib_symbols_then_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connected = []
    fetched = []

    class _Connection:
        def __init__(self, config: Any) -> None:
            connected.append(config.environment)

        def __enter__(self) -> object:
            return object()

        def __exit__(self, *exc: object) -> None:
            return None

    def _fake_fetch_and_store(ib: Any, symbol: str, start: date, end: date, environment: str, **kwargs: Any) -> None:
        fetched.append(symbol)
        _store(kwargs["root"], symbol, start=start, end=end)

    monkeypatch.setattr(prices_module, "ibkr_connection", _Connection)
    monkeypatch.setattr(prices_module, "fetch_and_store", _fake_fetch_and_store)

    prices = fetch_ibkr_price_data(["RELIANCE", "M&M"], "2026-09-01", "2026-10-23", root=tmp_path, min_observations=10)

    assert connected == ["paper"]
    assert fetched == ["RELIANCE", "MM"]
    assert list(prices.columns) == ["RELIANCE", "M&M"]
