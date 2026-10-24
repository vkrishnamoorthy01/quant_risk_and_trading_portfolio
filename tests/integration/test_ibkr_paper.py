"""Integration tests against paper TWS on port 7497. Opt-in: see conftest.py.

Each test prints what IB actually returned (run with -s to see it), since a
passing connection is not proof that the requested data is all there.
Market data only: nothing here places or modifies orders.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import time, timedelta
from pathlib import Path

import pytest
from ib_async import IB
from pandas import DateOffset, Timestamp

from data_providers.ibkr_connection import IBKRConfig, IBKREnvironmentMismatchError, ibkr_connection
from data_providers.ibkr_provider import IBKRContractError, get_historical_data, get_intraday_data, qualify_stock
from data_providers.ibkr_storage import fetch_and_store

TODAY = Timestamp.today().date()
THREE_YEARS_AGO = (Timestamp(TODAY) - DateOffset(years=3)).date()


def test_paper_connection_is_verified_and_disconnects(paper_config: IBKRConfig) -> None:
    with ibkr_connection(paper_config) as ib:
        assert ib.isConnected()
        accounts = ib.managedAccounts()
        print(f"\nconnected, {len(accounts)} paper account(s), server time {ib.reqCurrentTime()}")
        assert accounts and all(account.startswith("D") for account in accounts)
    assert not ib.isConnected()


def test_asking_for_live_on_the_paper_port_fails_closed(paper_config: IBKRConfig) -> None:
    config = replace(paper_config, environment="live")  # still port 7497: a paper TWS

    with pytest.raises(IBKREnvironmentMismatchError) as excinfo:
        with ibkr_connection(config):
            pytest.fail("the body must not run when the environment does not match")
    print(f"\nrefused as expected: {excinfo.value}")


def test_qualify_reliance(paper_ib: IB) -> None:
    contract = qualify_stock(paper_ib, "RELIANCE")

    print(f"\n{contract}")
    assert contract.conId > 0
    assert (contract.symbol, contract.primaryExchange, contract.currency) == ("RELIANCE", "NSE", "INR")


def test_unknown_symbol_is_rejected(paper_ib: IB) -> None:
    with pytest.raises(IBKRContractError) as excinfo:
        qualify_stock(paper_ib, "NOTANSESTOCKXYZ")
    print(f"\n{excinfo.value}")


def test_reliance_daily_bars_for_the_last_month(paper_ib: IB) -> None:
    result = get_historical_data(paper_ib, "RELIANCE", TODAY - timedelta(days=31), TODAY, "paper")
    meta = result.metadata

    print(f"\nRELIANCE 1M daily: {meta.n_bars} bars, {meta.first_date} to {meta.last_date}, "
          f"missing weekdays {[d.isoformat() for d in meta.missing_weekdays]}")
    assert meta.n_bars >= 15
    assert (TODAY - meta.last_date).days <= 7
    assert result.bars["date"].is_monotonic_increasing


@pytest.mark.parametrize("symbol", ["RELIANCE", "TATASTEEL", "HDFCBANK"])
def test_three_years_of_daily_bars(paper_ib: IB, symbol: str) -> None:
    result = get_historical_data(paper_ib, symbol, THREE_YEARS_AGO, TODAY, "paper")
    meta = result.metadata

    print(f"\n{symbol} 3Y daily: {meta.n_bars} bars, {meta.first_date} to {meta.last_date}, "
          f"{meta.n_requests} requests, {len(meta.missing_weekdays)} missing weekdays, "
          f"placeholders dropped {[ts.date().isoformat() for ts in meta.placeholder_bars_dropped]}, "
          f"warnings {meta.warnings or 'none'}")
    assert meta.n_bars >= 700  # about 245 NSE sessions a year
    assert (meta.first_date - THREE_YEARS_AGO).days <= 7
    assert (TODAY - meta.last_date).days <= 7
    assert not result.bars["date"].duplicated().any()
    paper_ib.sleep(2)


def test_tatasteel_five_minute_bars_for_the_last_week(paper_ib: IB) -> None:
    result = get_intraday_data(paper_ib, "TATASTEEL", TODAY - timedelta(days=7), TODAY, "paper")
    meta = result.metadata
    stamps = result.bars["timestamp"]

    print(f"\nTATASTEEL 5-min: {meta.n_bars} bars over {stamps.dt.date.nunique()} sessions, "
          f"{stamps.min()} to {stamps.max()}, incomplete sessions "
          f"{ {d.isoformat(): n for d, n in meta.incomplete_sessions.items()} }, "
          f"outside grid {meta.bars_outside_session}")
    assert meta.n_bars > 0
    assert str(stamps.dt.tz) == "Asia/Kolkata"
    assert not stamps.duplicated().any()
    assert ((stamps.dt.minute % 5) == 0).all()
    assert stamps.dt.time.between(time(9, 15), time(15, 25)).all()


def test_fetch_and_store_round_trip_does_not_download_twice(paper_ib: IB, tmp_path: Path) -> None:
    start = TODAY - timedelta(days=31)
    first = fetch_and_store(paper_ib, "RELIANCE", start, TODAY, "paper", root=tmp_path)
    second = fetch_and_store(paper_ib, "RELIANCE", start, TODAY, "paper", root=tmp_path)

    print(f"\nstored {len(first.bars)} rows at {first.path.relative_to(tmp_path)}, "
          f"coverage {first.info['coverage']}, downloads logged after repeat: {len(second.info['downloads'])}")
    assert second.bars.equals(first.bars)
    assert len(second.info["downloads"]) == 1
