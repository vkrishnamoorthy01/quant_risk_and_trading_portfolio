"""Unit tests for shared.validation and import-time side effects (offline)."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import pytest

from shared.validation import PriceDataError, validate_close_prices


def _prices() -> pd.DataFrame:
    return pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 50.5, 51.0]},
                        index=pd.bdate_range("2026-10-19", periods=3))


def test_complete_finite_prices_pass() -> None:
    validate_close_prices(_prices(), ["A", "B"])


def test_a_ticker_missing_entirely_is_rejected() -> None:
    # Kite's fetch_price_data silently leaves out a ticker whose fetch is empty.
    with pytest.raises(PriceDataError, match="C"):
        validate_close_prices(_prices(), ["A", "B", "C"])


def test_a_missing_close_is_rejected_by_default() -> None:
    prices = _prices()
    prices.iloc[1, 0] = np.nan
    with pytest.raises(PriceDataError, match="A"):
        validate_close_prices(prices, ["A", "B"])


def test_a_one_session_gap_can_be_carried_forward_explicitly() -> None:
    prices = _prices()
    prices.iloc[1, 0] = np.nan

    with pytest.warns(UserWarning, match="Carried the last close forward"):
        filled = validate_close_prices(prices, ["A", "B"], max_fill_sessions=1)

    assert filled.iloc[1, 0] == 100.0  # no trade: last close carried
    returns = filled.pct_change().dropna()
    assert returns.iloc[0, 0] == 0.0
    assert np.isclose(returns.iloc[1, 0], 0.02)  # the full 100 -> 102 move lands on the day it traded


def test_a_late_starting_ticker_is_not_filled_backwards() -> None:
    prices = _prices()
    prices.iloc[0, 1] = np.nan
    with pytest.raises(PriceDataError, match="B: no close before"):
        validate_close_prices(prices, ["A", "B"], max_fill_sessions=5)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, 0.0, -1.0])
def test_nonfinite_or_nonpositive_closes_are_rejected(bad: float) -> None:
    prices = _prices()
    prices.iloc[2, 1] = bad
    with pytest.raises(PriceDataError, match="B"):
        validate_close_prices(prices, ["A", "B"])


def test_importing_the_backtest_runner_does_not_change_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_chdir(path: str) -> None:
        raise AssertionError(f"import changed directory to {path}")

    monkeypatch.setattr(os, "chdir", _no_chdir)
    import signal_project.run_backtest as run_backtest

    importlib.reload(run_backtest)
