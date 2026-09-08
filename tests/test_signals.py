"""Unit tests for signal_project.signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_project.signals import mean_reversion_signal, momentum_signal


def _price_path(values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"A": values}, index=dates)


def test_momentum_signal_is_long_after_a_rising_path() -> None:
    prices = _price_path([100.0 + i for i in range(30)])  # steadily rising

    signal = momentum_signal(prices, lookback_months=1)  # 21 trading days

    assert signal["A"].iloc[-1]


def test_momentum_signal_is_flat_after_a_falling_path() -> None:
    prices = _price_path([130.0 - i for i in range(30)])  # steadily falling

    signal = momentum_signal(prices, lookback_months=1)

    assert not signal["A"].iloc[-1]


def test_momentum_signal_is_false_before_enough_history() -> None:
    prices = _price_path(list(np.linspace(100, 110, 10)))

    signal = momentum_signal(prices, lookback_months=1)  # needs 21 obs, only have 10

    assert not signal["A"].any()


def test_mean_reversion_signal_is_long_after_a_recent_drop() -> None:
    prices = _price_path([100.0] * 20 + [90.0])  # sharp one-day drop

    signal = mean_reversion_signal(prices, lookback_days=5)

    assert signal["A"].iloc[-1]


def test_mean_reversion_signal_is_flat_after_a_recent_rise() -> None:
    prices = _price_path([100.0] * 20 + [110.0])

    signal = mean_reversion_signal(prices, lookback_days=5)

    assert not signal["A"].iloc[-1]
