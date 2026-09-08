"""Unit tests for signal_project.backtest.

Uses a fake signal (always long, ignoring price/lookback) so tests can
isolate the day-by-day NAV/weight/risk-control mechanics from the actual
momentum/mean-reversion signal logic, which is covered separately in
test_signals.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_project.backtest import run_book_backtest


def _always_long(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return pd.DataFrame(True, index=prices.index, columns=prices.columns)


def test_nav_tracks_the_sole_active_ticker_once_a_position_is_open() -> None:
    # Small oscillation gives a stable, well-defined vol; the book should
    # fully allocate to A (the only active/uncapped name) once vol warms up,
    # and NAV growth should then track A's return one day later (weights
    # decided at close of day t apply to day t+1's return).
    dates = pd.bdate_range("2020-01-06", periods=15)
    prices = pd.DataFrame({"A": [100.0, 100.5] * 7 + [100.0]}, index=dates)
    returns = prices.pct_change().dropna()

    result = run_book_backtest(
        prices, returns, _always_long, lookback=1, book_notional=1_000_000.0,
        rebalance_freq="D", cap=1.0, vol_lookback=2,
    )

    nav_returns = result.nav.pct_change().dropna()
    # By the tail of the series the position has long been open and steady at 1.0.
    aligned_asset_returns = returns["A"].reindex(nav_returns.index)
    np.testing.assert_allclose(nav_returns.iloc[-5:], aligned_asset_returns.iloc[-5:])


def test_stop_loss_forces_a_same_day_exit_on_a_non_rebalance_day() -> None:
    # Weekly rebalance -> Mondays only (2020-01-07 as the first available
    # date since day 0 is dropped by pct_change, then 2020-01-13, 2020-01-20).
    # Position opens on 2020-01-13 (vol has warmed up by then). A crash on
    # 2020-01-15 (a Wednesday, not a rebalance day) should force weight to
    # 0 immediately via the stop-loss, independent of the weekly schedule.
    dates = pd.bdate_range("2020-01-06", periods=15)
    prices = [100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    prices_df = pd.DataFrame({"A": prices}, index=dates)
    returns = prices_df.pct_change().dropna()

    result = run_book_backtest(
        prices_df, returns, _always_long, lookback=1, book_notional=1_000_000.0,
        rebalance_freq="W", cap=1.0, vol_lookback=2, n_sigma=2.0,
    )

    weights = result.weights_history["A"]
    assert weights.loc["2020-01-13"] == 1.0  # position opened at the Monday rebalance
    assert weights.loc["2020-01-14"] == 1.0  # small dip, within the stop threshold
    assert weights.loc["2020-01-15"] == 0.0  # crash day: stop-loss forces exit
    assert weights.loc["2020-01-16"] == 0.0  # stays flat (no rebalance to reconsider)
    assert weights.loc["2020-01-17"] == 0.0


def test_drawdown_halt_blocks_reopening_a_stopped_out_position() -> None:
    # Two tickers, B never signals (always flat) except as a stable
    # placeholder; A gets stopped out by a crash large enough to also push
    # the whole book's NAV past the 15% drawdown halt threshold. On the
    # next rebalance, A's signal is true again but the halt should block
    # re-opening it since it's currently flat (a "new" entry).
    dates = pd.bdate_range("2020-01-06", periods=20)
    a_prices = [100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0, 70.0] + [70.5, 70.0] * 6
    prices_df = pd.DataFrame({"A": a_prices}, index=dates)
    returns = prices_df.pct_change().dropna()

    result = run_book_backtest(
        prices_df, returns, _always_long, lookback=1, book_notional=1_000_000.0,
        rebalance_freq="W", cap=1.0, vol_lookback=2, n_sigma=2.0,
    )

    weights = result.weights_history["A"]
    assert weights.loc["2020-01-15"] == 0.0  # ~30% crash stops the position out
    assert result.nav.loc["2020-01-15"] < 0.85 * result.nav.max()  # book drawdown breached 15%
    # Next weekly rebalance (2020-01-20): signal is true again but NAV is
    # still deep in drawdown, so the halt should keep this a flat position.
    assert weights.loc["2020-01-20"] == 0.0
