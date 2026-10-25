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


# --- audit regressions ----------------------------------------------------------

import pytest  # noqa: E402

import signal_project.backtest as backtest_module  # noqa: E402


def _fixed_weights(signal: pd.Series, vol: pd.Series, cap: float) -> pd.Series:
    """Equal weights across signaled names, ignoring vol, to isolate accounting."""
    active = signal[signal].index
    weights = pd.Series(0.0, index=signal.index)
    weights[active] = 1.0 / len(active)
    return weights


def test_weights_drift_with_prices_between_weekly_rebalances(monkeypatch: pytest.MonkeyPatch) -> None:
    # The audit's controlled example: 50/50 in A and B with no trades in
    # between; A goes 100 -> 200 -> 100, B stays at 100. Fixed shares must
    # bring NAV back to exactly 100. Re-applying constant 50/50 weights each
    # day (implicit daily rebalancing) gives 112.5 instead.
    monkeypatch.setattr(backtest_module, "vol_scaled_weights", _fixed_weights)
    monkeypatch.setattr(backtest_module, "realized_vol", lambda returns, lookback: returns * 0.0 + 0.01)
    dates = pd.to_datetime(["2020-01-10", "2020-01-13", "2020-01-14", "2020-01-15"])  # Fri, Mon, Tue, Wed
    prices = pd.DataFrame({"A": [100.0, 100.0, 200.0, 100.0], "B": [100.0, 100.0, 100.0, 100.0]}, index=dates)
    returns = prices.pct_change().dropna()

    result = run_book_backtest(prices, returns, _always_long, lookback=1, book_notional=100.0,
                               rebalance_freq="W", cap=1.0, n_sigma=1e9)

    assert result.weights_history.loc["2020-01-13"].tolist() == [0.5, 0.5]  # Monday rebalance
    np.testing.assert_allclose(result.nav.loc["2020-01-14"], 150.0)
    np.testing.assert_allclose(result.weights_history.loc["2020-01-14"].tolist(), [2 / 3, 1 / 3])  # drifted
    np.testing.assert_allclose(result.nav.loc["2020-01-15"], 100.0)


def _crash_path(crash_day: int, n: int = 30) -> pd.DataFrame:
    """Small oscillation, a 10% drop on `crash_day`, then small oscillation at the lower level."""
    base = [100.0, 100.5] * n
    path = base[:crash_day] + [90.0 + (0.5 if i % 2 else 0.0) for i in range(n * 2 - crash_day)]
    return pd.DataFrame({"A": path[:n]}, index=pd.bdate_range("2020-01-06", periods=n))


def test_a_stopped_name_is_not_reopened_by_the_same_days_daily_rebalance() -> None:
    prices = _crash_path(crash_day=10)
    returns = prices.pct_change().dropna()
    crash = prices.index[10]
    next_day = prices.index[11]

    result = run_book_backtest(prices, returns, _always_long, lookback=1, book_notional=1_000_000.0,
                               rebalance_freq="D", cap=1.0, vol_lookback=2, n_sigma=2.0)

    weights = result.weights_history["A"]
    assert weights.loc[prices.index[9]] == 1.0  # held going into the crash
    assert weights.loc[crash] == 0.0  # stopped, and not reopened at the same close
    assert weights.loc[next_day] == 1.0  # eligible again at the next scheduled rebalance


def test_a_stop_on_a_weekly_rebalance_day_keeps_the_name_out_until_the_next_week() -> None:
    # 2020-01-06 is a Monday; index 10 is Monday 2020-01-20, a weekly rebalance day.
    prices = _crash_path(crash_day=10)
    returns = prices.pct_change().dropna()

    result = run_book_backtest(prices, returns, _always_long, lookback=1, book_notional=1_000_000.0,
                               rebalance_freq="W", cap=1.0, vol_lookback=2, n_sigma=2.0)

    weights = result.weights_history["A"]
    assert weights.loc["2020-01-17"] == 1.0  # held into the crash (opened on 2020-01-13)
    assert (weights.loc["2020-01-20":"2020-01-24"] == 0.0).all()  # stopped Monday; out all week
    assert weights.loc["2020-01-27"] == 1.0  # re-entered at the next Monday rebalance


def test_a_missing_return_on_a_held_position_raises_instead_of_counting_as_zero() -> None:
    dates = pd.bdate_range("2020-01-06", periods=15)
    prices = pd.DataFrame({"A": [100.0, 100.5] * 7 + [100.0]}, index=dates)
    returns = prices.pct_change().dropna()
    returns.iloc[10, 0] = np.nan  # A is held by then

    with pytest.raises(ValueError, match="A"):
        run_book_backtest(prices, returns, _always_long, lookback=1, book_notional=1_000_000.0,
                          rebalance_freq="D", cap=1.0, vol_lookback=2)
