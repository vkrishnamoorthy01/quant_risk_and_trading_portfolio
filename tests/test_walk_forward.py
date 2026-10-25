"""Unit tests for signal_project.walk_forward."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_project.walk_forward import rolling_windows, run_walk_forward, sharpe_ratio, winner_change_rate


def test_rolling_windows_counts_and_boundaries() -> None:
    dates = pd.bdate_range("2015-01-01", "2020-01-01")

    windows = rolling_windows(dates, train_months=24, test_months=6)

    # (60 months of data - 24 month train) / 6 month step = 6 windows.
    assert len(windows) == 6

    train_start, train_end, test_start, test_end = windows[0]
    assert train_start == dates.min()
    assert abs((train_end - (train_start + pd.DateOffset(months=24))).days) <= 5
    assert test_start > train_end
    assert abs((test_end - (test_start + pd.DateOffset(months=6))).days) <= 5

    # Rolling (not expanding): each window's train length is ~24 months, and
    # train_start advances by ~6 months between consecutive windows.
    for i in range(1, len(windows)):
        prev_train_start = windows[i - 1][0]
        assert abs((windows[i][0] - (prev_train_start + pd.DateOffset(months=6))).days) <= 5


def test_sharpe_ratio_matches_manual_calc() -> None:
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])

    result = sharpe_ratio(returns, periods_per_year=252)

    expected = returns.mean() / returns.std() * (252**0.5)
    assert np.isclose(result, expected)


def test_sharpe_ratio_is_negative_infinity_for_zero_vol_returns() -> None:
    assert sharpe_ratio(pd.Series([0.0, 0.0, 0.0])) == float("-inf")


def test_winner_change_rate_counts_consecutive_changes() -> None:
    report = pd.DataFrame({"selected_lookback": [3, 3, 6, 6, 3]})

    # 3->3 (no change), 3->6 (change), 6->6 (no change), 6->3 (change): 2/4
    assert np.isclose(winner_change_rate(report), 0.5)


def _long_only_for_lookback(target_lookback: int):
    def _signal_fn(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return pd.DataFrame(lookback == target_lookback, index=prices.index, columns=prices.columns)

    return _signal_fn


def test_run_walk_forward_selects_the_only_profitable_candidate() -> None:
    # A signal that's only ever long when lookback == 6 (all other candidates
    # stay flat forever, giving a degenerate -inf Sharpe). On a generally
    # rising, noisy price path this guarantees lookback=6 wins every window.
    n = 900  # ~3.5 years of business days
    trend = 100.0 * (1.0002**np.arange(n))
    noise = 1.0 + 0.003 * np.sin(np.arange(n))
    prices = pd.DataFrame({"A": trend * noise}, index=pd.bdate_range("2015-01-01", periods=n))
    returns = prices.pct_change().dropna()

    result = run_walk_forward(
        prices, returns, _long_only_for_lookback(6), candidate_lookbacks=[1, 3, 6, 12],
        book_notional=1_000_000.0, rebalance_freq="D", cap=1.0,
    )

    assert len(result.window_report) >= 1
    assert (result.window_report["selected_lookback"] == 6).all()
    assert result.nav.index.is_monotonic_increasing
    assert not result.nav.index.has_duplicates


# --- audit regressions: non-overlapping, exactly-once stitching ------------------


def test_rolling_windows_test_periods_never_overlap_and_leave_no_gaps() -> None:
    # A plain weekday calendar over the IBKR backtest's span reproduces the
    # audit's overlapping windows with the old independently clipped boundaries.
    dates = pd.bdate_range("2015-01-05", "2026-10-23")

    windows = rolling_windows(dates, train_months=24, test_months=6)

    for (_, _, prev_start, prev_end), (train_start, train_end, test_start, test_end) in zip(windows, windows[1:]):
        assert test_start > prev_end, f"test window starting {test_start.date()} overlaps one ending {prev_end.date()}"
        # Contiguous: the next test period starts on the very next trading date.
        assert test_start == dates[dates.get_loc(prev_end) + 1]
    for train_start, train_end, test_start, test_end in windows:
        assert train_start <= train_end < test_start <= test_end
        assert train_end == dates[dates.get_loc(test_start) - 1]  # training ends the day before testing


def test_walk_forward_compounds_every_out_of_sample_day_exactly_once() -> None:
    # One asset, always long, cap 1.0, daily rebalance: once vol has warmed up
    # each test window is flat on its first day (it starts fresh) and fully
    # invested afterwards. So the correct stitched NAV is the notional times
    # the product of (1 + asset return) over every out-of-sample date that is
    # not the first date of its window. Overlapping windows compound the
    # overlap days twice and miss this number.
    n = 1500
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        {"A": 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, n))}, index=pd.bdate_range("2015-01-05", periods=n)
    )
    returns = prices.pct_change().dropna()

    result = run_walk_forward(
        prices, returns, _long_only_for_lookback(6), candidate_lookbacks=[6],
        book_notional=1_000_000.0, rebalance_freq="D", cap=1.0, n_sigma=1e9,
    )

    first_days = set(result.window_report["test_start"])
    earned = [returns.loc[t, "A"] for t in result.nav.index if t not in first_days]
    expected_final = 1_000_000.0 * np.prod(1.0 + np.array(earned))
    np.testing.assert_allclose(result.nav.iloc[-1], expected_final, rtol=1e-12)
    # And the stitched path has no jumps: every day's NAV return is either the
    # asset's return (invested) or zero (a window's flat first day).
    nav_returns = result.nav.pct_change().dropna()
    for t, r in nav_returns.items():
        assert np.isclose(r, 0.0 if t in first_days else returns.loc[t, "A"], rtol=1e-12, atol=1e-15)
