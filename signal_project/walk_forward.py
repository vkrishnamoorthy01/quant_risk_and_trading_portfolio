"""Rolling walk-forward lookback selection and out-of-sample stitching.

Each training window evaluates every candidate lookback in-sample on
Sharpe ratio, freezes the winner, and applies it unchanged over the
following out-of-sample test window. Each out-of-sample test window is
simulated as an independent, freshly-flat book (starting at the prior
window's ending NAV) rather than carrying open positions/stop state
across the train/test boundary — a deliberate simplification so windows
can be evaluated and stitched independently; see the project README.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from signal_project.backtest import SignalFn, run_book_backtest
from signal_project.risk_controls import STOP_LOSS_N_SIGMA
from signal_project.sizing import POSITION_CAP, VOL_LOOKBACK_DAYS

TRAIN_MONTHS = 24
TEST_MONTHS = 6
TRADING_PERIODS_PER_YEAR = 252


@dataclass
class WalkForwardResult:
    """Output of a full rolling walk-forward run for one book."""

    nav: pd.Series
    window_report: pd.DataFrame


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio (zero risk-free rate). -inf if returns are degenerate."""
    if returns.empty or returns.std() == 0:
        return float("-inf")
    return float(returns.mean() / returns.std() * (periods_per_year**0.5))


def _clip_to_index(target: pd.Timestamp, dates: pd.DatetimeIndex, side: str) -> pd.Timestamp:
    """Nearest actual trading date: >= target for "start", <= target for "end"."""
    if side == "start":
        pos = dates.searchsorted(target, side="left")
    else:
        pos = dates.searchsorted(target, side="right") - 1
    pos = min(max(pos, 0), len(dates) - 1)
    return dates[pos]


def rolling_windows(
    dates: pd.DatetimeIndex, train_months: int = TRAIN_MONTHS, test_months: int = TEST_MONTHS
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Rolling (train_start, train_end, test_start, test_end) windows.

    The training window is rolling (not expanding), advancing by
    `test_months` each iteration. Calendar-month boundaries are clipped to
    the nearest actual trading date in `dates`.
    """
    windows = []
    train_start = dates.min()
    while True:
        train_end = _clip_to_index(train_start + pd.DateOffset(months=train_months), dates, "end")
        test_start = _clip_to_index(train_end + pd.DateOffset(days=1), dates, "start")
        test_end = _clip_to_index(test_start + pd.DateOffset(months=test_months), dates, "end")
        if test_end <= test_start or test_end > dates.max():
            break
        windows.append((train_start, train_end, test_start, test_end))
        train_start = _clip_to_index(train_start + pd.DateOffset(months=test_months), dates, "start")
    return windows


def _window_sharpe(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_fn: SignalFn,
    lookback: int,
    book_notional: float,
    rebalance_freq: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cap: float,
    vol_lookback: int,
    n_sigma: float,
) -> tuple[float, pd.Series]:
    """Run one window and return (Sharpe, NAV) restricted to [start_date, end_date]."""
    result = run_book_backtest(
        prices.loc[:end_date],
        returns.loc[:end_date],
        signal_fn,
        lookback,
        book_notional,
        rebalance_freq,
        cap=cap,
        vol_lookback=vol_lookback,
        n_sigma=n_sigma,
        start_date=start_date,
    )
    return sharpe_ratio(result.nav.pct_change().dropna()), result.nav


def run_walk_forward(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_fn: SignalFn,
    candidate_lookbacks: list[int],
    book_notional: float,
    rebalance_freq: str,
    cap: float = POSITION_CAP,
    vol_lookback: int = VOL_LOOKBACK_DAYS,
    n_sigma: float = STOP_LOSS_N_SIGMA,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
) -> WalkForwardResult:
    """Roll training-window lookback selection forward across all available history."""
    windows = rolling_windows(returns.index, train_months, test_months)

    nav_segments = []
    report_rows = []
    current_notional = book_notional

    for train_start, train_end, test_start, test_end in windows:
        in_sample_scores = {
            lookback: _window_sharpe(
                prices, returns, signal_fn, lookback, book_notional, rebalance_freq,
                train_start, train_end, cap, vol_lookback, n_sigma,
            )[0]
            for lookback in candidate_lookbacks
        }
        winner = max(in_sample_scores, key=in_sample_scores.get)

        out_of_sample_sharpe, test_nav = _window_sharpe(
            prices, returns, signal_fn, winner, current_notional, rebalance_freq,
            test_start, test_end, cap, vol_lookback, n_sigma,
        )
        nav_segments.append(test_nav)
        current_notional = float(test_nav.iloc[-1])

        report_rows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "selected_lookback": winner,
                "in_sample_sharpe": in_sample_scores[winner],
                "out_of_sample_sharpe": out_of_sample_sharpe,
                **{f"in_sample_sharpe_{lb}": score for lb, score in in_sample_scores.items()},
            }
        )

    nav = pd.concat(nav_segments)
    nav = nav[~nav.index.duplicated(keep="first")]
    return WalkForwardResult(nav=nav, window_report=pd.DataFrame(report_rows))


def winner_change_rate(window_report: pd.DataFrame) -> float:
    """Fraction of consecutive windows where the selected lookback changed."""
    if len(window_report) < 2:
        return 0.0
    changed = window_report["selected_lookback"].diff().fillna(0) != 0
    return float(changed.iloc[1:].mean())
