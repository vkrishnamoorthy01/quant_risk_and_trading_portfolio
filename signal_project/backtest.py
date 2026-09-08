"""Day-by-day book simulation: signal -> vol-scaled sizing -> risk controls -> NAV.

Convention: weights decided using information available through the close
of day t are applied to day t+1's return (no look-ahead). Positions persist
unchanged between rebalance dates except for a same-day forced exit from
the per-position stop-loss, which is checked every day regardless of the
signal's rebalance frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from signal_project.risk_controls import STOP_LOSS_N_SIGMA, DrawdownHalt, stop_loss_triggered
from signal_project.sizing import POSITION_CAP, VOL_LOOKBACK_DAYS, realized_vol, vol_scaled_weights

SignalFn = Callable[[pd.DataFrame, int], pd.DataFrame]


@dataclass
class BacktestResult:
    """Output of one book backtest."""

    nav: pd.Series
    weights_history: pd.DataFrame  # weights decided at each date, applied to the next day's return


def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    """First trading day of each rebalance period ("D" daily or "W" weekly)."""
    if freq == "D":
        return set(dates)
    if freq == "W":
        return set(pd.Series(dates, index=dates).groupby(dates.to_period("W")).first())
    raise ValueError(f"Unknown rebalance_freq: {freq!r}. Use 'D' or 'W'.")


def _apply_halt(current_weights: pd.Series, target_weights: pd.Series, is_halted: bool) -> pd.Series:
    """Block newly-opened positions while halted; leave exits/resizes untouched."""
    if not is_halted:
        return target_weights
    adjusted = target_weights.copy()
    newly_opened = (current_weights == 0) & (target_weights > 0)
    adjusted[newly_opened] = 0.0
    return adjusted


def run_book_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_fn: SignalFn,
    lookback: int,
    book_notional: float,
    rebalance_freq: str,
    cap: float = POSITION_CAP,
    vol_lookback: int = VOL_LOOKBACK_DAYS,
    n_sigma: float = STOP_LOSS_N_SIGMA,
    start_date: pd.Timestamp | None = None,
) -> BacktestResult:
    """Simulate one signal book over the date range of `prices`/`returns`.

    Signal and vol are computed over the *full* `prices`/`returns` history
    passed in, then the simulation itself only runs (and NAV starts fresh,
    flat) from `start_date` onward — so a caller can hand in extra history
    before `start_date` purely to warm up long lookback/vol windows without
    it affecting the simulated NAV path. This is what the walk-forward
    module uses to evaluate a window without truncating its lookback.

    Args:
        prices: Daily close prices, one column per ticker.
        returns: Daily simple returns aligned to `prices` (one row shorter).
        signal_fn: `momentum_signal` or `mean_reversion_signal`.
        lookback: Lookback parameter passed through to `signal_fn`.
        book_notional: Starting NAV for this book.
        rebalance_freq: "D" (mean-reversion) or "W" (momentum).
        cap: Per-stock position cap, as a fraction of book NAV.
        vol_lookback: Trailing window for the realized-vol estimate used by
            both sizing and the stop-loss.
        n_sigma: Stop-loss threshold, in multiples of vol at entry.
        start_date: First date to simulate (book starts flat here). Defaults
            to the earliest date in `returns`.

    Returns:
        BacktestResult with the NAV series and the weights decided at each
        date (applied to the following day's return), covering
        [start_date, end of `returns`].
    """
    tickers = prices.columns
    full_dates = returns.index
    signal = signal_fn(prices, lookback).reindex(full_dates)
    vol = realized_vol(returns, vol_lookback)
    dates = full_dates[full_dates >= start_date] if start_date is not None else full_dates
    rebalance_dates = _rebalance_dates(dates, rebalance_freq)

    weights = pd.Series(0.0, index=tickers)
    entry_price: dict[str, float] = {}
    entry_vol: dict[str, float] = {}
    halt = DrawdownHalt()
    nav = book_notional

    nav_history = {}
    weights_rows = {}

    for t in dates:
        day_return = returns.loc[t].reindex(tickers).fillna(0.0)
        nav *= 1.0 + float((weights * day_return).sum())
        nav_history[t] = nav
        is_halted = halt.update(nav)

        current_price = prices.loc[t]
        for ticker in list(entry_price):
            if weights[ticker] > 0 and stop_loss_triggered(
                entry_price[ticker], current_price[ticker], entry_vol[ticker], n_sigma
            ):
                weights[ticker] = 0.0
                del entry_price[ticker]
                del entry_vol[ticker]

        if t in rebalance_dates:
            target = vol_scaled_weights(signal.loc[t], vol.loc[t], cap)
            target = _apply_halt(weights, target, is_halted)

            newly_opened = (weights == 0) & (target > 0)
            for ticker in newly_opened[newly_opened].index:
                entry_price[ticker] = current_price[ticker]
                entry_vol[ticker] = vol.loc[t, ticker]
            for ticker in weights.index:
                if target[ticker] == 0 and ticker in entry_price:
                    del entry_price[ticker]
                    del entry_vol[ticker]

            weights = target

        weights_rows[t] = weights.copy()

    return BacktestResult(
        nav=pd.Series(nav_history, name="nav"),
        weights_history=pd.DataFrame(weights_rows).T,
    )
