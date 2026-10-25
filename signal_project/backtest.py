"""Day-by-day book simulation: signal -> vol-scaled sizing -> risk controls -> NAV.

Convention: weights decided using information available through the close
of day t are applied to day t+1's return (no look-ahead). Between rebalance
dates the book holds fixed shares, not fixed weights: each position's
weight drifts with its own return, and the uninvested remainder (cash)
earns nothing. This is the weight form of holdings/cash accounting, so a
weekly book is not silently rebalanced every day.

The per-position stop-loss is checked every day regardless of the signal's
rebalance frequency and forces a same-day exit. A name stopped on day t is
excluded from day t's rebalance (if t is a rebalance date) and becomes
eligible again only at the next scheduled rebalance, so a stop can't be
undone at the same close. For the daily book that is the next day; for the
weekly book, the next week's first trading day.

A missing return for a held position raises an error rather than being
treated as a zero return; missing returns for names not held don't affect
the book and are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
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
        day_return = returns.loc[t].reindex(tickers)
        held = weights[weights != 0].index
        unknown = [ticker for ticker in held if not isfinite(day_return[ticker])]
        if unknown:
            raise ValueError(
                f"No valid return on {t.date()} for held position(s) {unknown}; "
                "refusing to treat an unknown return as zero."
            )
        day_return = day_return.where(weights != 0, 0.0)  # names not held don't move the book
        portfolio_return = float((weights * day_return).sum())
        nav *= 1.0 + portfolio_return
        nav_history[t] = nav
        # Fixed shares between trades: each holding grows with its own
        # return while NAV grows with the portfolio return.
        if portfolio_return != -1.0:
            weights = weights * (1.0 + day_return) / (1.0 + portfolio_return)
        is_halted = halt.update(nav)

        current_price = prices.loc[t]
        stopped_today = []
        for ticker in list(entry_price):
            if weights[ticker] > 0 and stop_loss_triggered(
                entry_price[ticker], current_price[ticker], entry_vol[ticker], n_sigma
            ):
                weights[ticker] = 0.0
                del entry_price[ticker]
                del entry_vol[ticker]
                stopped_today.append(ticker)

        if t in rebalance_dates:
            eligible = signal.loc[t].copy()
            eligible[stopped_today] = False  # a stop holds until the next scheduled rebalance
            target = vol_scaled_weights(eligible, vol.loc[t], cap)
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
