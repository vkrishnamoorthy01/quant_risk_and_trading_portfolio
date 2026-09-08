"""One-off check: is the data run_backtest.py is about to use actually clean?

Run from the repo root:
    python checks/check_data_completeness.py

Fetches the full universe + benchmark over the same window
run_backtest.py uses, and reports index health (range, monotonic,
duplicates), per-ticker gap fractions, and weekend dates. None of this
replaces the warnings shared.data.fetch_price_data already raises on its
own (rate-limit gaps, weekend dates) -- this is a fast, standalone look
at the numbers *before* spending the time on a full walk-forward run.

Weekend dates deserve a specific note: NSE occasionally holds real
Saturday/Sunday sessions (Union Budget day when it lands on a weekend,
Diwali "Muhurat" trading, SEBI-mandated disaster-recovery drills), so a
weekend date in the data is not automatically a bug -- see
KNOWN_SPECIAL_SESSIONS below, confirmed against real NSE history during
development. This script flags any weekend date NOT already in that list
as worth investigating; a low participation count (e.g. 2 of 17 tickers)
on an otherwise-known date is expected too -- thinly-traded stocks can
simply see zero trades during a short special session.
"""

from __future__ import annotations

import os

os.chdir(r"D:\Personal_projects\quant_risk_and_trading_portfolio")

import pandas as pd

from signal_project.portfolio import MOMENTUM_BOOK
from signal_project.run_backtest import BENCHMARK_TICKER, HISTORY_START
from shared.data import compute_returns, fetch_price_data

KNOWN_SPECIAL_SESSIONS = {
    "2015-02-28": "Union Budget day (pre-2017 Feb 28 date), Saturday",
    "2016-10-30": "Diwali Muhurat trading, Sunday",
    "2019-10-27": "Diwali Muhurat trading, Sunday",
    "2020-02-01": "Union Budget day, Saturday",
    "2020-11-14": "Diwali Muhurat trading, Saturday",
    "2023-11-12": "Diwali Muhurat trading, Sunday",
    "2024-01-20": "Make-up session for the Jan 22, 2024 Ram Mandir holiday, Saturday",
    "2024-03-02": "SEBI-mandated disaster-recovery site switchover drill, Saturday",
    "2024-05-18": "SEBI-mandated disaster-recovery site switchover drill, Saturday",
    "2025-02-01": "Union Budget day, Saturday",
    "2026-02-01": "Union Budget day, Sunday",
}

if __name__ == "__main__":
    end = pd.Timestamp.today().normalize().date().isoformat()
    tickers = list(MOMENTUM_BOOK.tickers) + [BENCHMARK_TICKER]

    print(f"Fetching {len(tickers)} tickers from {HISTORY_START} to {end} via Kite Connect...")
    prices = fetch_price_data(tickers, HISTORY_START, end)
    returns = compute_returns(prices, method="simple")

    print(f"\nRows: {len(returns)}, range: {returns.index.min()} to {returns.index.max()}")
    print(f"Monotonic: {returns.index.is_monotonic_increasing}, duplicates: {returns.index.has_duplicates}")

    print("\nPer-ticker gap fraction over the returned range:")
    gap_fraction = prices.isna().mean().sort_values(ascending=False)
    for ticker, fraction in gap_fraction.items():
        flag = "  <-- check this" if fraction > 0.05 else ""
        print(f"  {ticker}: {fraction:.2%}{flag}")

    print("\nWeekend dates (checked against KNOWN_SPECIAL_SESSIONS):")
    weekend_rows = prices[prices.index.dayofweek >= 5]
    participation = weekend_rows.notna().sum(axis=1)
    for day, count in participation.items():
        key = day.date().isoformat()
        label = KNOWN_SPECIAL_SESSIONS.get(key, "UNRECOGNIZED -- investigate this one")
        print(f"  {key}: {count}/{len(tickers)} tickers -- {label}")
