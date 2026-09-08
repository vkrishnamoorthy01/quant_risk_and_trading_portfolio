"""One-off check: is the data run_backtest.py is about to use actually clean?

Run from the repo root:
    python checks/check_data_completeness.py

Fetches the full universe + benchmark over the same window
run_backtest.py uses, and reports index health (range, monotonic,
duplicates) plus per-ticker gap fractions. shared.data.fetch_price_data
already warns loudly on its own if a ticker comes back with an unusual
gap (see the throttle/retry fix in shared/data.py), so this script's
job is a fast, standalone look at the numbers *before* spending the time
on a full walk-forward run, not a replacement for that warning.
"""

from __future__ import annotations

from signal_project.portfolio import MOMENTUM_BOOK
from signal_project.run_backtest import BENCHMARK_TICKER, HISTORY_START
from shared.data import compute_returns, fetch_price_data

if __name__ == "__main__":
    end = "2026-09-08"
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
