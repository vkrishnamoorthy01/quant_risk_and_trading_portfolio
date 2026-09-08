"""End-to-end signal_project backtest: fetch prices, walk-forward each book,
and print a summary to fill in the README's Results section.

Run this yourself from the repo root:

    python -m signal_project.run_backtest

The first Kite Connect call each day needs an interactive login: it prints
a login URL, and after you log in and get redirected, you paste back the
`request_token` from the redirect URL when prompted. The access token is
then cached for the rest of the day (see shared/data.py). This script only
uses the PRODUCTION Kite client for historical data — no orders are placed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from signal_project.portfolio import MEAN_REVERSION_BOOK, MOMENTUM_BOOK
from signal_project.signals import mean_reversion_signal, momentum_signal
from signal_project.walk_forward import run_walk_forward, sharpe_ratio, winner_change_rate
from shared.data import compute_returns, fetch_price_data

RESULTS_DIR = Path(__file__).resolve().parent / "results"
HISTORY_START = "2015-01-01"
MOMENTUM_LOOKBACK_GRID = [3, 6, 9, 12]  # months
MEAN_REVERSION_LOOKBACK_GRID = [3, 5, 10]  # trading days
BENCHMARK_TICKER = "NIFTY 50"


def _report_book(name: str, notional: float, result, rebalance_freq: str) -> None:
    total_return = result.nav.iloc[-1] / notional - 1.0
    oos_returns = result.nav.pct_change().dropna()
    print(f"\n=== {name} book (rebalanced {rebalance_freq}) ===")
    print(f"  Out-of-sample window: {result.nav.index.min().date()} to {result.nav.index.max().date()}")
    print(f"  Starting notional: Rs {notional:,.0f}")
    print(f"  Ending NAV: Rs {result.nav.iloc[-1]:,.0f} (total return {total_return:.2%})")
    print(f"  Annualized Sharpe (stitched OOS): {sharpe_ratio(oos_returns):.2f}")
    print(f"  Windows run: {len(result.window_report)}")
    print(f"  Lookback winner changed between {winner_change_rate(result.window_report):.0%} of consecutive windows")
    print(result.window_report[["train_start", "train_end", "test_start", "test_end", "selected_lookback", "out_of_sample_sharpe"]].to_string(index=False))


def main() -> None:
    end = pd.Timestamp.today().normalize().date().isoformat()
    tickers = MOMENTUM_BOOK.tickers

    print(f"Fetching {len(tickers)} tickers from {HISTORY_START} to {end} via Kite Connect...")
    prices = fetch_price_data(tickers, HISTORY_START, end)
    returns = compute_returns(prices, method="simple")

    print(f"Fetching {BENCHMARK_TICKER} for the same window (market context)...")
    benchmark_prices = fetch_price_data([BENCHMARK_TICKER], HISTORY_START, end)

    print("\nRunning walk-forward for the momentum book...")
    momentum_result = run_walk_forward(
        prices, returns, momentum_signal, MOMENTUM_LOOKBACK_GRID,
        MOMENTUM_BOOK.notional, rebalance_freq="W",
    )

    print("Running walk-forward for the mean-reversion book...")
    mean_reversion_result = run_walk_forward(
        prices, returns, mean_reversion_signal, MEAN_REVERSION_LOOKBACK_GRID,
        MEAN_REVERSION_BOOK.notional, rebalance_freq="D",
    )

    _report_book("Momentum", MOMENTUM_BOOK.notional, momentum_result, "weekly")
    _report_book("Mean-reversion", MEAN_REVERSION_BOOK.notional, mean_reversion_result, "daily")

    oos_start = min(momentum_result.nav.index.min(), mean_reversion_result.nav.index.min())
    oos_end = max(momentum_result.nav.index.max(), mean_reversion_result.nav.index.max())
    benchmark_window = benchmark_prices.loc[oos_start:oos_end, BENCHMARK_TICKER]
    benchmark_return = benchmark_window.iloc[-1] / benchmark_window.iloc[0] - 1.0
    print(f"\n=== Market context ===")
    print(f"  {BENCHMARK_TICKER} return over the same {oos_start.date()} to {oos_end.date()} window: {benchmark_return:.2%}")
    print("  Both books are long-only and execution-constrained (see README) -- they inherit market")
    print("  beta, so this benchmark return should be read alongside each book's result, not ignored.")

    RESULTS_DIR.mkdir(exist_ok=True)
    momentum_result.nav.to_csv(RESULTS_DIR / "momentum_nav.csv", header=["nav"])
    momentum_result.window_report.to_csv(RESULTS_DIR / "momentum_window_report.csv", index=False)
    mean_reversion_result.nav.to_csv(RESULTS_DIR / "mean_reversion_nav.csv", header=["nav"])
    mean_reversion_result.window_report.to_csv(RESULTS_DIR / "mean_reversion_window_report.csv", index=False)
    print(f"\nSaved NAV and window-report CSVs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
