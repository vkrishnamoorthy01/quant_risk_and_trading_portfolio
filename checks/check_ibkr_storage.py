"""One-off check: download IB bars into the local Parquet store, then confirm a repeat run downloads nothing.

Run from the repo root (D:\\Personal_projects\\quant_risk_and_trading_portfolio),
with TWS running and logged in:
    python -m checks.check_ibkr_storage                          # paper
    python -m checks.check_ibkr_storage --years 1 --symbols TATASTEEL

Stores daily bars for each symbol and 5-minute bars for the first symbol under
data/cache/ibkr/ (gitignored), runs the same requests a second time, and
prints what is on disk. Market data only: nothing here places or modifies orders.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import timedelta
from logging import INFO, WARNING, basicConfig, getLogger
from time import perf_counter

from pandas import DateOffset, Timestamp

from data_providers.ibkr_connection import ibkr_connection, load_ibkr_config
from data_providers.ibkr_storage import fetch_and_store, load_bars

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment", choices=["paper", "live"], default="paper")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE", "TATASTEEL", "HDFCBANK"])
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--intraday-days", type=int, default=7)
    args = parser.parse_args()

    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # ib_async logs startup positions at INFO with the full, unmasked account ID.
    getLogger("ib_async").setLevel(WARNING)

    today = Timestamp.today().normalize()
    daily_start = (today - DateOffset(years=args.years)).date()
    intraday_start = (today - timedelta(days=args.intraday_days)).date()
    requests = [(symbol, "1 day", daily_start) for symbol in args.symbols]
    requests.append((args.symbols[0] if args.symbols != ["RELIANCE", "TATASTEEL", "HDFCBANK"] else "TATASTEEL",
                     "5 mins", intraday_start))

    with ibkr_connection(load_ibkr_config(args.environment)) as ib:
        for run in ("first", "repeat"):
            print(f"\n--- {run} run ---")
            for symbol, bar_size, start in requests:
                began = perf_counter()
                stored = fetch_and_store(ib, symbol, start, today.date(), args.environment, bar_size=bar_size)
                print(f"{symbol} {bar_size}: {len(stored.bars)} rows, coverage {stored.info['coverage']}, "
                      f"downloads logged {len(stored.info['downloads'])}, {perf_counter() - began:.1f}s")

    print("\n--- reloaded from disk (no IB connection) ---")
    for symbol, bar_size, _ in requests:
        bars = load_bars(symbol, args.environment, bar_size=bar_size)
        time_column = "date" if bar_size == "1 day" else "timestamp"
        print(f"{symbol} {bar_size}: {len(bars)} rows, {bars[time_column].min()} to {bars[time_column].max()}, "
              f"dtype {bars[time_column].dtype}, duplicates {bars[time_column].duplicated().sum()}")
