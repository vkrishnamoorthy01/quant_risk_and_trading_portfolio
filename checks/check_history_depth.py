"""One-off check: how much clean daily history does production Kite give us?

Run from the repo root (D:\\Personal_projects\\quant_risk_and_trading_portfolio):
    python "<path to this file>"

Requires the same interactive Kite login as any other production call
(prints a login URL, prompts for a pasted request_token) unless a
same-day .kite_session.json cache already exists.

Kite's "day" interval historical_data call rejects a single request
spanning more than ~2000 days, so this chunks the requested range into
1900-day windows and concatenates them.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

import os
os.chdir(r"D:\Personal_projects\quant_risk_and_trading_portfolio")


from shared.data import fetch_price_data

TICKERS = ["TCS", "HDFCBANK"]
START = date(2015, 1, 1)
END = date(2026, 9, 8)
CHUNK_DAYS = 1900

if __name__ == "__main__":
    chunks: list[pd.DataFrame] = []
    chunk_start = START
    while chunk_start <= END:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), END)
        print(f"Fetching {chunk_start} to {chunk_end} ...")
        chunks.append(fetch_price_data(TICKERS, chunk_start.isoformat(), chunk_end.isoformat()))
        chunk_start = chunk_end + timedelta(days=1)

    prices = pd.concat(chunks).sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    print()
    print(f"Requested range: {START} to {END}")
    print(f"Columns returned: {list(prices.columns)}")
    print(f"Rows (trading days): {len(prices)}")
    print(f"Actual date range: {prices.index.min()} to {prices.index.max()}")
    print()
    print("Per-ticker first/last non-null observation:")
    for ticker in TICKERS:
        series = prices[ticker].dropna()
        print(f"  {ticker}: {series.index.min()} to {series.index.max()} ({len(series)} obs)")
    print()
    gaps = prices.index.to_series().diff().dt.days
    print(f"Largest gap between consecutive trading days: {gaps.max()} calendar days")
