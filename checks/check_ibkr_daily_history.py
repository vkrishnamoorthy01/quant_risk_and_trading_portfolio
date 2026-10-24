"""One-off check: three years of daily bars from IB for a few NSE stocks.

Run from the repo root (D:\\Personal_projects\\quant_risk_and_trading_portfolio),
with TWS running and logged in:
    python -m checks.check_ibkr_daily_history                    # paper
    python -m checks.check_ibkr_daily_history --years 1 --symbols TATASTEEL

Prints bar counts, date ranges, weekdays with no bar (mostly NSE holidays)
and any warnings, plus the closes either side of a known corporate action so
the adjustment convention can be checked by eye. Nothing is saved to disk.
Market data only: nothing here places or modifies orders.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date
from logging import INFO, WARNING, basicConfig, getLogger

from pandas import DateOffset, Timestamp

from data_providers.ibkr_connection import ibkr_connection, load_ibkr_config
from data_providers.ibkr_provider import IBKRDataError, get_historical_data

# RELIANCE's 1:1 bonus went ex on 2024-10-28: unadjusted closes would roughly halve that day.
CORPORATE_ACTION_CHECKS = {"RELIANCE": date(2024, 10, 28)}

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment", choices=["paper", "live"], default="paper")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE", "TATASTEEL", "HDFCBANK"])
    parser.add_argument("--years", type=int, default=3)
    args = parser.parse_args()

    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # ib_async logs startup positions at INFO with the full, unmasked account ID.
    getLogger("ib_async").setLevel(WARNING)

    end = Timestamp.today().normalize()
    start = (end - DateOffset(years=args.years)).date()
    end = end.date()

    with ibkr_connection(load_ibkr_config(args.environment)) as ib:
        for symbol in args.symbols:
            print(f"\n=== {symbol} ({start} to {end}) ===")
            try:
                result = get_historical_data(ib, symbol, start, end, environment=args.environment)
            except IBKRDataError as exc:
                print(f"FAILED: {type(exc).__name__}: {exc}")
                continue

            meta = result.metadata
            print(f"con_id={meta.con_id} bars={meta.n_bars} first={meta.first_date} last={meta.last_date} "
                  f"requests={meta.n_requests} duplicates_dropped={meta.duplicates_dropped}")
            print(f"empty_chunks={meta.empty_chunks}")
            print(f"placeholder bars dropped: {[ts.date().isoformat() for ts in meta.placeholder_bars_dropped]}")
            print(f"missing weekdays ({len(meta.missing_weekdays)}): {[d.isoformat() for d in meta.missing_weekdays]}")
            print(f"warnings: {meta.warnings or 'none'}")
            print(result.bars[["date", "open", "high", "low", "close", "volume"]].tail(3).to_string(index=False))

            if symbol in CORPORATE_ACTION_CHECKS:
                ex_date = Timestamp(CORPORATE_ACTION_CHECKS[symbol])
                window = result.bars[(result.bars["date"] >= ex_date - DateOffset(days=6))
                                     & (result.bars["date"] <= ex_date + DateOffset(days=4))]
                print(f"Closes around the {ex_date.date()} corporate action:")
                print(window[["date", "close", "volume"]].to_string(index=False))
