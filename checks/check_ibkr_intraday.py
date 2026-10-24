"""One-off check: a small window of 5-minute bars from IB for one NSE stock.

Run from the repo root (D:\\Personal_projects\\quant_risk_and_trading_portfolio),
with TWS running and logged in:
    python -m checks.check_ibkr_intraday                         # paper, TATASTEEL, last 7 days
    python -m checks.check_ibkr_intraday --days 30 --symbol RELIANCE

Prints per-session bar counts and first/last bar times (IST), incomplete
sessions, bars outside the 09:15-15:30 grid, and compares each session's last
5-minute close with IB's daily close for the same session. Nothing is saved.
Market data only: nothing here places or modifies orders.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import timedelta
from logging import INFO, WARNING, basicConfig, getLogger

from pandas import DataFrame, Timestamp

from data_providers.ibkr_connection import ibkr_connection, load_ibkr_config
from data_providers.ibkr_provider import IBKRDataError, get_historical_data, get_intraday_data

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment", choices=["paper", "live"], default="paper")
    parser.add_argument("--symbol", default="TATASTEEL")
    parser.add_argument("--days", type=int, default=7, help="calendar days back from today")
    args = parser.parse_args()

    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # ib_async logs startup positions at INFO with the full, unmasked account ID.
    getLogger("ib_async").setLevel(WARNING)

    end = Timestamp.today().date()
    start = end - timedelta(days=args.days)

    with ibkr_connection(load_ibkr_config(args.environment)) as ib:
        try:
            intraday = get_intraday_data(ib, args.symbol, start, end, environment=args.environment)
            daily = get_historical_data(ib, args.symbol, start, end, environment=args.environment)
        except IBKRDataError as exc:
            raise SystemExit(f"FAILED: {type(exc).__name__}: {exc}")

    meta = intraday.metadata
    bars = intraday.bars
    print(f"\n=== {args.symbol} 5-minute bars ({start} to {end}) ===")
    print(f"con_id={meta.con_id} bars={meta.n_bars} requests={meta.n_requests} "
          f"duplicates_dropped={meta.duplicates_dropped} bars_outside_session={meta.bars_outside_session}")
    print(f"timestamp dtype: {bars['timestamp'].dtype}")
    print(f"placeholder bars dropped: {[ts.isoformat() for ts in meta.placeholder_bars_dropped]}")
    print(f"missing weekdays: {[d.isoformat() for d in meta.missing_weekdays]}")
    print(f"incomplete sessions: {meta.incomplete_sessions or 'none'}")
    print(f"warnings: {meta.warnings or 'none'}")

    by_session = bars.groupby(bars["timestamp"].dt.date)
    summary = DataFrame({
        "bars": by_session.size(),
        "first_bar": by_session["timestamp"].min().dt.strftime("%H:%M"),
        "last_bar": by_session["timestamp"].max().dt.strftime("%H:%M"),
        "last_5min_close": by_session["close"].last(),
        "volume_5min_sum": by_session["volume"].sum(),
    })
    daily_bars = daily.bars.set_index(daily.bars["date"].dt.date)
    summary["daily_close"] = daily_bars["close"]
    summary["daily_volume"] = daily_bars["volume"]
    print("\nPer session:")
    print(summary.to_string())
    print("\nFirst and last bars of the latest session:")
    latest = bars[bars["timestamp"].dt.date == bars["timestamp"].dt.date.max()]
    print(latest[["timestamp", "open", "high", "low", "close", "volume", "bar_count"]].iloc[[0, 1, -2, -1]].to_string(index=False))
