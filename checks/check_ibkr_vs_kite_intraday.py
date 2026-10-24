"""One-off check: compare IB and Kite 5-minute bars for the same NSE sessions.

Run from the repo root (D:\\Personal_projects\\quant_risk_and_trading_portfolio),
with paper TWS running:
    python -m checks.check_ibkr_vs_kite_intraday
    python -m checks.check_ibkr_vs_kite_intraday --symbol RELIANCE --sessions 2026-10-23

IB is fetched first. The Kite part then needs the usual interactive Kite login
(prints a login URL, prompts for a pasted request_token) unless a same-day
.kite_session.json cache exists. Kite is used read-only through shared.data;
nothing there is modified. Nothing is saved to disk.

For each session it prints both providers' first and last bar times, what
Kite shows for 15:15-15:30 (the period missing from IB), price differences on
the bars both providers have, and the IB/Kite volume ratio.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date, datetime, time, timedelta
from logging import INFO, WARNING, basicConfig, getLogger

from pandas import DataFrame, Timestamp, to_datetime

from data_providers.ibkr_connection import ibkr_connection, load_ibkr_config
from data_providers.ibkr_provider import EXCHANGE_TIMEZONE, get_intraday_data
from shared.data import _get_instrument_token, get_kite_client

PRICE_COLUMNS = ["open", "high", "low", "close"]


def _kite_session(kite: object, token: int, session: date) -> DataFrame:
    """Kite 5-minute candles for one session, with a tz-aware IST "timestamp" column."""
    candles = DataFrame(kite.historical_data(
        token, datetime.combine(session, time(9, 0)), datetime.combine(session, time(16, 0)), "5minute"
    ))
    if candles.empty:
        return candles
    candles["timestamp"] = to_datetime(candles["date"]).dt.tz_convert(EXCHANGE_TIMEZONE)
    return candles.drop(columns="date")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="TATASTEEL")
    parser.add_argument("--sessions", nargs="+", default=["2026-10-23", "2025-10-23"])
    args = parser.parse_args()
    sessions = [date.fromisoformat(s) for s in args.sessions]

    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # ib_async logs startup positions at INFO with the full, unmasked account ID.
    getLogger("ib_async").setLevel(WARNING)

    ib_results = {}
    with ibkr_connection(load_ibkr_config("paper")) as ib:
        for session in sessions:
            ib_results[session] = get_intraday_data(ib, args.symbol, session, session, environment="paper")
            ib.sleep(2)

    print("\nIB done. Logging in to Kite next.")
    kite = get_kite_client()
    token = _get_instrument_token(kite, args.symbol)

    for session in sessions:
        ib_result = ib_results[session]
        ib_bars = ib_result.bars.set_index("timestamp")
        kite_bars = _kite_session(kite, token, session)
        print(f"\n=== {args.symbol} {session} ===")
        print(f"IB:   {len(ib_bars)} bars, {ib_bars.index.min():%H:%M} to {ib_bars.index.max():%H:%M}; "
              f"placeholders dropped: {[f'{ts:%H:%M}' for ts in ib_result.metadata.placeholder_bars_dropped]}")
        if kite_bars.empty:
            print("Kite: no candles returned")
            continue
        kite_bars = kite_bars.set_index("timestamp")
        print(f"Kite: {len(kite_bars)} bars, {kite_bars.index.min():%H:%M} to {kite_bars.index.max():%H:%M}")

        late = kite_bars[kite_bars.index.time >= time(15, 15)]
        late_share = late["volume"].sum() / kite_bars["volume"].sum()
        print(f"Kite 15:15-15:30 bars (missing from IB), {late_share:.1%} of Kite's session volume:")
        print(late[[*PRICE_COLUMNS, "volume"]].to_string())

        common = ib_bars.index.intersection(kite_bars.index)
        diff = (ib_bars.loc[common, PRICE_COLUMNS] - kite_bars.loc[common, PRICE_COLUMNS]).abs()
        print(f"Bars in both: {len(common)}. Max abs price difference: "
              + ", ".join(f"{col} {diff[col].max():.2f}" for col in PRICE_COLUMNS)
              + f"; bars with identical close: {(diff['close'] == 0).sum()} of {len(common)}")
        ratio = ib_bars.loc[common, "volume"] / kite_bars.loc[common, "volume"]
        print(f"IB/Kite volume per bar: median {ratio.median():.1%}, min {ratio.min():.1%}, max {ratio.max():.1%}; "
              f"session total {ib_bars['volume'].sum() / kite_bars['volume'].sum():.1%}")
        only_kite = kite_bars.index.difference(ib_bars.index)
        print(f"Bar times in Kite but not IB: {[f'{ts:%H:%M}' for ts in only_kite]}")
