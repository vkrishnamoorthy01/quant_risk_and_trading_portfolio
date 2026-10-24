"""One-off check: can we connect to TWS read-only and pull a month of daily bars?

Run from the repo root (D:\\Personal_projects\\quant_risk_and_trading_portfolio),
with TWS running and logged in:
    python -m checks.check_ibkr_connection                       # paper, RELIANCE
    python -m checks.check_ibkr_connection --environment live --symbol TATASTEEL

Same request as the original check_ib_api_*_account_original.py scripts (1 M of
"1 day" TRADES bars, useRTH=True), but through data_providers.ibkr_connection,
so it also checks the environment and disconnects cleanly. Market data only:
nothing here places or modifies orders.
"""

from __future__ import annotations

from argparse import ArgumentParser
from logging import INFO, WARNING, basicConfig, getLogger

from ib_async import Stock

from data_providers.ibkr_connection import ibkr_connection, load_ibkr_config

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment", choices=["paper", "live"], default="paper")
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--port", type=int, default=None, help="override the default TWS port")
    parser.add_argument("--client-id", type=int, default=None)
    args = parser.parse_args()

    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # ib_async logs startup positions at INFO with the full, unmasked account ID.
    getLogger("ib_async").setLevel(WARNING)
    config = load_ibkr_config(args.environment, port=args.port, client_id=args.client_id)

    with ibkr_connection(config) as ib:
        print("Server time:", ib.reqCurrentTime())

        contracts = ib.qualifyContracts(Stock(args.symbol, "NSE", "INR"))
        if not contracts:
            raise RuntimeError(f"Could not qualify {args.symbol}'s NSE contract.")
        contract = contracts[0]
        print("Contract:", contract)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1 M",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
            timeout=30,
        )
        print(f"Received {len(bars)} daily bars")
        if bars:
            print(f"Range: {bars[0].date} to {bars[-1].date}")
            for bar in bars[-5:]:
                print(bar.date, "Close:", bar.close, "Volume:", bar.volume)
        else:
            print("No data returned. Check the IB error messages above.")
