from ib_async import IB, Stock

ib = IB()

try:
    ib.connect(
    "127.0.0.1",
    7497,
    clientId=17,
    timeout=10,
    readonly=True,
)
    print("Connected:", ib.isConnected())
    print("Server time:", ib.reqCurrentTime())

    contracts = ib.qualifyContracts(
        Stock("RELIANCE", "NSE", "INR")
    )
    if not contracts:
        raise RuntimeError("Could not resolve RELIANCE's NSE contract.")

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
    for bar in bars[-5:]:
        print(bar.date, "Close:", bar.close, "Volume:", bar.volume)

    if not bars:
        print("No data returned. Check the IB error messages above.")

finally:
    ib.disconnect()