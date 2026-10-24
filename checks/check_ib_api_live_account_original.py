from ib_async import IB, Stock

ib = IB()

try:
    # Connect to live IB Trader Workstation
    ib.connect(
        "127.0.0.1",
        7496,           # Live TWS port
        clientId=18,
        timeout=10,
        readonly=True,
    )

    print("Connected:", ib.isConnected())
    print("Server time:", ib.reqCurrentTime())

    # Qualify TATASTEEL contract on NSE
    contracts = ib.qualifyContracts(
        Stock("TATASTEEL", "NSE", "INR")
    )

    if not contracts:
        raise RuntimeError("Could not qualify TATASTEEL contract")

    contract = contracts[0]
    print("Contract:", contract)

    # Request one month of historical daily prices
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

    print(f"\nReceived {len(bars)} daily bars")

    # Display last five trading sessions
    for bar in bars[-5:]:
        print(
            bar.date,
            "Close:", bar.close,
            "Volume:", bar.volume
        )

    if not bars:
        print("No data returned. Check IB error messages.")

finally:
    ib.disconnect()
    print("\nDisconnected from IB.")