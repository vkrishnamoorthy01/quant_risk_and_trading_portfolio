# Interactive Brokers data provider (`data_providers/`)

A second, independent source of historical NSE prices next to Kite, so `risk_project` and `signal_project` can run on either. Kite stays the default, and `shared/data.py` is unchanged. Market data only: nothing here places, modifies or cancels orders.

## Setup

- `pip install -r requirements.txt`. The IBKR code needs `ib_async` and `pyarrow`.
- Run Trader Workstation and log in to the paper or live account. Under Global Configuration → API → Settings:
  - enable socket clients;
  - set port 7497 (paper) or 7496 (live);
  - keep **Read-Only API** ticked. That setting, not the code, is what blocks orders at the broker.
- The account needs NSE market-data permissions.
- Optional `IBKR_*` variables in `.env` change ports, client ID and timeout. No credentials go there; TWS handles the login.

```
python -m risk_project.run_report --source ibkr_live         # or ibkr_paper
python -m signal_project.run_backtest --source ibkr_live     # add --cache-only to skip TWS
```

IBKR runs write their result files with an `_ibkr_paper` or `_ibkr_live` suffix, so the Kite files are never overwritten.

## How it works

- **Connection (`ibkr_connection.py`):** connects read-only and checks that the account matches the requested environment (paper accounts start with "D"). There is no fallback between paper and live.
- **Bars (`ibkr_provider.py`):** daily and 5-minute TRADES bars, regular hours only. Long ranges are downloaded in chunks, with pacing and one retry. IB's error codes are kept, so "no data" can be told apart from a permission or pacing problem. Holiday placeholder bars (zero volume) are dropped; missing bars are reported, never filled.
- **Storage (`ibkr_storage.py`):** Parquet under `data/cache/ibkr/{paper|live}/` (gitignored). Each file records what was attempted, which expected sessions came back missing, and what is verified, against an NSE calendar (`nse_sessions.csv`). A refresh re-downloads a few stored sessions next to each gap. If IB has revised them after a split, the whole history is downloaded again, so old and new adjustments never mix.
- **Feeding the projects (`ibkr_prices.py`):** returns the same close-price frame as the Kite path and maps IB's symbols (HINDUNILV, MM, ULTRACEMC) to the NSE names. It refuses the data if a ticker is missing, starts late, ends early or lacks a close on a trading day. A missing price would otherwise count as a zero return. NIFTY 50 comes from IB's index contract.

## Key findings about the data

- **Adjusted for splits and bonuses, not dividends,** so these are price returns. RELIANCE's and HDFCBANK's 1:1 bonuses pass through smoothly.
- **Clean over 2015-01-02 to 2026-10-23:** all 16 stocks and NIFTY 50 are complete and aligned, with no unadjusted corporate actions. The largest one-day move is SBIN's 27.7% on 2017-10-25, the PSU bank recapitalization. RELIANCE's last five closes matched an independent source.
- **Paper and live closes are identical** (16 stocks × 740 sessions), so both give the same results. Open/high/low differ in 28 places, and about 30% of volumes differ by up to 3%, so the two are stored separately.
- **IB's volume is not exchange volume:** about 4–5% of NSE's for RELIANCE. Don't use it for liquidity.
- **Closing auction:** since 2026-08-03, IB's intraday bars end at 15:15. Daily closes are the official closes and are unaffected.
- **IB is missing 14 real NSE sessions:** some New Year's Days and seven short special weekend sessions. They're recorded in the calendar, so they don't count as gaps.

## Limitations

- **Price returns only.** IB's dividend-adjusted series doesn't work with chunked downloads.
- **The session calendar is derived from the data** and stops at 2026-10-23. Later holidays show up as gaps until it's extended.
- **The 15:15 rule is a single date,** not a per-stock check. All 16 portfolio stocks are in the auction.
- **Only daily and 5-minute bars** are supported.

## Testing

126 mocked unit tests cover the connection, contracts, normalization, sessions, gaps, storage, split revisions and errors. None of them needs TWS. The integration tests against paper TWS are opt-in (`IBKR_PAPER_TESTS=1`). There are deliberately no live-account tests; live checks are run by hand with the `checks/check_ibkr_*.py` scripts.

## Future work

- Compare IB and Kite 5-minute bars session by session, including the closing auction.
- A one-command daily refresh of the cache.
- Measure how much ignoring dividends moves VaR.
- Rerun the Kite backtest on the fixed engine and compare it with IBKR on identical dates.
