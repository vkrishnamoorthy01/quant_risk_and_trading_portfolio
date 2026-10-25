# Interactive Brokers data provider (`data_providers/`)

This is the full documentation for the IBKR data provider; the README has a short summary.

## Problem statement

Add Interactive Brokers (IBKR) as a second, independent source of
historical NSE prices alongside Kite Connect, so `risk_project` and
`signal_project` can run on either. Kite stays the default, and its code
(`shared/data.py`) is unchanged. This is market data only: nothing here
places, modifies, or cancels orders.

## Setup

- Install the dependencies in your Python environment with
  `pip install -r requirements.txt`. The IBKR code needs `ib_async`
  (the TWS API client) and `pyarrow` (Parquet storage).
- Run Trader Workstation and log in to the account you want (paper or
  live). Under **Global Configuration → API → Settings**:
  - tick **Enable ActiveX and Socket Clients**;
  - set the **Socket port** to 7497 for paper or 7496 for live (the
    defaults this code expects);
  - keep **Read-Only API** ticked. That setting, not the code, is what
    stops orders at the broker.
- The account needs NSE market-data permissions for historical bars.
- Ports, client ID, and timeout can be changed with the optional
  `IBKR_*` variables in `.env` (see `.env.example`). No credentials go
  there: TWS handles the login.

## Methodology

**Connection.** `data_providers/ibkr_connection.py` connects to TWS
through `ib_async`, read-only, defaulting to paper (port 7497; live is
7496). Once connected, it checks the account IDs TWS reports against the
requested environment (IB paper accounts start with "D") and refuses to
go on if they don't match. There is no fallback between paper and live.
`readonly=True` only stops `ib_async` from requesting order state; it is
not enforced by the broker, so the real safeguard is TWS's Read-Only API
setting, which stays on. Ports, client ID, and timeout can be overridden
with optional `IBKR_*` variables (see `.env.example`). TWS handles the
login, so no credentials are stored.

**Historical bars.** `ibkr_provider.py` downloads daily
(`get_historical_data()`) and 5-minute (`get_intraday_data()`) bars.
Long ranges are split into chunks (365 days for daily, 7 for
5-minute) with a pause between requests, and a timeout or pacing
violation is retried once. `ib_async` returns an empty list on failure
rather than raising, so IB's error codes are captured separately to tell
"no data" apart from a permission problem, pacing, or a timeout. The
request conventions are fixed and recorded with every download:

- `whatToShow="TRADES"`, `useRTH=True`. TRADES history is adjusted for
  splits and bonus issues but not dividends. I checked this on RELIANCE's
  1:1 bonus (ex 2024-10-28) and HDFCBANK's (ex 2025-08-26), and both
  pass through smoothly. So these are price returns, not total returns.
- Daily bars keep the exchange session date with no time component.
  Intraday bars are timezone-aware Asia/Kolkata times, each the start
  of its bar.
- IB returns zero-volume, zero-trade placeholder bars on some holidays
  (2024-11-20, an election holiday, in all 16 stocks). These are dropped
  and logged, since leaving them in would add a fake zero-return day.
  Missing bars are reported, never filled in.

**Local storage.** `ibkr_storage.py` keeps bars in Parquet under
`data/cache/ibkr/{paper|live}/{daily|5min}/`, which is already
gitignored. Each file records the request settings, a log of every
download, and three separate views of completeness: the ranges a
download was **attempted** for, the **unresolved sessions** inside them
(expected sessions that came back without a bar), and the verified
**coverage** (attempted minus unresolved). "Expected" comes from an
explicit NSE calendar, `data_providers/nse_sessions.csv`: 167 weekday
closures, 11 special weekend sessions (Budget days, Muhurat trading,
disaster-recovery drills), and 14 sessions NSE held that IB has no data
for. So a missing ordinary Wednesday is retried on the next fetch, a
holiday isn't, and a Budget-day Saturday is requested.
`fetch_and_store()` downloads ranges never attempted plus unresolved
sessions.

Refreshes keep split adjustment consistent. IB back-adjusts its whole
TRADES history after a split or bonus, so appending new dates to an old
cache could splice two adjustment vintages (cached 100s followed by a
post-split 50 would invent a −50% day). Every refresh therefore
re-downloads up to five already-stored sessions on each side of a gap.
If they no longer match, the full stored range is downloaded again and
replaces the file in a single write, and the download log records
`revision_detected`. (`on_revision="raise"` stops instead and writes
nothing.) Every write is validated, read back, and compared before it
replaces the old file, and today's bars aren't stored before 16:00 IST.

**Feeding the risk engine.** `ibkr_prices.fetch_ibkr_price_data()`
returns the same close-price frame as `shared.data.fetch_price_data()`:
session dates as the index, one column per NSE tradingsymbol. Three
symbols are spelled differently on IB (HINDUNILVR is HINDUNILV, M&M is
MM, ULTRACEMCO is ULTRACEMC); the adapter maps them and keeps the NSE
names. It is stricter than the Kite path on gaps on purpose:
`portfolio_returns` sums across tickers, so a missing price would
silently count as a zero return. The adapter refuses the data if a
ticker is missing, starts late, ends early, or lacks a close on a day
the others traded, or if there are fewer than 250 aligned sessions.

```
python -m risk_project.run_report                       # Kite (default)
python -m risk_project.run_report --source ibkr_paper   # IBKR paper TWS
python -m risk_project.run_report --source ibkr_live    # IBKR live TWS
```

IBKR runs write `scenario_report_ibkr_paper.csv` /
`backtest_report_ibkr_paper.csv` (or `_ibkr_live`), so the Kite report
files are never overwritten. The VaR, factor, stress, and backtest code
is the same for all three sources.

`signal_project` works the same way:

```
python -m signal_project.run_backtest                       # Kite (default)
python -m signal_project.run_backtest --source ibkr_live    # or ibkr_paper
```

IBKR runs write the NAV and window-report CSVs with the same
`_ibkr_paper` / `_ibkr_live` suffix. The NIFTY 50 benchmark comes from
IB's `NIFTY50` index contract. Index bars have no volume or trade count,
so an index holiday placeholder is recognized as a bar that never moved
(open = high = low = close) instead. The signal, sizing, and
walk-forward code is the same for all three sources.

## Limitations and assumptions

- **Price returns only.** IB's TRADES closes aren't dividend-adjusted.
  IB's `ADJUSTED_LAST` series is, but it only works with an open end
  date, which rules out the chunked downloads used here.
- **IB's NSE volume is not exchange volume.** For RELIANCE in October
  2026 it was roughly 4-5% of the NSE figure, and the ratio varied by
  day. IB's daily volume equals the sum of its own intraday bars. Fine
  to ignore for close-based risk work; don't use it for liquidity or ADV.
- **Closing auction from 2026-08-03.** NSE started a Closing Auction
  Session for stocks with derivatives on that date: continuous trading
  ends at 15:15 and an auction sets the official close. IB's intraday
  bars cover continuous trading only, so the expected 5-minute session
  is 09:15-15:15 from 2026-08-03 and 09:15-15:30 before it. A year of
  live TATASTEEL bars shows the switch cleanly (complete through
  2026-07-31, ending at 15:15 every session since), and RELIANCE
  matches. Daily closes aren't affected: IB's daily close is the
  official close. The date rule assumes the stock is in the auction,
  which all 16 portfolio stocks are.
- **Paper and live data aren't quite the same.** Closes match exactly
  (16 stocks x 740 sessions), but there are 28 differences in
  open/high/low, and about 30% of daily volumes differ, by up to ~3%.
  The two are stored separately for that reason.
- **IB is missing a few real NSE sessions.** New Year's Day in 2015,
  2016 and 2018 to 2021 and the 2018-11-07 Diwali Muhurat session (NSE
  traded on all of them) come back from IB as zero-trade placeholders or
  no bar at all, so those days are absent and IB's history starts on
  2015-01-02. IB also has none of the seven short special weekend
  sessions from 2016-10-30 to 2024-05-18 (Muhurat trading, the
  2024-01-20 make-up session, two disaster-recovery drills). The four
  Budget-day weekend sessions are there. All 14 are recorded as
  `omitted:IBKR` in `nse_sessions.csv`, so they don't show up as
  unresolved gaps.
- **The session calendar is derived, and stops at 2026-10-23.** Its
  closures are the weekdays with no bar for any of the 17 cached
  instruments, minus the IB omissions above, spot-checked against known
  holidays but not reconciled line by line with NSE's circulars. After
  2026-10-23 every weekday counts as a session, so a new holiday shows up
  as an unresolved gap (and is retried) until the calendar is extended.
  The intraday 15:15 closing-auction rule is also a single date rule, not
  a per-instrument eligibility check.
- Only `1 day` and `5 mins` bars are supported.

## Testing

126 mocked unit tests (`tests/test_ibkr_connection.py`,
`test_ibkr_provider.py`, `test_ibkr_storage.py`, `test_ibkr_prices.py`,
`test_nse_calendar.py`, `test_run_report.py`, `test_run_backtest.py`)
cover connection config and the paper/live check, contract qualification
(stocks and indices), normalization, dates and sessions, duplicates,
gaps, placeholder bars, non-finite values, storage and conflicts,
attempted vs verified coverage, unresolved-session retries, special
weekend sessions, split revisions, empty responses, and error handling.
None of them need TWS running.

`tests/integration/` holds opt-in tests against a real paper TWS. They
are skipped unless you set `IBKR_PAPER_TESTS=1`:

```
$env:IBKR_PAPER_TESTS = "1"; python -m pytest tests/integration -v -s
```

There are deliberately no live-account tests, so nothing can run against
live automatically. Live checks are run by hand with the `checks/`
scripts, each with an `--environment live` option:

- `check_ibkr_connection.py`: connect, verify the environment, pull a
  month of daily bars.
- `check_ibkr_daily_history.py`: three years of daily bars for a few
  stocks, with gaps, placeholders, and a corporate-action spot check.
- `check_ibkr_intraday.py`: a small window of 5-minute bars, checked
  session by session against the daily bars.
- `check_ibkr_storage.py`: fill the local store, then confirm a repeat
  run downloads nothing.
- `check_ibkr_risk_inputs.py`: build the 16-stock price frame and run
  VaR on it, printing only.

Two more scripts sit alongside them. `check_ibkr_vs_kite_intraday.py`
compares IB and Kite 5-minute bars for the same sessions; it needs a
Kite login and paper TWS, and hasn't been run yet (see Future work).
`check_ib_api_paper_account_original.py` and
`check_ib_api_live_account_original.py` are the first manual scripts
used to confirm API access, kept as they were and renamed so pytest
doesn't collect them (they connect as soon as they're imported).

## Results

**risk_project.** Run on 2026-10-24/25 against both paper and live TWS
over the trailing 3-year window (2023-10-25 to 2026-10-23, 740
sessions). All 16 stocks
came back complete and aligned, with no gaps. The 45 weekdays without a
bar are NSE holidays. The largest single-day move in any stock is 16.8%
(ONGC on 2024-06-04, election results day), so there are no unadjusted
corporate actions in the data. RELIANCE's last five closes matched an
independent source exactly.

Because paper and live closes are identical, both IBKR reports give the
same numbers:

| Method     | 95% VaR   | 99% VaR   |
|------------|----------:|----------:|
| Historical | ₹133,093  | ₹204,174  |
| Parametric | ₹134,999  | ₹190,932  |

Factor model (2023-10-26 to 2025-12-31, 539 observations): MF 0.833,
SMB -0.332, HML -0.111, WML -0.056, R² 0.839. These are close to the
Kite results in `risk_project`'s Results section. The windows differ
(the Kite run ends 2026-08-30), so the comparison is a sanity check, not
a reconciliation. All four Kupiec tests pass, as they do on Kite data.

**signal_project.** `python -m signal_project.run_backtest --source
ibkr_live`, run on 2026-10-25 over 2015-01-02 to 2026-10-23 (2,912
aligned sessions for all 16 stocks and NIFTY 50, no gaps). The largest
single-day move in any stock over the whole period is SBIN's 27.7% on
2017-10-25, the day the PSU bank recapitalization was announced. Every
other stock stays under 19% (about half of those peaks are in March
2020), and TATASTEEL's 2022
1:10 split is adjusted, so there are no unadjusted corporate actions.

Corrected results, after the October 2026 audit fixes, rerun offline
from the cached inputs with `python -m signal_project.run_backtest
--source ibkr_live --cache-only --end 2026-10-23 --tag corrected`
(results in `*_ibkr_live_corrected.csv`). Out-of-sample 2017-01-05 to
2026-10-23, 20 contiguous 6-month test windows per book:

| Book                   | Total return | Sharpe | Lookback winner changed |
|------------------------|-------------:|-------:|------------------------:|
| Momentum (weekly)      | 259.18%      | 1.13   | 37% of windows          |
| Mean-reversion (daily) | 104.15%      | 0.61   | 42% of windows          |
| NIFTY 50               | 173.15%      |        |                         |

Same cached inputs, before and after the fixes:

| Book           | Before (audited engine) | After (corrected) | Same lookback chosen |
|----------------|------------------------:|------------------:|---------------------:|
| Momentum       | 218.09%, Sharpe 1.04    | 259.18%, Sharpe 1.13 | 15 of 20 windows  |
| Mean-reversion | 76.96%, Sharpe 0.50     | 104.15%, Sharpe 0.61 | 17 of 20 windows  |

Fix by fix, on the same inputs (a harness that reproduces both the
before and after figures exactly):

| Step                           | Momentum          | Mean-reversion    |
|--------------------------------|------------------:|------------------:|
| Audited engine                 | 218.09% / 1.04    | 76.96% / 0.50     |
| + non-overlapping windows      | 258.00% / 1.12    | 94.05% / 0.57     |
| + fixed-share drift            | 258.53% / 1.12    | 94.05% / 0.57     |
| + stop exclusion (corrected)   | 259.18% / 1.13    | 104.15% / 0.61    |

Read the increases with care. Most of momentum's gain comes from the
window schedule, but not mainly from removing double-counted days:
moving the boundaries by a few sessions changes the selected lookback in
three of the 20 windows (from 9/6/6 to 12/12/12), and those different
choices drive most of the difference. That sensitivity to where window
edges fall is itself a warning about how stable the selection is. Drift
changes the weekly momentum book only slightly and the daily book not at
all, as it should (a book rebalanced every day has no days in between
for weights to drift). The stop fix matters most for daily
mean-reversion, where the old engine would sell a stopped name and buy
it straight back at the same close.

The audited engine's report had 8 overlapping window boundaries
(e.g. a test window ending 2021-01-13 followed by one starting
2021-01-11); the corrected schedule has none, and each test window now
starts on the trading day after the previous one ends. Both books still
lose money in their last two windows (2026-01-05 to 2026-10-23).

The earlier Kite comparison (momentum 232.12% vs 226.08%,
mean-reversion 103.72% vs 79.75% to 2026-09-08) compared two runs of the
old engine, so it no longer says anything about the corrected strategy.
A like-for-like Kite comparison needs a fresh Kite run with the
corrected engine, which hasn't been done (Kite prices aren't cached).

## Future work

- Compare IB and Kite 5-minute bars session by session, including what
  NSE prints during the closing auction.
- A one-command daily refresh of the local store for the portfolio.
- Pull a dividend-adjusted series (`ADJUSTED_LAST`) and measure how much
  ignoring dividends moves VaR.
- Rerun the Kite backtest with the corrected engine, then compare it with
  the IBKR run on identical session dates, price set against price set.
