"""Checks on a close-price panel before returns are computed, for any data source.

risk_project's portfolio_returns sums weighted returns with pandas' default
NaN skipping, and the original signal engine filled missing returns with
zero, so a missing ticker or a missing close used to flow through as a
plausible-looking zero return. These checks make that explicit:

  - every expected ticker must be present (Kite's fetch_price_data leaves out
    a ticker whose fetch came back empty);
  - closes must be finite and positive;
  - a ticker may not start late (NaN before its first close);
  - a short run of missing closes inside the history (a session where a thin
    stock printed no trade, e.g. a special weekend session) may be filled by
    carrying the last close forward, up to `max_fill_sessions` in a row, with
    a warning. No trade means no change in the close, and the next real close
    then books the whole move on the day it happened. Longer runs raise.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from warnings import warn

import pandas as pd


class PriceDataError(ValueError):
    """A close-price panel failed a completeness or validity check."""


def validate_close_prices(
    prices: pd.DataFrame,
    expected_tickers: Sequence[str],
    max_fill_sessions: int = 0,
) -> pd.DataFrame:
    """Validate a close-price panel; return it with any allowed gaps filled.

    Args:
        prices: Close prices, indexed by date, one column per ticker.
        expected_tickers: Tickers that must all be present, in output order.
        max_fill_sessions: Longest run of consecutive missing closes (after a
            ticker's first close) that may be carried forward. 0 means any
            missing close raises.

    Raises:
        PriceDataError naming the tickers (and dates) at fault.
    """
    expected = list(expected_tickers)
    absent = [ticker for ticker in expected if ticker not in prices.columns]
    if absent:
        raise PriceDataError(f"No prices at all for {absent}; the data source returned nothing for them.")
    if prices.empty:
        raise PriceDataError("The price panel is empty.")
    panel = prices[expected].sort_index()

    bad = [ticker for ticker in expected
           if any(not isfinite(v) or v <= 0 for v in panel[ticker].dropna())]
    if bad:
        raise PriceDataError(f"Non-finite or non-positive closes for {bad}.")

    problems = []
    filled = 0
    for ticker in expected:
        column = panel[ticker]
        if column.isna().all():
            problems.append(f"{ticker}: no closes")
            continue
        first_valid = column.first_valid_index()
        if column.loc[:first_valid].isna().any():
            problems.append(f"{ticker}: no close before {first_valid.date()}")
        after = column.loc[first_valid:]
        runs = after.isna().astype(int).groupby(after.notna().cumsum()).sum()
        longest = int(runs.max()) if len(runs) else 0
        if longest > max_fill_sessions:
            gap_dates = after.index[after.isna()]
            problems.append(f"{ticker}: {int(after.isna().sum())} missing close(s), e.g. "
                            f"{[d.date().isoformat() for d in gap_dates[:3]]}")
        filled += int(after.isna().sum())
    if problems:
        raise PriceDataError("Incomplete price data: " + "; ".join(problems))

    if filled:
        warn(f"Carried the last close forward over {filled} missing close(s) "
             f"(at most {max_fill_sessions} session(s) in a row).", stacklevel=2)
        panel = panel.ffill(limit=max_fill_sessions)
    return panel
