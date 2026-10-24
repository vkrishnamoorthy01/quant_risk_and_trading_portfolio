"""IBKR daily closes in the shape risk_project expects, as an alternative to shared.data.fetch_price_data.

The output matches fetch_price_data's: a DataFrame of close prices indexed
by session date (timezone-naive, named "date"), one column per NSE
tradingsymbol, in the order given. Kite remains the default source; nothing
here touches shared/data.py.

Differences from the Kite path, by design:
  - Prices come from the local IBKR store (data_providers.ibkr_storage), which
    fetch_ibkr_price_data fills from TWS first, downloading only what is missing.
  - Closes are IB TRADES closes: split- and bonus-adjusted, not
    dividend-adjusted, so returns are price returns, not total returns.
  - Missing data fails loudly. risk_project.var_models.portfolio_returns sums
    across tickers, which would treat a missing price as a zero return, so
    every ticker must be stored, cover the requested window, and have a close
    on every session the others traded (unless max_missing_fraction allows a
    little slack, in which case the gaps stay NaN and a warning is raised).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from logging import getLogger
from pathlib import Path
from typing import Any
from warnings import warn

from pandas import DataFrame, Timestamp

from data_providers.ibkr_connection import ibkr_connection, load_ibkr_config
from data_providers.ibkr_provider import IBKRDataError
from data_providers.ibkr_storage import DEFAULT_ROOT, fetch_and_store, load_bars

logger = getLogger(__name__)

# NSE tradingsymbol -> IB symbol, where they differ. IB caps NSE symbols at 9
# characters and drops "&". Each was checked against IB's contract search.
IB_SYMBOLS: dict[str, str] = {
    "HINDUNILVR": "HINDUNILV",
    "M&M": "MM",
    "ULTRACEMCO": "ULTRACEMC",
}


class IBKRInsufficientDataError(IBKRDataError):
    """Stored IBKR data does not cover every ticker over the requested window."""


def ib_symbol(ticker: str) -> str:
    """The IB symbol for an NSE tradingsymbol."""
    return IB_SYMBOLS.get(ticker, ticker)


def build_close_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
    environment: str = "paper",
    exchange: str = "NSE",
    min_observations: int = 250,
    max_missing_fraction: float = 0.0,
    max_edge_gap_days: int = 7,
    root: Path = DEFAULT_ROOT,
) -> DataFrame:
    """Close prices for `tickers` over [start, end] from the local IBKR store, with alignment checks.

    Args:
        tickers: NSE tradingsymbols, e.g. PORTFOLIO.tickers; also the output column names.
        start, end: "YYYY-MM-DD", inclusive.
        min_observations: Fewest aligned sessions accepted (250 is about one NSE year).
        max_missing_fraction: Share of sessions a ticker may lack while others traded.
            The default 0.0 means any such gap raises.
        max_edge_gap_days: How far (calendar days) a ticker's first or last close may sit
            inside the requested window before its history counts as too short or stale.

    Raises:
        IBKRInsufficientDataError: naming each ticker and what is wrong with it.
    """
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    series = {}
    problems = []
    for ticker in tickers:
        try:
            bars = load_bars(ib_symbol(ticker), environment, "1 day", exchange, start_date, end_date, root=root)
        except FileNotFoundError:
            problems.append(f"{ticker}: nothing stored")
            continue
        if bars.empty:
            problems.append(f"{ticker}: no bars between {start} and {end}")
            continue
        first, last = bars["date"].iloc[0].date(), bars["date"].iloc[-1].date()
        if (first - start_date).days > max_edge_gap_days:
            problems.append(f"{ticker}: history starts {first}, after {start}")
        if (end_date - last).days > max_edge_gap_days:
            problems.append(f"{ticker}: last close {last}, well before {end}")
        series[ticker] = bars.set_index("date")["close"]
    if problems:
        raise IBKRInsufficientDataError("IBKR data is insufficient: " + "; ".join(problems))

    prices = DataFrame(series)[list(tickers)].sort_index()
    prices.index.name = "date"
    prices = prices.dropna(how="all")

    gap_fraction = prices.isna().mean()
    too_gappy = gap_fraction[gap_fraction > max_missing_fraction]
    if not too_gappy.empty:
        details = [
            f"{ticker} missing {prices[ticker].isna().sum()} session(s), "
            f"e.g. {[ts.date().isoformat() for ts in prices.index[prices[ticker].isna()][:3]]}"
            for ticker in too_gappy.index
        ]
        raise IBKRInsufficientDataError("Tickers lack closes on sessions others traded: " + "; ".join(details))
    for ticker, fraction in gap_fraction[gap_fraction > 0].items():
        warn(f"{ticker} is missing {fraction:.1%} of sessions; left as NaN, not filled.", stacklevel=2)

    if len(prices) < min_observations:
        raise IBKRInsufficientDataError(
            f"Only {len(prices)} aligned sessions between {start} and {end}; need at least {min_observations}."
        )

    prices.attrs.update(source="IBKR", environment=environment, what_to_show="TRADES",
                        adjustment="splits and bonus issues; not dividends")
    logger.info("ibkr close prices tickers=%d sessions=%d first=%s last=%s",
                len(tickers), len(prices), prices.index[0].date(), prices.index[-1].date())
    return prices


def fetch_ibkr_price_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    environment: str = "paper",
    exchange: str = "NSE",
    currency: str = "INR",
    root: Path = DEFAULT_ROOT,
    **checks: Any,
) -> DataFrame:
    """IBKR counterpart to shared.data.fetch_price_data(tickers, start, end), daily closes only.

    Connects to TWS for `environment`, downloads into the local store whatever
    is not already there, then returns build_close_prices(...). Extra keyword
    arguments go to build_close_prices (min_observations, max_missing_fraction,
    max_edge_gap_days).
    """
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    with ibkr_connection(load_ibkr_config(environment)) as ib:
        for ticker in tickers:
            fetch_and_store(ib, ib_symbol(ticker), start_date, end_date, environment,
                            exchange=exchange, currency=currency, root=root)
    return build_close_prices(tickers, start, end, environment, exchange, root=root, **checks)
