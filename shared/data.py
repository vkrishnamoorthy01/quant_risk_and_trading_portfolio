"""Price data retrieval and return calculation utilities, backed by Kite Connect."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

_SESSION_CACHE_PATH = Path(__file__).resolve().parent.parent / ".kite_session.json"

_instruments_cache: dict[str, pd.DataFrame] = {}


def _load_cached_access_token(api_key: str) -> str | None:
    """Return today's cached access token for `api_key`, or None if absent/stale.

    Kite access tokens expire daily, so a token cached under a previous date
    is treated as missing and the login flow runs again.
    """
    if not _SESSION_CACHE_PATH.exists():
        return None
    cached = json.loads(_SESSION_CACHE_PATH.read_text())
    if cached.get("api_key") != api_key or cached.get("date") != date.today().isoformat():
        return None
    return cached.get("access_token")


def _cache_access_token(api_key: str, access_token: str) -> None:
    """Persist today's access token so later calls in the same day skip login."""
    _SESSION_CACHE_PATH.write_text(
        json.dumps({"api_key": api_key, "date": date.today().isoformat(), "access_token": access_token})
    )


def get_kite_client() -> KiteConnect:
    """Authenticate with Kite Connect and return a ready-to-use client.

    Reads KITE_API_KEY and KITE_API_SECRET from the environment (via a local
    .env file). Reuses a same-day cached access token if one exists; otherwise
    walks through Kite's three-legged login: it prints a login URL, and you
    paste back the `request_token` from the redirect URL after logging in.
    """
    api_key = os.environ["KITE_API_KEY"]
    api_secret = os.environ["KITE_API_SECRET"]

    kite = KiteConnect(api_key=api_key)

    access_token = _load_cached_access_token(api_key)
    if access_token is None:
        print(f"Log in, then paste the request_token from the redirect URL:\n{kite.login_url()}")
        request_token = input("request_token: ").strip()
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]
        _cache_access_token(api_key, access_token)

    kite.set_access_token(access_token)
    return kite


def _get_instrument_token(kite: KiteConnect, tradingsymbol: str, exchange: str = "NSE") -> int:
    """Look up the numeric instrument token Kite's historical API requires.

    The full instrument list for `exchange` is fetched once per process and
    cached in memory, since it is a large download shared across tickers.
    """
    if exchange not in _instruments_cache:
        _instruments_cache[exchange] = pd.DataFrame(kite.instruments(exchange))
    instruments = _instruments_cache[exchange]
    match = instruments[instruments["tradingsymbol"] == tradingsymbol]
    if match.empty:
        raise ValueError(f"No instrument token found for {tradingsymbol!r} on {exchange!r}")
    return int(match.iloc[0]["instrument_token"])


def fetch_price_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    interval: str = "day",
    exchange: str = "NSE",
) -> pd.DataFrame:
    """Download close prices for a list of tradingsymbols via Kite Connect.

    Args:
        tickers: Exchange trading symbols, e.g. "TCS", "HDFCBANK" (no suffix).
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        interval: Kite candle interval (default "day"); also accepts
            "minute", "3minute", "5minute", "10minute", "15minute",
            "30minute", "60minute".
        exchange: Exchange segment for instrument lookup (default "NSE").

    Returns:
        DataFrame of close prices, indexed by date, one column per ticker.
    """
    kite = get_kite_client()
    series = {}
    for ticker in tickers:
        token = _get_instrument_token(kite, ticker, exchange)
        candles = pd.DataFrame(kite.historical_data(token, start, end, interval))
        if candles.empty:
            continue
        candles["date"] = pd.to_datetime(candles["date"]).dt.tz_localize(None)
        series[ticker] = candles.set_index("date")["close"]

    prices = pd.DataFrame(series)
    return prices.dropna(how="all")


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute periodic returns from a price DataFrame.

    Args:
        prices: DataFrame of prices, one column per asset.
        method: "log" for log returns or "simple" for arithmetic returns.

    Returns:
        DataFrame of returns, one row shorter than the input.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'log' or 'simple'.")
    return returns.dropna(how="all")
