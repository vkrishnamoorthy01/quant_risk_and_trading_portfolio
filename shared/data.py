"""Price data retrieval and return calculation utilities, backed by Kite Connect."""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import date, timedelta
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


_MAX_DAY_INTERVAL_SPAN = 1900  # Kite's "day" interval historical_data call rejects spans over ~2000 days


def _chunk_date_range(start: str, end: str, max_days: int = _MAX_DAY_INTERVAL_SPAN) -> list[tuple[str, str]]:
    """Split [start, end] into <=max_days-long (start, end) chunks, both "YYYY-MM-DD"."""
    chunk_start = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    chunks = []
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=max_days), end_date)
        chunks.append((chunk_start.isoformat(), chunk_end.isoformat()))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


_REQUEST_THROTTLE_SECONDS = 0.5  # baseline delay between successive historical_data calls
_MAX_FETCH_ATTEMPTS = 4
_MAX_EXPECTED_GAP_FRACTION = 0.05  # warn if a ticker is missing more of the requested span than this


def _fetch_chunk(kite: KiteConnect, token: int, chunk_start: str, chunk_end: str, interval: str) -> pd.DataFrame:
    """One historical_data call, throttled and retried with growing backoff if it comes back empty.

    An empty response for a chunk that's well within a stock's trading
    history is far more likely a rate-limit hiccup than genuinely missing
    data — fetching 16 tickers x several date chunks fires enough
    back-to-back requests to hit Kite's per-second limit, and a throttled
    call here returns an empty list rather than raising, so it has to be
    retried explicitly or it silently turns into missing history. A single
    quick retry wasn't enough in practice (still saw a consistent ~20% gap
    across most tickers), so this backs off further on each attempt rather
    than retrying once at a fixed delay.
    """
    for attempt in range(_MAX_FETCH_ATTEMPTS):
        candles = pd.DataFrame(kite.historical_data(token, chunk_start, chunk_end, interval))
        time.sleep(_REQUEST_THROTTLE_SECONDS * (attempt + 1))
        if not candles.empty:
            return candles
    return candles


def fetch_price_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    interval: str = "day",
    exchange: str = "NSE",
) -> pd.DataFrame:
    """Download close prices for a list of tradingsymbols via Kite Connect.

    A "day"-interval request spanning more than Kite's per-call limit
    (~2000 days) is transparently split into consecutive chunks and
    concatenated; other intervals are not chunked (out of scope here,
    since nothing in this repo pulls multi-year intraday history). Calls
    are throttled and each chunk retried with backoff on an empty response (see
    `_fetch_chunk`), and a UserWarning is raised for any ticker still
    missing an unusually large fraction of the requested trading days
    afterward, so a rate-limit-induced gap fails loudly instead of
    silently propagating into downstream results.

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
    date_chunks = _chunk_date_range(start, end) if interval == "day" else [(start, end)]

    series = {}
    for ticker in tickers:
        token = _get_instrument_token(kite, ticker, exchange)
        chunk_frames = [
            _fetch_chunk(kite, token, chunk_start, chunk_end, interval) for chunk_start, chunk_end in date_chunks
        ]
        chunk_frames = [frame for frame in chunk_frames if not frame.empty]
        if not chunk_frames:
            continue
        candles = pd.concat(chunk_frames)
        candles["date"] = pd.to_datetime(candles["date"]).dt.tz_localize(None)
        series[ticker] = candles.drop_duplicates(subset="date").set_index("date")["close"].sort_index()

    prices = pd.DataFrame(series)
    prices = prices.dropna(how="all")

    if not prices.empty:
        gap_fraction = prices.isna().mean()
        for ticker, fraction in gap_fraction[gap_fraction > _MAX_EXPECTED_GAP_FRACTION].items():
            warnings.warn(
                f"{ticker} is missing {fraction:.0%} of the requested {start} to {end} range "
                "even after a retry -- this is more likely a rate-limit-induced gap than "
                "genuinely absent history; verify before trusting downstream results.",
                stacklevel=2,
            )

    return prices


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
