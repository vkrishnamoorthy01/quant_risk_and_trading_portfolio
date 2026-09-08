"""Sandbox authentication and order translation for live paper trading.

Two separate authenticated Kite clients are used across the project:
`shared.data.get_kite_client()` for production historical data, and
`get_sandbox_kite_client()` here for order placement against
sandbox.kite.trade. The sandbox only accepts LIMIT orders and
auto-rejects anything priced too far from LTP (see Kite's sandbox docs),
so every order here is priced a small margin away from the latest quote
rather than at the market. Long-only: a reduction never sells more than
is currently held, since there is no short leg to open.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

import pandas as pd
from kiteconnect import KiteConnect

SANDBOX_ROOT = "https://sandbox.kite.trade"
CALLBACK_PORT = 13173
LIMIT_PRICE_MARGIN = 0.02  # stay safely inside the sandbox's ~2%-from-LTP price-band check

_SESSION_CACHE_PATH = Path(__file__).resolve().parent.parent / ".kite_sandbox_session.json"


class OrderPlacingClient(Protocol):
    """The subset of KiteConnect execution.py depends on, for test doubles."""

    def quote(self, instruments: list[str]) -> dict: ...

    def place_order(self, **kwargs: object) -> str: ...

    def cancel_order(self, variety: str, order_id: str) -> str: ...


@dataclass(frozen=True)
class TargetOrder:
    """One order needed to move a ticker from its current to target position."""

    tradingsymbol: str
    transaction_type: str  # "BUY" or "SELL"
    quantity: int


def limit_price_for(transaction_type: str, ltp: float, margin: float = LIMIT_PRICE_MARGIN) -> float:
    """A resting LIMIT price close enough to LTP to clear the price-band check.

    A BUY rests slightly below LTP and a SELL slightly above — the
    direction a resting order needs in order to have any chance of
    filling — while staying inside the sandbox's roughly 2%-from-LTP
    tolerance.
    """
    if transaction_type == "BUY":
        return round(ltp * (1 - margin), 1)
    if transaction_type == "SELL":
        return round(ltp * (1 + margin), 1)
    raise ValueError(f"Unknown transaction_type: {transaction_type!r}")


def target_orders(
    current_shares: dict[str, int],
    target_weights: pd.Series,
    prices: pd.Series,
    book_notional: float,
) -> list[TargetOrder]:
    """Diff current share holdings against target weights into whole-share orders.

    Args:
        current_shares: Shares currently held per ticker (0 if absent).
        target_weights: Desired weight per ticker, as a fraction of `book_notional`.
        prices: Latest price per ticker, used to convert weight to share count.
        book_notional: Current book NAV that `target_weights` is a fraction of.

    Returns:
        One TargetOrder per ticker whose share count needs to change; tickers
        already at their target are omitted.
    """
    orders = []
    for ticker, weight in target_weights.items():
        price = prices[ticker]
        target_shares = int(weight * book_notional / price) if price > 0 else 0
        current = current_shares.get(ticker, 0)
        delta = target_shares - current
        if delta > 0:
            orders.append(TargetOrder(ticker, "BUY", delta))
        elif delta < 0:
            sell_quantity = min(-delta, current)  # never sell more than currently held (no shorting)
            if sell_quantity > 0:
                orders.append(TargetOrder(ticker, "SELL", sell_quantity))
    return orders


def place_target_orders(
    client: OrderPlacingClient, orders: list[TargetOrder], exchange: str = "NSE"
) -> list[str]:
    """Place each TargetOrder as a sandbox LIMIT order, priced off the latest quote.

    Returns the resulting order ids, in the same order as `orders`.
    """
    order_ids = []
    for order in orders:
        quote_key = f"{exchange}:{order.tradingsymbol}"
        ltp = client.quote([quote_key])[quote_key]["last_price"]
        price = limit_price_for(order.transaction_type, ltp)
        order_id = client.place_order(
            variety="regular",
            exchange=exchange,
            tradingsymbol=order.tradingsymbol,
            transaction_type=order.transaction_type,
            quantity=order.quantity,
            product="MIS",
            order_type="LIMIT",
            price=price,
            tag="signalproject",
        )
        order_ids.append(order_id)
    return order_ids


def _patch_for_sandbox(kite: KiteConnect) -> None:
    """Prefix all SDK routes with /oms except the instrument dumps (sandbox requirement)."""
    passthrough = {"market.instruments.all", "market.instruments"}
    kite._routes = {
        key: (value if key in passthrough else "/oms" + value) for key, value in kite._routes.items()
    }


def _load_cached_sandbox_token(api_key: str) -> str | None:
    if not _SESSION_CACHE_PATH.exists():
        return None
    cached = json.loads(_SESSION_CACHE_PATH.read_text())
    if cached.get("api_key") != api_key or cached.get("date") != date.today().isoformat():
        return None
    return cached.get("access_token")


def _cache_sandbox_token(api_key: str, access_token: str) -> None:
    _SESSION_CACHE_PATH.write_text(
        json.dumps({"api_key": api_key, "date": date.today().isoformat(), "access_token": access_token})
    )


class _CallbackHandler(BaseHTTPRequestHandler):
    """Captures request_token off the sandbox's OAuth redirect to localhost."""

    request_token: str | None = None
    captured = threading.Event()

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        _CallbackHandler.request_token = query.get("request_token", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Login captured, you can close this tab.")
        _CallbackHandler.captured.set()

    def log_message(self, *args: object) -> None:  # silence default request logging
        pass


def _get_sandbox_request_token(login_url: str, timeout: float = 180.0) -> str:
    _CallbackHandler.captured.clear()
    _CallbackHandler.request_token = None
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    threading.Thread(target=server.handle_request, daemon=True).start()

    print(f"Opening sandbox login URL:\n{login_url}")
    webbrowser.open(login_url)

    if not _CallbackHandler.captured.wait(timeout=timeout):
        raise TimeoutError(f"No sandbox login redirect received within {timeout}s.")
    server.server_close()
    assert _CallbackHandler.request_token is not None
    return _CallbackHandler.request_token


def get_sandbox_kite_client() -> KiteConnect:
    """Authenticate against sandbox.kite.trade and return a ready-to-use client.

    Reads KITE_SANDBOX_API_KEY / KITE_SANDBOX_API_SECRET from the
    environment, defaulting to Kite's published shared demo values
    (sandboxdemo / sandboxdemo-secret — safe to use in plain text, not
    tied to real money) if unset. Reuses a same-day cached access token if
    one exists; otherwise opens the sandbox login URL in a browser and
    captures the redirect on a local callback server.
    """
    api_key = os.environ.get("KITE_SANDBOX_API_KEY", "sandboxdemo")
    api_secret = os.environ.get("KITE_SANDBOX_API_SECRET", "sandboxdemo-secret")

    kite = KiteConnect(api_key=api_key, root=SANDBOX_ROOT)
    _patch_for_sandbox(kite)

    access_token = _load_cached_sandbox_token(api_key)
    if access_token is None:
        login_url = f"{SANDBOX_ROOT}/connect/login?api_key={api_key}"
        request_token = _get_sandbox_request_token(login_url)
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]
        _cache_sandbox_token(api_key, access_token)

    kite.set_access_token(access_token)
    return kite
