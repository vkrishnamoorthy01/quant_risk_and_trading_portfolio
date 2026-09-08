"""One-off check: does a paper order actually go through on sandbox.kite.trade?

Run from anywhere with the repo's venv active:
    python check_sandbox_order.py

What it does, end to end, in isolation from the rest of the codebase:
  1. Prints a sandbox login URL and starts a tiny local HTTP server on
     localhost:13173 to catch the OAuth redirect (mirrors Kite's own
     agent-setup callback-app pattern, without adding a Flask dependency
     for a throwaway check).
  2. Exchanges the captured request_token for an access_token.
  3. Applies the required /oms route patch and sets root to
     https://sandbox.kite.trade.
  4. Fetches LTP for one test symbol, places a single LIMIT order priced
     ~2% away from LTP (the only order type the sandbox accepts), and
     prints back the resulting order status.
  5. Best-effort cancels the resting order afterward so nothing is left
     open in the sandbox account.

This only exercises order placement end to end — it is NOT the real
signal_project pipeline (no strategy, no position sizing, no risk
controls). It exists purely to confirm sandbox execution actually works,
and incidentally to observe whether anything IP-related blocks it, before
building the full pipeline on top of it.

Sandbox credentials default to Kite's published shared demo app
(sandboxdemo / sandboxdemo-secret per their own docs, safe to use in
plain text) unless KITE_SANDBOX_API_KEY / KITE_SANDBOX_API_SECRET are set
in the environment.
"""

from __future__ import annotations

import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect

SANDBOX_ROOT = "https://sandbox.kite.trade"
CALLBACK_PORT = 13173
TEST_SYMBOL = "INFY"
API_KEY = os.environ.get("KITE_SANDBOX_API_KEY", "sandboxdemo")
API_SECRET = os.environ.get("KITE_SANDBOX_API_SECRET", "sandboxdemo-secret")

_request_token: str | None = None
_captured = threading.Event()


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        global _request_token
        query = parse_qs(urlparse(self.path).query)
        _request_token = query.get("request_token", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Login captured, you can close this tab.")
        _captured.set()

    def log_message(self, *args: object) -> None:  # silence default request logging
        pass


def _patch_for_sandbox(kite: KiteConnect) -> None:
    passthrough = {"market.instruments.all", "market.instruments"}
    kite._routes = {
        key: (value if key in passthrough else "/oms" + value) for key, value in kite._routes.items()
    }


def _get_request_token() -> str:
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    login_url = f"{SANDBOX_ROOT}/connect/login?api_key={API_KEY}"
    print(f"Opening login URL (log in with the sandbox demo flow):\n{login_url}")
    webbrowser.open(login_url)

    if not _captured.wait(timeout=180):
        raise TimeoutError("No login redirect received within 180s.")
    server.server_close()
    assert _request_token is not None
    return _request_token


if __name__ == "__main__":
    request_token = _get_request_token()

    kite = KiteConnect(api_key=API_KEY, root=SANDBOX_ROOT)
    _patch_for_sandbox(kite)

    session = kite.generate_session(request_token, api_secret=API_SECRET)
    kite.set_access_token(session["access_token"])
    print(f"Authenticated. user_id={session.get('user_id')}")

    quote_key = f"NSE:{TEST_SYMBOL}"
    ltp = kite.quote([quote_key])[quote_key]["last_price"]
    price = round(ltp * 0.98, 1)
    print(f"LTP for {TEST_SYMBOL}: {ltp}. Placing BUY LIMIT @ {price} (2% below LTP).")

    order_id = kite.place_order(
        variety="regular",
        exchange="NSE",
        tradingsymbol=TEST_SYMBOL,
        transaction_type="BUY",
        quantity=1,
        product="MIS",
        order_type="LIMIT",
        price=price,
        tag="signalprojectcheck",
    )
    print(f"Order placed: order_id={order_id}")

    history = kite.order_history(order_id)
    print("Order history:")
    for entry in history:
        print(f"  {entry['order_timestamp']}: status={entry['status']} ({entry.get('status_message')})")

    try:
        kite.cancel_order(variety="regular", order_id=order_id)
        print("Resting order cancelled (cleanup).")
    except Exception as exc:  # noqa: BLE001 - best-effort cleanup only
        print(f"Cleanup cancel skipped/failed (likely already filled or terminal): {exc}")
