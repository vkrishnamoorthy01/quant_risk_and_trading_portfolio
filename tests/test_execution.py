"""Unit tests for signal_project.execution.

Covers only the pure order-translation logic. Authentication and the
sandbox HTTP callback (get_sandbox_kite_client) require a live interactive
login and are deliberately not exercised here — see check_sandbox_order.py
for the live smoke test, run manually.
"""

from __future__ import annotations

import pandas as pd
import pytest

from signal_project.execution import TargetOrder, limit_price_for, place_target_orders, target_orders


class _FakeKiteClient:
    """Records place_order calls instead of hitting any network."""

    def __init__(self, ltp_by_symbol: dict[str, float]) -> None:
        self._ltp_by_symbol = ltp_by_symbol
        self.placed_orders: list[dict] = []

    def quote(self, instruments: list[str]) -> dict:
        return {key: {"last_price": self._ltp_by_symbol[key.split(":")[1]]} for key in instruments}

    def place_order(self, **kwargs: object) -> str:
        self.placed_orders.append(kwargs)
        return f"order-{len(self.placed_orders)}"


def test_limit_price_for_buy_rests_below_ltp() -> None:
    assert limit_price_for("BUY", ltp=100.0, margin=0.02) == 98.0


def test_limit_price_for_sell_rests_above_ltp() -> None:
    assert limit_price_for("SELL", ltp=100.0, margin=0.02) == 102.0


def test_limit_price_for_rejects_unknown_transaction_type() -> None:
    with pytest.raises(ValueError):
        limit_price_for("SHORT", ltp=100.0)


def test_target_orders_buys_to_open_a_new_position() -> None:
    orders = target_orders(
        current_shares={},
        target_weights=pd.Series({"TCS": 0.5}),
        prices=pd.Series({"TCS": 1000.0}),
        book_notional=100_000.0,
    )

    assert orders == [TargetOrder("TCS", "BUY", 50)]  # 0.5 * 100,000 / 1000 = 50 shares


def test_target_orders_sells_to_flatten_an_exited_position() -> None:
    orders = target_orders(
        current_shares={"TCS": 50},
        target_weights=pd.Series({"TCS": 0.0}),
        prices=pd.Series({"TCS": 1000.0}),
        book_notional=100_000.0,
    )

    assert orders == [TargetOrder("TCS", "SELL", 50)]


def test_target_orders_never_sells_more_than_currently_held() -> None:
    # Long-only: even if the (impossible, here contrived) target implied a
    # larger reduction than what's held, the sell is capped at current shares.
    orders = target_orders(
        current_shares={"TCS": 10},
        target_weights=pd.Series({"TCS": -1.0}),
        prices=pd.Series({"TCS": 1000.0}),
        book_notional=100_000.0,
    )

    assert orders == [TargetOrder("TCS", "SELL", 10)]


def test_target_orders_skips_tickers_already_at_target() -> None:
    orders = target_orders(
        current_shares={"TCS": 50},
        target_weights=pd.Series({"TCS": 0.5}),
        prices=pd.Series({"TCS": 1000.0}),
        book_notional=100_000.0,
    )

    assert orders == []


def test_place_target_orders_prices_off_the_latest_quote_and_returns_order_ids() -> None:
    client = _FakeKiteClient(ltp_by_symbol={"TCS": 1000.0, "INFY": 500.0})
    orders = [TargetOrder("TCS", "BUY", 10), TargetOrder("INFY", "SELL", 5)]

    order_ids = place_target_orders(client, orders)

    assert order_ids == ["order-1", "order-2"]
    assert client.placed_orders[0]["price"] == 980.0  # BUY, 2% below LTP
    assert client.placed_orders[0]["order_type"] == "LIMIT"
    assert client.placed_orders[1]["price"] == 510.0  # SELL, 2% above LTP
