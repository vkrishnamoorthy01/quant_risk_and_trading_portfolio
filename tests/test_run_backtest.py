"""Unit tests for signal_project.run_backtest's price-source selection. Fakes
both price fetchers; nothing here hits Kite or TWS.
"""

from __future__ import annotations

from typing import Any

import pytest
from pandas import DataFrame

import data_providers.ibkr_prices as ibkr_prices_module
import signal_project.run_backtest as backtest_module


def test_kite_is_the_default_source_and_keeps_the_original_file_names(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(backtest_module, "fetch_price_data", lambda tickers, *args: calls.append(list(tickers)) or DataFrame())

    backtest_module._fetch_prices("kite", ["NIFTY 50"], "2015-01-01", "2026-10-24")

    assert backtest_module.PRICE_SOURCES[0] == "kite"
    assert calls == [["NIFTY 50"]]
    assert backtest_module._results_suffix("kite") == ""


@pytest.mark.parametrize(("source", "environment"), [("ibkr_paper", "paper"), ("ibkr_live", "live")])
def test_ibkr_sources_use_the_ibkr_adapter_and_separate_file_names(
    monkeypatch: pytest.MonkeyPatch, source: str, environment: str
) -> None:
    calls = []

    def _fake_ibkr(tickers: Any, start: str, end: str, environment: str) -> DataFrame:
        calls.append((list(tickers), start, environment))
        return DataFrame()

    monkeypatch.setattr(backtest_module, "fetch_price_data", lambda *args: pytest.fail("Kite must not be called"))
    monkeypatch.setattr(ibkr_prices_module, "fetch_ibkr_price_data", _fake_ibkr)

    backtest_module._fetch_prices(source, ["NIFTY 50"], "2015-01-01", "2026-10-24")

    assert calls == [(["NIFTY 50"], "2015-01-01", environment)]
    assert backtest_module._results_suffix(source) == f"_ibkr_{environment}"


def test_an_unknown_source_is_refused() -> None:
    with pytest.raises(ValueError, match="ibkr_live"):
        backtest_module._fetch_prices("ibkr", ["RELIANCE"], "2015-01-01", "2026-10-24")
