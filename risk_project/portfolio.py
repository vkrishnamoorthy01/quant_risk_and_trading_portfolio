"""Portfolio configuration for the risk project."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    """Equal-weighted NSE equity portfolio definition."""

    tickers: tuple[str, ...]
    notional: float
    benchmark: str

    @property
    def weights(self) -> pd.Series:
        """Equal weight per ticker, summing to 1.0."""
        weight = 1.0 / len(self.tickers)
        return pd.Series(weight, index=list(self.tickers), name="weight")


PORTFOLIO = PortfolioConfig(
    tickers=(
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "TCS",
        "INFY",
        "WIPRO",
        "RELIANCE",
        "ONGC",
        "HINDUNILVR",
        "ITC",
        "MARUTI",
        "TATAMOTORS",
        "SUNPHARMA",
        "DRREDDY",
        "TATASTEEL",
        "ULTRACEMCO",
    ),
    notional=10_000_000.0,
    benchmark="NIFTY 50",
)
