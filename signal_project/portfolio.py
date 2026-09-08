"""Book configuration for the signal project."""

from __future__ import annotations

from dataclasses import dataclass

from risk_project.portfolio import PORTFOLIO


@dataclass(frozen=True)
class BookConfig:
    """A single independently-run signal book: same universe, own notional."""

    name: str
    tickers: tuple[str, ...]
    notional: float


TOTAL_NOTIONAL = PORTFOLIO.notional  # ₹1 crore, same universe as risk_project

MOMENTUM_BOOK = BookConfig(name="momentum", tickers=PORTFOLIO.tickers, notional=TOTAL_NOTIONAL / 2)
MEAN_REVERSION_BOOK = BookConfig(name="mean_reversion", tickers=PORTFOLIO.tickers, notional=TOTAL_NOTIONAL / 2)
