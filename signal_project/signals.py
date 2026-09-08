"""Per-stock time-series momentum and mean-reversion signals.

Both signals are computed independently for each stock (its own trend or
reversion state), not as a cross-sectional ranking across the universe —
see the project README for the rationale. Both are long-only: the signal
is a boolean "be long" / "be flat" state, never "be short".
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_MONTH = 21


def momentum_signal(prices: pd.DataFrame, lookback_months: int) -> pd.DataFrame:
    """Time-series momentum: long a stock while its trailing return is positive.

    Trailing return is measured over `lookback_months` (converted to
    trading days at 21/month), following the standard time-series
    momentum convention (position = sign of trailing return), restricted
    to long-only by dropping the short leg.

    Args:
        prices: Daily close prices, one column per ticker.
        lookback_months: Lookback window in months.

    Returns:
        Boolean DataFrame, same shape as `prices`, True where the signal
        is long. Insufficient-history cells are False.
    """
    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    trailing_return = prices / prices.shift(lookback_days) - 1
    return (trailing_return > 0).fillna(False)


def mean_reversion_signal(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Time-series mean-reversion: long a stock after a recent price drop.

    Contrarian counterpart to momentum: long while the trailing return
    over `lookback_days` is negative (expecting reversion), flat once it
    is no longer negative, restricted to long-only.

    Args:
        prices: Daily close prices, one column per ticker.
        lookback_days: Lookback window in trading days.

    Returns:
        Boolean DataFrame, same shape as `prices`, True where the signal
        is long. Insufficient-history cells are False.
    """
    trailing_return = prices / prices.shift(lookback_days) - 1
    return (trailing_return < 0).fillna(False)
