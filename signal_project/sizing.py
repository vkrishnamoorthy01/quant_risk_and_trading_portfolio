"""Volatility-scaled position sizing with a per-stock cap.

Sizing is a separate, continuously-updated calculation from signal-lookback
selection: it uses its own trailing realized-vol window, refreshed at every
rebalance, and is not part of the walk-forward parameter search.
"""

from __future__ import annotations

import pandas as pd

VOL_LOOKBACK_DAYS = 20
POSITION_CAP = 0.175  # ~2.5-3x equal-weight share at 16 stocks (~17.5% of book capital)


def realized_vol(returns: pd.DataFrame, lookback: int = VOL_LOOKBACK_DAYS) -> pd.DataFrame:
    """Trailing realized volatility (rolling std of daily returns) per ticker."""
    return returns.rolling(lookback).std()


def _cap_and_renormalize(weights: pd.Series, cap: float) -> pd.Series:
    """Clip weights at `cap`, redistributing the excess across uncapped names.

    Iterative waterfall with an explicit lock set: redistributing excess
    can push previously-uncapped names over the cap too, so this repeats
    until nothing uncapped exceeds it. A name that hits the cap is locked
    at that value for good — without the lock, a name sitting exactly at
    `cap` would look indistinguishable from one that was never capped, and
    a later round could hand it more weight and push it back over, cycling
    indefinitely instead of converging. If every active name ends up
    locked, the book is left partially in cash rather than breaching the
    cap — the cap is a concentration control, so it takes priority over
    full investment in that edge case.
    """
    weights = weights.copy()
    locked = pd.Series(False, index=weights.index)
    while True:
        newly_over = ~locked & (weights > cap)
        if not newly_over.any():
            return weights
        excess = (weights[newly_over] - cap).sum()
        weights[newly_over] = cap
        locked |= newly_over
        unlocked = ~locked
        if not unlocked.any():
            return weights
        weights[unlocked] += excess * (weights[unlocked] / weights[unlocked].sum())


def vol_scaled_weights(signal: pd.Series, vol: pd.Series, cap: float = POSITION_CAP) -> pd.Series:
    """Capped inverse-volatility weights for one rebalance date.

    Only tickers with a True signal and a known (non-NaN, positive) vol
    estimate participate; weights among them are inverse-vol, sum to 1.0,
    then capped and renormalized per `_cap_and_renormalize`. All other
    tickers get weight 0.

    Args:
        signal: Boolean Series, index of tickers, True where long.
        vol: Trailing realized vol per ticker (same index universe).
        cap: Maximum weight per ticker, as a fraction of the book.

    Returns:
        Weight Series over `signal`'s full index, summing to at most 1.0.
    """
    active = signal[signal].index
    active_vol = vol.reindex(active).dropna()
    active_vol = active_vol[active_vol > 0]

    weights = pd.Series(0.0, index=signal.index)
    if active_vol.empty:
        return weights

    inverse_vol = 1.0 / active_vol
    raw_weights = inverse_vol / inverse_vol.sum()
    weights.loc[raw_weights.index] = _cap_and_renormalize(raw_weights, cap)
    return weights
