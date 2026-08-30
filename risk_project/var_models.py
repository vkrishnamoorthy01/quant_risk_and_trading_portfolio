"""Historical and parametric (variance-covariance) Value-at-Risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute a weighted portfolio return series from individual asset returns.

    Args:
        returns: DataFrame of asset returns, one column per ticker.
        weights: Portfolio weights indexed by ticker, summing to 1.0.

    Returns:
        Series of portfolio returns aligned to `returns`' index.
    """
    aligned_weights = weights.reindex(returns.columns)
    return returns.mul(aligned_weights, axis=1).sum(axis=1)


def historical_var(returns: pd.Series, confidence_level: float, notional: float = 1.0) -> float:
    """Compute historical (empirical) VaR from a portfolio return series.

    Takes the empirical quantile of realized returns directly, so it makes no
    distributional assumption but is only as good as the historical sample.

    Args:
        returns: Portfolio return series.
        confidence_level: Confidence level, e.g. 0.95 or 0.99.
        notional: Portfolio notional value used to scale VaR into currency terms.

    Returns:
        VaR as a positive number representing potential loss.
    """
    quantile = 1.0 - confidence_level
    var_return = returns.quantile(quantile)
    return -notional * var_return


def parametric_var(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    confidence_level: float,
    notional: float = 1.0,
) -> float:
    """Compute parametric (variance-covariance) VaR assuming normally distributed returns.

    Derives portfolio variance from the asset covariance matrix and weights
    (w' Sigma w), then scales by the normal quantile for `confidence_level`.
    Assumes zero mean return, standard for short-horizon VaR.

    Args:
        weights: Portfolio weights indexed by ticker, summing to 1.0.
        cov_matrix: Covariance matrix of asset returns, indexed and columned by ticker.
        confidence_level: Confidence level, e.g. 0.95 or 0.99.
        notional: Portfolio notional value used to scale VaR into currency terms.

    Returns:
        VaR as a positive number representing potential loss.
    """
    aligned_cov = cov_matrix.loc[weights.index, weights.index]
    portfolio_variance = weights.to_numpy() @ aligned_cov.to_numpy() @ weights.to_numpy()
    portfolio_std = np.sqrt(portfolio_variance)
    z_score = norm.ppf(confidence_level)
    return notional * z_score * portfolio_std
