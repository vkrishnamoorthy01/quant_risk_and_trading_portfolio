"""VaR backtesting: breach counts and the Kupiec proportion-of-failures test."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

KUPIEC_SIGNIFICANCE = 0.05


@dataclass(frozen=True)
class BacktestResult:
    """Backtest of a single VaR estimate against realized returns."""

    method: str
    confidence_level: float
    n_obs: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    kupiec_statistic: float
    kupiec_p_value: float

    @property
    def kupiec_reject(self) -> bool:
        """Whether Kupiec's test rejects the VaR model at the 5% significance level."""
        return self.kupiec_p_value < KUPIEC_SIGNIFICANCE


def count_breaches(returns: pd.Series, var_estimate: float, notional: float = 1.0) -> int:
    """Count days where the realized loss exceeded a fixed VaR estimate.

    Args:
        returns: Realized portfolio return series.
        var_estimate: VaR as a positive loss number, in the same currency
            units as `notional`.
        notional: Portfolio notional value used to convert returns to
            currency loss.

    Returns:
        Number of days where realized loss (-return * notional) exceeded
        var_estimate.
    """
    losses = -returns * notional
    return int((losses > var_estimate).sum())


def kupiec_test(n_obs: int, n_breaches: int, expected_rate: float) -> tuple[float, float]:
    """Kupiec's proportion-of-failures likelihood-ratio test.

    Tests whether the observed breach rate is statistically consistent with
    a VaR model's target exceedance rate. The statistic is asymptotically
    chi-squared with 1 degree of freedom under the null hypothesis that the
    model is correctly calibrated; `xlogy` handles the 0*log(0) edge cases
    at zero or full breach rates by the standard convention (treated as 0).

    Args:
        n_obs: Number of observations backtested.
        n_breaches: Number of VaR breaches observed.
        expected_rate: Target exceedance rate implied by the VaR confidence
            level (e.g. 0.05 for 95% VaR, 0.01 for 99% VaR).

    Returns:
        Tuple of (LR statistic, p-value).
    """
    observed_rate = n_breaches / n_obs
    log_l0 = xlogy(n_obs - n_breaches, 1 - expected_rate) + xlogy(n_breaches, expected_rate)
    log_l1 = xlogy(n_obs - n_breaches, 1 - observed_rate) + xlogy(n_breaches, observed_rate)
    lr_statistic = -2 * log_l0 + 2 * log_l1
    p_value = 1 - chi2.cdf(lr_statistic, df=1)
    return float(lr_statistic), float(p_value)


def backtest_var(
    returns: pd.Series,
    var_estimate: float,
    confidence_level: float,
    method: str,
    notional: float = 1.0,
) -> BacktestResult:
    """Backtest a single fixed VaR estimate against realized returns.

    This is an in-sample backtest: the VaR estimate and the return series
    it's tested against cover the same window, rather than a rolling
    out-of-sample comparison. See README limitations for the implication.

    Args:
        returns: Realized portfolio return series over the backtest window.
        var_estimate: VaR as a positive loss number.
        confidence_level: VaR confidence level, e.g. 0.95 or 0.99.
        method: Label for the VaR method, e.g. "Historical" or "Parametric".
        notional: Portfolio notional value used to convert returns to
            currency loss.

    Returns:
        BacktestResult with breach counts, breach rate, and the Kupiec
        test statistic/p-value.
    """
    n_obs = len(returns)
    n_breaches = count_breaches(returns, var_estimate, notional)
    expected_rate = 1.0 - confidence_level
    kupiec_statistic, kupiec_p_value = kupiec_test(n_obs, n_breaches, expected_rate)

    return BacktestResult(
        method=method,
        confidence_level=confidence_level,
        n_obs=n_obs,
        n_breaches=n_breaches,
        breach_rate=n_breaches / n_obs,
        expected_rate=expected_rate,
        kupiec_statistic=kupiec_statistic,
        kupiec_p_value=kupiec_p_value,
    )


def build_backtest_report(results: list[BacktestResult]) -> pd.DataFrame:
    """Assemble a backtest report table from BacktestResults.

    Args:
        results: BacktestResults from `backtest_var`.

    Returns:
        DataFrame with one row per (method, confidence level), including
        breach counts/rates and the Kupiec statistic, p-value, and reject
        flag.
    """
    return pd.DataFrame(
        [
            {
                "Method": r.method,
                "Confidence": r.confidence_level,
                "N": r.n_obs,
                "Breaches": r.n_breaches,
                "Breach Rate": r.breach_rate,
                "Expected Rate": r.expected_rate,
                "Kupiec LR": r.kupiec_statistic,
                "Kupiec p-value": r.kupiec_p_value,
                "Kupiec Reject": r.kupiec_reject,
            }
            for r in results
        ]
    )
