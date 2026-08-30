"""Multi-factor model regressing portfolio returns on Fama-French-India factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

FACTOR_COLUMNS = ["MF", "SMB", "HML", "WML"]

DEFAULT_FACTOR_PATH = Path(__file__).resolve().parent / "data" / "iima_ff_india_daily.csv"


def load_iima_factors(path: str | Path = DEFAULT_FACTOR_PATH) -> pd.DataFrame:
    """Load IIMA's daily Fama-French-India factor data.

    IIMA publishes factor returns in percentage points (e.g. 1.38 means
    1.38%); this converts them to decimal fractions so they sit on the same
    scale as portfolio returns from shared.data.compute_returns.

    Args:
        path: Path to the IIMA daily factor CSV (columns: Date, SMB, HML,
            WML, MF, RF).

    Returns:
        DataFrame indexed by date, with SMB, HML, WML, MF, RF as decimal
        fractions. Rows missing any factor are dropped (the dataset's first
        row has no MF or RF).
    """
    factors = pd.read_csv(path, na_values="NA", parse_dates=["Date"])
    factors = factors.set_index("Date").sort_index()
    factors = factors[["SMB", "HML", "WML", "MF", "RF"]] / 100.0
    return factors.dropna(how="any")


@dataclass(frozen=True)
class FactorModelResult:
    """Result of regressing portfolio excess returns on factor returns."""

    alpha: float
    betas: pd.Series
    r_squared: float
    n_obs: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp


def fit_factor_model(portfolio_returns: pd.Series, factors: pd.DataFrame) -> FactorModelResult:
    """Regress portfolio excess returns on SMB, HML, WML, and MF.

    The regression window is the overlap between `portfolio_returns` and
    `factors`, so it is capped by whichever series is shorter. In practice
    this means the IIMA factor data's coverage (through Dec 2025) bounds the
    window, not the availability of the price data.

    Args:
        portfolio_returns: Daily portfolio returns indexed by date.
        factors: Factor DataFrame from `load_iima_factors`, indexed by date,
            including RF.

    Returns:
        FactorModelResult with intercept, factor loadings, R-squared, and the
        date range actually used.
    """
    aligned = pd.concat([portfolio_returns.rename("portfolio"), factors], axis=1, join="inner")
    aligned = aligned.dropna()
    if aligned.empty:
        raise ValueError("No overlapping dates between portfolio returns and factor data.")

    excess_return = aligned["portfolio"] - aligned["RF"]
    regressors = sm.add_constant(aligned[FACTOR_COLUMNS])
    model = sm.OLS(excess_return, regressors).fit()

    return FactorModelResult(
        alpha=model.params["const"],
        betas=model.params[FACTOR_COLUMNS],
        r_squared=model.rsquared,
        n_obs=int(model.nobs),
        window_start=aligned.index.min(),
        window_end=aligned.index.max(),
    )
