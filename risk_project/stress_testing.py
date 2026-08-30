"""Stress testing: historical crisis replay and a statistically calibrated hypothetical shock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_project.factor_model import FACTOR_COLUMNS

# 2008 GFC: Lehman Brothers' collapse (2008-09-15) through the approximate
# global equity market trough in early 2009.
GFC_WINDOW = ("2008-09-01", "2009-03-31")

# 2020 COVID crash: pre-selloff peak through the NIFTY 50's trough on
# 2020-03-23.
COVID_WINDOW = ("2020-02-20", "2020-03-23")


def historical_factor_shock(factors: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Sum realized daily factor returns over a historical window.

    Summing (rather than compounding) daily factor returns is the
    aggregation consistent with the linear factor model: since
    excess_return_t = alpha + sum(beta_i * factor_i,t) + residual_t, summing
    factor_i,t over the window preserves a clean per-factor decomposition of
    cumulative impact when later multiplied through by static betas. This
    ignores compounding effects and any alpha/idiosyncratic contribution to
    the portfolio's actual historical P&L over the window.

    Args:
        factors: Factor DataFrame from `load_daily_factors`.
        start: Window start date, "YYYY-MM-DD".
        end: Window end date, "YYYY-MM-DD" (inclusive).

    Returns:
        Series of summed factor returns over the window, indexed by
        FACTOR_COLUMNS.
    """
    window = factors.loc[start:end, FACTOR_COLUMNS]
    if window.empty:
        raise ValueError(f"No factor data between {start} and {end}.")
    return window.sum()


def hypothetical_factor_shock(
    factors: pd.DataFrame,
    betas: pd.Series,
    start: str,
    end: str,
    n_sigma: float = 3.0,
) -> pd.Series:
    """Build a statistically calibrated adverse factor shock.

    Each factor is shocked by `n_sigma` times its own historical daily
    volatility, estimated over [start, end] — the same window used for the
    VaR analysis, so the shock size is calibrated to observed risk rather
    than an arbitrary guess. Each factor's shock direction is adverse to
    this portfolio: opposite the sign of its estimated beta, so a
    positive-beta factor is shocked negative and vice versa.

    Args:
        factors: Factor DataFrame from `load_daily_factors`.
        betas: Factor loadings from `fit_factor_model`, indexed by
            FACTOR_COLUMNS.
        start: VaR window start date, "YYYY-MM-DD".
        end: VaR window end date, "YYYY-MM-DD" (inclusive).
        n_sigma: Number of standard deviations for the shock (default 3.0).

    Returns:
        Series of signed shock sizes, indexed by FACTOR_COLUMNS.
    """
    window = factors.loc[start:end, FACTOR_COLUMNS]
    if window.empty:
        raise ValueError(f"No factor data between {start} and {end}.")
    daily_vol = window.std()
    adverse_sign = -np.sign(betas.reindex(FACTOR_COLUMNS))
    return n_sigma * daily_vol * adverse_sign


def scenario_impact(
    betas: pd.Series, factor_shock: pd.Series, notional: float
) -> tuple[pd.Series, float]:
    """Decompose a scenario's P&L impact into per-factor contributions.

    Args:
        betas: Factor loadings from `fit_factor_model`, indexed by
            FACTOR_COLUMNS.
        factor_shock: Shock size per factor, from `historical_factor_shock`
            or `hypothetical_factor_shock`.
        notional: Portfolio notional value.

    Returns:
        Tuple of (per-factor P&L contribution, total P&L impact). A
        negative value represents a loss.
    """
    aligned_shock = factor_shock.reindex(betas.index)
    contributions = betas * aligned_shock * notional
    return contributions, contributions.sum()


@dataclass(frozen=True)
class ScenarioResult:
    """A stress scenario's P&L impact, decomposed by factor, with a VaR breach flag."""

    name: str
    pnl_impact: float
    factor_contributions: pd.Series
    worst_factor: str
    var_breach: bool


def run_scenario(
    name: str,
    betas: pd.Series,
    factor_shock: pd.Series,
    notional: float,
    var_99: float,
) -> ScenarioResult:
    """Run one stress scenario and flag whether its implied loss breaches VaR.

    Args:
        name: Scenario label, e.g. "2008 GFC".
        betas: Factor loadings from `fit_factor_model`.
        factor_shock: Per-factor shock size for this scenario.
        notional: Portfolio notional value.
        var_99: Parametric 99% VaR (a positive loss threshold) to compare
            the scenario's implied loss against.

    Returns:
        ScenarioResult with the P&L impact, per-factor breakdown, worst
        factor contributor, and whether the implied loss exceeds var_99.
    """
    contributions, pnl_impact = scenario_impact(betas, factor_shock, notional)
    loss = -pnl_impact
    return ScenarioResult(
        name=name,
        pnl_impact=pnl_impact,
        factor_contributions=contributions,
        worst_factor=contributions.idxmin(),
        var_breach=bool(loss > var_99),
    )


def build_scenario_report(results: list[ScenarioResult]) -> pd.DataFrame:
    """Assemble a scenario report table from a list of ScenarioResults.

    Args:
        results: ScenarioResults from `run_scenario`.

    Returns:
        DataFrame indexed by scenario name with columns "P&L Impact",
        "VaR Breach", and "Worst Factor".
    """
    return pd.DataFrame(
        {
            "P&L Impact": {r.name: r.pnl_impact for r in results},
            "VaR Breach": {r.name: r.var_breach for r in results},
            "Worst Factor": {r.name: r.worst_factor for r in results},
        }
    )
