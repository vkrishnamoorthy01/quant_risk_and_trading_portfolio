"""Unit tests for risk_project.var_models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from risk_project.var_models import historical_var, parametric_var, portfolio_returns


def test_portfolio_returns_matches_manual_weighted_sum() -> None:
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [-0.01, 0.02, 0.00]})
    weights = pd.Series({"A": 0.4, "B": 0.6})

    result = portfolio_returns(returns, weights)

    expected = 0.4 * returns["A"] + 0.6 * returns["B"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_historical_var_matches_empirical_quantile() -> None:
    returns = pd.Series(np.linspace(-0.10, 0.10, 101))  # includes exact 5th/1st percentiles

    result = historical_var(returns, confidence_level=0.95, notional=1_000.0)

    expected = -1_000.0 * returns.quantile(0.05)
    assert np.isclose(result, expected)
    assert result > 0  # a loss, reported as a positive number


def test_historical_var_scales_linearly_with_notional() -> None:
    returns = pd.Series(np.linspace(-0.10, 0.10, 101))

    var_1x = historical_var(returns, confidence_level=0.99, notional=1_000.0)
    var_10x = historical_var(returns, confidence_level=0.99, notional=10_000.0)

    assert np.isclose(var_10x, 10 * var_1x)


def test_parametric_var_matches_closed_form_for_uncorrelated_assets() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    variances = pd.Series({"A": 0.04, "B": 0.09})  # std of 0.20 and 0.30
    cov_matrix = pd.DataFrame(np.diag(variances), index=variances.index, columns=variances.index)

    result = parametric_var(weights, cov_matrix, confidence_level=0.95, notional=1.0)

    expected_variance = 0.5**2 * 0.04 + 0.5**2 * 0.09
    expected = norm.ppf(0.95) * np.sqrt(expected_variance)
    assert np.isclose(result, expected)


def test_parametric_var_increases_with_confidence_level() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    cov_matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )

    var_95 = parametric_var(weights, cov_matrix, confidence_level=0.95, notional=1.0)
    var_99 = parametric_var(weights, cov_matrix, confidence_level=0.99, notional=1.0)

    assert var_99 > var_95


def test_parametric_var_ignores_weight_order_via_alignment() -> None:
    cov_matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    weights_ab = pd.Series({"A": 0.3, "B": 0.7})
    weights_ba = pd.Series({"B": 0.7, "A": 0.3})

    var_ab = parametric_var(weights_ab, cov_matrix, confidence_level=0.95, notional=1.0)
    var_ba = parametric_var(weights_ba, cov_matrix, confidence_level=0.95, notional=1.0)

    assert np.isclose(var_ab, var_ba)
