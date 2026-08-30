"""Unit tests for risk_project.backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_project.backtesting import (
    backtest_var,
    build_backtest_report,
    count_breaches,
    kupiec_test,
)


def test_count_breaches_counts_days_exceeding_var() -> None:
    returns = pd.Series([-0.06, -0.02, 0.01, -0.10, 0.03])  # losses: 0.06,0.02,-0.01,0.10,-0.03
    notional = 1_000.0

    result = count_breaches(returns, var_estimate=50.0, notional=notional)

    assert result == 2  # 60 and 100 exceed 50; 20 does not


def test_kupiec_test_near_zero_when_breach_rate_matches_expected() -> None:
    lr_statistic, p_value = kupiec_test(n_obs=100, n_breaches=5, expected_rate=0.05)

    assert np.isclose(lr_statistic, 0.0, atol=1e-8)
    assert p_value > 0.99


def test_kupiec_test_rejects_when_breach_rate_far_from_expected() -> None:
    lr_statistic, p_value = kupiec_test(n_obs=250, n_breaches=25, expected_rate=0.01)

    assert lr_statistic > 3.841  # chi2 critical value at 5% significance, df=1
    assert p_value < 0.05


def test_kupiec_test_handles_zero_breaches_without_error() -> None:
    lr_statistic, p_value = kupiec_test(n_obs=100, n_breaches=0, expected_rate=0.05)

    assert np.isfinite(lr_statistic)
    assert 0.0 <= p_value <= 1.0


def test_kupiec_test_handles_all_breaches_without_error() -> None:
    lr_statistic, p_value = kupiec_test(n_obs=10, n_breaches=10, expected_rate=0.05)

    assert np.isfinite(lr_statistic)
    assert 0.0 <= p_value <= 1.0


def test_backtest_var_rejects_when_miscalibrated() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.05, 250))  # much wider than VaR assumes

    result = backtest_var(returns, var_estimate=0.01, confidence_level=0.99, method="Parametric")

    assert result.n_obs == 250
    assert np.isclose(result.expected_rate, 0.01)
    assert result.kupiec_reject is True


def test_backtest_var_no_reject_when_well_calibrated() -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, 500))
    var_estimate = -np.quantile(returns, 0.05)  # exactly the 95% empirical VaR

    result = backtest_var(returns, var_estimate=var_estimate, confidence_level=0.95, method="Historical")

    assert result.kupiec_reject is False


def test_build_backtest_report_shape_and_values() -> None:
    returns = pd.Series(np.linspace(-0.05, 0.05, 200))
    result_a = backtest_var(returns, var_estimate=0.02, confidence_level=0.95, method="Historical")
    result_b = backtest_var(returns, var_estimate=0.03, confidence_level=0.99, method="Parametric")

    report = build_backtest_report([result_a, result_b])

    assert list(report["Method"]) == ["Historical", "Parametric"]
    assert list(report.columns) == [
        "Method",
        "Confidence",
        "N",
        "Breaches",
        "Breach Rate",
        "Expected Rate",
        "Kupiec LR",
        "Kupiec p-value",
        "Kupiec Reject",
    ]
