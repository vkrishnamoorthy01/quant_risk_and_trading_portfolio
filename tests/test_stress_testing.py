"""Unit tests for risk_project.stress_testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_project.stress_testing import (
    build_scenario_report,
    historical_factor_shock,
    hypothetical_factor_shock,
    run_scenario,
    scenario_impact,
)


def _synthetic_factors(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "SMB": rng.normal(0, 0.01, n),
            "HML": rng.normal(0, 0.02, n),
            "WML": rng.normal(0, 0.015, n),
            "MF": rng.normal(0.0005, 0.012, n),
            "RF": np.full(n, 0.0001),
        },
        index=dates,
    )


def test_historical_factor_shock_sums_window() -> None:
    factors = _synthetic_factors(n=20)
    start, end = factors.index[5], factors.index[10]

    result = historical_factor_shock(factors, start, end)

    expected = factors.loc[start:end, ["MF", "SMB", "HML", "WML"]].sum()
    pd.testing.assert_series_equal(result, expected)


def test_historical_factor_shock_raises_on_empty_window() -> None:
    factors = _synthetic_factors(n=20)

    with pytest.raises(ValueError, match="No factor data"):
        historical_factor_shock(factors, "1900-01-01", "1900-01-31")


def test_hypothetical_factor_shock_direction_is_adverse_to_betas() -> None:
    factors = _synthetic_factors(n=250)
    betas = pd.Series({"MF": 1.2, "SMB": -0.4, "HML": 0.0, "WML": -0.6})
    start, end = factors.index[0], factors.index[-1]

    shock = hypothetical_factor_shock(factors, betas, start, end, n_sigma=3.0)

    assert shock["MF"] < 0  # positive beta -> adverse shock is negative
    assert shock["SMB"] > 0  # negative beta -> adverse shock is positive
    assert shock["WML"] > 0
    assert shock["HML"] == 0.0  # zero beta -> zero-sign shock magnitude is 0


def test_hypothetical_factor_shock_magnitude_uses_n_sigma() -> None:
    factors = _synthetic_factors(n=250)
    betas = pd.Series({"MF": 1.0, "SMB": 1.0, "HML": 1.0, "WML": 1.0})
    start, end = factors.index[0], factors.index[-1]

    shock = hypothetical_factor_shock(factors, betas, start, end, n_sigma=3.0)

    expected_vol = factors.loc[start:end, ["MF", "SMB", "HML", "WML"]].std()
    assert np.allclose(shock.abs(), 3.0 * expected_vol)


def test_scenario_impact_decomposition_sums_to_total() -> None:
    betas = pd.Series({"MF": 1.0, "SMB": 0.5, "HML": -0.3, "WML": 0.2})
    shock = pd.Series({"MF": -0.03, "SMB": 0.02, "HML": 0.02, "WML": 0.015})
    notional = 10_000_000.0

    contributions, total = scenario_impact(betas, shock, notional)

    assert np.isclose(total, contributions.sum())
    assert np.isclose(contributions["MF"], 1.0 * -0.03 * notional)


def test_run_scenario_flags_breach_when_loss_exceeds_var() -> None:
    betas = pd.Series({"MF": 1.0, "SMB": 0.0, "HML": 0.0, "WML": 0.0})
    shock = pd.Series({"MF": -0.10, "SMB": 0.0, "HML": 0.0, "WML": 0.0})
    notional = 10_000_000.0  # loss = 1_000_000

    result = run_scenario("big loss", betas, shock, notional, var_99=500_000.0)

    assert result.var_breach is True
    assert result.worst_factor == "MF"
    assert np.isclose(result.pnl_impact, -1_000_000.0)


def test_run_scenario_no_breach_when_loss_within_var() -> None:
    betas = pd.Series({"MF": 1.0, "SMB": 0.0, "HML": 0.0, "WML": 0.0})
    shock = pd.Series({"MF": -0.01, "SMB": 0.0, "HML": 0.0, "WML": 0.0})
    notional = 10_000_000.0  # loss = 100_000

    result = run_scenario("small loss", betas, shock, notional, var_99=500_000.0)

    assert result.var_breach is False


def test_build_scenario_report_shape_and_values() -> None:
    betas = pd.Series({"MF": 1.0, "SMB": 0.0, "HML": 0.0, "WML": 0.0})
    notional = 10_000_000.0
    result_a = run_scenario(
        "A", betas, pd.Series({"MF": -0.10, "SMB": 0.0, "HML": 0.0, "WML": 0.0}), notional, 500_000.0
    )
    result_b = run_scenario(
        "B", betas, pd.Series({"MF": -0.01, "SMB": 0.0, "HML": 0.0, "WML": 0.0}), notional, 500_000.0
    )

    report = build_scenario_report([result_a, result_b])

    assert list(report.index) == ["A", "B"]
    assert list(report.columns) == ["P&L Impact", "VaR Breach", "Worst Factor"]
    assert report.loc["A", "VaR Breach"] == True  # noqa: E712 (numpy bool scalar, not Python bool)
    assert report.loc["B", "VaR Breach"] == False  # noqa: E712
