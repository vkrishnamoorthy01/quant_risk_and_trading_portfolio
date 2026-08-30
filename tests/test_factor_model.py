"""Unit tests for risk_project.factor_model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk_project.factor_model import fit_factor_model, load_iima_factors


def test_load_iima_factors_converts_percent_to_decimal_and_drops_na(tmp_path: Path) -> None:
    csv_path = tmp_path / "factors.csv"
    csv_path.write_text(
        "Date,SMB,HML,WML,MF,RF\n"
        "1993-10-01,1.0,2.0,-1.0,NA,NA\n"
        "1993-10-04,0.5,1.0,-0.5,2.0,0.02\n"
        "1993-10-05,-0.5,-1.0,0.5,-1.0,0.02\n"
    )

    factors = load_iima_factors(csv_path)

    assert len(factors) == 2  # first row (NA MF/RF) dropped
    assert list(factors.columns) == ["SMB", "HML", "WML", "MF", "RF"]
    row = factors.loc[pd.Timestamp("1993-10-04")]
    assert np.isclose(row["SMB"], 0.005)
    assert np.isclose(row["MF"], 0.02)
    assert np.isclose(row["RF"], 0.0002)


def _synthetic_factors(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "SMB": rng.normal(0, 0.01, n),
            "HML": rng.normal(0, 0.01, n),
            "WML": rng.normal(0, 0.01, n),
            "MF": rng.normal(0.0005, 0.01, n),
            "RF": np.full(n, 0.0001),
        },
        index=dates,
    )


def test_fit_factor_model_recovers_known_betas_with_no_noise() -> None:
    factors = _synthetic_factors(n=60)
    true_alpha = 0.0002
    true_betas = {"MF": 1.1, "SMB": 0.3, "HML": -0.2, "WML": 0.4}

    excess_return = (
        true_alpha
        + true_betas["MF"] * factors["MF"]
        + true_betas["SMB"] * factors["SMB"]
        + true_betas["HML"] * factors["HML"]
        + true_betas["WML"] * factors["WML"]
    )
    portfolio_returns = excess_return + factors["RF"]

    result = fit_factor_model(portfolio_returns, factors)

    assert np.isclose(result.alpha, true_alpha, atol=1e-8)
    for factor, beta in true_betas.items():
        assert np.isclose(result.betas[factor], beta, atol=1e-6)
    assert result.r_squared > 0.999
    assert result.n_obs == 60


def test_fit_factor_model_window_matches_overlap() -> None:
    factors = _synthetic_factors(n=30)
    portfolio_returns = pd.Series(
        np.full(40, 0.001),
        index=pd.date_range("2022-01-03", periods=40, freq="B"),
    )

    result = fit_factor_model(portfolio_returns, factors)

    assert result.n_obs == 30
    assert result.window_start == factors.index.min()
    assert result.window_end == factors.index.max()


def test_fit_factor_model_raises_on_no_overlap() -> None:
    factors = _synthetic_factors(n=10)
    non_overlapping_returns = pd.Series(
        np.full(10, 0.001),
        index=pd.date_range("2030-01-01", periods=10, freq="B"),
    )

    with pytest.raises(ValueError, match="No overlapping dates"):
        fit_factor_model(non_overlapping_returns, factors)
