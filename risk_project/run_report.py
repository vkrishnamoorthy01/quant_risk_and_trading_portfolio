"""End-to-end risk report: fetch prices, run VaR/factor/stress/backtest
modules in sequence, and print a summary to fill in the README's Results
section.

Run this yourself from the repo root:

    python -m risk_project.run_report

The first Kite Connect call each day needs an interactive login: it prints
a login URL, and after you log in and get redirected, you paste back the
`request_token` from the redirect URL when prompted. The access token is
then cached for the rest of the day (see shared/data.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_project.backtesting import backtest_var, build_backtest_report
from risk_project.factor_model import fit_factor_model, load_daily_factors
from risk_project.portfolio import PORTFOLIO
from risk_project.stress_testing import (
    COVID_WINDOW,
    GFC_WINDOW,
    build_scenario_report,
    historical_factor_shock,
    hypothetical_factor_shock,
    run_scenario,
)
from risk_project.var_models import historical_var, parametric_var, portfolio_returns
from shared.data import compute_returns, fetch_price_data

RESULTS_DIR = Path(__file__).resolve().parent / "results"
VAR_LOOKBACK_YEARS = 3
CONFIDENCE_LEVELS = (0.95, 0.99)


def _var_window() -> tuple[str, str]:
    """Trailing VAR_LOOKBACK_YEARS window ending today, as ISO date strings."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=VAR_LOOKBACK_YEARS)
    return start.date().isoformat(), end.date().isoformat()


def main() -> None:
    start, end = _var_window()

    print(f"Fetching {len(PORTFOLIO.tickers)} tickers from {start} to {end} via Kite Connect...")
    prices = fetch_price_data(PORTFOLIO.tickers, start, end)

    # Simple (not log) returns: portfolio return = weighted sum of asset
    # returns only holds exactly for simple returns.
    returns = compute_returns(prices, method="simple")
    port_returns = portfolio_returns(returns, PORTFOLIO.weights)
    cov_matrix = returns.cov()

    print("\n=== Value at Risk (trailing 3-year window) ===")
    var_estimates: dict[tuple[str, float], float] = {}
    for confidence in CONFIDENCE_LEVELS:
        hist_var = historical_var(port_returns, confidence, PORTFOLIO.notional)
        param_var = parametric_var(PORTFOLIO.weights, cov_matrix, confidence, PORTFOLIO.notional)
        var_estimates[("Historical", confidence)] = hist_var
        var_estimates[("Parametric", confidence)] = param_var
        print(f"  {int(confidence * 100)}% Historical VaR:  Rs {hist_var:,.0f}")
        print(f"  {int(confidence * 100)}% Parametric VaR:  Rs {param_var:,.0f}")

    print("\n=== Factor Model ===")
    factors = load_daily_factors()
    factor_result = fit_factor_model(port_returns, factors)
    print(
        f"  Window: {factor_result.window_start.date()} to {factor_result.window_end.date()} "
        f"({factor_result.n_obs} obs)"
    )
    print(f"  Alpha (daily): {factor_result.alpha:.6f}")
    print("  Betas:")
    print(factor_result.betas.to_string())
    print(f"  R-squared: {factor_result.r_squared:.4f}")

    print("\n=== Stress Testing ===")
    var_99_parametric = var_estimates[("Parametric", 0.99)]
    scenarios = []
    for name, window in [("2008 GFC", GFC_WINDOW), ("2020 COVID", COVID_WINDOW)]:
        shock = historical_factor_shock(factors, *window)
        scenarios.append(
            run_scenario(name, factor_result.betas, shock, PORTFOLIO.notional, var_99_parametric)
        )

    hypothetical_shock = hypothetical_factor_shock(factors, factor_result.betas, start, end, n_sigma=3.0)
    scenarios.append(
        run_scenario(
            "Hypothetical 3-sigma shock",
            factor_result.betas,
            hypothetical_shock,
            PORTFOLIO.notional,
            var_99_parametric,
        )
    )
    scenario_report = build_scenario_report(scenarios)
    print(scenario_report.to_string())

    print("\n=== VaR Backtesting ===")
    backtest_results = [
        backtest_var(port_returns, var_estimates[(method, confidence)], confidence, method, PORTFOLIO.notional)
        for method in ("Historical", "Parametric")
        for confidence in CONFIDENCE_LEVELS
    ]
    backtest_report = build_backtest_report(backtest_results)
    print(backtest_report.to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    scenario_report.to_csv(RESULTS_DIR / "scenario_report.csv")
    backtest_report.to_csv(RESULTS_DIR / "backtest_report.csv")
    print(f"\nSaved scenario_report.csv and backtest_report.csv to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
