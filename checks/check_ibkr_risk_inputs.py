"""One-off check: build the risk portfolio's price input from IBKR and run VaR on it.

Run from the repo root (D:\\Personal_projects\\quant_risk_and_trading_portfolio),
with TWS running and logged in:
    python -m checks.check_ibkr_risk_inputs                      # paper

Downloads (or reuses from data/cache/ibkr/) three years of daily bars for
every PORTFOLIO ticker, builds the close-price frame with
data_providers.ibkr_prices, and computes historical and parametric VaR the
same way risk_project.run_report does. It only prints: nothing is written to
risk_project/results/, so the Kite-based report files are left alone.
Market data only: nothing here places or modifies orders.
"""

from __future__ import annotations

from argparse import ArgumentParser
from logging import INFO, WARNING, basicConfig, getLogger

from pandas import DateOffset, Timestamp

from data_providers.ibkr_prices import IBKRInsufficientDataError, fetch_ibkr_price_data
from risk_project.portfolio import PORTFOLIO
from risk_project.var_models import historical_var, parametric_var, portfolio_returns
from shared.data import compute_returns

CONFIDENCE_LEVELS = (0.95, 0.99)  # same as run_report

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment", choices=["paper", "live"], default="paper")
    parser.add_argument("--years", type=int, default=3)
    args = parser.parse_args()

    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # ib_async logs startup positions at INFO with the full, unmasked account ID.
    getLogger("ib_async").setLevel(WARNING)

    end = Timestamp.today().normalize()
    start = (end - DateOffset(years=args.years)).date().isoformat()
    end = end.date().isoformat()

    try:
        prices = fetch_ibkr_price_data(PORTFOLIO.tickers, start, end, environment=args.environment)
    except IBKRInsufficientDataError as exc:
        raise SystemExit(f"\nFAILED (as designed, rather than feeding gaps to the engine):\n{exc}")

    print(f"\nPrices: {prices.shape[0]} sessions x {prices.shape[1]} tickers, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}, index '{prices.index.name}' {prices.index.dtype}")
    print(f"Columns: {list(prices.columns)}")
    print(f"Any NaN: {prices.isna().any().any()}; provenance: {prices.attrs}")

    returns = compute_returns(prices, method="simple")
    port_returns = portfolio_returns(returns, PORTFOLIO.weights)
    cov_matrix = returns.cov()
    print(f"\n=== Value at Risk from IBKR prices ({len(port_returns)} daily returns) ===")
    for confidence in CONFIDENCE_LEVELS:
        hist_var = historical_var(port_returns, confidence, PORTFOLIO.notional)
        param_var = parametric_var(PORTFOLIO.weights, cov_matrix, confidence, PORTFOLIO.notional)
        print(f"  {int(confidence * 100)}% Historical VaR:  Rs {hist_var:,.0f}")
        print(f"  {int(confidence * 100)}% Parametric VaR:  Rs {param_var:,.0f}")
