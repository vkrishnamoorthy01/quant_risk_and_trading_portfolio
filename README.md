# Quant Risk & Trading Portfolio

Two linked projects built to demonstrate applied quant research and risk
engineering: a portfolio risk framework (VaR, factor exposure, stress
testing) and a systematic trading signal (momentum, walk-forward
backtesting, live paper trading). Work in progress — `risk_project` is
under active development; `signal_project` has not yet been started.

## risk_project

### Problem statement

Quantify and stress-test the risk of a ₹1 crore, equal-weighted, 16-stock
NSE portfolio: how much can it lose on a bad day (VaR), what macro/style
factors drive its risk (factor model), and how would it have performed in
past and hypothetical crises (stress testing)?

### Methodology

**Value at Risk.** Two VaR methods are computed side by side, each with
different tradeoffs:

- **Historical VaR** takes the empirical quantile of realized portfolio
  returns directly. It makes no distributional assumption, but is only as
  informative as the historical sample — a calm 3-year window will
  understate tail risk.
- **Parametric (variance-covariance) VaR** assumes portfolio returns are
  normally distributed and derives portfolio variance from the asset
  covariance matrix and portfolio weights (`w' Σ w`), scaled by the normal
  quantile for the target confidence level. This is fast and smooth, but
  understates risk when actual return distributions are fat-tailed —
  common in equity markets, especially during stress.

Both are computed at 95% and 99% confidence over a trailing 3-year daily
return window.

**Factor model.** Portfolio excess returns (return minus the risk-free
rate) are regressed via OLS on IIMA's daily Fama-French-India factors —
Market (MF), Size (SMB), Value (HML), and Momentum (WML) — to decompose
portfolio risk into systematic factor exposures versus idiosyncratic
alpha.

### Limitations and assumptions

- **Factor exposures are as of December 2025, not live.** IIMA's daily
  Fama-French-India dataset runs through 2025-12-31, while the portfolio's
  price data runs closer to the present. The factor regression window is
  therefore capped by factor-data availability, not by a modeling choice —
  betas reflect the portfolio's factor loadings as of the end of that
  window, not its current exposure.
- Parametric VaR assumes normally distributed returns and zero mean daily
  return; both understate tail risk relative to fat-tailed, skewed equity
  return distributions.
- Historical VaR is bounded by whatever regime the trailing 3-year sample
  happened to cover, and will miss tail events outside that window (which
  is exactly why stress testing is run separately against 2008 and 2020).

### Results

TBD — to be filled in once stress testing and VaR backtesting are
complete.

### Future work / production considerations

- **Monthly five-factor robustness check.** `risk_project/data/ashoka_ff5_monthly.csv`
  (Foujdar, Juneja, Kumar, Prabhala, 2026; Ashoka University) provides a
  five-factor monthly dataset (adds RMW and CMA to SMB/HML/WML, plus a
  market factor and risk-free rate) with coverage through May 2026. It has
  not been wired into `factor_model.py` — it's a different frequency
  (monthly vs. daily) and factor set from the primary IIMA model — but is a
  natural candidate for a monthly robustness check on factor loadings, and
  usefully extends coverage past IIMA's Dec-2025 cutoff.
- Refresh the factor regression as IIMA (or an equivalent daily source)
  publishes newer data, to close the vintage gap noted above.
- Extend VaR to a Monte Carlo / historical-simulation hybrid to address the
  normality assumption in parametric VaR.

## signal_project

Not yet started.
