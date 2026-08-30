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

**Stress testing.** Three scenarios are run through the fitted factor
betas and reported in a common P&L-impact / VaR-breach / worst-factor
table:

- **2008 GFC** (2008-09-01 to 2009-03-31: Lehman Brothers' collapse through
  the approximate global equity market trough) and **2020 COVID crash**
  (2020-02-20 to 2020-03-23: pre-selloff peak through the NIFTY 50's
  trough) replay realized factor returns from the IIMA dataset over each
  window. Daily factor returns are *summed* rather than compounded — this
  is the aggregation consistent with the linear factor model (excess
  return = alpha + Σ beta·factor + residual), since it preserves a clean
  per-factor decomposition when multiplied through by the static betas.
  It ignores compounding effects and the alpha/idiosyncratic contribution
  to the portfolio's actual historical P&L — the reported scenario impact
  is the portion explained by systematic factor exposure alone.
- **Hypothetical factor shock (21-day horizon)**, calibrated statistically
  rather than by picking arbitrary numbers: each of MF, SMB, HML, and WML
  is shocked by **3 standard deviations of its own historical daily
  volatility**, estimated over the same trailing 3-year window used for
  the VaR analysis, then scaled to a cumulative 21-trading-day
  (~1-month) move via sqrt-time scaling (`daily_std * sqrt(21)`) —
  consistent with the summed, not compounded, aggregation used for the
  historical scenarios above. Each factor's shock direction is adverse to
  this portfolio — opposite the sign of its estimated beta. The 21-day
  horizon was chosen to match the COVID window's length, so that scenario
  is horizon-comparable; **the GFC window is 139 trading days (~6.5
  months) and is a longer, structurally different sustained-drawdown
  event, not a sharp shock, so it remains a different horizon from the
  hypothetical scenario even after this fix.** An earlier version of this
  shock used a single day's volatility with no horizon scaling, which
  understated it by roughly √21 ≈ 4.6x relative to the multi-day historical
  scenarios — a horizon mismatch, not a difference in underlying severity.

For all three scenarios, P&L impact is Σ(beta_i × shock_i) × notional, and
a scenario is flagged as a **VaR breach** when its implied loss exceeds
the parametric 99% VaR. Because the three scenarios span different
horizons (139 trading days, 21 trading days, and 21 trading days
respectively), their P&L impacts are not on equal footing and should be
read as horizon-specific rather than directly ranked against each other.

**VaR backtesting.** Each VaR estimate (historical and parametric, at 95%
and 99%) is checked against the same trailing 3-year window's realized
daily returns:

- **Breach count / breach rate** — the fraction of days where the realized
  loss exceeded the VaR estimate, compared against the rate the model
  targets (5% for 95% VaR, 1% for 99% VaR).
- **Kupiec's proportion-of-failures (POF) test** — a likelihood-ratio test
  of whether the observed breach rate is statistically consistent with the
  target rate. The statistic is asymptotically χ² with 1 degree of
  freedom under the null hypothesis that the model is correctly
  calibrated. **Interpretation:** a low p-value (conventionally < 0.05)
  rejects that null — the model's breach rate is too far from what it
  claims, in either direction (too many breaches means the model
  understates risk; too few can mean it's overly conservative). A
  high p-value means the observed breach rate is not statistically
  distinguishable from the target — the test finds no evidence the model
  is miscalibrated, which is a pass, not proof the model is correct.

This is an **in-sample** backtest — the VaR estimate and the returns it's
tested against cover the same window — rather than a rolling
out-of-sample comparison (see Limitations).

### Limitations and assumptions

- Tata Motors was excluded from the portfolio universe due to its October
  2025 demerger (renaming to TMPV, with the commercial vehicle business
  spun off as a separate listed entity), which would have complicated
  continuous historical data over the 3-year window. Mahindra & Mahindra
  was substituted to preserve auto-sector representation.
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
- Stress scenario P&L is computed purely through factor betas (Σ beta ×
  shock × notional). Historical scenarios sum rather than compound daily
  factor returns (see Methodology above), and neither historical nor
  hypothetical scenarios include the portfolio's alpha or idiosyncratic
  (non-factor) return component — actual historical P&L in a real crash
  would also reflect stock-specific effects the factor model doesn't
  capture.
- VaR backtesting here is in-sample (the same 3-year window produces both
  the VaR estimate and the returns it's tested against), which is
  optimistic relative to a true out-of-sample test: a rolling backtest
  (re-estimating VaR daily/monthly and checking only forward-looking
  breaches) would be a more rigorous validation and is noted under future
  work.

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
- Replace the in-sample VaR backtest with a rolling out-of-sample backtest
  (re-estimate VaR on a trailing window, check only forward breaches) for
  a more rigorous validation than the current in-sample Kupiec test.

## signal_project

Not yet started.
