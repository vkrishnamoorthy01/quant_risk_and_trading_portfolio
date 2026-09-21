# Quant Risk and Trading Portfolio

Two linked projects on the same ₹1 crore, 16-stock NSE universe. `risk_project` measures how risky an equal-weighted portfolio is (VaR, factor exposure, stress testing). `signal_project` tests two long-only signals, momentum and mean-reversion, with walk-forward validation and paper trading against live Kite Connect quotes.

`risk_project` is complete. `signal_project`'s modeling pipeline is built, unit-tested, and has produced walk-forward results against production Kite data. Paper-trading execution is planned but not yet built.

```
python -m risk_project.run_report
python -m signal_project.run_backtest
```

## risk_project

**The question.** How much can a ₹1 crore equal-weighted portfolio of 16 NSE stocks lose on a bad day, what drives that risk, and how would it have held up in a crisis?

**VaR.** I compute historical and parametric VaR at 95% and 99% over a trailing 3-year window of daily returns. Historical VaR makes no distributional assumption but only knows what the window contained. Parametric VaR uses the covariance matrix and assumes normal returns, which tends to understate tail risk.

**Factor model.** Portfolio excess returns are regressed on IIMA's daily FF factors (market, size, value) plus a momentum factor: MF, SMB, HML, WML. We work with the Carhart four-factor model, not Fama-French.

**Stress tests.** Three scenarios run through the fitted betas: the 2008 GFC (139 trading days), the 2020 COVID crash (21 days), and a hypothetical shock of 3 standard deviations on each factor, scaled to 21 days by root-time and set against the sign of each beta. Historical factor returns are summed rather than compounded, consistent with the linear model. P&L is Σ(beta × shock) × notional, so it only captures the factor-explained part of a loss, not alpha or stock-specific moves. A scenario is flagged a breach when its loss exceeds parametric 99% VaR. Because the horizons differ (139 vs. 21 vs. 21 days), the three P&L figures shouldn't be ranked against each other.

**VaR backtest.** Each VaR estimate is checked against realized returns using breach rates and Kupiec's proportion-of-failures test. A low p-value means the breach rate is too far from the target, in either direction. A high one means no evidence of miscalibration, not proof the model is right. This test is in-sample.

### Results

Run 2026-08-30, over 2023-08-30 to 2026-08-30 (743 trading days).

| Method | 95% VaR | 99% VaR |
|---|---:|---:|
| Historical | ₹1,31,269 | ₹2,03,524 |
| Parametric | ₹1,33,333 | ₹1,88,576 |

The two agree at 95%. At 99%, parametric is lower than historical, the usual signature of fat tails.

Factor regression covers 2023-08-31 to 2025-12-31 (580 observations), cut short by IIMA's data cutoff. 
| Factor | Beta |
|---|---:|
| $\beta_{MF}$ | 0.834 |
| $\beta_{SMB}$ | -0.335 |
| $\beta_{HML}$ | -0.111 |
| $\beta_{WML}$ | -0.048 |

$\alpha$ (daily): 0.000053. $R^2$: 0.838.

The portfolio is mostly a market bet with modest negative tilts to size, value, and momentum.

| Scenario | P&L impact | Breaches 1-day 99% VaR | Worst factor |
|---|---:|:---:|:---:|
| 2008 GFC (139 days) | -₹24,13,797 | Yes | MF |
| 2020 COVID (21 days) | -₹28,29,409 | Yes | MF |
| Hypothetical 3σ (21 days) | -₹15,51,619 | Yes | MF |

All three breach the 1-day 99% VaR, which is expected: a 21- to 139-day crisis move will almost mechanically exceed a single day's risk estimate, so this is more a sanity check than a stress-test finding. The market factor drives every loss.

All four VaR estimates pass the Kupiec test. Breach rates: 5.11% and 1.08% (historical, 95%/99%), 4.58% and 1.35% (parametric), p-values between 0.37 and 0.89. Since this is in-sample, that shows internal consistency, not out-of-sample validation.

### Limitations

- Tata Motors excluded due to its October 2025 demerger, which complicates the price history. Mahindra & Mahindra substituted to keep auto representation.
- Factor exposures are as of December 2025, not current, capped by IIMA's data.
- Parametric VaR assumes normal returns and zero mean.
- Historical VaR can't see events outside its 3-year window.
- Stress P&L excludes alpha and idiosyncratic returns.
- The VaR backtest is in-sample.

### Next steps

- A rolling out-of-sample VaR backtest.
- A Monte Carlo or historical-simulation hybrid VaR.
- A monthly five-factor robustness check using the Ashoka dataset (Foujdar, Juneja, Kumar, Prabhala, 2026), which runs through May 2026 and is in `risk_project/data/`.
- Refresh the daily factor regression once newer data is available.

## signal_project

**The setup.** The ₹1 crore splits into two independent ₹50 lakh books, one momentum and one mean-reversion. Both trade all 16 stocks, so a stock can be held in one book, both, or neither.

Signals are per stock, not cross-sectional. Sixteen names is too thin a universe for a ranking to mean much, and per-stock signals map directly onto the orders execution needs. Both books are long-only, since NSE cash delivery can't carry an overnight short. That means both inherit market beta, so every result below should be read against the NIFTY 50.

| Book | Rule | Candidate lookbacks | Rebalance |
|---|---|---|---|
| Momentum | Long while trailing return is positive | 3, 6, 9, 12 months | Weekly |
| Mean-reversion | Long while trailing return is negative | 3, 5, 10 trading days | Daily |

**Sizing.** Inverse-volatility weights using trailing 20-day realized vol, refreshed each rebalance. Each stock is capped at 17.5% of its book. Weight above the cap redistributes across the other names, and a name that hits the cap locks there for that redistribution (without the lock, weight cycles back onto a capped name and never converges, caught by a unit test). If every active name in a book hits the cap, the book sits partly in cash rather than breach it.

**Risk controls.** A per-position stop-loss at 2 standard deviations of the 20-day volatility observed at entry, checked daily regardless of rebalance frequency. A book-level halt on new entries once drawdown from peak exceeds 15%, lifted once it recovers to 10%. The halt never force-liquidates existing positions.

**Walk-forward validation.** Train on a rolling 24 months, pick the lookback with the best in-sample Sharpe, freeze it, apply unchanged over the next 6-month test window, then slide forward 6 months and repeat. Each test window starts flat at the prior window's ending NAV, so windows stitch into one continuous out-of-sample curve. Production Kite gives clean daily history back to 2015-01-01, about 11.7 years, which yields 20 rolling test windows per book.

**Two data bugs surfaced along the way.**

1. Fetching 16 tickers across 3 date chunks fires 48 back-to-back calls. Kite throttled some and returned empty instead of raising, which surfaced three layers downstream as zero walk-forward windows. `shared/data.py` now throttles every call and retries with backoff on an empty response.
2. A roughly 20% gap survived that fix. Kite's older daily candles are timestamped at market open (09:15 IST) while newer ones sit at midnight, so two tickers ingested under different conventions don't align on the same trading day. Day candles are now normalized to midnight (minute intervals are left alone, since normalizing those would collapse distinct intraday candles). My first history-depth check missed this because it checked each ticker's completeness in isolation rather than comparing date indices across tickers.

After the fix, 11 weekend dates remain in the data, and each checks out against real NSE events: Union Budget sessions, Diwali Muhurat trading, a make-up session for the 22 Jan 2024 Ram Mandir holiday, and two SEBI-mandated disaster-recovery drills. `checks/check_data_completeness.py` now flags any weekend date not already on that confirmed list, so a new alignment bug would still stand out.

**Execution.** One authenticated Kite Connect client, production, used for historical data. Live paper-trading execution is planned but not built.

**Testing.** Signals, sizing (including the cap-redistribution edge cases), risk controls, the backtest engine, walk-forward logic, and order translation are unit tested on synthetic data; none of it needs a live connection.

### Results

Run 2026-09-08. Stitched out-of-sample window 2017-01-03 to 2026-09-08 (20 rolling 6-month test windows per book).

| Book | Ending NAV (from ₹50L) | Total return | Sharpe (OOS, annualized) |
|---|---:|---:|---:|
| Momentum (weekly) | ₹1,66,06,066 | 232.12% | 1.07 |
| Mean-reversion (daily) | ₹1,01,85,844 | 103.72% | 0.61 |

The NIFTY 50 returned 188.69% over the same window. Since both books are long-only and inherit market beta by construction, this comparison matters: momentum modestly beat the index and mean-reversion clearly did not. How much of either result is signal versus market beta needs a proper factor decomposition, the natural link back to `risk_project`, not just this raw comparison.

**Parameter stability.** The winning lookback changed between consecutive windows 37% of the time, for both signals.

| Signal | Most-selected lookback | Selection counts (of 20 windows) |
|---|---|---|
| Momentum | 12 months | 12mo: 14, 9mo: 4, 6mo: 1, 3mo: 1 |
| Mean-reversion | 5 days | 5d: 11, 3d: 7, 10d: 2 |

Long lookbacks winning for momentum and short ones for mean-reversion makes directional sense. But with only 3-4 candidates and 20 windows, some of that switching is estimation noise rather than a genuine regime change, and I wouldn't lean on it as a strong finding.

### Limitations

- Small candidate grids: window-to-window switching partly reflects noise, especially for momentum at roughly 26 weekly observations per test window.
- In-sample selection frozen for out-of-sample application reduces overfitting risk but doesn't eliminate it.
- Each test window restarts flat, so results differ from what one continuously running book would have produced.
- Long-only means returns include market beta, not just signal.
- Live paper-trading execution isn't built yet.
- Combined risk across the two books isn't addressed here.

### Next steps

- Build and validate live paper-trading execution.
- A combined-book risk view using the `risk_project` framework.
- A continuous parameter search in place of the fixed grids.
- A long-short or market-neutral variant via F&O or SLB.

*This project's code, tests, and documentation were drafted with AI tools in the loop, including Claude Code, Sonnet 5, OpenAI Codex, and GPT-6 Astra. The modeling choices and judgment calls are mine, and so are the errors.*
