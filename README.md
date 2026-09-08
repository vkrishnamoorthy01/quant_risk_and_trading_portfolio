# Quant Risk & Trading Portfolio

Two linked projects built to demonstrate applied quant research and risk
engineering: a portfolio risk framework (VaR, factor exposure, stress
testing) and a systematic trading signal (momentum and mean-reversion,
walk-forward backtesting, live sandbox paper trading). `risk_project` is
complete. `signal_project`'s modeling pipeline (signals, sizing, risk
controls, backtest engine, walk-forward validation, order-translation
logic) is built and unit-tested; the live sandbox connection and an
actual backtest run against production data are both pending (see
`signal_project`'s Limitations section).

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

Produced by `python -m risk_project.run_report` on 2026-08-30, over the
trailing 3-year price window 2023-08-30 to 2026-08-30 (743 trading days).

**Value at Risk** (₹1 crore notional):

| Method     | 95% VaR   | 99% VaR   |
|------------|----------:|----------:|
| Historical | ₹131,269  | ₹203,524  |
| Parametric | ₹133,333  | ₹188,576  |

The two methods agree closely at 95%, but diverge at 99%: parametric VaR
(₹188,576) is *lower* than historical VaR (₹203,524) at the 99% level,
consistent with the normality assumption understating tail risk relative
to the fatter-tailed empirical return distribution (see Limitations).

**Factor model.** Regression window: 2023-08-31 to 2025-12-31, **580
observations** — truncated from the full 743-day price window by IIMA's
factor-data cutoff at end-2025, as documented above.

| Factor | Beta    |
|--------|--------:|
| MF     |  0.834  |
| SMB    | -0.335  |
| HML    | -0.111  |
| WML    | -0.048  |

Alpha (daily): 0.000053. R²: **0.838** — the portfolio is overwhelmingly
market-driven (beta ≈ 0.83 on MF), with modest negative tilts to size,
value, and momentum, and a high proportion of return variance explained
by the four factors.

**Stress testing:**

| Scenario                                    | P&L Impact  | VaR Breach | Worst Factor |
|----------------------------------------------|------------:|:----------:|:------------:|
| 2008 GFC (139 trading days)                   | -₹24,13,797 | Yes        | MF           |
| 2020 COVID (21 trading days)                  | -₹28,29,409 | Yes        | MF           |
| Hypothetical 3σ shock (21-day horizon)        | -₹15,51,619 | Yes        | MF           |

All three scenarios breach the parametric 99% VaR (₹188,576) — expected
and, in fact, the point of running them: a multi-week crisis move is
naturally an order of magnitude larger than a single-day 99% VaR estimate
drawn from a comparatively calm 3-year sample. The market factor (MF)
dominates every scenario's loss, consistent with the portfolio's high
market beta.

**VaR backtesting.** All four VaR estimates pass the Kupiec test — no
evidence of miscalibration over the backtest window:

| Method     | Confidence | N   | Breaches | Breach Rate | Expected Rate | Kupiec LR | p-value | Reject |
|------------|-----------:|----:|---------:|------------:|---------------:|----------:|--------:|:------:|
| Historical | 95%        | 743 | 38       | 5.11%       | 5%             | 0.020     | 0.887   | No     |
| Historical | 99%        | 743 | 8        | 1.08%       | 1%             | 0.043     | 0.836   | No     |
| Parametric | 95%        | 743 | 34       | 4.58%       | 5%             | 0.289     | 0.591   | No     |
| Parametric | 99%        | 743 | 10       | 1.35%       | 1%             | 0.810     | 0.368   | No     |

This is a clean result: breach rates land close to their targets across
both methods and both confidence levels, and Kupiec p-values (0.37-0.89)
are all comfortably above the 0.05 rejection threshold — the data gives
no reason to doubt either VaR method's calibration over this window. (As
noted above, this is an in-sample backtest, so "clean" here means
internally consistent, not validated out-of-sample.)

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

### Problem statement

Build two independent, per-stock time-series trading signals (momentum and
mean-reversion) on the same ₹1 crore, 16-stock NSE universe used in
`risk_project`, run them as two parallel ₹50 lakh books with
volatility-scaled position sizing, validate signal-parameter choices via
rolling walk-forward testing, and execute live paper trades through Kite
Connect's sandbox.

### Methodology

**Universe, notional, and book structure.** Same 16-stock universe as
`risk_project`. The ₹1 crore total notional is split 50/50 into two
independently-run books — momentum and mean-reversion — that both trade
all 16 stocks independently: a stock can carry a position in one book,
both, or neither, purely on what each book's own signal says.

**Time-series, not cross-sectional, signals.** Both signals are computed
per-stock (each stock's own trend/reversion state), not as a
cross-sectional ranking (long top decile / short bottom decile) across
the universe. Sixteen names is too thin a cross-section for a
ranking-based long-short spread to be statistically meaningful, and
per-stock signals map directly onto the per-instrument orders execution
needs.

**Signal definitions.** Both follow the standard sign-based time-series
convention (position = sign of trailing return), restricted to long-only
by dropping the short leg:

- **Momentum** (`signal_project/signals.py::momentum_signal`): long a
  stock while its trailing return over the lookback is positive.
  Candidate lookbacks: 3, 6, 9, 12 months. Rebalanced weekly.
- **Mean-reversion** (`momentum_signal`'s contrarian counterpart,
  `mean_reversion_signal`): long a stock while its trailing return over
  the lookback is *negative* (expecting reversion), i.e. the sign-flipped
  version of the same rule. Candidate lookbacks: 3, 5, 10 trading days.
  Rebalanced daily.

**Long-only, both signals.** A "sell" signal exits an existing long back
to flat and never opens a new short. This is driven by execution
reality, not a strategy preference: NSE cash-segment delivery trades
cannot carry an overnight short. Long-only momentum and long-only
reversal are consequently different from their canonical
long-short/market-neutral counterparts in the literature — those hedge
out general market exposure to isolate the pure factor, while these
long-only books inherit market beta (see Limitations).

**Position sizing** (`signal_project/sizing.py`). Inverse-volatility
weighting within each book: lower-vol stocks get larger positions,
higher-vol stocks smaller, for a given signal strength, using trailing
20-day realized vol refreshed at every rebalance. This is a separate,
continuously-updated calculation, not part of the walk-forward parameter
search below.

*Position cap:* each stock is capped at 17.5% of that book's capital
(~2.8x its 6.25% equal-weight share at 16 stocks), so the signal's
selection effect and the vol-scaling effect can't compound into a
concentrated bet on a handful of low-vol names. Mechanically: compute
inverse-vol weights, clip anything above the cap, and redistribute the
clipped excess proportionally across the remaining uncapped names —
iteratively, since redistributing excess can itself push a previously
uncapped name over the cap. A name that hits the cap is locked there for
the rest of that redistribution (without the lock, weight can cycle back
onto an already-capped name and never converge — caught by a unit test
before it became a live bug). If every active name in a book ends up
locked at the cap simultaneously (few active names, low cap), the book
is left partially in cash rather than breaching the cap — the cap is a
concentration control and takes priority over full investment in that
edge case.

**Walk-forward validation** (`signal_project/walk_forward.py`). Tests
whether the "best" lookback for each signal is stable across time or an
artifact of a particular window:

| Parameter | Value |
|---|---|
| Training window length | 24 months, rolling (not expanding) |
| Test window / step size | 6 months |
| Selection process | Evaluate every candidate lookback in-sample on Sharpe ratio over the training window; freeze the winner; apply it unchanged over the following 6-month out-of-sample test window |
| Roll forward | Slide the training window forward by 6 months; repeat |

The spec flagged a risk here: 24-month train + 6-month step needs
meaningfully more than 2.5-3 years of history to produce more than one or
two windows. `checks/check_history_depth.py` (a standalone, one-off
script, not part of the pipeline) confirmed production Kite actually
gives clean daily history back to **2015-01-01** — about 11.7 years, no
gaps beyond ordinary weekends/holidays — so `run_backtest.py` uses the
full 2015-to-present history rather than the 3-year window
`risk_project` uses for VaR. That's roughly 19 rolling windows, enough to
say something about parameter stability rather than just demonstrate the
mechanism on one or two.

That full-history fetch surfaced two real bugs during development, in
sequence:

1. **A rate-limit issue.** Pulling 16 tickers across 3 date chunks each
   fires 48 back-to-back `historical_data` calls, enough to hit Kite's
   rate limit; a throttled call returns empty instead of raising, and the
   original chunking code treated that as "no data for this chunk." This
   propagated three layers downstream into zero walk-forward windows and
   an opaque `pd.concat` error inside `walk_forward.py`, nowhere near the
   actual cause. Fixed in `shared/data.py` by throttling every call and
   retrying with backoff on an empty response.
2. **The actual root cause of the ~20% gap that remained even after fix
   #1** (identical missing rows regardless of retry strategy — the tell
   that it wasn't transient): Kite's "day" candles aren't timestamped
   consistently across history. Older data comes back at market-open time
   (09:15 IST) rather than midnight. Left un-normalized, two tickers whose
   data was ingested under different conventions end up with different
   timestamps for what's really the same trading day — e.g. `2015-02-28
   09:15:00` for one ticker and `2015-02-28 00:00:00` for another — so
   combining them creates a wall of spurious gaps rather than aligning the
   dates. This is also why the original `check_history_depth.py` run
   looked clean with 2 tickers: it only checked each ticker's own
   completeness in isolation and never compared their date indices to
   each other, so a cross-ticker misalignment wouldn't have shown up
   until more tickers were combined. Fixed by normalizing "day" interval
   timestamps to midnight before indexing (minute intervals are
   deliberately left alone — normalizing those would collapse distinct
   intraday candles onto the same timestamp).

Both fixes are independently useful and both stayed in: the rate-limit
throttle/retry is real, general-purpose defensive practice regardless of
whether it was the cause of this particular gap, and the loud
`UserWarning` on a residual gap (from fix #1) still catches whatever this
timestamp fix doesn't. `checks/check_data_completeness.py` is a fast,
standalone sanity check on exactly this (index health, per-ticker gap
fractions) worth running before a full walk-forward pass.

Each out-of-sample test window is simulated as an independent,
freshly-flat book — starting at the prior window's ending NAV rather than
carrying open positions or stop/halt state across the train/test
boundary — so windows can be evaluated and stitched into one continuous
OOS equity curve independently of each other. This is a deliberate
simplification (see Limitations), not an attempt to replicate exactly
what a continuously-running book would have done across window
boundaries.

**Risk controls** (`signal_project/risk_controls.py`), both reusing
estimates the rest of the pipeline already computes rather than
introducing disconnected parameters:

- *Per-position stop-loss:* vol-scaled, not a flat percentage — exit a
  position if it moves against the entry price by more than 2 standard
  deviations of the same trailing 20-day realized vol used for sizing,
  fixed at the vol estimate observed when the position was opened.
  Checked every day regardless of the signal's own rebalance frequency,
  so a weekly-rebalanced momentum position can still be stopped out
  intraweek.
- *Book-level drawdown halt:* pause new entries in a book once its NAV
  drawdown from peak exceeds 15%; resume once drawdown recovers to 10% or
  better. Existing positions are not force-liquidated — the halt blocks
  new entries only. "New entry" is interpreted here as opening a position
  in a ticker currently at zero weight; resizing or exiting an existing
  position is unaffected by the halt.

**Execution: Kite Connect sandbox paper trading**
(`signal_project/execution.py`). Two independent, separately-authenticated
Kite clients are used across the project: `shared.data.get_kite_client()`
against production (`api.kite.trade`) for historical data, and
`get_sandbox_kite_client()` against the sandbox (`sandbox.kite.trade`) for
order placement — a design constraint from the spec, kept even though the
sandbox uses shared public demo credentials (`sandboxdemo` /
`sandboxdemo-secret`, published by Zerodha as safe to use in plain text)
rather than a personal registered app. The sandbox client requires an
`/oms` prefix patched onto every SDK route except the instrument dumps,
and only accepts resting `LIMIT` orders — no `MARKET` order type, and an
order priced too far from LTP is auto-rejected by the exchange price-band
check. `execution.py` prices every order a small, explicit margin off the
latest quote (BUY below LTP, SELL above) to stay inside that band, and
enforces long-only at the order-translation level: a reduction never
sells more shares than are currently held, since there is no short leg to
open.

### Limitations and assumptions

- **Small candidate grids.** Four candidates for momentum and three for
  mean-reversion are enough to demonstrate the walk-forward mechanism
  working, but window-to-window "winner changes" will partly reflect
  estimation noise rather than genuine regime change, especially for
  momentum at roughly 26 weekly observations per 6-month test window.
  Read the walk-forward results in the Results section below with that in
  mind, rather than over-claiming a parameter-stability finding.
- **In-sample selection, frozen out-of-sample application** reduces but
  does not eliminate overfitting risk — the selection step itself is
  still an in-sample choice, just not re-touched during the test window.
- **Each out-of-sample test window restarts flat**, rather than carrying
  a continuously-running book's open positions and stop/halt state across
  the train/test boundary (see Methodology). This keeps windows
  independently evaluable and stitchable, at the cost of not exactly
  reproducing what one continuously-running book would have experienced.
- **Long-only inherits market beta.** Because both books are long-only
  (execution-constrained, not a design choice), reported returns are not
  purely attributable to the signal; a rising market over the backtest
  period lifts both books regardless of signal quality. The Results
  section below reports the NIFTY 50's return over the same out-of-sample
  window alongside each book's result for this reason.
- **Live paper-trading execution risk** (API stability, order rejects,
  corporate actions/dividends over an extended history) is a different
  risk category from a pure backtest.
- **Sandbox IP-allowlisting is unconfirmed.** Kite's sandbox
  documentation does not state whether the static-IP allowlisting
  required for production order placement also applies to
  `sandbox.kite.trade`. `execution.py` and `check_sandbox_order.py` were
  built and tested (via a mocked Kite client — see Testing) without a
  live sandbox connection, because sandbox login access was blocked by a
  Zerodha helpdesk ticket during development. Whether IP allowlisting
  applies, and whether the order-placement flow works end-to-end against
  the real sandbox, is confirmed by running `check_sandbox_order.py` once
  access clears, not by anything in this codebase today.
- **No live-runner reconciliation script.** `execution.py` provides the
  tested order-translation and sandbox-client primitives (diff target
  weights against current holdings into long-only LIMIT orders), but a
  daily/weekly script that pulls real sandbox positions and NAV,
  reconciles them against the frozen walk-forward lookback's live signal,
  and calls `execution.py` to act was deliberately not built yet. Writing
  that reconciliation logic against `kite.positions()` / `kite.margins()`
  response shapes we have never actually observed (sandbox access being
  blocked) risks encoding wrong assumptions about the sandbox's own P&L
  accounting; it belongs after the first successful `check_sandbox_order.py`
  run, not before.
- Two independently-sized books sharing one notional pool means combined
  portfolio-level risk (e.g., correlation between the two books' P&L) is
  not addressed here — a natural extension linking back to `risk_project`.

### Testing

Signals, sizing (including the cap-redistribution edge cases), risk
controls, the day-by-day backtest engine, walk-forward window generation
and selection, and order-translation logic are covered by unit tests on
synthetic data (`tests/test_signals.py`, `test_sizing.py`,
`test_risk_controls.py`, `test_backtest.py`, `test_walk_forward.py`,
`test_execution.py`) — none of them hit live Kite, matching
`risk_project`'s testing convention. `get_sandbox_kite_client()` and the
sandbox HTTP callback are the one piece that can only be verified live;
`checks/check_sandbox_order.py` is that live smoke test, run manually
once sandbox access is available. `checks/` holds standalone, one-off
verification scripts like this one — not part of the pipeline itself, and
each runnable independently of the others — including
`check_data_completeness.py`, a fast sanity check on index health and
per-ticker gap fractions over the full fetch, worth running before a full
walk-forward pass rather than after one fails on bad data (see the
rate-limit gap caught and fixed during development, above).

### Results

Pending an actual `python -m signal_project.run_backtest` run against
production Kite data (blocked on the same-day interactive login, not on
anything in this codebase).

### Future work

- The live-runner reconciliation script described in Limitations, once
  the sandbox connection is confirmed working.
- Combined-book risk view linking `signal_project`'s two books back to
  `risk_project`'s factor/VaR framework.
- Extend beyond the fixed candidate grids to a continuous parameter
  search, once the fixed-grid version is validated.
- A genuine long-short/market-neutral variant of either signal, via F&O
  or SLB, as a natural extension once the long-only version is validated.

*Note: this project's code, tests, and documentation were drafted with AI
tools in the loop — Claude Code, Sonnet 5, OpenAI Codex, and GPT-6 Astra,
among them. The modeling choices and judgment calls are mine, and so are
the errors.*
