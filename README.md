# Quant Risk and Trading Portfolio

Two linked projects on the same ₹1 crore, equal-weighted, 16-stock NSE portfolio. `risk_project` measures its risk: VaR, factor exposure and stress tests. `signal_project` tests two long-only signals, momentum and mean-reversion, with walk-forward validation. Prices come from Kite (the default) or Interactive Brokers.

```
python -m risk_project.run_report [--source ibkr_live]
python -m signal_project.run_backtest [--source ibkr_live]
```

## risk_project

How much can the portfolio lose on a bad day, what drives that risk, and how would it have held up in a crisis?

- **VaR:** historical and parametric, 95% and 99%, over three years of daily returns, backtested with Kupiec's test.
- **Factors:** a Carhart four-factor regression on IIMA's daily factors (market, size, value, momentum).
- **Stress tests:** the 2008 GFC, the 2020 COVID crash, and a hypothetical 3σ factor shock, run through the fitted betas.

**Key findings** (IBKR live data, 2023-10-25 to 2026-10-23):

| | 95% VaR | 99% VaR |
|---|---:|---:|
| Historical | ₹1,33,093 | ₹2,04,174 |
| Parametric | ₹1,34,999 | ₹1,90,932 |

- At 99%, parametric VaR is lower than historical: the usual sign of fat tails.
- The portfolio is mostly a market bet. Market beta is 0.83, with small negative tilts to size, value and momentum (R² 0.84; factor data runs to 2025-12-31).
- Crisis losses are driven by the market factor: ₹24.3 lakh (2008, 139 days), ₹28.3 lakh (COVID, 21 days), ₹15.7 lakh (3σ shock, 21 days).
- All four VaR estimates pass Kupiec's test, in-sample.

**Limitations:**
- factor data ends December 2025;
- parametric VaR assumes normal returns;
- stress losses cover only the factor-explained part;
- the VaR backtest is in-sample;
- Tata Motors is excluded because of its 2025 demerger, with M&M in its place.

## signal_project

Two independent ₹50 lakh long-only books, each trading all 16 stocks:

| Book | Long when | Lookbacks tried | Rebalance |
|---|---|---|---|
| Momentum | trailing return is positive | 3, 6, 9, 12 months | weekly |
| Mean-reversion | trailing return is negative | 3, 5, 10 days | daily |

- **Sizing:** inverse-volatility weights, capped at 17.5% per stock.
- **Risk controls:** a 2σ stop-loss per position, and no new entries while a book is more than 15% below its peak.
- **Walk-forward:** pick the best lookback on 24 months, apply it unchanged to the next 6, then roll forward.

**Key findings** (IBKR live data, out-of-sample 2017-01-05 to 2026-10-23):

| Book | Total return | Sharpe |
|---|---:|---:|
| Momentum | 259.18% | 1.13 |
| Mean-reversion | 104.15% | 0.61 |
| NIFTY 50 | 173.15% | |

- Momentum beat the index and mean-reversion didn't. Both are long-only, so their returns include market beta.
- The lookback choice is fragile: shifting the window boundaries by a few days flipped three of momentum's choices and moved its return noticeably.
- These are results after an October 2026 audit fixed engine bugs (overlapping windows, weight drift, stops). The older Kite results haven't been rerun yet.

**Limitations:**
- the small lookback grids make the selection noisy;
- each test window restarts flat;
- long-only returns include beta;
- paper-trading execution isn't built;
- combined risk across the two books isn't covered.

## Data and checks

- **Prices:** Kite, or IBKR through TWS. Both are read-only market data and never place orders.
- **IBKR bars:** adjusted for splits and bonuses, not dividends. They're cached under `data/cache/ibkr/` (gitignored) and checked against an NSE session calendar. Details: [docs/ibkr_data_provider.md](docs/ibkr_data_provider.md).
- **Validation:** every price panel goes through `shared/validation.py`. A missing ticker or a long gap stops the run.
- **Tests:** offline unit tests run on GitHub Actions on every push.
- **Sharing:** to share the repo, use `python -m tools.export_distribution`, not a ZIP of the folder, because the folder holds credentials.

## Possible extensions

- An out-of-sample VaR backtest.
- Re-running the Kite results on the fixed engine.
- A combined-book risk view.
- Paper-trading execution.
- A long-short variant via F&O.

---

*Code, tests and docs were drafted with AI tools in the loop (Claude Code, Sonnet 5, OpenAI Codex, GPT-6 Astra). The modeling choices and judgment calls are mine, and so are the errors.*
