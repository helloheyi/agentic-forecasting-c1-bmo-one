# BAA10Y multivariate forecasting (leak-safe covariates)

> **Reference implementation 1 of 4.** Recommended order: [getting_started](../getting_started/) → **S&P 500** → [food CPI](../food_price_forecasting/) → [energy / WTI](../energy_oil_forecasting/) → [BoC rate decisions](../boc_rate_decisions/). Each stands on its own.

The **corporate-credit-markets** reference: a head-to-head comparison of
conventional time-series methods on a daily investment-grade credit-spread
series, all reading the **same leak-safe covariate panel**, plus an LLM-Process
forecaster that can read those covariates in its prompt. It is the template for
evaluated prediction on market series with exogenous covariates.

The headline question:

> Given the same macro/market observations, **which method forecasts short-horizon
> BAA10Y spread changes best — and can an LLM-Process, handed those covariates,
> keep up with gradient boosting?**

**How this differs from the energy/oil reference.** Energy forecasts a
*univariate* price trajectory with news-grounded, code-executing, and adaptive
**agents**. This reference has no agents and no news in its systematic backtest —
it is a clean, reproducible **numerical-methods bake-off across a multivariate
covariate panel**, scored with CRPS and direction metrics.

---

## Forecasting task

The targets are **cumulative BAA10Y spread changes** in basis points, registered
one series per horizon (window `N` in business days):

$$
\Delta^{(N)}_t = 100 \times \left(s_t - s_{t-N}\right)
$$

where \(s_t\) is the FRED `BAA10Y` spread level in percentage points. The factor
of 100 converts the change to basis points.

Forecasting `baa10y_change_{N}b` exactly `N` business days ahead resolves to the
**forward** cumulative spread change over the next `N` business days — a clean
single-marginal forecast at each horizon:

| Target | Horizon | Actionable framing |
|--------|---------|--------------------|
| `baa10y_change_1b` | 1 business day | next-session widening/tightening and credit-risk monitoring |
| `baa10y_change_5b` | 5 business days | weekly credit-risk and relative-value positioning |
| `baa10y_change_21b` | 21 business days | monthly portfolio-risk and credit-regime assessment |

**Frequency:** business (`B`). Spread changes, rather than the spread level,
provide a more suitable target for a conventional-methods comparison.

A positive value means **spread widening**: Moody's Seasoned Baa Corporate Bond
Yield increased relative to the 10-Year Treasury yield. A negative value means
**spread tightening**.

**What's forecastable at daily resolution.** The conditional mean of short-run
spread changes is often near zero, while volatility, tail risk, and widening
risk can change sharply during periods of market stress. A VIX-led
macro/market panel may therefore help most during stressed regimes and at short
horizons.

---

## Methods compared

| Family | Predictors | Covariates? |
|--------|-----------|-------------|
| Naive floor | `LastValuePredictor` | — |
| Classical | `DartsExponentialSmoothingPredictor` (ETS), `DartsKalmanForecasterPredictor`, `DartsAutoARIMAPredictor` | — (univariate) |
| ML regression | `DartsLinearRegressionPredictor`, `DartsLightGBMPredictor` | ✅ optional past covariates |
| LLM-Process | `SampledTrajectoryLLMPredictor` | ✅ optional covariate prompt blocks |

The **LLMP (target)** vs **LLMP + cov** rows are the centerpiece: the covariate
variant serializes labeled covariate-history blocks into the prompt (the
`covariate_series_ids=` passed to `build_baa10y_llmp_sampled_trajectory`), so
their CRPS gap measures whether an LLM can use the same exogenous observations
the ML methods do.

---

## Canonical covariates (when enabled)

| Series ID (registered) | Economic meaning |
|------------------------|------------------|
| `vix_level_l1b` | VIX level, lagged 1 business day |
| `vix_log_ret_1b_l1b` | VIX log return, lagged |
| `ust10y_level_l1b` | 10Y Treasury yield |
| `ust2y10y_spread_l1b` | 2Y–10Y spread |
| `fed_funds_level_l1b` | Fed funds effective rate |
| `cpi_mom_logdiff_l1b` | CPI MoM log-diff |
| `unemployment_rate_l1b` | Unemployment rate |
| `oil_log_ret_1b_l1b` | Oil futures log return |
| `gold_log_ret_1b_l1b` | Gold log return (skipped if FRED series unavailable) |
| `dollar_index_log_ret_1b_l1b` | Broad dollar index log return |
| `nasdaq_log_ret_1b_l1b` | NASDAQ composite log return |

Optional high-yield-credit covariates are defined separately in
`HYOAS_OPTIONAL_COVARIATE_SERIES_IDS`:

| Series ID | Economic meaning |
|-----------|------------------|
| `hyoas_observed_change_1b_bps_l1b` | Observed ICE BofA US High Yield OAS daily change, lagged 1 business day |
| `hyoas_hyg_dgs3_proxy_change_1b_bps_l1b` | HYG–DGS3 duration-based high-yield spread-change proxy, lagged 1 business day |

The observed HYOAS series and its proxy represent related high-yield credit-risk
information; they should not be treated as independent signals.

Exact adapters and transforms live in `data.py` (`DEFAULT_COVARIATE_SERIES_IDS`).
Yahoo covariates use `YFinanceDailyAdapter` (parquet under `data/yfinance/` at the
repo root); FRED series use `FREDAdapter` (`data/fred/`). Warm both caches to the
present before running the 2025/2026 windows (see Prerequisites).

---

## Cutoff-aware evaluation (read this)

This is the methodological heart of the comparison, and easy to get wrong.

- **Numerical methods are cutoff-safe by construction.** Naive, ETS, Kalman,
  AutoARIMA, LinReg and LightGBM only ever see the series up to the forecast
  origin (`ForecastContext` enforces it), so they can be backtested on *any*
  historical window.
- **An LLM is not.** Gemini's training cutoff is ~**January 2025**, so it has
  effectively memorised pre-2025 outcomes. Scoring an LLM-Process on a pre-cutoff
  origin measures recall, not forecasting, and silently flatters it in the
  head-to-head.

So the LLM-inclusive comparison lives **after the cutoff** — a **2025 backtest**
for iteration and a **protected 2026 eval** as the honest scoreboard (mirroring
the energy reference and `getting_started`'s `backtest()` → `evaluate()` split).
The 2020 COVID window is kept as a **numerical-only** stress test.

---

## No-leakage design

- Every covariate is shifted by **one business day** before registration.
- Macro series use **conservative release proxies** before daily expansion;
  rows carry `released_at` suitable for `ForecastContext` cutoffs.
- Backtests enforce **information available at `as_of`**.

Missing optional feeds are **skipped with warnings** by default
(`strict_covariates=False`). Set `strict_covariates=True` to fail fast.

---

## Specs — windows and tasks (experiment design only)

Four co-located YAML configs. Each spec carries **only the experiment design** —
the window (`start`/`end`/`stride`/`warmup`) and one single-horizon task per
`sp500_logret_{N}b` target (`horizons: [N]`, `frequency: B`). The first three are
`MultiTargetBacktestSpec`; the eval spec is a `MultiTargetEvalSpec` that adds
`max_runs`. **Which predictors run, and all their hyperparameters (including the
covariate panel), live in the notebook — not the spec.**

```text
specs/
├── baa10y_smoke.yaml         # fast late-2025 smoke run
├── baa10y_backtest_2025.yaml # main weekly 2025 comparison
├── baa10y_eval_2026.yaml     # protected held-out 2026 evaluation
└── baa10y_stress_2020.yaml   # COVID stress test; numerical methods only
```

The notebook runs the 2025 backtest (Section 5) and the protected 2026 eval
(Section 7); set `EXPERIMENT_CONFIG = "stress_2020"` to study the volatile regime
with the cutoff-safe methods (the predictors cell drops the LLMP rows
automatically). Copy a spec and edit the window/tasks to pose a new study.

---

## Module layout

```text
implementations/BAA10Y_forecasting/
├── data.py                    # BAA10Y spread-change targets and covariate IDs
├── predictors/                # BAA10Y LLMP recipe
├── leaderboard.py             # cached results → RESULTS_DF; forecast-vs-actual frame
├── analysis.py                # direction metrics and styled leaderboards
├── plots.py                   # target history, CRPS, forecast-vs-actual charts
├── starter_agent/             # optional news-search/code-execution agent
├── specs/                     # smoke, backtest, evaluation, and stress specs
├── 00_baa10y_data_exploration.ipynb
├── 01_BAA10Y_multivariate_backtest.ipynb
├── 99_starter_agent.ipynb
└── README.md
```

Unit tests for data helpers live under
`implementations/tests/sp500_forecasting/test_data.py`.

---

## Adding a method

The roster is meant to grow, and it's all just code now — no registry or dispatch
to edit. In the notebook's predictors cell:

1. Instantiate any `Predictor` and append it to `all_predictors`. For a new Darts
   model, mirror `aieng-forecasting/aieng/forecasting/methods/numerical/darts_classical.py`
   (univariate, probabilistic via `num_samples`, per-horizon quantiles) and export
   it from `methods/numerical/__init__.py` and `methods/__init__.py` first.
2. Add a `PREDICTOR_LABELS` entry (the leaderboard "model" column). If it reads the
   covariate panel, also add a `PREDICTOR_COVARIATES` entry so the leaderboard's
   covariate columns are correct.

For a tuned LLM-Process variant, add a builder to `predictors/` (mirror
`predictors/llmp_sampled_trajectory.py`) so the prompt framing is reusable.

Keep numerical models **fast** (sub-second per origin) and **probabilistic** (CRPS
needs a distribution — deterministic models like Theta need a conformal/residual
wrapper first).

---

## Prerequisites

From the **repository root**, run `uv sync` once so `BAA10Y_forecasting` is on the
interpreter path (same pattern as `food_price_forecasting` / `energy_oil_forecasting`).
Use the project `.venv` as the Jupyter kernel — imports are `from BAA10Y_forecasting import ...`.

Warm caches at the repo root (gitignored) to the **present** — the 2025/2026
windows need coverage through today:

```bash
uv run python scripts/fetch_sp500_market.py --refresh   # ^GSPC / ^VIX / ^IXIC (Yahoo)
uv run python scripts/fetch_fred.py                     # macro covariates (FRED)
```

`fetch_fred.py` requires a **FRED API key** in your repo-root `.env` (`FRED_API_KEY=...`).
FRED keys are free but must be requested individually — **we cannot provide one for you**.
Request yours at <https://fred.stlouisfed.org/docs/api/api_key.html> (approval is usually
quick, but allow some time). A description like "Requesting an API key to explore the
effectiveness of various forecasting techniques on economic data." works well.

The `llmp_*` rows call the Vector proxy, so a populated repo-root `.env` (with
`OPENAI_BASE_URL` / `OPENAI_API_KEY`) is required when those rows are enabled.

**How to run:** begin with `00_baa10y_data_exploration.ipynb`, then open
`01_BAA10Y_multivariate_backtest.ipynb` and run the `"smoke"` configuration.
The protected 2026 evaluation runs in Section 7.

---

