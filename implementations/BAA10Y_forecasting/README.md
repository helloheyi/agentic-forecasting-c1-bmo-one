# BAA10Y multivariate forecasting 

A forecasting and credit-market analysis implementation for BAA10Y, the FRED
series measuring Moody's Seasoned Baa Corporate Bond Yield relative to the
10-Year Treasury Constant Maturity Rate. The main benchmark compares classical
time-series models, machine-learning models, and sampled LLM forecasts.
Separate notebooks demonstrate a configurable forecast agent, a credit-market
analyst agent, and an adaptive agent that uses Optuna to tune selected methods.

---

## Forecasting task

The targets are **cumulative changes in BAA10Y**, the spread between Moody's
Seasoned Baa Corporate Bond Yield and the 10-Year Treasury Constant Maturity
Rate. FRED quotes the spread in percentage points; targets are converted to
**basis points**:

$$
\Delta S_t^{(N)} = 100\left(S_t-S_{t-N}\right)
$$

Forecasting `baa10y_change_{N}b` exactly `N` business days ahead resolves to the
**forward cumulative spread change** over that window. Each task is scored at its matching horizon; forecasts of the already-cumulative
target must not be summed again.

| Target | Horizon | Interpretation |
|--------|---------|----------------|
| `baa10y_change_1b` | 1 business day | Next-day spread change and direction |
| `baa10y_change_5b` | 5 business days | Forward one-week spread change |
| `baa10y_change_21b` | 21 business days | Forward one-month spread change |

**Positive = widening; negative = tightening.** A spread move from `1.50` to
`1.60` is `+10 bp`. Frequency is business (`B`, Monday–Friday); missing target
levels are forward-filled before changes are calculated.

---



## Covariates

Default covariates are defined in `data.py` and lagged by one observation
before registration. Examples include:

| Series ID (registered) | Economic meaning |
|------------------------|------------------|
| `vix_level_l1b` | VIX level |
| `vix_log_ret_1b_l1b` | VIX log return |
| `ust10y_level_l1b` | 10Y Treasury yield |
| `ust2y10y_spread_l1b` | 10Y minus 2Y Treasury yield |
| `fed_funds_level_l1b` | Effective federal funds rate |
| `cpi_mom_logdiff_l1b` | CPI monthly log change |
| `unemployment_rate_l1b` | Unemployment rate |

FRED and Yahoo inputs are cached under `data/fred/` and
`data/yfinance/` at the repository root.

### HYOAS extension

Two optional features add information about high-yield credit conditions:

| Series ID | Definition |
|-----------|------------|
| `hyoas_observed_change_1b_bps_l1b` | Observed daily change in FRED `BAMLH0A0HYM2`, in basis points |
| `hyoas_hyg_dgs3_proxy_change_1b_bps_l1b` | HYG return / 3Y Treasury yield-change proxy, in basis points |

The proxy uses a fixed three-year duration to approximate spread changes from
HYG adjusted-close returns after removing the Treasury-yield contribution
(`hyoas_proxy.py`). Both features are lagged before registration. The proxy is an approximation,
not an observed OAS series.

Named tuning panels are **`target_only`**, **`default`**, and
**`default_plus_hyoas`**. The extended panel includes **both** observed and proxy
HYOAS changes. Notebook 01 compares LightGBM with and without this extension.

---

## Cutoff-aware evaluation

The main workflow uses a **2025 backtest** for comparison and a **2026 evaluation**
for selected finalists. The **2020 COVID window** is used for stress testing.

---

## Specs — windows and tasks

Specs contain the **experiment design**: origin window, stride, warmup, and one
single-horizon task per target.


---

## Module layout

| Module | Responsibility |
|--------|----------------|
| `data.py` | Target construction, covariate transformations, and `build_baa10y_multivariate_service()` |
| `hyoas_proxy.py` | HYG–DGS3 spread-change proxy |
| `predictors/` | LLMP recipe, fixed benchmark parameters, and adaptive candidates |
| `analyst_agent/` | Credit-market analyst, data/news tools, and analysis skills |
| `tasks.py` | Continuous forecast, widening-event, and scenario output contracts |
| `adaptive_agent/` | Tuning agent, Optuna optimizer, evaluation tools, and persistent state |
| `baa10y_benchmarks.py` | Historical statistical benchmarks for analyst context |
| `leaderboard.py` | Cached results to `RESULTS_DF`; forecast-versus-actual frames |
| `analysis.py` / `plots.py` | Direction metrics, styled tables, and charts |
| `starter_agent/` | Configurable agent template |
| `specs/` | Backtest, tuning, validation, and evaluation designs |

### Notebooks

| Notebook | Start here for… |
|----------|-----------------|
| [`00_baa10y_data_exploration.ipynb`](00_baa10y_data_exploration.ipynb) | Data exploration |
| [`01_BAA10Y_multivariate_backtest.ipynb`](01_BAA10Y_multivariate_backtest.ipynb) | Main benchmark, evaluation, and HYOAS comparison |
| [`02_BAA10Y_backtest_comparison.ipynb`](02_BAA10Y_backtest_comparison.ipynb) | Numerical comparisons across backtest windows |
| [`03_BAA10Y_analysis_agent_tests.ipynb`](03_BAA10Y_analysis_agent_tests.ipynb) | Analyst integration and skill tests |
| [`04_BAA10Y_adaptive_agent_tests.ipynb`](04_BAA10Y_adaptive_agent_tests.ipynb) | Adaptive tuning with the default panel |
| [`04_BAA10Y_adaptive_agent_tests_withHYOAS.ipynb`](04_BAA10Y_adaptive_agent_tests_withHYOAS.ipynb) | Adaptive tuning with HYOAS features |
| [`99_starter_agent.ipynb`](99_starter_agent.ipynb) | Build and test a custom agent |

---

## Prerequisites and running

From the **repository root**, install the workspace dependencies with Python
**3.12+** and `uv`:

```bash
uv sync
```

Set `FRED_API_KEY` in the environment or repository-root `.env`. 

Use the following command line to launch the starter agent:
```bash
cd /home/coder/agentic-forecasting/implementations/BAA10Y_forecasting
uv run adk web starter_agent/
```

---

## Forecast, analyst, and adaptive agents

The three agents support different parts of the forecasting workflow:

| | Forecast agent — `starter_agent/` | Analyst agent — `analyst_agent/` | Adaptive agent — `adaptive_agent/` |
|---|---|---|---|
| Main question | **What is the likely spread change and its uncertainty?** | **What do market conditions imply for spreads, and why?** | **Which forecasting parameters perform better in backtests?** |
| User input | Forecast origin, horizon, and optional covariate/tool settings | Analysis question, information cutoff, and horizon | Model family, horizon, covariate panel, and trial budget |
| Main work | Use historical observations and optional tools to generate a forecast distribution | Diagnose market conditions, interpret drivers, and combine statistical/news evidence | Search parameters, compare CRPS, freeze a candidate, and validate it |
| Tools and skills | Forecasting skill credit spread movement | Statistical-analysis and credit-driver-analysis skills; data loader and configurable news/code/model tools | Agent tools for parameter search, diagnostics, and validation; Optuna TPE or grid search; persistent experiment state |
| Result | Point forecast and quantiles in basis points, or an interactive response | Market analysis, scenarios, or a structured forecast | Parameters, scores, diagnostics, and a promotion/rejection decision |



These are **separate workflows**. The current implementation does not automatically
pass an analyst report into adaptive search or install a promoted configuration
into either forecasting agent. Integration could be a next step.

### Where to start

- **Forecast agent:** Open `99_starter_agent.ipynb` for interactive questions and individual forecast evaluation. Configure tools in `starter_agent/agent.py`.
- **Analyst agent:** Open `03_BAA10Y_analysis_agent_tests.ipynb` for statistical and credit-driver analysis. Agent configurations and skills are in `analyst_agent/`.
- **Adaptive agent:** Open `04_BAA10Y_adaptive_agent_tests.ipynb` for parameter tuning, or `04_BAA10Y_adaptive_agent_tests_withHYOAS.ipynb` for the HYOAS panel. Set the method, horizon, covariate panel, and trial budget in the notebook.
---
