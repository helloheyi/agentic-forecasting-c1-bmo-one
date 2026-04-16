# Food Price CPI Forecasting

This use case replicates and extends the **Canada's Food Price Report (CFPR)** forecasting experiment — an annual prediction of Canadian food price changes across 9 food CPI categories. The original CFPR is published jointly by several Canadian universities and uses food CPI data from Statistics Canada.

---

## Forecasting Task

**Target variable:** Consumer Price Index (CPI) for food products and sub-categories in Canada (index, 2002=100). Year-over-year percentage changes are derived from index forecasts at reporting time.

**Target categories (9):**

| Series ID | Description |
|-----------|-------------|
| `cpi_food_canada` | Overall Food |
| `cpi_bakery_cereal_canada` | Bakery and cereal products (excl. baby food) |
| `cpi_dairy_eggs_canada` | Dairy products and eggs |
| `cpi_fish_seafood_canada` | Fish, seafood and other marine products |
| `cpi_restaurants_canada` | Food purchased from restaurants |
| `cpi_fruit_preparations_nuts_canada` | Fruit, fruit preparations and nuts |
| `cpi_meat_canada` | Meat |
| `cpi_other_food_nonalcoholic_canada` | Other food products and non-alcoholic beverages |
| `cpi_vegetables_preparations_canada` | Vegetables and vegetable preparations |

**Data source:** Statistics Canada table 18-10-0004-11. Populated via `scripts/fetch_cpi.py`.

---

## Exogenous Covariates (FRED)

The following FRED series can be used as past covariates by predictors that support them (e.g. `DartsAutoARIMAPredictor(covariate_series_ids=[...])`). Populated via `scripts/fetch_fred.py`.

| Series ID | FRED ID | Description |
|-----------|---------|-------------|
| `fred_us_cpi_food_at_home` | CPIFABSL | US CPI: Food at Home |
| `fred_us_cpi_meats_poultry_fish_eggs` | CUSR0000SAF112 | US CPI: Meats, Poultry, Fish, Eggs |
| `fred_us_cpi_fruits_vegetables` | CUSR0000SAF113 | US CPI: Fruits and Vegetables |
| `fred_canada_10yr_bond_yield` | IRLTLT01CAM156N | Canada 10-year government bond yield |
| `fred_canada_us_exchange_rate` | EXCAUS | Canada/US exchange rate (CAD per USD) |
| `fred_sp100_volatility_vxo` | VXOCLS | S&P 100 Volatility Index (VXO) |
| `fred_wilshire_5000` | WILL5000IND | Wilshire 5000 Total Market Index |

**Note:** FRED requires an API key. Set `FRED_API_KEY` in `.env` before running `scripts/fetch_fred.py`.

---

## Experiment Structure

Two experiments run in parallel, differing only in horizon and evaluation window:

| Experiment | Horizon | Backtest stride | Eval window | Eval origins |
|-----------|---------|----------------|-------------|--------------|
| 18-month (CFPR-replica) | 18 months | 6 months (Jan/Jul) | Jul 2022 → Jul 2024 | ~5 |
| 3-month (denser eval) | 3 months | 3 months (quarterly) | Jan 2024 → Jan 2026 | ~9 |

### Why two horizons?

The 18-month horizon replicates the CFPR evaluation cycle. The 3-month horizon provides a denser eval sample in recent history — ~9 eval origins vs ~5 — without pushing origins so far back that structural changes in food price dynamics make the comparison less relevant. More eval origins means:

- More reliable CRPS estimates (lower variance)
- Better method discrimination
- More quantile-coverage data for calibration assessment

---

## Evaluation Paradigm

This experiment separates backtest and eval modes:

- **Backtest (open):** Run as many times as you want against `food_cpi_{horizon}m_backtest.yaml`. Use for tuning, learning, and exploration.
- **Eval (protected, budget-limited):** Run `multi_evaluate()` against `food_cpi_{horizon}m_eval.yaml`. One call consumes one budget run (`max_runs=5`) across all 9 categories simultaneously. This simulates a live exercise — the eval error distribution is a proxy for expected live performance.

The budget constraint follows the same logic as Kaggle's public/private leaderboard split: use backtest freely, spend eval deliberately.

---

## Reference Specs

```
reference_specs/food_cpi/
├── food_cpi_18m_backtest.yaml   # MultiTargetBacktestSpec, 18m, 2000–2026
├── food_cpi_18m_eval.yaml       # MultiTargetEvalSpec, 18m, Jul 2022–Jul 2024
├── food_cpi_3m_backtest.yaml    # MultiTargetBacktestSpec, 3m, 2000–2026
└── food_cpi_3m_eval.yaml        # MultiTargetEvalSpec, 3m, Jan 2024–Jan 2026
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `food_data_exploration.ipynb` | Register and verify all series; inspect date ranges, gaps, seasonality; confirm StatCan label mappings; visualise food CPI history and FRED covariates |
| `food_cpi_18m_experiment.ipynb` | 18-month CFPR-replica: multi-target backtest, CRPS comparison table, MAPE (post-hoc), eval run, YoY derivation |
| `food_cpi_3m_experiment.ipynb` | 3-month companion: same structure, denser eval; discussion of what more eval origins enables |

---

## Prerequisites

```bash
# Populate data caches
uv run python scripts/fetch_cpi.py
uv run python scripts/fetch_fred.py   # requires FRED_API_KEY in .env
```

---

## Key Design Decisions

- **Forecast target is the raw CPI index.** Year-over-year percentage changes are derived at reporting time: `yoy_pct = (forecast - actual_12m_ago) / actual_12m_ago * 100`. This keeps the modelling target straightforward and separable from the reporting presentation.
- **No ensemble model selection.** We compare individual methods on CRPS + MAPE (median). Ensemble search is out of scope for this iteration.
- **MAPE on median.** MAPE is computed post-hoc in notebooks using `prediction.payload.point_forecast` (the median of the predictive distribution). It is not part of `BacktestResult`/`EvalResult`; the primary metric is CRPS.
- **Covariates are predictor-level.** Which FRED series to include is a predictor decision (`covariate_series_ids`), not a task definition. Tasks define the question; predictors decide how to answer it.
