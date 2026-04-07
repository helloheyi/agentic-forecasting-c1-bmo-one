# Economic Forecasting — CPI Use Case

This directory contains reference implementations and notebooks for the
**Canada CPI economic forecasting** use case.

The forecasting task: predict Canada-wide CPI All-items (StatCan, 2002=100)
12 months ahead, evaluated at January and July origins from 2000 to present.

---

## Before you start

Populate the local data cache (required before any backtests or notebooks):

```bash
uv run python scripts/fetch_cpi.py
```

This registers 47 Canada-wide CPI series from StatCan table 18-10-0004-11
into `data/statcan/`. The cache is gitignored; re-run after a fresh clone.

---

## Learning path

### 1. Understand the data

Open `cpi_data_exploration.ipynb`. It walks through loading CPI series,
applying information-cutoff filtering, plotting historical trends, and
computing year-over-year changes across key categories.

### 2. Run a backtest end-to-end

Open `cpi_backtest_demo.ipynb`. It walks through:

- Loading a reference `BacktestSpec` from YAML
- Defining a predictor by implementing the `Predictor` ABC
- Running `backtest()` and inspecting the `BacktestResult`
- Visualizing predictions against ground truth

The predictor is defined **inline in the notebook** so you can see the
complete implementation in the linear flow.

### 3. Write your own predictor

Copy `predictors/predictor_template.py`. It is an annotated last-value
naive baseline with step-by-step comments explaining each required piece:
`predictor_id`, fetching series from the context, building a
`ContinuousForecast`, computing the forecast date, and returning a
`Prediction`. Replace the forecast logic with your own model — swap in
any Darts model, an LLM call, or anything else that implements `predict()`.

### 4. Compare predictors

Run `backtest()` with two or more predictors against the same
`BacktestSpec`. Compare `result.mean_crps` across predictors; lower is
better. The reference spec is:

```
reference_specs/cpi_allitems_12m.yaml   # Jan/Jul origins, 2000–2026
```

### 5. Spend an eval run

Once you have a predictor you're happy with, test it against the
held-out evaluation window:

```
reference_specs/cpi_allitems_eval_2yr.yaml   # 2024–2026, max_runs: 5
```

Use `evaluate()` instead of `backtest()`, and pass an `EvalTracker` to
enforce the budget. Spend your runs deliberately — each one counts against
the `max_runs` limit in the spec.

---

## Directory layout

```
economic_forecasting/
├── README.md                        # this file
├── predictors/
│   └── predictor_template.py        # annotated starting point — copy this
├── cpi_data_exploration.ipynb       # data exploration and visualization
└── cpi_backtest_demo.ipynb          # end-to-end backtest walkthrough
```

Reference specs (at the repo root, shared across use cases):

```
reference_specs/
├── cpi_allitems_12m.yaml            # backtest spec — use freely
└── cpi_allitems_eval_2yr.yaml       # eval spec — 5 runs max
```

---

## Key interfaces (from `aieng-forecasting`)

```python
from aieng.forecasting.evaluation import (
    Predictor,          # ABC — implement this
    backtest,           # run a backtest, returns BacktestResult
    evaluate,           # run against the held-out eval window
    BacktestSpec,       # loaded from reference_specs/ YAML
    EvalSpec,           # loaded from reference_specs/ YAML
    EvalTracker,        # file-backed run counter
    ContinuousForecast, # forecast payload (point + quantiles)
    Prediction,         # full prediction record (payload + metadata)
    STANDARD_QUANTILES, # [0.05, 0.10, ..., 0.90, 0.95]
)
from aieng.forecasting.data import DataService  # register series, create contexts
```
