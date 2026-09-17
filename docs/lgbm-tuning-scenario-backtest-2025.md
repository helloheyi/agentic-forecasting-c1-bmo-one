# Tuning scenario: `backtest_2025`

**Read `lightgbm-quantile-tuning-guide.md` first** for the general tuning
mechanics (per-quantile interpolation, the `separate` flag, save/resume
modes). This doc is scenario-specific: what settings `backtest_2025` uses,
why, and — critically — why they **cannot** be shared with the `stress_2020`
scenario (see the sibling doc, `lgbm-tuning-scenario-stress-2020.md`).

## What this scenario is

`EXPERIMENT_CONFIG = "backtest_2025"` in `01_BAA10Y_multivariate_backtest.ipynb`
— `specs/baa10y_backtest_2025.yaml`, ~50 weekly origins across all of 2025,
post-cutoff (safe for the LLM-Process rows too). This is the main, open-iteration
comparison — the one you tune on repeatedly before spending the protected 2026
eval.

## Why this scenario needs its own, separate Optuna tuning run

The no-leakage rule (guide §5): tuning must validate against a window of
already-elapsed origins **strictly before** the live forecast's cutoff. For
`backtest_2025`, the live origins start 2025-01-06 — so tuning can legitimately
use validation data all the way through **late 2024**, right up to that
boundary. Using less recent data than that is safe but wasteful — it throws
away years of relevant, legitimately-available information.

This is a **different, later** validation window than `stress_2020` can ever
use (that scenario's origins start 2020-02-03, so its validation data must stop
years earlier — see the sibling doc). Because the two scenarios have
fundamentally different "how much history can I legitimately look at"
boundaries, one shared tuning run cannot be simultaneously optimal-or-even-valid
for both. `backtest_2025` should be tuned to use its full legitimate window;
`stress_2020` cannot use that same window without leaking the future.

## Current settings (as actually configured in the notebook)

```python
EXPERIMENT_CONFIG = "backtest_2025"
TUNING_TASK_ID = "baa10y_change_5b"
LGBM_TUNING_MODE = "resume"          # was "scratch" for the first run
LGBM_TUNING_DB = ROOT / "data" / "lgbm_tuning" / "optuna_studies.db"

validation_end = datetime(2024, 12, 20)   # right up against the live cutoff
cutoff = datetime(2025, 1, 6)             # = backtest_spec.start (enforced, raises if violated)
validation_window = 120                    # trailing business days of known outcomes
n_trials = 67
n_jobs = 4
```

`param_ranges` is overridden with `CHEAPER_PARAM_RANGES` (narrower `num_leaves`/
`n_estimators` bounds than the library default) — the original wider range was
costing approximately 2.5 hours/trial on the covariate variant; halving the
base and slope ranges for the costliest params brought this down without
changing the interpolation mechanism itself (guide §3).

## The danger this doc exists to prevent

`tune_lightgbm_configs` auto-derives its saved-study name as:
```
f"{task.task_id}_{'covariate' if covariate_series_ids else 'univariate'}"
```
— keyed **only** by `TUNING_TASK_ID` and whether covariates are present.
`TUNING_TASK_ID` is `"baa10y_change_5b"` for *both* scenarios. **If a
`stress_2020` tuning run reuses this same `LGBM_TUNING_DB` file, it will load
this exact study** — the one validated on data through 2024-12-20 — for a
scenario whose live origins start in *2020*. That's not merely suboptimal, it's
a genuine leakage violation: the hyperparameters would have been selected using
a validation window that is entirely in `stress_2020`'s future.
`tune_lightgbm_configs` does not detect or warn about this (guide §7's "not
auto-detected" caveat) — nothing raises an error, it just silently returns the
wrong-vintage config.

**The fix, already applied in practice**: `stress_2020` tuning uses a
**separate storage file** (see the sibling doc) — never `LGBM_TUNING_DB` above.

## Covariate panel

Whatever covariate panel decision is made (the union-vs-swap comparison
involving the new H.4.1-derived series, `QTIntensity`/
`LiquidityNarrativeScore_median` — still pending as of this doc) applies here
too, and is an **independent axis** from the validation-window question above:
changing `covariate_series_ids` between sessions also requires
`LGBM_TUNING_MODE = "scratch"` for the same reason — the saved study's trials
were sampled against the old feature space and aren't valid for a different one
(guide §7).
