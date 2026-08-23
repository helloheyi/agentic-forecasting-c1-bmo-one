# LightGBM Per-Quantile Hyperparameter Tuning

Why `DartsLightGBMPredictor` fits 13 independent boosters per forecast, why one
shared config across them (and across the univariate/covariate variants) is a
compromise, and how the tuning module in this repo searches for better ones
without leaking the future into the search. Read this before touching
`per_quantile_kwargs`, `lgbm_quantile_tuning.py`, or the `separate` flag.

All of this is implemented in
[`aieng-forecasting/aieng/forecasting/methods/numerical/darts_regression.py`](../aieng-forecasting/aieng/forecasting/methods/numerical/darts_regression.py)
(the predictor-side seam) and
[`aieng-forecasting/aieng/forecasting/methods/numerical/lgbm_quantile_tuning.py`](../aieng-forecasting/aieng/forecasting/methods/numerical/lgbm_quantile_tuning.py)
(the search itself).

> **Status.** Wired into `01_BAA10Y_multivariate_backtest.ipynb`'s predictors
> cell (`LIGHTGBM_MODE`/`TUNING_TASK_ID`/`LGBM_TUNING_MODE`); not yet into
> `02_BAA10Y_backtest_comparison.ipynb`. Every pre-existing
> `DartsLightGBMPredictor` call site elsewhere is unaffected unless it opts
> in. See §11.

---

## 1. Why per-quantile hyperparameters

`DartsLightGBMPredictor` uses Darts' `likelihood="quantile"` mode, which fits
one independent LightGBM booster per quantile level in `_TRAINING_QUANTILES`
(13 levels, 0.025 through 0.975). Confirmed by reading Darts' internals
(`darts/models/forecasting/lgbm.py`): `LightGBMModel.fit()` loops over those 13
quantiles and calls `self._create_model(**self.kwargs)` fresh each time —
**identical kwargs every call**, only the quantile level (`alpha`) changes.
There was previously no way to give the median booster a different config than
the tail boosters, even though they're solving different problems (tail
quantiles see less effective training signal per split and often want more
conservative regularization).

The same "one shared config" limitation also applies **across** the
univariate and covariate predictor variants, which see meaningfully different
feature counts:

| Variant | Feature count |
|---|---|
| Univariate (`covariate_series_ids=None`) | `lags` (default 12) |
| Covariate (`covariate_series_ids=[...]`) | `lags + lags_past_covariates * len(covariate_series_ids)` |

More features generally wants more capacity (`num_leaves`, `max_depth`) *and*
more regularization (`min_data_in_leaf`, `lambda_l1`, `lambda_l2`) to avoid
overfitting the larger, noisier space — and the optimal `learning_rate` /
`n_estimators` pairing shifts too. A single global config forces a compromise
that's suboptimal for at least one variant.

---

## 2. The predictor-side seam: `per_quantile_kwargs`

`DartsLightGBMPredictor` takes an optional `per_quantile_kwargs: dict[float,
dict[str, Any]] | None` constructor argument, keyed by quantile level. When
set, `predict()` builds the model from a small mixin
(`_PerQuantileLightGBMModel`) composed with Darts' `LightGBMModel`, overriding
`_create_model` to merge that quantile's override dict on top of the shared
`lgbm_kwargs` before delegating to Darts.

`DartsLightGBMPredictor` itself has **no notion of tuning** — it just applies
whatever `per_quantile_kwargs` dict it's given. `per_quantile_kwargs=None` (the
default, and every pre-existing call site) preserves the original
single-shared-config behavior byte-for-byte. `predictor_id` gains a `_tuned`
suffix when a config is supplied, so tuned and untuned `BacktestResult`s never
collide under the same id.

```python
DartsLightGBMPredictor(
    lags=12,
    per_quantile_kwargs={
        0.025: {"num_leaves": 16, "min_data_in_leaf": 40},
        0.5: {"num_leaves": 64, "min_data_in_leaf": 10},
        0.975: {"num_leaves": 16, "min_data_in_leaf": 40},
        # ... remaining 10 quantile levels
    },
)
```

Writing that dict by hand for 13 quantiles × 7 params is tedious and easy to
get inconsistent — that's what `lgbm_quantile_tuning.py` builds automatically.

---

## 3. The interpolation trick

Searching 13 quantiles × 7 params = 91 raw numbers is an intractable space for
a handful of Optuna trials. Instead, each tunable param is parameterized as a
straight line in "distance from the median":

```
value(q) = base + slope * tail_distance(q)
tail_distance(q) = |q - 0.5| / 0.475
```

`tail_distance` is 0.0 at the median (`q=0.5`) and 1.0 at the extreme tails
(`q=0.025`/`q=0.975`) — and **symmetric**: 0.025 and 0.975 always get the same
value, so one `slope` moves both tails together rather than letting them
diverge independently. That's 2 coefficients per param instead of 13 raw
values, so the search space is 14-dimensional (7 params × 2) instead of 91.

| Param | Integer-valued | Floor |
|---|---|---|
| `num_leaves` | yes | 2 |
| `max_depth` | yes | 1 |
| `min_data_in_leaf` | yes | 1 |
| `n_estimators` | yes | 1 |
| `lambda_l1` | no | 0.0 |
| `lambda_l2` | no | 0.0 |
| `learning_rate` | no | 1e-4 |

`n_estimators` is tuned alongside `learning_rate` even though it wasn't in the
originally-requested param list — the two are tightly coupled (a smaller
learning rate typically wants more estimators to reach the same effective fit),
so tuning one without the other rarely moves CRPS much on its own.

Values are clamped to the floor above (so a large negative slope can never
produce e.g. `num_leaves <= 1`), then rounded to `int` for the integer-valued
params. See `_expand_to_per_quantile` and `_PARAM_MINIMUMS` /
`_INT_PARAMS` in `lgbm_quantile_tuning.py`.

---

## 4. Shared vs. separate tuning

`tune_lightgbm_configs(..., separate: bool = True)` is the entry point for
choosing between one config and two:

- **`separate=True` (default).** Runs two independent Optuna studies — one
  with `covariate_series_ids=None`, one with the real covariate list —
  producing potentially different `per_quantile_kwargs` for each variant.
  Recommended, since §1 established the feature spaces genuinely differ, and
  because the near-term plan is to add a new covariate (a RAG-extracted
  Fed regime-change signal) that widens that gap further — the covariate
  variant is the one this project most needs to perform well, so it should
  get a config actually suited to its feature space rather than one tuned
  for a smaller univariate problem.
- **`separate=False`.** Tunes once against the univariate variant and reuses
  the same `per_quantile_kwargs` for the covariate variant — half the Optuna
  trials, at the cost of the covariate variant not getting its own search.

Both variants are always present in the returned `dict[str, TuningResult]`
(keys `"univariate"` / `"covariate"`) — under `separate=False` they carry
identical `per_quantile_kwargs` but distinct `predictor_variant` labels, so
downstream code doesn't need to branch on which mode was used.

This toggle is deliberately a plain function argument, not a flag on
`DartsLightGBMPredictor` or a link between two predictor instances — the
predictor class stays completely unaware that tuning exists (see §2), and the
"configure it in the notebook" convention this repo already uses for every
other predictor hyperparameter (see `sp500_forecasting/leaderboard.py`)
extends naturally to "configure the tuning mode in the notebook" too.

---

## 5. The no-leakage rule (nested walk-forward)

CRPS is scored against the **realized outcome**, which is not known at the
origin being forecast. Tuning that validates against the same origin(s) it
will ultimately be used to predict — or against origins after that point in
time — leaks the future into model selection.

The rule: tuning must always validate against a window of **already-elapsed**
origins strictly before the live forecast's cutoff.

```
history ────────────[ validation_start .. validation_end ]──── cutoff ──▶ live forecast origin(s)
                      (validation_window origins, known outcomes)          (unknown outcome — never
                                                                             used for tuning)
```

`tune_lightgbm_quantile_config(..., cutoff=...)` enforces this: if `cutoff` is
given (normally the live forecast's `as_of`) and `validation_end > cutoff`, it
raises `ValueError` immediately rather than silently running a leaky study.
`cutoff` is optional only because standalone/exploratory tuning runs — not
feeding a specific live prediction — have no such cutoff to check against.

---

## 6. Cost and runtime caveats

Each Optuna trial runs a full `backtest()` over `validation_window` origins,
each of which fits 13 LightGBM boosters. Total cost for one study is
approximately:

```
n_trials × validation_window origins × 13 boosters
```

With the defaults (`n_trials=30`, `validation_window=60`), that's ~23,400
booster fits per study — and `separate=True` runs two studies. Start smaller
during development (`n_trials=5-10`, `validation_window=12-20`) before scaling
up once you've confirmed the pipeline works end-to-end.

`num_samples` defaults to 200 during tuning (vs. 500 for a production
predictor) — tuning only needs CRPS *ranking* between candidate configs to be
stable, not final-answer calibration precision, so a smaller Monte Carlo
sample keeps each trial cheaper.

**Parallelism: raise `n_jobs`, not LightGBM's `num_threads`.** The real
parallelism in this workload is *across* the ~23,400+ independent booster
fits (trials × origins × quantiles), not *within* any single fit — each fit
is only ~6,500 rows × ~60 columns for BAA10Y-scale data, too small for
LightGBM's own thread-level parallelism to buy much. `tune_lightgbm_quantile_config(..., n_jobs=N)`
runs `N` trials concurrently via Optuna (this works despite Python's GIL
because LightGBM's `fit()` is a native call that releases it). Whenever
`n_jobs != 1`, each fit's own `num_threads` is automatically capped to 1
(unless already set in `base_lgbm_kwargs`) — otherwise `N` concurrent trials
would each try to claim every core, oversubscribing and typically running
*slower* than sequential. Set `n_jobs` to roughly the number of available
CPU cores; for scaling beyond one machine, Optuna's storage-backed
multi-process pattern (separate worker processes against a shared RDB) is
the more robust upgrade path.

---

## 7. Saving and resuming tuning sessions

A study built at BAA10Y scale takes real wall-clock time — see §6. By
default every study is **in-memory only**: closing the notebook kernel
discards it, and the next session pays the full cost again. Passing
`storage_path` (a SQLite file) persists the study across sessions, and
`mode` controls what happens with what's saved:

| Mode | Cost | Use it when | Resulting `TuningResult.n_trials` |
|---|---|---|---|
| `"scratch"` | Full `n_trials` | First run, or after changing `param_ranges`/`lags`/`covariate_series_ids` | `n_trials` |
| `"resume"` | Only the shortfall | Want more confidence than last session gave, without losing it | `max(n_trials, trials already saved)` |
| `"reuse"` | ~Zero (no new trials) | Just want last session's config back | Whatever was already saved |

`storage_path` is a plain SQLite file — one file can hold multiple
independently-named studies (`tune_lightgbm_configs`'s two per-variant calls
share one file safely, via an auto-derived `study_name` per variant). Put it
under `<implementation>/data/...` (e.g. next to `PREDICTIONS_DIR` in the
BAA10Y notebook) — the repo's existing broad `/data/` and
`implementations/**/data/` `.gitignore` rules already cover it, no new
entries needed.

**`n_trials`'s meaning changes with `mode`.** For `"scratch"` it's the count
to run, same as always. For `"resume"` it's a **lifetime budget across
sessions** — if a saved study already has 8 trials and you pass `n_trials=15`,
only 7 more run, not 15 more. For `"reuse"` it's ignored entirely (zero new
trials, whatever's saved is returned as-is). The returned
`TuningResult.n_trials` always reflects the study's actual total afterward,
not an echo of the input.

**Caveat — not auto-detected.** If you change `param_ranges`, `lags`, or
`covariate_series_ids` between sessions while reusing/resuming the *same*
`study_name`, Optuna will happily keep sampling into what's now an
incompatible search space — this module doesn't guard against it (a general,
well-known HPO footgun, not specific to this code). Use `mode="scratch"` (or
pass a different `study_name`) whenever the search space itself changes.

```python
from pathlib import Path

db = Path("data/lgbm_tuning/optuna_studies.db")

# First session: build the study from nothing.
tuned = tune_lightgbm_configs(..., storage_path=db, mode="scratch", n_trials=15)

# A later session: extend it to 30 trials total (only 15 more actually run).
tuned = tune_lightgbm_configs(..., storage_path=db, mode="resume", n_trials=30)

# A read-only session: just load what's saved, no new trials.
tuned = tune_lightgbm_configs(..., storage_path=db, mode="reuse")
```

---

## 8. How to call it

```python
from datetime import datetime
from aieng.forecasting.methods.numerical import DartsLightGBMPredictor, tune_lightgbm_configs

tuned = tune_lightgbm_configs(
    task=task,
    data_service=svc,
    validation_end=datetime(2025, 1, 1),
    covariate_series_ids=COVARIATES,
    lags=LAGS,
    base_lgbm_kwargs=LGBM_KWARGS,
    n_trials=10,            # start small; see §6
    validation_window=20,
    # separate=True by default
)

lightgbm = DartsLightGBMPredictor(
    lags=LAGS, covariate_series_ids=None, num_samples=NUM_SAMPLES,
    lgbm_kwargs=LGBM_KWARGS, per_quantile_kwargs=tuned["univariate"].per_quantile_kwargs,
)
lightgbm_cov = DartsLightGBMPredictor(
    lags=LAGS, lags_past_covariates=LAGS, covariate_series_ids=COVARIATES, num_samples=NUM_SAMPLES,
    lgbm_kwargs=LGBM_KWARGS, per_quantile_kwargs=tuned["covariate"].per_quantile_kwargs,
)
```

To tune a single variant directly (e.g. only the covariate one), call
`tune_lightgbm_quantile_config` instead of `tune_lightgbm_configs`.

---

## 9. Testing

`aieng-forecasting/tests/aieng/forecasting/methods/numerical/test_lgbm_quantile_tuning.py`
covers the interpolation math and the `_PerQuantileLightGBMModel` override
directly (pure-function tests, no Optuna study needed), plus the
shared-vs-separate orchestration logic with `tune_lightgbm_quantile_config`
mocked out. One smoke test does run a real (tiny: `n_trials=1`, small
validation window) end-to-end study against synthetic data — that's the only
place the expensive path actually executes; it is not run at production scale
in CI.

---

## 10. Checklist for adding a new tunable param

1. Add it to `_DEFAULT_PARAM_RANGES` in `lgbm_quantile_tuning.py` with sensible
   `base`/`slope` bounds.
2. If it's integer-valued or has a hard floor, add it to `_INT_PARAMS` and/or
   `_PARAM_MINIMUMS`.
3. Add a unit test asserting the floor/rounding behavior (mirror the existing
   `test_expand_to_per_quantile_*` tests).
4. Add a row to the table in §3 above.

---

## 11. Current status / open follow-ups

- **Wired into `01_BAA10Y_multivariate_backtest.ipynb`, not yet into
  `02_BAA10Y_backtest_comparison.ipynb`.** Notebook 01's predictors cell has
  a tuning cell (`LIGHTGBM_MODE`/`TUNING_TASK_ID`/`LGBM_TUNING_MODE`) using
  §7's save/resume support. Notebook 02 is a separate, independent
  predictor-instantiation-and-rerun notebook (not a cache reader — its
  `run_experiment(...)` defaults `force_refresh=True`), so nothing carries
  over automatically; wiring it in is a small follow-up now that
  `mode="reuse"` makes loading an already-tuned config near-free.
- **`_DEFAULT_PARAM_RANGES` bounds are strawman defaults**, not calibrated
  against real BAA10Y data — expect to retune after a first real run.
- **The RAG regime-change covariate itself** (extracting a signal from Fed
  PDFs and registering it as a covariate series) is separate, not-yet-started
  work this tuning module is meant to support once it lands — it's the
  concrete reason `separate=True` is the default (see §4).
