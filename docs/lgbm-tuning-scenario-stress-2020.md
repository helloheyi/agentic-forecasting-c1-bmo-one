# Tuning scenario: `stress_2020`

**Read `lightgbm-quantile-tuning-guide.md` first** for the general tuning
mechanics. This doc is scenario-specific: what settings `stress_2020` needs,
why they differ from `backtest_2025`'s, and why the two **cannot** share a
saved Optuna study. See the sibling doc, `lgbm-tuning-scenario-backtest-2025.md`,
for that scenario's own settings and reasoning — read both, not just one, since
the collision risk between them is the whole point of having two files.

## What this scenario is

`EXPERIMENT_CONFIG = "stress_2020"` — `specs/baa10y_stress_2020.yaml`, daily
origins across the COVID crash (Feb-Apr 2020), **numerical methods only** (the
predictors cell drops the LLM-Process rows automatically — 2020 predates
Gemini's training cutoff, so an LLM would be reciting memorized outcomes, not
forecasting). This is the spec's own stated purpose: "study when do covariates
help among conventional methods" — a volatile regime where a genuine covariate
edge should be most visible. It's the natural place to test whether the new
H.4.1-derived series (`QTIntensity`, `LiquidityNarrativeScore_median`, etc.) —
built specifically to carry regime/stress signal — actually earn their keep,
since a calm-period backtest like `backtest_2025` may under-represent their
value (their correlation with the target was measurably weaker in a 2025-only
window than in the full 2000-2026 history that includes 2008/2020/2023).

## Why this scenario needs its own, separate Optuna tuning run

Same no-leakage rule as always (guide §5), but with a much tighter constraint
than `backtest_2025` faces: `stress_2020`'s live origins start **2020-02-03**.
Validation data must stop *strictly before* that — there is no legitimate way
to use 2021-2024 data to tune a model being evaluated on Feb-Apr 2020, no
matter how tempting that data is. This is a much shorter, older validation
window than `backtest_2025` gets to use, and that's not a choice — it's forced
by when this scenario's origins actually are.

## Recommended settings for the first run (not yet executed as of this doc)

```python
EXPERIMENT_CONFIG = "stress_2020"
TUNING_TASK_ID = "baa10y_change_5b"       # same task id as backtest_2025 --
                                            # this is exactly why the storage
                                            # path below MUST differ
LGBM_TUNING_MODE = "scratch"               # always "scratch" for a first run
LGBM_TUNING_DB_STRESS = ROOT / "data" / "lgbm_tuning" / "optuna_studies_stress2020.db"
# ^ a SEPARATE SQLite file from backtest_2025's optuna_studies.db --
#   see "The collision this file exists to prevent" below for why this can't
#   just be a different study_name in the same file (tune_lightgbm_configs
#   doesn't expose study_name as a parameter at all).

validation_end ≈ datetime(2019, 11, 15)    # close to the original, pre-drift value
cutoff = datetime(2020, 2, 3)              # = stress_2020 spec's start (enforced)
validation_window ≈ 45-60                  # start smaller than backtest_2025's 120
n_trials ≈ 15-20                           # start smaller than backtest_2025's 67
n_jobs = 4
```

Start smaller than `backtest_2025`'s settings on purpose — this is a fresh,
unverified setup (new storage file, first run), and the guide's own advice
(§6) is to confirm a pipeline works end-to-end before scaling `n_trials`/
`validation_window` up. Reuse `CHEAPER_PARAM_RANGES` from the `backtest_2025`
cell too, unless a first run shows it's unnecessary here.

## The collision this file exists to prevent

`tune_lightgbm_configs` auto-derives its saved-study name as:
```
f"{task.task_id}_{'covariate' if covariate_series_ids else 'univariate'}"
```
This depends only on `TUNING_TASK_ID` (`"baa10y_change_5b"`, identical in both
scenarios) and whether covariates are present — **not** on `validation_end`,
`cutoff`, or `EXPERIMENT_CONFIG`. `tune_lightgbm_configs` does not expose a
`study_name` parameter to override this (by design — see its docstring: this
keeps the univariate/covariate pair from colliding with each other within one
file, but it means nothing distinguishes *this* scenario's intended study from
`backtest_2025`'s except which file they're saved to).

**Concretely: if this scenario's tuning cell points at the same
`LGBM_TUNING_DB` file `backtest_2025` uses, running it with `mode="reuse"` or
`"resume"` will silently load `backtest_2025`'s study** — the one validated
through 2024-12-20 — and apply hyperparameters selected using years of data
that postdate the crash this scenario is supposed to be testing blind. Nothing
raises an error; `tune_lightgbm_configs` has no way to know the two calls were
meant to represent different scenarios. The only safeguard is using a
genuinely separate file, as specified above.

## Covariate panel

Same covariate-panel decision as `backtest_2025` (union vs. swap involving the
new H.4.1-derived series) should be evaluated here too, and for the reason
this whole comparison exists — a crisis-flag-like feature such as
`FacilityStress` or a regime-signal series may show its real value here even
if it looked inert in the calm 2025 window. Keep the panel choice consistent
between the two scenarios' runs unless deliberately testing whether a
regime-specific panel does better than one fixed choice — that's a separate,
larger question (see the earlier discussion on union vs. swap) and not
something to conflate with the tuning-window separation this doc covers.
