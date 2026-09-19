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

## Recommended settings (current — notebook cells 7 and 16)

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

DATA_HISTORY_START = "2007-01-01"          # section 1 -- NOT the old 2016-01-01
validation_end = datetime(2009, 8, 31)     # back near the post-GFC "new normal"
cutoff = datetime(2020, 2, 3)              # = stress_2020 spec's start (enforced)
validation_window = 281                    # ~2008-08-01 -> 2009-08-31 (business days)
stride = 4                                 # subsample -- ~71 origins, not ~281
n_trials ≈ 15-20                           # start smaller than backtest_2025's 67
n_jobs = 4
```

Start smaller than `backtest_2025`'s settings on purpose — this is a fresh,
unverified setup (new storage file, first run), and the guide's own advice
(§6) is to confirm a pipeline works end-to-end before scaling `n_trials`/
`validation_window` up. Reuse `CHEAPER_PARAM_RANGES` from the `backtest_2025`
cell too, unless a first run shows it's unnecessary here.

### Why the window moved into the 2008 GFC (not "close to the original, pre-drift value")

The original plan validated at `2019-11-15` — a calm, pre-COVID window chosen
to stay close in time to the live `2020-02-03` origins ("pre-drift"). That
choice quietly worked against this scenario's own stated purpose: with
`DATA_HISTORY_START` still at `2016-01-01`, every origin's training set —
tuning *and* live evaluation alike — was expanding-window from 2016, so
LightGBM never once trained on a genuine stress episode before being scored
on the COVID crash. A crisis-flag-like feature such as `FacilityStress` or
`QTIntensity` can look like pure noise to a tree that has only ever seen calm
data, regardless of how good it might be during an actual regime shift.

Two changes fix this together (both are needed — either alone is a partial
fix):

- `DATA_HISTORY_START` moved to `2007-01-01` (section 1 of the notebook,
  shared across all `EXPERIMENT_CONFIG`s). All of the default covariates and
  the H.4.1-RAG series (`data/h41_rag/*.parquet`) have history back to at
  least 2000, and `BAA10Y` itself back to 1986, so this is safe. This alone
  gets 2008 into every origin's *training* set, but tuning would still be
  picking hyperparameters based on calm-period validation CRPS.
- `validation_end`/`validation_window` moved to cover `2008-08-01` ->
  `2009-08-31`. The first attempt at this tried to bracket just the acute
  spike with 15 calm days on each side (`validation_end = 2008-12-31`,
  `validation_window = 65`) — but per `data/fred/BAA10Y.parquet` there's no
  genuine calm that close to the spike: spreads were already at ~3.2 in
  August 2008 (Bear Stearns had markets jittery since March), the spread
  widens from ~2.9 in September to a peak of 6.16 on 2008-12-04, and stays at
  5.0-5.9 through Q1 2009 — real calm doesn't return until ~August 2009
  (~2.9-3.0, a "new normal" still above 2007's ~1.6-2.1 baseline), about 8
  months after the peak. `BacktestSpec` validation windows are a single
  contiguous trailing range (`start = validation_end − window`, no gaps), so
  reaching genuine calm on both sides means the window has to span the whole
  year, not just the acute core. `stride = 4` keeps the actual number of
  walk-forward retrains (~71 origins) comparable to what the tighter 65-day
  window would have cost, while covering onset → peak → recovery instead of
  only the acute middle. `tune_lightgbm_configs`/`tune_lightgbm_quantile_config`
  both already accept `stride` — it just wasn't threaded through this
  notebook's tuning cell before.

`cutoff` stays at `2020-02-03`: it's what the no-leakage guard checks against
(`validation_end` must not exceed it), and with `validation_end` now in 2009
it's no longer the binding constraint, but it's still correct to keep passing
it — see the next section for what happens if a scenario's tuning cell
doesn't enforce this.

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
