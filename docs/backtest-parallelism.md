# Backtest parallelism: rationale and rejected alternatives

## The problem

`LightGBM + cov` (and its tuned variant) is by far the slowest predictor in
`01_BAA10Y_multivariate_backtest.ipynb`'s roster — the tuning cell's own
comment cites ~2.5 hours/trial *before* `CHEAPER_PARAM_RANGES` narrowed
things down. Every backtest, eval, and Optuna trial refits it from scratch at
every forecast origin (no warm-starting across origins, since each origin's
model must only see data up to that origin — walk-forward correctness
requires this). Per origin, that's 13 independent per-quantile LightGBM
fits (one per `_TRAINING_QUANTILES` level), repeated once per task (1b/5b/21b
are separately refit models, since `output_chunk_length` differs), repeated
again across however many origins the spec has (~60 daily origins for
`stress_2020`'s live backtest, ~71 for its tuning-validation window with
`stride=4`).

## What we implemented

Two independent, composable layers of parallelism, both **additive and
opt-in** (default behavior is unchanged everywhere they aren't explicitly
used):

1. **Origin-level parallelism**, in `aieng-forecasting`'s
   `run_eval_loop`/`backtest`/`multi_backtest` (and the caching/eval
   wrappers `cached_backtest`/`cached_multi_backtest`/`evaluate`/
   `multi_evaluate`) — a new keyword-only `n_jobs: int = 1` parameter.
   `run_eval_loop`'s per-origin work was factored into a `_run_origin()`
   helper; with `n_jobs=1` it runs in a plain list comprehension (byte-for-byte
   the historical code path, zero pool overhead); with `n_jobs>1` the same
   helper is dispatched across a `ThreadPoolExecutor` via `Executor.map`,
   which preserves origin order in the returned `(predictions, scores)` lists
   regardless of which origin's job finishes first.
2. **Predictor-level parallelism**, notebook-only, in Section 5's backtest
   loop (`01_BAA10Y_multivariate_backtest.ipynb`, cell `0aa4e777`): the
   `for predictor in all_predictors: cached_multi_backtest(...)` loop now
   submits every predictor's backtest to a `ThreadPoolExecutor` up front and
   collects results via `as_completed`, instead of running predictors one at
   a time.

Both axes are exposed as `PREDICTOR_N_JOBS`/`ORIGIN_N_JOBS`, computed from
`os.cpu_count()` rather than hardcoded (see "Sizing" below), and `n_jobs`
is threaded into every Section 5/7/8 call site (`cached_multi_backtest` in
Sections 5 and 8, `multi_evaluate` in Section 7).

## Why threads, not processes

`ProcessPoolExecutor` was the more obvious-looking choice for CPU-bound work,
and we rejected it:

- **Picklability risk.** Dispatching to separate processes requires
  `predictor`, `task`, and `data_service` to cross a pickle boundary. The
  `Predictor` protocol is implemented by arbitrary subclasses — including the
  LLM-Process predictors, which hold network clients — and `DataService`
  itself hasn't been audited for picklability. A thread pool needs none of
  this; everything stays in shared memory in the same process.
- **LightGBM's own compute releases the GIL.** The concern with threads for
  CPU-bound work is the GIL serializing Python bytecode — but LightGBM's
  actual tree-building happens in its C++ extension, which releases the GIL
  during that work. So threads still get real parallelism for the part that
  actually costs time, without the pickling risk.
- **LLM-Process predictors are I/O-bound**, not CPU-bound (network calls to
  an external proxy) — threads are the *better* fit for those regardless of
  the LightGBM question, and `run_eval_loop` is shared by every predictor
  type, not just LightGBM.
- **Windows process-spawn overhead.** `ProcessPoolExecutor` uses the "spawn"
  start method on Windows, re-importing `darts`/`lightgbm`/`pandas`/etc. in
  every worker process at pool creation — a real, avoidable cost that threads
  don't pay.

> **Correction, added after this went live:** the GIL-release argument above
> is true but incomplete — it covers LightGBM's *C++ tree-building*, not
> Darts' own Python-level model construction, which turned out to have a real
> thread-safety bug of its own. See "A real bug this parallelism change
> introduced" below. Threads were still the right call; the fix is a small
> lock around model construction, not a reason to switch to processes.

## Why origin-level, not quantile-level (the 13 per-quantile fits)

Each origin's `predict()` call fits 13 independent LightGBM boosters
internally — one per quantile. We deliberately did **not** try to
parallelize *those*, for two reasons:

- **Darts doesn't expose that axis.** We traced Darts' actual fit path: the
  per-quantile loop is internal to `LightGBMModel.fit()` (confirmed via
  `SKLearnModel.fit()`'s source — no `joblib`/`Parallel` anywhere near it).
  The only similarly-named parameter, `n_jobs_multioutput_wrapper`,
  parallelizes across **horizon steps** via sklearn's `MultiOutputRegressor`
  wrapper — a different axis entirely, not the 13 quantiles. Parallelizing
  the quantile loop would mean monkey-patching/reimplementing Darts' internal
  fit orchestration ourselves, with real risk of drifting from Darts' own
  implementation across library upgrades.
- **It wouldn't help even if we did it.** With ~60–95 origins and roughly a
  dozen CPU cores on a typical machine, origin-level parallelism alone
  already has far more independent units of work than there are cores to run
  them on — the machine is already saturated. Nesting quantile-level
  parallelism on top would only help in the *opposite* regime (few origins,
  idle cores), which isn't our situation; it would just add process/thread
  management overhead for no measurable speedup.

## Why not also task-level parallelism (the 3 horizon tasks)

`multi_backtest`/`multi_evaluate` loop over `spec.specs()`/`spec.tasks`
(1b/5b/21b) sequentially, each calling `backtest()`/`run_eval_loop()` with
`n_jobs` threaded through. We left that outer loop sequential rather than
also parallelizing across tasks, for the same "already saturated" reasoning
as above — only 3 tasks vs. dozens of origins per task, so origin-level
parallelism is already the dominant, sufficient lever. Adding task-level
parallelism on top would only matter if origin-level parallelism weren't
already consuming all available cores, which it is.

## Why not per-fit LightGBM multithreading (`num_threads`/`n_jobs` inside `LGBM_KWARGS`)

This was the *first* idea explored, and we walked it back after further
discussion — worth documenting explicitly since it's the most tempting wrong
answer:

- LightGBM's intra-fit multithreading splits histogram-building work across
  threads *within one fit*, with a synchronization barrier every boosting
  iteration. That only pays off with enough raw work per thread to amortize
  the barrier cost — typically hundreds of thousands of rows, or many
  features/deep trees. This dataset's fits are small: by a 2020 origin, the
  expanding window from `DATA_HISTORY_START=2007-01-01` is only ~3,300 rows,
  with at most ~65 features (5 lags + up to 12 covariates × 5 lags) and
  `num_leaves` capped at ~64–96 by `CHEAPER_PARAM_RANGES`. LightGBM's own
  documentation warns that "using too many threads can result in poor
  performance due to overhead" at this scale.
- **It would actively fight the tuning cell's existing parallelism.**
  `LGBM_KWARGS = {"num_threads": 1, "n_jobs": 1, ...}` (cell `2e1da155`) is
  *already* deliberately pinned to single-threaded, specifically because
  `tune_lightgbm_configs(..., n_jobs=4)` (cell `0bbef83c`) runs 4 Optuna
  trials concurrently — see `_resolve_lgbm_kwargs`'s docstring in
  `lgbm_quantile_tuning.py`. If each of those 4 concurrent trials' LightGBM
  fits also grabbed multiple threads for itself, the same cores would be
  oversubscribed by both layers at once — typically *slower*, not faster.
  We left `LGBM_KWARGS` exactly as-is; it is **not** touched by this change,
  in either the tuning cell (`base_lgbm_kwargs=LGBM_KWARGS`) or the plain
  backtest predictors (`lightgbm`/`lightgbm_cov`/`lightgbm_tuned`/
  `lightgbm_cov_tuned`).

The mechanism that actually helped is the *outer-loop* one: Optuna's
`n_jobs=4` parallelizes across fully independent trials with no
synchronization between them, which is structurally the same shape as
origin-level parallelism (many small, independent fits) — not intra-fit
threading (one fit, split across threads with per-iteration barriers).

## Sizing: avoiding oversubscription between the two axes

`PREDICTOR_N_JOBS` and `ORIGIN_N_JOBS` are two independent knobs that can be
*simultaneously* busy — e.g. if all 4 LightGBM variants
(`lightgbm`/`lightgbm_cov`/`lightgbm_tuned`/`lightgbm_cov_tuned`) happen to
be running concurrently at the predictor level, each is *also* running its
own origin-level thread pool. Worst-case concurrent threads =
`PREDICTOR_N_JOBS × ORIGIN_N_JOBS`. The notebook computes both from
`os.cpu_count()` to land near, not several multiples over, that count:

```python
_CPU_COUNT = os.cpu_count() or 4
PREDICTOR_N_JOBS = min(4, _CPU_COUNT)
ORIGIN_N_JOBS = max(1, _CPU_COUNT // PREDICTOR_N_JOBS)
```

`PREDICTOR_N_JOBS` is capped at 4 regardless of core count, since most of the
roster (Naive, ETS, Kalman, LinReg) finishes quickly — 4 concurrent slots is
enough to keep the queue moving without over-provisioning threads that will
mostly sit on fast predictors. `ORIGIN_N_JOBS` then gets whatever's left per
slot. Deliberately computed at runtime rather than hardcoded — a stale
hardware assumption ("this machine's 4 physical cores, minus 1 free," written
next to the tuning cell's own `n_jobs=4`) was already found drifting from
actual `os.cpu_count()` output earlier in this project, so a fixed number
here would risk the same problem on a different machine.

## Verification

- Full existing `aieng-forecasting` test suite (103 tests across
  `test_backtest.py`, `test_artifacts.py`, `test_multi_target.py`,
  `test_eval.py`, `test_binary_scoring.py`, `test_categorical_scoring.py`)
  passes unmodified — confirms the default `n_jobs=1` path is behaviorally
  identical to the pre-change code for every existing caller.
- A caller-impact audit across the whole repo (~40 call sites of
  `run_eval_loop`/`backtest`/`multi_backtest`, including tests) confirmed
  none pass more positional arguments than the pre-change signatures — the
  new `n_jobs` parameter is keyword-only and purely additive, so no caller
  needed to change.
- A standalone check (not part of the permanent test suite) confirmed
  `backtest(..., n_jobs=1)` and `backtest(..., n_jobs=4)` produce identical
  predictions, in the same order, with the same `mean_score`, against a
  synthetic deterministic predictor — and that the retry-on-failure path
  (`max_retries`) still works correctly when dispatched through the thread
  pool (a flaky predictor that fails once then succeeds was retried and
  scored correctly for every origin, under `n_jobs=4`).
- **Gap in the above, found afterward:** that standalone check used a
  synthetic `ConstantPredictor`/`FlakyPredictor` — it verified
  `run_eval_loop`'s own dispatch/retry/ordering logic correctly, but never
  actually constructed a real Darts model under concurrency, which is exactly
  where the bug in the next section lives. Confirming a framework is correct
  is not the same as confirming every predictor built on top of it is safe
  under the concurrency the framework now allows.

## A real bug this parallelism change introduced (found and fixed)

Origin-level parallelism (`ORIGIN_N_JOBS>1`) surfaced a genuine race
condition in Darts itself, seen as intermittent failures like:

```
predict() failed at origin 2020-03-20 (attempt 1/3): 'ExponentialSmoothing' object has no attribute '_model_call' — retrying in 2s
predict() failed at origin 2020-02-19 (attempt 1/3): 'KalmanForecaster' object has no attribute '_model_call' — retrying in 2s
```

**Root cause:** every Darts forecasting model is built with a `ModelMeta`
metaclass (`darts/models/forecasting/forecasting_model.py`) that stashes
constructor arguments on the **class** (not the instance) as a hand-off from
`__call__` to `__init__`:

```python
# ModelMeta.__call__
cls._model_call = all_params
return super().__call__(**all_params)

# ForecastingModel.__init__ (elsewhere)
model_params = copy.deepcopy(self._model_call)
del self.__class__._model_call
```

This is not thread-safe. If two threads both call, say, `ExponentialSmoothing(...)`
at nearly the same moment — exactly what origin-level parallelism does, since
`DartsExponentialSmoothingPredictor.predict()`/`DartsKalmanForecasterPredictor.predict()`
construct a fresh model on every origin, and ETS/Kalman fits are fast enough
("well under a second," per their own docstrings) that many origins'
constructions land in the same tiny window — one thread's `__init__` can find
`_model_call` already deleted by the other thread's `__init__`, raising
`AttributeError`. Every Darts model shares this same metaclass (including
`LightGBMModel`), so the race isn't specific to ETS/Kalman; they just hit it
far more often because their whole fit is so fast that many origins' worth of
construction calls pile up in a short window, unlike LightGBM where each
construction is comparatively rare relative to its own fit time. The race is
also not confined to one predictor's own origin loop: `lightgbm` and
`lightgbm_cov` both build plain `LightGBMModel` (only the *tuned* variant
uses the separate per-quantile subclass), so Section 5's predictor-level
parallelism running both at once could race them against each other too — a
lock scoped to a single file/predictor would not have caught that case.

**Why this wasn't caught by the "Verification" section above:** that
verification exercised `run_eval_loop`'s own logic with a synthetic
predictor, not real Darts model construction under concurrency — see the new
bullet added there.

**In this notebook's specific usage, low risk of silent corruption:** each
predictor is constructed with the same arguments on every origin (e.g.
`KalmanForecaster(dim_x=self._dim_x)` — `dim_x` doesn't vary by origin), so
even a cross-thread parameter mix-up would swap identical dicts. The
observable symptom here was crashes (caught by `run_eval_loop`'s existing
retry logic, not silent wrong results) — but that's a property of how this
notebook happens to call these predictors, not a guarantee the underlying
race is harmless in general.

**Fix:** a single shared `threading.Lock()`
(`aieng/forecasting/methods/numerical/_darts_construction_lock.py`,
`DARTS_MODEL_CONSTRUCTION_LOCK`), imported by every predictor wrapper that
constructs a Darts model (`darts_classical.py`, `darts_arima.py`,
`darts_regression.py`), wrapped around just the constructor call — not
`.fit()`/`.predict()`. This serializes only the brief, buggy hand-off window;
the actual fit/predict computation, where origin-level and predictor-level
parallelism's real benefit lives, still runs fully concurrently. One shared
lock (not one per file) is required specifically because of the
cross-predictor case above (`lightgbm` vs. `lightgbm_cov` both building
`LightGBMModel`) — three independent locks would not protect against two
different files' predictors racing on the same underlying Darts class.

**Verification:** reproduced the race directly — 400 concurrent
`ExponentialSmoothing(...)` constructions across 16 threads, no lock: 5
failures with the exact `AttributeError` above. Same test with
`DARTS_MODEL_CONSTRUCTION_LOCK` held during construction: 0 failures across
400 calls. Full `aieng-forecasting/tests/aieng/forecasting/methods/numerical`
suite (28 tests) still passes.

## A pre-existing, unrelated bug found (and deliberately not fixed) while verifying this

Running the full `aieng-forecasting` test suite as part of verifying this
change surfaced 2 failures, both in
`tests/aieng/forecasting/methods/agentic/test_adk_runner.py::TestResponseExtraction`:
`test_returns_empty_string_when_stream_has_no_final_event` and
`test_returns_empty_string_when_final_event_has_no_content`. Both raise
`UnboundLocalError: cannot access local variable 'final_text'` from
`adk_runner.py:306`.

**Root cause:** in `AdkTextRunner.run_text_async`'s inner `drain_run()`,
`final_text` is only assigned inside
`if event.is_final_response() and event.content and event.content.parts:`
— so a response stream that never satisfies that condition (empty stream, or
a final event with no content/parts) leaves `final_text` unbound, and
`return final_text or ""` raises instead of returning `""` as the function's
own docstring promises.

**Confirmed unrelated to this change:** `adk_runner.py` is part of the
agentic/Google-ADK predictor runner — a different module entirely from
`backtest.py`/`artifacts.py`/`eval.py`, which this change touches. The failing
test file has zero references to `run_eval_loop`, `backtest`, or `n_jobs`.
Not fixed here — it's pre-existing (present before this work started) and out
of scope for a parallelism change. It's flagged inline at the bug site in
`adk_runner.py` (a `BUG(pre-existing)` comment pointing to this doc and the
two failing tests) so it's discoverable by whoever next touches that
function, without bundling an unrelated fix into this change.

## Summary of what was rejected, and why

| Option | Rejected because |
|---|---|
| Per-fit LightGBM multithreading (`num_threads`/`n_jobs` in `LGBM_KWARGS`) | Fits are too small (few thousand rows, <100 features) for intra-fit thread sync to pay off; would also oversubscribe against Optuna's own `n_jobs=4` in the tuning cell |
| `ProcessPoolExecutor` instead of threads | Picklability risk for `DataService`/`Predictor`/LLMP network clients; Windows spawn re-imports the whole stack per worker; LightGBM releases the GIL anyway, so threads already get the real speedup |
| Quantile-level (13-model) parallelism | Not exposed by Darts without monkey-patching its internal fit loop; origin-level parallelism already saturates available cores, so it wouldn't add real speedup |
| Task-level parallelism (the 3 horizon tasks) | Only 3× ceiling vs. dozens of origins per task; same "already saturated" reasoning as quantile-level |
