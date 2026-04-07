## Apr 7, 2026 — Implementations architecture: follow-up decisions [Ethan & Agent]

### Decided: remove `darts_arima.py`

The standalone `implementations/economic_forecasting/predictors/darts_arima.py`
is redundant now that `predictor_template.py` fills the "copy this to write
your own" role and the demo notebook shows the full implementation inline.
Keeping both creates a maintenance tax (two copies of the same code to keep
in sync) and a decision burden for participants ("which file do I start from?").

**Decision:** delete `darts_arima.py`. The pair of `predictor_template.py`
(starting point) + inline notebook definition (working demonstration) is
the clean, non-overlapping split.

When experiment scripts need an importable predictor, add the `.py` file at
that point with clear purpose — not speculatively.

### Decided: no mid-level `implementations/predictors/` layer yet

A top-level `implementations/predictors/` (method implementations not tied
to any use case) is the right model *once* the same predictor is needed in
two different use-case folders. That duplication hasn't appeared yet.
**Decision:** don't create the middle layer until there's a concrete reason.
Apply the same "two concrete instances before abstracting" rule used
everywhere else.

### Noted: agent backbone belongs in the package

When agentic predictors are built, the ADK setup, tool definitions, and
prompt scaffolding are reusable infrastructure — not use-case demonstrations.
Those should live in `aieng-forecasting` (e.g. `aieng/forecasting/agents/`),
while task-specific agent configuration lives in `implementations/`. This is
a cleaner cut than a mid-level `implementations/` layer for that case.

---

## Apr 7, 2026 — Lift predictors to implementations [Agent]

### What changed

**Architectural decision:** the `aieng-forecasting` package now owns zero
concrete predictor implementations. Darts ships models; the package ships
the interface. Reference implementations of how to use those models with
our `Predictor` ABC belong in `implementations/`, visible to bootcamp
participants in the same place as the notebooks and experiments.

**Deleted from the package:**
- `aieng-forecasting/aieng/forecasting/evaluation/predictors/arima.py`
- `aieng-forecasting/aieng/forecasting/evaluation/predictors/__init__.py`
- `ARIMAPredictor` removed from `evaluation/__init__.py` exports

**New files in `implementations/economic_forecasting/`:**
- `predictors/darts_arima.py` — `DartsAutoARIMAPredictor`, the moved
  implementation renamed to make the Darts dependency explicit
- `predictors/predictor_template.py` — annotated last-value naive baseline;
  the "start here" file for a participant writing their first predictor
- `README.md` — use case description, learning path, and interface reference

**Updated:**
- `cpi_backtest_demo.ipynb` — predictor now defined inline in the notebook
  (identical to `predictors/darts_arima.py`) so participants see the full
  implementation in the linear flow
- `technical-design.md` — package structure diagram updated; predictor
  placement principle documented
- Docstrings in `backtest.py` and `eval.py` no longer reference
  `ARIMAPredictor` by name

No test changes required — all tests use a locally-defined `ConstantPredictor`.
74 tests, `make lint` clean.

### Decisions made

- **Package = infrastructure only.** `Predictor` ABC + evaluation engines +
  data layer. No concrete implementations.
- **Darts predictors are not wrapped once and packaged.** The pattern of
  "here is how you wrap a Darts model in 20 lines" is demonstrated in
  `implementations/`, not encoded as a library class.
- **Agentic predictor backbone (future).** When we build ADK-based
  forecasting agents, the agent definition/tooling will live in the
  package (since it's genuinely reusable infrastructure). The task-specific
  configuration and experiments will live in `implementations/`.

---

## Apr 3, 2026 (session 4) — Faithfulness review: all TODOs resolved [Agent]

All 11 items from the session 3 audit were fixed in this session. Summary:

**Category A (doc/code inconsistencies):**
- `technical-design.md` Unified Loop: corrected `question_id` → `task_id`, `horizon` → `forecast_date`, added `as_of`
- `technical-design.md` `BacktestResult` snippet: added `skipped_origins: int`
- `arima.py`: replaced deprecated `datetime.utcnow()` with `datetime.now(tz=timezone.utc).replace(tzinfo=None)` (consistent with `backtest.py` and `eval.py`)
- `task.py` + `technical-design.md`: documented `resolution_fn` explicitly as a placeholder; the harness currently ignores it and always uses the default observed-value strategy

**Category B (stale documentation):**
- `technical-design.md` package structure: added `eval.py` and `backtest.py` to the diagram
- `technical-design.md`: removed stale inline "Build sequence for this layer" list from the Backtesting section (the ✅-marked Phase 1 sequence at the bottom is the authoritative tracker)
- `service.py` docstring example: updated table ID from `18-10-0004-13` → `18-10-0004-11`
- `ContinuousForecast` docstring: corrected quantile constraint ("keys must be in (0,1); standard levels recommended but not enforced")

**Category C (design debt):**
- Extracted `_compute_origins()` shared utility to `backtest.py`; both `BacktestSpec.origins()` and `EvalSpec.origins()` now delegate to it (DRY)
- Renamed `_run_eval_loop` → `run_eval_loop` (dropped private prefix since it intentionally crosses module boundaries into `eval.py`)
- Removed unused `import pandas as pd` from `eval.py` (pandas was only needed for the now-extracted origins logic)

74 tests, `make lint` clean.

---

## Apr 3, 2026 (session 3) — Faithfulness review: TODOs from doc/code audit [Agent]

The following items were found during a thorough cross-check of `technical-design.md`,
`planning-notes.md`, and all code under `aieng-forecasting/`. Grouped by severity.
Nothing was edited directly — these are TODOs to address in a future session.

### Category A — Genuine doc/code inconsistencies (fix promptly)

**TODO A1 — `technical-design.md`: stale field names in the Unified Loop section**
The Prediction bullet in "Unified Loop" reads:
> `question_id`, `predictor_id`, `issued_at`, `horizon`
The actual `Prediction` model uses `task_id` (not `question_id`) and `forecast_date`
(not `horizon`). `as_of` is also missing. Update that bullet to match the real fields.

**TODO A2 — `technical-design.md`: `BacktestResult` code snippet is incomplete**
The inline Python snippet in the "BacktestResult" subsection (inside "Backtesting:
User Model and Interfaces") is missing the `skipped_origins: int` field that exists
in the actual code and matters for understanding skip behaviour. Add it to the snippet.

**TODO A3 — `arima.py` line 125: `datetime.utcnow()` is deprecated**
`ARIMAPredictor.predict()` uses `datetime.utcnow()`, which is deprecated since
Python 3.12 and will warn (or break in future versions). Both `backtest.py` and
`eval.py` already use `datetime.now(tz=timezone.utc).replace(tzinfo=None)`. Fix
`arima.py` to match.

**TODO A4 — `ForecastingTask.resolution_fn` is defined but never used**
`ForecastingTask` has a `resolution_fn: str` field (default
`"observed_value_at_resolution_timestamp"`), but `_resolve()` in `backtest.py`
ignores it entirely — it always looks up the observed series value directly.
The field is currently dead config. Either: (a) document it explicitly as a
planned-but-unimplemented feature in both the task docstring and `technical-design.md`,
or (b) remove it and add it back when the dispatch is implemented. Leaving it silent
will confuse implementors who configure it expecting it to do something.

### Category B — Stale documentation (clean up when convenient)

**TODO B1 — `technical-design.md`: package structure diagram missing `eval.py`**
The diagram under "Package: aieng-forecasting" shows:
```
└── evaluation/
    └── predictors/
```
but `evaluation/eval.py` was added. Update the diagram to include it.

**TODO B2 — `technical-design.md`: "Build sequence for this layer" inside the
Backtesting section is stale planning content**
The numbered list (`1. ContinuousForecast + Prediction models … 7. End-to-end run`)
inside the Backtesting subsection was the pre-build plan and is now fully done. The
authoritative tracker is the ✅-marked Phase 1 Build Sequence at the bottom of the
document. The inline one adds no information and could confuse readers. Consider
removing it or replacing it with a pointer to the Phase 1 sequence.

**TODO B3 — `service.py` docstring example uses the old table ID `"18-10-0004-13"`**
The `DataService` class docstring example shows `table_id="18-10-0004-13"`, but the
canonical current table ID (corrected Apr 1) is `"18-10-0004-11"`. Both normalise to
the same zip, but for clarity the example should match what `scripts/fetch_cpi.py`
actually uses.

**TODO B4 — `ContinuousForecast` docstring overstates the validator constraint**
The docstring says `quantiles` "Must contain at least the standard levels defined in
`STANDARD_QUANTILES`", but the actual validator only checks that keys are in `(0, 1)`.
There is no enforcement of standard level presence. Fix the docstring to match the
real constraint, or add the enforcement if presence of standard levels is actually
required.

### Category C — Design debt (low priority, track for later)

**TODO C1 — `EvalSpec.origins()` and `BacktestSpec.origins()` are identical**
Both classes implement the same striding logic (`pd.date_range` + slice + `to_pydatetime()`).
Minor DRY violation. Could extract to a shared private utility or a common base class
when the design stabilises.

**TODO C2 — `_run_eval_loop` is a private function crossing module boundaries**
`eval.py` imports `_run_eval_loop` from `backtest.py` by its private name. A private
function in the public API of another module is an unusual pattern that could confuse
contributors. Consider either making it semi-public (drop the leading underscore) or
moving it to a shared internal module (e.g. `evaluation/_loop.py`) so the boundary is
explicit.

**TODO C3 — `eval.py`: `import pandas as pd` is misplaced in the import block**
`import pandas as pd` appears at line 49, after the pydantic/local imports, instead of
grouped with the other third-party imports. Doesn't fail lint (isort not configured)
but is inconsistent with project style.

---

## Apr 3, 2026 (session 2) — Prediction metadata + eval mode [Agent]

### What we built

Two additive features on top of the Phase 1 backtest layer.

**`Prediction.metadata`** (`aieng/forecasting/evaluation/prediction.py`):
- Added `metadata: dict[str, Any]` field to `Prediction`, defaulting to `{}`.
- The harness never reads or validates it — passes through transparently to `BacktestResult.predictions` and `EvalResult.predictions`.
- Predictors populate it with whatever structured side-channel data they want (token counts, source lists, Langfuse trace IDs, etc.). No schema enforced beyond `dict[str, Any]`.
- `Predictor` ABC docstring updated to document this as the canonical pattern for "things that travel with a prediction."

**Eval mode** (`aieng/forecasting/evaluation/eval.py`):
- `EvalSpec` — mirrors `BacktestSpec` with `spec_id` (tracker key) and `max_runs` (optional budget cap encoded directly in the spec YAML).
- `EvalResult` — analogous to `BacktestResult`, adds `run_number` provenance (which run against this spec this was).
- `EvalTracker` — file-backed YAML counter. `runs_for(spec_id)` / `record(spec_id, ran_at)`. Survives process restarts. Path is caller-supplied; wiring to per-user identity is deferred.
- `EvalBudgetExceededError` — `ValueError` subclass with a clear message when a budget is exhausted.
- `evaluate()` — checks budget, runs the shared `_run_eval_loop()` (also used by `backtest()`), records the run, returns `EvalResult` with `run_number`.
- `reference_specs/cpi_allitems_eval_2yr.yaml` — 2024–2026 held-out window, `max_runs: 5`, as a worked example.

**Infra:** extracted `_run_eval_loop()` from `backtest()` to avoid duplication. `evaluation/__init__.py` exports all new symbols. 23 new tests (74 total). `make lint` clean. `technical-design.md` updated.

### Decisions made in discussion

- **Predictor side-effects are free** — predictors may write logs, traces, or any other artifacts as side-effects without the harness caring. `Prediction.metadata` is only for structured data that should travel *with* each prediction.
- **`metadata` stays generic** — `dict[str, Any]`. No schema enforced at the interface level; users define structure internally.
- **Eval mode is the "validation set" concept** — backtesting is for learning/tuning (run freely); eval is the held-out window (spend deliberately). `max_runs` on the spec + `EvalTracker` enforce this.
- **Notebook polish tabled** — the demo notebook runs well. Confidence interval shading and multi-series comparison are deferred.

### What's next

1. **Second predictor** — add a second variant (seasonal naive or fixed-order ARIMA via Darts) to make the comparison the Phase 1 plan called for. This will also be the first real use of `Prediction.metadata` in practice.
2. **Pass 2 — Metaculus** — `BinaryForecast`, `BinaryPredictor` ABC, discrete event evaluation loop.
3. **Per-user eval tracking** — defer until bootcamp infrastructure is more defined, but the hook (`EvalTracker` path) is ready.
4. **Notebook polish** — confidence interval shading, multi-series CPI comparison (Food, Shelter, Water/fuel/electricity). Tabled for now but worth revisiting before the bootcamp.

---

## Apr 3, 2026 (session 1) [Ethan]

A couple of questions on my mind today. These might not be actual problems, but things to think about against our design.

- We're working towards standardizing interfaces for the backtesting/eval engines. I want to make sure we're thinking about all the kinds of artifacts that a Predictor might produce over the course of its work. Should we be defining interfaces for these? I'm thinking not. There is probably a basic level of data we expect a Predictor to produce (... the predictions ...) but I think beyond that we might not want to over-design things. Maybe we should leave it to the user to build/adapt their predictors so that they produce the side-effects/outputs they desire. BUT if there is a reasonable pattern for Predictors to be able to return additional artifacts as they're used in an experiment, that might be good to do. Overall I am thinking: if users want to implement Predictors that create additional artifacts (like agent traces, statistics, plots, other data, etc.) -- do we want to support that in the official interfaces or leave that to users to deal with? Is there some basic pattern that makes sense for cases like this, or are we better to just completely separate backtest/experiments/evals from whatever else might be going on in a predictor implementation? I definitely need some advice here!

- Another thing I want to consider is building in support for something in between backtesting and live testing. I'm imagining right now if we applied a meta learning algorithm (or even just an agentic loop) over the backtesting process, there is lots of opportunity for bias/data leakage. We might want to add a third mode (validation? test? ???) with the expectation that users would run very many backtesting runs for learning/tuning/exploration purposes, but then want to run relatively few checks against a slice of recent data. This could look more like rolling/time series cross validation, but where we are especially trying to limit how much we might "learn" from the most recent data. Of course, there will be no substitute for true live testing, especially with agentic predictors that might access live sources of information (i.e. even if we try to cut off news searches past a certain date, it will be really hard to control information leakage, but we will try).

- And just a small thing, but I want to make sure we iterate on the base reference experiment and notebook a bit. I would love to actually see the full prediction confidence intervals as shaded regions, clean up some of the overlapping label, and focus just on predictions for the last 10 years. Perhaps we could expand it to a few more CPI time series, e.g. I would love to see some of the main categories compared: "Food" "Shelter" "Water, fuel and electricity" in a nice, clean visual analysis.

## Apr 2, 2026 (session 5) — CPI backtest end-to-end implementation [Agent]

### What we built

Full Phase 1 evaluation layer — the repo now supports a complete, runnable backtest.

**New evaluation layer** (`aieng/forecasting/evaluation/`):
- `prediction.py` — `ContinuousForecast` (point + quantiles at 0.05…0.95), `Prediction` (metadata wrapper). Both YAML-serializable Pydantic models.
- `predictor.py` — `Predictor` ABC with `predict(task, context) -> Prediction` and `predictor_id` property.
- `backtest.py` — `BacktestSpec`, `BacktestResult`, `backtest()` function. Derives origins from spec, enforces warmup, resolves ground truth, scores with CRPS (`properscoring`).
- `predictors/arima.py` — `ARIMAPredictor` using Darts `AutoARIMA`. Fits per-origin, generates 500 Monte Carlo samples, extracts quantiles at standard levels.

**Data layer fix:**
- `StatCanAdapter.fetch()` now populates `released_at = timestamp + 21 days` to approximate StatCan's publication lag. Removes the optimistic bias in backtests.

**Reference spec:** `reference_specs/cpi_allitems_12m.yaml` — CPI All-items, 12-month horizon, January and July origins, 2000–2026, warmup=24.

**Demo notebook:** `implementations/economic_forecasting/cpi_backtest_demo.ipynb` — 7-cell walkthrough from data registration through serialized result YAML.

**New dependencies:** `darts==0.43.0`, `properscoring==0.1` (+ transitive: scipy, statsmodels, scikit-learn, numba, etc.).

**Tests:** 51 total (was 32), all passing. `make lint` clean.

### Confirmed full pipeline

```python
import yaml
from aieng.forecasting.evaluation import ARIMAPredictor, BacktestSpec, backtest

with open("reference_specs/cpi_allitems_12m.yaml") as f:
    spec = BacktestSpec.model_validate(yaml.safe_load(f))

results = backtest(predictor=ARIMAPredictor(), spec=spec, data_service=svc)
print(f"Mean CRPS: {results.mean_crps:.4f}")
```

### What's next

1. **Run the actual backtest** — execute `cpi_backtest_demo.ipynb` end-to-end, record the real mean CRPS as a baseline number.
2. **Second predictor** — add a second variant (e.g., fixed-order ARIMA or a seasonal naive via Darts) to make the comparison promised in the plan.
3. **Pass 2 — Metaculus** — `BinaryForecast`, `BinaryPredictor` ABC, discrete event evaluation loop.

---

## Apr 2, 2026 (session 4) — Backtest interface design [Ethan & Agent]

### Design direction decided (no code yet)

**How users run backtests.** Users invoke backtests directly — `backtest(predictor, spec, data_service)` — and get results back in-process. No submission engine. This is right for the bootcamp: low friction, immediate feedback, easy iteration. The submission-based model (ForecastBench, Numerai) is deferred; it layers on top naturally once `BacktestResult` is serializable.

**`BacktestSpec` separates *what* from *when*.** `ForecastingTask` defines the prediction problem (target, horizon, frequency). `BacktestSpec` wraps a task and adds the evaluation window (`start`, `end`, `stride`, `warmup`). Both are Pydantic models, both YAML-serializable. Reference specs for canonical tasks will live in `reference_specs/` (YAML, versioned). Participants use them as-is or customize.

**`BacktestResult` is a first-class, serializable object.** Not just a DataFrame of scores — a Pydantic model containing the full spec, predictor identity, list of `Prediction` objects, per-origin scores, and summary stats. Design goals: YAML-roundtrippable (for persistence and versioning), passable to agents as structured context, comparable across predictors on the same spec, and forward-compatible with a future submission/leaderboard mechanism.

**The bridge to live evaluation:** "submitting a backtest result" in a future competition just means serializing this object and sending it somewhere. Nothing in the backtest-first design forecloses that.

### Updated next steps (Phase 1 build sequence)

1. `ContinuousForecast` + `Prediction` Pydantic models — YAML-serializable, hashable
2. `Predictor` ABC — `predict(task, context) -> Prediction`
3. Naive baseline predictor (Darts) — forcing function for the interface
4. `BacktestSpec` + `BacktestResult` Pydantic models — define interfaces before writing the loop
5. `backtest()` function
6. `released_at` fix for StatCan CPI (removes optimistic bias)
7. Reference spec YAML for CPI All-items (`reference_specs/cpi_allitems.yaml`)
8. End-to-end run: two predictor variants on CPI All-items, compare CRPS

---

## Apr 2, 2026 (session 3) — ForecastContext: interface design + implementation [Ethan & Agent]

### What we discussed

This session was a deliberate, slow design conversation before building. Key threads:

- **Backtesting first, live evaluation later.** The bootcamp is the near-term target. Live evaluation (including the longer-term on-chain / public immutable prediction vision) is kept in sight but not built yet. The design goal: don't commit to anything in the backtesting layer that would make the live evaluation path harder.
- **The `DataService` naming problem.** "Service" implies a running process/API. What we have is an in-process Python object. The name was misleading to future participants. More importantly, `DataService` was doing two things: (1) registration/management of series, and (2) providing a data view to predictors. These deserve to be distinct.
- **The `as_of` footgun.** The proposed predictor signature `predict(task, data_service, as_of)` had a subtle flaw: `as_of` and `DataService` are separate arguments, making cutoff enforcement opt-in. A predictor (especially an agentic one using tool calls) could forget to pass `as_of` when querying. The fix: bake `as_of` into the object the predictor receives.
- **`ForecastContext` decided.** The clean solution is a lightweight, read-only, cutoff-scoped companion to `DataService`. The harness creates it via `DataService.context(as_of)`. Predictors receive a `ForecastContext` and call `get_series()` without ever managing the cutoff date themselves.
- **What `DataService` is for.** Registration (called by setup scripts), ad-hoc notebook queries, and `summary()`. Not passed to predictors.
- **Information discipline for agentic predictors.** LLM-based predictors using live tools (news, web search) cannot be retroactively cut off. This is inherent and known — it's part of the challenge, and part of what will cause poor backtest-to-live generalization. Not a system flaw.
- **Feast / point-in-time correctness analogy.** The closest prior art is ML feature stores (Feast's "point-in-time join"). No forecasting library we're aware of has this as a first-class concept — this is a genuine differentiator of our architecture.

### What we built

- **`ForecastContext`** (`aieng/forecasting/data/context.py`): read-only, cutoff-scoped data view. `get_series()` always enforces `as_of`. `get_metadata()` and `series_ids` also delegated.
- **`DataService.context(as_of)`**: factory method that creates a `ForecastContext` from the underlying `SeriesStore`.
- **8 new tests** (`TestForecastContext` + 2 `TestDataService` factory tests): 32 total, all passing. `make lint` clean.
- **`technical-design.md` updated**: `ForecastContext` section added to evaluation architecture; `DataService` architecture diagram updated; predictor interface contract documented.

### Confirmed predictor interface

```python
def predict(task: ForecastingTask, context: ForecastContext) -> Prediction:
    series = context.get_series(task.target_series_id)
    # series is already filtered to context.as_of — can't leak future data
    ...
```

### What's next

1. **Define `Prediction` payload types** — `ContinuousForecast` Pydantic model (point + quantiles, `predictor_id`, `task_id`, `issued_at`, `as_of`, `horizon`). Design for serializability from day one (YAML-roundtrippable, hashable) so persistence / on-chain submission is easy to add later. `BinaryForecast` can wait for the Metaculus pass.
2. **Define `Predictor` ABC** — abstract base class with `predict(task, context) -> Prediction`. Probably lives in `aieng/forecasting/evaluation/predictor.py`. Keep it minimal.
3. **First baseline predictor** — naive (last known value) or seasonal naive via Darts, implementing the `Predictor` ABC. This is the forcing function that validates the interface end-to-end.
4. **Backtesting harness** — iterate over historical origins for a `ForecastingTask`, call `predictor.predict()`, collect `Prediction` objects, resolve against the series, score with CRPS. Goal: run two predictor variants on the CPI All-items task and compare results. This is the first complete end-to-end backtest.
5. **`released_at` for StatCan** — StatCan CPI is published ~3 weeks after the reference month. Fix this before running the backtests so results are not optimistically biased.

---

## Apr 2, 2026 (session 2) [Ethan]

- I've now played around with the statcan code a bit and found it flexible enough to start with. We can download historical data into series just fine.
- I think the next step is to think about how we could start working with an actual forecasting problem. I'm thinking today about backtesting, evaluation, and how we might design the live evaluation mechanism.
-- If we do move towards "live evaluation" how will forecasters "submit" predictions? Where will those be stored? How will predictors receive feedback? I think there are still some common elements to backtesting here, but I think we need to do some thinking before committing to an architecture.
- Some other/related things that are on my mind:
-- We might want to separate logging/tracing/observability from forecast submission and evaluation. LangFuse really might not be the best tool to try to do everything. We should at least challenge this before committing to it. I think it will be important that the mechanism for submitting forecasts is super simple and easy to follow. Just like the backtesting engine.
-- I still have it in mind that if we define really sound interfaces for predictors, users should be able to participate in backtesting and live evaluation without any trouble.
-- I think this is something I want to think about: would it make sense for the data service and/or backtesting engine to determine and limit (well, try to limit) what data are available to the predictor, and then the predictor can just do whatever it does under the hood, then generate a prediction? I guess my question is: are the underlying interfaces and the contracts between the components in this system really possible to keep simple, elegant, and effective? I want to do some good, slow thinking about this before we get much further.
-- And at the end of this, I would love to start with a build goal of running a backtest on two variations of a backtest on a reference forecasting task, just to establish that it works.

## Apr 1, 2026 — CPI series expansion and notebook update (session 1) [Agent]

### What we completed

- **CPI series expanded from 9 → 47**: `scripts/fetch_cpi.py` now registers all product-group series visible at [table 18-10-0004-11](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000411). Table ID updated from `18-10-0004-13` to `18-10-0004-11` (both normalise to the same `18100004.zip` download; the suffix is a variant label). All 47 series register with 0 failures. Series span food sub-components, shelter sub-components, energy, transportation, clothing, health, recreation, alcohol/tobacco, and special aggregates (core, ex-gasoline, etc.).
- **`cpi_data_exploration.ipynb` updated**: now selects 6 representative series — All-items, Core (ex-food & energy), Shelter, Food, Energy, Gasoline — and demos both a combined overlay plot and a small-multiples view. YoY change plot updated to show all 6 series together.
- **Notebook outputs policy decided**: `nbstripout` removed from pre-commit config. Contributors control output stripping per-notebook. `technical-design.md` updated with this decision. Exploration notebooks like `cpi_data_exploration.ipynb` may commit outputs for readability.
- **`fetch_statcan.ipynb` cleaned up**: hardcoded absolute `CACHE_DIR` path replaced with a relative path.

### Current state

- 47 Canada-wide CPI series loadable from StatCan table 18-10-0004-11
- `cpi_data_exploration.ipynb` is a runnable demo of series selection and visualisation
- Notebook output policy is explicit and documented

### What's next (unchanged from Mar 31 - session 6)

1. **First baseline predictor** — naive/seasonal naive via Darts, forcing definition of `ContinuousForecast` and `Predictor` ABC.
2. **Backtesting loop** — iterate over historical forecast origins, collect predictions, resolve, score with CRPS.
3. **`released_at` for StatCan** — CPI is published ~3 weeks after the reference month; `released_at=None` introduces optimistic backtest bias.
4. **Second data source** — FRED adapter.

---

## Mar 31, 2026 — Bugfix, cleanup, and test review (session 6) [Agent]

### What we completed

- **stats_can / pandas-3 incompatibility fixed**: `stats_can v3.1.0` calls `pd.to_datetime(..., errors="ignore")` which was removed in pandas 3.0. Fixed by bypassing `zip_table_to_dataframe()` entirely — we now use `stats_can.sc.download_tables()` for the download step and read the CSV from the zip directly with `_read_zip()`. All 9 CPI series now register successfully (`scripts/fetch_cpi.py` works end-to-end).
- **Test suite trimmed**: 41 → 24 tests. Removed Pydantic construction tests, trivial Python dict behavior, and mock-interaction assertions. Kept tests for non-obvious logic (cutoff fallback rules), defensive copy contracts, error paths with meaningful messages, and the fetch/filter contract. Also fixed a `test_get_returns_copy` that was accidentally passing under pandas 3 Copy-on-Write semantics.
- **`nbstripout` added** as a pre-commit hook: notebook outputs are automatically stripped on commit. Cleaned existing notebook outputs from the repo. Also suppressed `B018` (bare expression) in `nbqa-ruff` since lone expressions are idiomatic for DataFrame display in Jupyter cells.
- **Cleanup**: fixed `scripts/setup.sh` (still referenced `aieng-template-implementation`), ensured test fixtures use `tmp_path` so no stray `data/` directories are created under `aieng-forecasting/`, and anchored the `.gitignore` entry accordingly.
- **Tech doc corrected**: removed stale `past_covariate_ids` reference from `ForecastingTask` description (those are predictor concerns), and corrected `SeriesMetadataStore` to reflect that metadata lives inside `SeriesStore`.

### Current state of the codebase

The data service foundation is fully functional:
- `DataService` → `SeriesStore` + `CutoffEnforcer` + `StatCanAdapter`
- `ForecastingTask` model defined (problem spec only)
- 9 Canada-wide CPI series loadable from StatCan
- 24 tests, all passing; pre-commit hooks all green

### What's next (suggested)

1. **First baseline predictor** — A naive/seasonal naive predictor using Darts, implementing a `Predictor` interface. This forces us to define the `Prediction` payload type (`ContinuousForecast`) and the `Predictor` ABC. The payoff: we can run end-to-end eval on CPI.
2. **Backtesting loop** — Once a predictor exists, wire up the backtesting harness: iterate over historical forecast origins for a `ForecastingTask`, collect `Prediction` objects, resolve against `SeriesStore`, score with CRPS.
3. **`released_at` for StatCan** — StatCan CPI is published ~3 weeks after the reference month; we currently set `released_at=None` (falls back to timestamp). Fix this to remove the optimistic bias in backtests.
4. **Second data source** — FRED adapter for US economic series (simple once StatCan pattern is established).

---

## Mar 31, 2026 — Data service design + long-term vision (session 3) [Ethan & Agent]

Key decisions and design refinements; full details in `technical-design.md`.

- **Canonical internal format**: each series stored as a `(timestamp, value, released_at?)` DataFrame; `series_id` is the store key, not a column. `released_at` is optional, defaults to `timestamp` — this handles both official datasets with known release lags and custom bring-your-own datasets with no lag.
- **Single-value-column convention**: one quantity per series object. Multivariate data = multiple registered series. Series relationships (covariates) are declared in `ForecastingTask`, not in the data format.
- **`ForecastingTask`**: new Pydantic model that parameterizes the evaluation loop — binds `target_series_id`, `horizon`, `frequency`, `past_covariate_ids`, `future_covariate_ids`, `gap_fill_strategy`, and `resolution_fn`. This is how series relationships and covariate structure are captured. It should be easy for users to create variants of these.
- **Gap-filling at conversion boundary**: `SeriesStore` makes no regularity guarantees. Gap-filling (ffill, interpolate, etc.) is an explicit step when converting to `darts.TimeSeries`, governed by `ForecastingTask.gap_fill_strategy` (or this could be method dependent and up to the user). LLM predictors may skip this entirely.
- **Adapter protocol**: `BaseAdapter` requires one method — `fetch() -> pd.DataFrame`. `LocalCSVAdapter` is the first-class path for custom datasets, requiring only column-name mappings.
- **Series relationships open question**: task-scoped covariate declarations via `ForecastingTask` handle the immediate need. A global covariate/series relationship registry (for discovery across tasks) is deferred.
- **Long-term vision confirmed**: the project should serve both the bootcamp (learning + experimentation) and an ongoing forecasting benchmark/competition. The evaluation loop's backtest/live symmetry is the architectural property that makes this feasible.

---

## Mar 31, 2026 — First build: StatCan CPI data service (session 5) [Agent]

Implemented the data service layer and StatCan CPI adapter. All 35 unit tests passing.

- **Package**: renamed template `aieng-topic-impl` → `aieng-forecasting`. Import namespace: `aieng.forecasting`. Registered in uv workspace.
- **Data service** (`aieng/forecasting/data/`): `SeriesRecord`, `SeriesMetadata` (Pydantic), `SeriesStore`, `CutoffEnforcer`, `DataService`, `BaseAdapter` (ABC), `StatCanAdapter`.
- **Evaluation stub** (`aieng/forecasting/evaluation/`): `ForecastingTask` Pydantic model.
- **StatCan CPI**: `StatCanAdapter` for table `18-10-0004-13` (Canada-wide CPI by product group, 2002=100 baseline). `member_filter` dict selects one series per instance.
- **Data cache**: `data/statcan/` (gitignored). Scripts: `scripts/fetch_cpi.py` registers 9 Canada-wide CPI series, prints summary.
- **Notebook**: `implementations/economic_forecasting/cpi_data_exploration.ipynb` — demonstrates registration, cutoff filtering, plotting, YoY change, and `ForecastingTask` definition.
- **Note on released_at**: StatCan CPI is published ~3 weeks after the reference month. `StatCanAdapter` currently sets `released_at=None`, so `CutoffEnforcer` falls back to `timestamp`. This introduces a slight optimistic bias in backtests; fixing it requires populating `released_at` from StatCan's release schedule.

---

## Mar 31, 2026 — ForecastingTask / Predictor separation (session 4) [Ethan & Agent]

- Clarified that `ForecastingTask` defines the *problem* only: `task_id`, `target_series_id`, `horizon`, `frequency`, `resolution_fn`, `description`. It says nothing about how to solve the problem.
- Covariate selection, gap-fill strategy, and model choice are all `Predictor` responsibilities. A predictor requests whatever series it wants from the `DataService` (subject to cutoff); the task doesn't constrain this.
- Series relationships (e.g., CPI sub-components) live in dataset documentation and config files — no formal global registry needed at our scale. Explicitly deferred.
- Removed global covariate registry from open questions in `technical-design.md`.

---

## Mar 31, 2026 — Architecture decisions (session 2) [Ethan & Agent]

Key decisions from this session are now recorded in `technical-design.md`. Summary:

- **Darts** selected as primary numerical forecasting library (over sktime). Reasons: consistent API, first-class backtest utilities, modular install, lower support burden.
- **Evaluation architecture**: unified loop — `Predictor → Prediction → Resolution → Score`. Backtesting and live evaluation share the same architecture; they differ only in whether ground truth is already known.
- **Two prediction payload types**: `ContinuousForecast` (values + quantiles) and `BinaryForecast` (probability). Metaculus conventions followed for the latter.
- **Data service** is a standalone package. Deterministic data (historical series, resolution targets) is pre-populated locally; stochastic context (news, web search) is live at call time. Components: `SeriesStore`, `ResolutionStore`, `CutoffEnforcer`, and provider adapters for StatCan, FRED, and yfinance.
- **Information cutoff discipline** via `CutoffEnforcer` is the unifying teaching concept across both paradigms.
- **Langfuse** for tracing, integrated at the Predictor level.
- **Build plan**: two concrete passes (StatCan economic series first, then Metaculus) before extracting shared abstractions.
- **Open question**: how the data service handles new monthly data releases (important for live benchmark extension).

Also created `technical-design.md` as the technical source of truth, and updated `AGENTS.md` with maintenance instructions.

---

## Mar 31, 2026 [Ethan]

I am indeed thinking it makes sense for me to just start building around (1) the Canada's Food Price Report (CFPR) forecasting task and (2) Metaculus forecasting questions. These cover two distinct forecasting modalities: multivariate/multi-target time series forecasting and discrete event prediction.

I just updated the bootcamp-project-charter. A couple of ideas are coming together:
- LLMPs are starting to look a lot like a special case of forecasting agent. A basic LLMP might be something closer to an "LLMFunction" than a full agent, where an LLMFunction is a configured LLM call where the prompts, examples, input data and output format are all specified. I've used this repo in the past: https://github.com/567-labs/instructor  (Note: might want to look at Pydantic AI -- https://ai.pydantic.dev/#why-use-pydantic-ai) But generally speaking, it might be good for us to unify around Google ADK and build as much as possible using its native/built-in features. In fact, let's take the approach of: try to build it with ADK, and only if we're blocked should we introduce additional dependencies.
- So, the "baseline" LLMP could be a simple agent that has some basic access to historical data (perhaps it can get fixed observations via a tool) and contains instructions in the system prompt for how to produce a structured forecast as output, which should be defined (and validated!) by a Pydantic dataclass.
- Then more advanced agentic forecasters including hybrids will look more like modern agents with tools, code execution, agent skills, etc. LLMPs might use tools and/or code to build additional context before producing a forecast.
- One specific example of a hybrid numerical/LLMP could be an agent that uses tools or code to produce a numerical forecast (or even ensemble of forecasts) and can additionally fetch context from a number of sources. This way, the additional context could be used to condition the numerical forecasts, and a "challenge" could be to find the right agent design/configuration to best leverage these sources of data.
- At the highest level, we could just have open-ended coding agents that could do *any kind of analysis* they think is helpful before producing a forecast. This could be a super interesting search space for participants to consider and for us to consider in longer-running experiments.

- This leads into some early thinking about how we should support backtesting and live evaluation.
-- We should definitely separate the prediction task from the methods.
-- Doesn't matter whether the forecast comes from an LLM or a stats model: we have to define the interfaces for "submitting predictions"
-- The interfaces will differ depending on the forecasting task, but we should try to set standards as early as possible.
-- We shouldn't invent anything here -- I'm imagining point forecasts at one or more discrete future time points, variations with confidence percentiles (or similar -- let's try to follow what others are doing per task type, like for discrete event forecasting we should just follow exactly what Metaculus uses.)

- I wonder if this is actually the FIRST thing we should do: define the forecasting tasks for two broad types of tasks, i.e. economic/financial forecasting and Metaculus predictions, starting with the former, and we can be even more specific and zoom in on the CPI and simlar series from StatCan as targets.

- We can use these to create both the backtesting engine and "live" prediction resolution engine, then baseline test it with methods from Darts.
- In fact, after some basic review, I think we can make an early decision to lean more into Darts as the default/reference forecasting library that will will support.
-- (We could even consider building a set of agent skills that would enable agents to use Darts more effectively...)

## Mar 30, 2026 [Ethan]
TODOs

- Dig into metaculus “data” to see what we can get. Will likely need to reach out to their team to get access for research purposes.
- I want to get to a clear articulation of problems / use cases that will let us start to focus on building.
- Meeting: when to get data office involved? I need to make some requests e.g. to Metaculus

Thinking and planning

- Design architecture for the repo. How do we want it to work, exactly? Think about this from the user’s perspective.
- Start framing some basic forecasting questions around accessible data sources.
- How will backtesting work?
- How will “live” forecasting work?
- Do we need a data service?
- What will the common interfaces look like?
- How do we want to organize the repo – around datasets? Techniques? We’ll probably want to be able to experiment with combinations. This makes me think we should separate src code based on techniques.
- Do we want to build this with tight langfuse integration? Make this possible with an adapter of some sort?

Data services and interfaces. Without creating too much complexity, I think we’ll want to enable flexible use of data sources. Data could be used for several purposes: prediction targets, historical observations and targets, exogenous features, etc. We’ll want to support our own reference datasets and make it very clear how a custom dataset could be integrated with our platform. Maybe we should think about this more as an experiment platform than a collection of reference implementations. The platform should enable participants to try different things during the bootcamp. Ideally this will largely be about exploring and evaluating techniques. It could be with the goal of improving accuracy or ensuring interpretability. Teams could be interested in creating an interpretability / experiment results dashboard for example.

But ideally we’ll be focusing on exploring the things that (hopefully) improve forecasting accuracy and consistency.

So perhaps a big thing to start with are the “no regrets” datasets and forecasting tasks we want to consider. Could make it easy on myself and just replicate the Canada’s Food Price Report forecasting task. It’s established and serves as a quite clean “reference implementation” – further this could be a really great collaboration opportunity.
