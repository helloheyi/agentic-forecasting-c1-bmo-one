## Apr 1, 2026 — CPI series expansion and notebook update (session 7)

### What we completed

- **CPI series expanded from 9 → 47**: `scripts/fetch_cpi.py` now registers all product-group series visible at [table 18-10-0004-11](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000411). Table ID updated from `18-10-0004-13` to `18-10-0004-11` (both normalise to the same `18100004.zip` download; the suffix is a variant label). All 47 series register with 0 failures. Series span food sub-components, shelter sub-components, energy, transportation, clothing, health, recreation, alcohol/tobacco, and special aggregates (core, ex-gasoline, etc.).
- **`cpi_data_exploration.ipynb` updated**: now selects 6 representative series — All-items, Core (ex-food & energy), Shelter, Food, Energy, Gasoline — and demos both a combined overlay plot and a small-multiples view. YoY change plot updated to show all 6 series together.
- **Notebook outputs policy decided**: `nbstripout` removed from pre-commit config. Contributors control output stripping per-notebook. `technical-design.md` updated with this decision. Exploration notebooks like `cpi_data_exploration.ipynb` may commit outputs for readability.
- **`fetch_statcan.ipynb` cleaned up**: hardcoded absolute `CACHE_DIR` path replaced with a relative path.

### Current state

- 47 Canada-wide CPI series loadable from StatCan table 18-10-0004-11
- `cpi_data_exploration.ipynb` is a runnable demo of series selection and visualisation
- Notebook output policy is explicit and documented

### What's next (unchanged from session 6)

1. **First baseline predictor** — naive/seasonal naive via Darts, forcing definition of `ContinuousForecast` and `Predictor` ABC.
2. **Backtesting loop** — iterate over historical forecast origins, collect predictions, resolve, score with CRPS.
3. **`released_at` for StatCan** — CPI is published ~3 weeks after the reference month; `released_at=None` introduces optimistic backtest bias.
4. **Second data source** — FRED adapter.

---

## Mar 31, 2026 — Bugfix, cleanup, and test review (session 6)

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

## Mar 31, 2026 — Data service design + long-term vision (session 3)

Key decisions and design refinements; full details in `technical-design.md`.

- **Canonical internal format**: each series stored as a `(timestamp, value, released_at?)` DataFrame; `series_id` is the store key, not a column. `released_at` is optional, defaults to `timestamp` — this handles both official datasets with known release lags and custom bring-your-own datasets with no lag.
- **Single-value-column convention**: one quantity per series object. Multivariate data = multiple registered series. Series relationships (covariates) are declared in `ForecastingTask`, not in the data format.
- **`ForecastingTask`**: new Pydantic model that parameterizes the evaluation loop — binds `target_series_id`, `horizon`, `frequency`, `past_covariate_ids`, `future_covariate_ids`, `gap_fill_strategy`, and `resolution_fn`. This is how series relationships and covariate structure are captured. It should be easy for users to create variants of these.
- **Gap-filling at conversion boundary**: `SeriesStore` makes no regularity guarantees. Gap-filling (ffill, interpolate, etc.) is an explicit step when converting to `darts.TimeSeries`, governed by `ForecastingTask.gap_fill_strategy` (or this could be method dependent and up to the user). LLM predictors may skip this entirely.
- **Adapter protocol**: `BaseAdapter` requires one method — `fetch() -> pd.DataFrame`. `LocalCSVAdapter` is the first-class path for custom datasets, requiring only column-name mappings.
- **Series relationships open question**: task-scoped covariate declarations via `ForecastingTask` handle the immediate need. A global covariate/series relationship registry (for discovery across tasks) is deferred.
- **Long-term vision confirmed**: the project should serve both the bootcamp (learning + experimentation) and an ongoing forecasting benchmark/competition. The evaluation loop's backtest/live symmetry is the architectural property that makes this feasible.

---

## Mar 31, 2026 — First build: StatCan CPI data service (session 5)

Implemented the data service layer and StatCan CPI adapter. All 35 unit tests passing.

- **Package**: renamed template `aieng-topic-impl` → `aieng-forecasting`. Import namespace: `aieng.forecasting`. Registered in uv workspace.
- **Data service** (`aieng/forecasting/data/`): `SeriesRecord`, `SeriesMetadata` (Pydantic), `SeriesStore`, `CutoffEnforcer`, `DataService`, `BaseAdapter` (ABC), `StatCanAdapter`.
- **Evaluation stub** (`aieng/forecasting/evaluation/`): `ForecastingTask` Pydantic model.
- **StatCan CPI**: `StatCanAdapter` for table `18-10-0004-13` (Canada-wide CPI by product group, 2002=100 baseline). `member_filter` dict selects one series per instance.
- **Data cache**: `data/statcan/` (gitignored). Scripts: `scripts/fetch_cpi.py` registers 9 Canada-wide CPI series, prints summary.
- **Notebook**: `implementations/economic_forecasting/cpi_data_exploration.ipynb` — demonstrates registration, cutoff filtering, plotting, YoY change, and `ForecastingTask` definition.
- **Note on released_at**: StatCan CPI is published ~3 weeks after the reference month. `StatCanAdapter` currently sets `released_at=None`, so `CutoffEnforcer` falls back to `timestamp`. This introduces a slight optimistic bias in backtests; fixing it requires populating `released_at` from StatCan's release schedule.

---

## Mar 31, 2026 — ForecastingTask / Predictor separation (session 4)

- Clarified that `ForecastingTask` defines the *problem* only: `task_id`, `target_series_id`, `horizon`, `frequency`, `resolution_fn`, `description`. It says nothing about how to solve the problem.
- Covariate selection, gap-fill strategy, and model choice are all `Predictor` responsibilities. A predictor requests whatever series it wants from the `DataService` (subject to cutoff); the task doesn't constrain this.
- Series relationships (e.g., CPI sub-components) live in dataset documentation and config files — no formal global registry needed at our scale. Explicitly deferred.
- Removed global covariate registry from open questions in `technical-design.md`.

---

## Mar 31, 2026 — Architecture decisions (session 2)

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

## Mar 31, 2026

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

## Mar 30, 2026
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
