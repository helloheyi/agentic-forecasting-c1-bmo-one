# Agentic Forecasting — Development Backlog

This file is a plain-text complement to ClickUp. It captures the current set of development tasks with enough detail to hand off to a new team member. Tasks are grouped into the active sprint first, then the holding queue. Update this file when tasks are started, completed, re-scoped, or reprioritized.

**Primary deliverable:** Bootcamp readiness. All sprint decisions should be made against this target first.

**Kaggle note:** Gemma 4 Good Hackathon final submission deadline is May 18, 2026. This is a "nice to have" that must not disrupt the bootcamp critical path. Task T6 is the relevant item; it is explicitly lower priority than T1–T5.

---

## Active Sprint

### T1 — CFPR Reference Experiment & Data Pipeline

**Theme:** Use case / Reference experiment
**Estimated effort:** ~1 week
**Dependencies:** None
**Owner:** TBD (good onboarding task — data-heavy, minimal package internals required)

**Context:**
Canada's Food Price Report (CFPR) is an annual report that forecasts food price inflation categories for the coming year. It is a well-understood, real-world forecasting task with publicly available historical predictions and ground-truth outcomes. It could be the primary use case for the Kaggle submission (T6) and a clean reference for comparing all forecaster types.

**Scope:**
- Source and document historical CFPR publications/predictions and ground-truth annual food CPI outcomes
- Ingest into the data service, likely via `LocalCSVAdapter` with a checked-in config YAML
- Define a `ForecastingTask` that mirrors the CFPR's actual prediction structure (annual horizon, category-level series — Food total, Bakery, Dairy, etc.)
-- Ethan can provide more details, but historically we used point predictions over an 18 month horizon with the prediction date up to about the August or September CPI release date. We typically used point predictions of what the observed CPI will be for 8 or 9 food categories. You can see in https://github.com/VectorInstitute/foodprice-forecasting how we set up this experiment (5 years ago!). We can use this to extract the reference forecasting experiment/task, but the methods themselves (including forecasts and evaluations) are outdated.
- Write a reference `BacktestSpec` YAML in `reference_specs/`
- Produce a demo notebook under `implementations/experiments/cfpr/` showing `DartsAutoARIMAPredictor` (imported from `methods/`) applied to the new task
- Write a `README.md` for the use case folder (learning path, task description, data provenance)

**Acceptance criteria:**
- `DartsAutoARIMAPredictor` runs end-to-end on the CFPR task with a reported mean CRPS
- Data provenance is documented (source, licence, how to refresh)
- A new team member can follow the notebook start-to-finish without consulting the package source

---

### T2 — Base LLMP Predictor

**Theme:** Reference method implementation
**Estimated effort:** ~1 week
**Dependencies:** None (applies to existing CPI task; CFPR application follows naturally)
**Owner:** TBD (requires comfort with LLM APIs and Pydantic; more senior member or Ethan)

**Context:**
The "base LLMP" (LLM Process) is the minimal viable LLM-based predictor: an "LLMFunction" rather than a full agent. It receives a structured prompt containing historical observations and a natural-language task description, and produces a validated `ContinuousForecast` via Pydantic. No agent framework side-effects, no hidden injections — just a configured LLM call with structured output. This is the prerequisite for T6 (fine-tuning) and the simplest entry point for the LLM forecasting paradigm in the bootcamp.

**Design question to resolve (document the decision):** Use LiteLLM directly, or use Google ADK in a non-agentic "LLMFunction" mode? If ADK, verify it introduces no hidden prompt injections or state side-effects. Only accept ADK if the implementation is demonstrably equivalent to a raw LiteLLM call. (Actually I really don't see why we wouldn't just build this from scratch using LiteLLM. I think it's good to keep things simple when possible.)

**Scope:**
- Implement `BaseLLMPredictor(Predictor)` in `implementations/methods/base_llmp.py`
- Prompt should include: serialized historical observations (tabular or JSON), task description, output format instructions
- Output parsing via Pydantic — the model must return a `ContinuousForecast` (point + quantiles)
- Populate `Prediction.metadata` with token counts, model name/version, and prompt hash
- Run a backtest on `cpi_allitems_12m` reference spec; report mean CRPS vs ARIMA baseline
- Document quantile elicitation strategy (asking for quantiles directly vs sampling)

**Acceptance criteria:**
- Backtest runs end-to-end without errors
- CRPS and calibration numbers reported alongside ARIMA baseline
- Implementation contains no hidden state or framework side-effects (documented in README or notebook)
- LiteLLM vs ADK decision recorded in `technical-design.md`

---

### T3 — Numerical Forecaster Expansion & Foundation Models

**Theme:** Reference method implementations + notebook polish
**Estimated effort:** ~1 week
**Dependencies:** None
**Owner:** TBD (good onboarding task — Darts/ML background helpful)

**Context:**
The current predictor library has one variant: `DartsAutoARIMAPredictor`. Before the bootcamp, we want a richer numerical forecaster leaderboard: a trivial baseline, a broader Darts model, and at least one time series foundation model. This gives participants clear reference points to beat and demonstrates the breadth of the numerical forecasting paradigm. Also includes long-deferred notebook polish.

**Scope:**
- Move `DartsAutoARIMAPredictor` from inline notebook definition to `implementations/methods/darts_arima.py` (it is cross-cutting and belongs in the methods library; update the CPI demo notebook to import it)
- `SeasonalNaivePredictor` in `implementations/methods/naive.py` — one level above the already-implemented `LastValuePredictor`; both are baselines that statistical and ML models should comfortably beat
- A second Darts model predictor (ETS or N-BEATS — pick whichever gives the better demo story; document the choice)
- `ChronosPredictor` or `TimesFMPredictor` — one time series foundation model via HuggingFace; aim for zero-shot application with no fine-tuning
- Apply all to `cpi_allitems_12m`; extend the comparison table in `cpi_backtest_demo.ipynb` (mean CRPS for all predictors)
- **Remaining notebook polish** for `cpi_backtest_demo.ipynb`: focus the plot window on the last 10 years, add a multi-series panel showing Food, Shelter, and Water/fuel/electricity alongside All-items *(CI shading and naive/ARIMA comparison are already done)*

**Acceptance criteria:**
- Five predictors runnable on the CPI reference spec with reported CRPS scores: `LastValuePredictor`, `SeasonalNaivePredictor`, `DartsAutoARIMAPredictor`, a second Darts model, and a foundation model
- `DartsAutoARIMAPredictor` lives in `methods/darts_arima.py` and is imported (not redefined) in the demo notebook
- Notebook renders cleanly with tidy labels and the last-10-years view
- Foundation model predictor documented with zero-shot framing rationale

---

### T4 — Pass 2: Binary Forecasting + BoC Reference Experiment

**Theme:** Second forecasting paradigm + reference experiment
**Estimated effort:** ~1 week
**Dependencies:** None
**Owner:** TBD (economics interest helpful; can work independently of T1–T3)

**Context:**
The current evaluation harness only supports `ContinuousForecast`. The project charter calls for a second paradigm: discrete event / binary forecasting. The Bank of Canada interest rate decision is the ideal first reference task — it has a well-defined, sparsely-resolved binary structure (cut / hold / hike), historical data is publicly available, and it is directly relevant to bootcamp sponsors. This lays the groundwork for ForecastBench integration (H3), which is the primary source of discrete event questions and resolutions for the bootcamp. Direct Metaculus API integration remains a future option but is not required.

**Scope:**
- `BinaryForecast` Pydantic model (probability estimate, follows Metaculus conventions)
- `BinaryPredictor` ABC (`predict(task, context) -> Prediction` where payload is `BinaryForecast`)
- Binary evaluation loop with Brier score (mirror the continuous harness; reuse `run_eval_loop` where possible)
- Export all new symbols from `evaluation/__init__.py`
- BoC interest rate decisions: source historical decisions (Bank of Canada publishes these publicly), ingest, define `ForecastingTask` for "Will BoC cut/hold/raise at the next announcement?"
- Reference spec YAML in `reference_specs/`
- Demo notebook under `implementations/experiments/boc_rate_decisions/`
- Document the ForecastBench integration point (no integration required yet — just document what the loader interface would look like; see H3)

**Acceptance criteria:**
- A `NaiveBinaryPredictor` (constant 50%) runs end-to-end and returns a Brier score
- BoC reference experiment notebook runs cleanly
- `technical-design.md` updated with `BinaryForecast` type and binary evaluation loop
- 74+ tests still passing; `make lint` clean

---

### T5 — Frontier Agentic Forecaster

**Theme:** Reusable agent infrastructure + reference method
**Estimated effort:** 1–2 weeks
**Dependencies:** None (but benefits from T2 being done for comparison)
**Owner:** Most senior available team member; Ethan if capacity allows

**Context:**
The "frontier agent" is the most powerful forecaster template: an ADK-based coding agent that can retrieve data via tools, write and execute code to produce numerical forecasts, and optionally search for context. The agent backbone (ADK setup, tool definitions, prompt scaffolding) is genuinely reusable infrastructure and belongs in the package (`aieng/forecasting/agents/`); task-specific configuration lives in `implementations/`. This becomes the template that bootcamp participants customize and compete with.

**Scope:**
- `aieng/forecasting/agents/` module: base agent definition, standard tool set
  - Data retrieval tool wrapping `ForecastContext.get_series()`
  - Code execution tool (via ADK's built-in code execution or a sandboxed equivalent)
  - Optional: web search / news retrieval tool
- `AgentPredictor(Predictor)` in the package: wraps the ADK agent, handles async execution and output parsing into `ContinuousForecast`
- Task-specific configuration (system prompt, enabled tools) in `implementations/experiments/<use-case>/`
- Demonstrate on CPI task; compare CRPS vs ARIMA and base LLMP
- **Timebox aggressively** — a working end-to-end demo with documented design decisions is more valuable than architectural completeness

**Acceptance criteria:**
- Agent runs end-to-end on CPI reference spec and produces a `Prediction`
- Agent backbone documented in `technical-design.md` (module location, tool interface, config pattern)
- Comparison result vs ARIMA and base LLMP reported

---

### T6 — Fine-Tunable LLMP + Kaggle Submission *(nice to have)*

**Theme:** Fine-tuning pipeline + competition submission
**Estimated effort:** 1–2 weeks
**Dependencies:** T1 (CFPR task), T2 (base LLMP)
**Owner:** Ethan (requires deepest project context; Kaggle submission narrative also needs it)

**Context:**
The Gemma 4 Good Hackathon (final deadline May 18, 2026) is a natural vehicle for publishing our fine-tuning work. The core research question: does fine-tuning a small open model (Gemma 4 via Unsloth) on historical forecasting I/O examples improve predictive performance relative to a zero-shot base LLMP, and in what conditions? This is a genuinely interesting question for the bootcamp too. **This task must not block bootcamp readiness.** If T1 and T2 are not complete with time to spare before May 18, deprioritize or scope down to fine-tuning scaffold only.

**Scope:**
- I/O example extraction: given a `BacktestSpec` and `DataService`, generate (prompt, `ContinuousForecast`) pairs for all origins up to a given cutoff date — this becomes the fine-tuning dataset
- Unsloth integration: fine-tune Gemma 4 on the extracted examples; wrap the fine-tuned model as a new `Predictor` variant
- Apply to CFPR task (T1); compare CRPS vs base LLMP (T2) and ARIMA (T1)
- Kaggle submission: notebook or writeup framing the CFPR task, the fine-tuning approach, and the comparative results

**Acceptance criteria:**
- Fine-tuning pipeline runs end-to-end on CFPR data
- Fine-tuned predictor runs against CFPR reference spec and reports CRPS
- Comparison table: ARIMA vs base LLMP vs fine-tuned LLMP
- Kaggle submission submitted before May 18 (conditional on T1 + T2 completion with margin)

---

## Holding Queue

These tasks are scoped and understood but not yet in an active sprint. Reorder priorities freely.

---

### H3 — ForecastBench Integration *(next priority after T4)*

**Theme:** Data / Use case
**Estimated effort:** ~1 week
**Dependencies:** T4 (binary evaluation harness)
**Owner:** TBD

Integrate ForecastBench as the primary source of discrete event forecasting questions and resolutions. ForecastBench provides direct download access under CC-BY-SA-4.0 — no outreach or API key required. Data includes historical questions, resolutions, and published community predictions from Metaculus, FRED, Yahoo Finance, and Rand Forecasting. T4's BoC reference experiment provides the binary evaluation harness; this task plugs in a curated question set from ForecastBench on top of it.

ForecastBench data is not a time series and does not flow through the `ProviderAdapter` / `SeriesStore` path. Integration will be a separate loader that populates questions and resolutions into the Pass 2 binary evaluation infrastructure.

Direct Metaculus API integration remains a future option (e.g. for live question feeds) but is no longer needed for a reference experiment.

**Decision date:** Apr 10, 2026.

---

### H1 — S&P500 / Equities Reference Experiment (Behnoosh)

**Theme:** Use case / Reference experiment
**Estimated effort:** ~1 week
**Dependencies:** None
**Owner:** Behnoosh

Implement a reference experiment for S&P500 and/or Canadian equities using the yfinance adapter. Define `ForecastingTask` variants (e.g. 30-day return distribution, earnings-beat binary). This also covers the FRED adapter for macro covariates. Once T1 is complete, the pattern for standing up a new use case is established and this should be straightforward to replicate.

---

### H2 — NYISO Reference Experiment (Behnoosh, after H1)

**Theme:** Use case / Reference experiment
**Estimated effort:** ~1 week
**Dependencies:** H1 (pattern established; share infrastructure)
**Owner:** Behnoosh

NYISO (New York Independent System Operator) replaces IESO (Ontario electricity) as the energy dataset. Behnoosh identified it as a better fit for classical multivariate forecasting. Define hourly demand/price `ForecastingTask` variants, NYISO data adapter, reference spec, demo notebook. By the time this is tackled, the use-case scaffolding pattern will be well-established and most of the effort is data ingestion + task framing.

---

### H4 — BoC Rate Decisions: Discrete Event Framing *(may merge with T4)*

*If T4 defines the BoC task as a continuous series (next rate value), this task adds the discrete event reframing ("Will BoC cut at the next announcement?"). May be in-scope for T4 itself — defer the decision until T4 is started.*

---

### H5 — Per-User Eval Tracking

**Theme:** Infrastructure
**Estimated effort:** ~0.5 week
**Dependencies:** T4 (or later)

Wire `EvalTracker` to per-participant identity for the bootcamp leaderboard. The hook (`EvalTracker` path is caller-supplied) is already in place; this task decides on the identity mechanism and writes the wiring. Deferred until bootcamp infrastructure is more defined.

---

### H6 — ForecastBench Historical Predictions: ICL / Fine-Tuning Research

**Theme:** Research / Future work
**Estimated effort:** TBD (exploratory)
**Dependencies:** H3 (ForecastBench integration), T4 (binary evaluation harness)
**Owner:** TBD

ForecastBench publishes historical community predictions alongside questions and resolutions. This opens several research directions not needed for the bootcamp but worth recording now:

- **ICL (in-context learning):** Can a discrete event forecasting agent use published historical predictions and resolutions as few-shot examples to improve its own calibration?
- **Fine-tuning:** Can fine-tuning on ForecastBench historical prediction data improve base model performance on discrete event tasks?
- **Hypothesis formation and resolution feedback:** Can an agent learn to form, test, and revise hypotheses by observing past resolution outcomes — i.e. a simulation of the superforecaster update loop?
- **Backtest-based strategy evaluation:** Can we replay different agent strategies against historical ForecastBench questions to identify which approaches generalize to live questions?

Not in scope for Phase 1. Documented here as a future research agenda that connects to Pass 2 infrastructure and the long-term agentic research direction.

**Decision date:** Apr 10, 2026.

---

## Conventions

- Tasks move from **Holding Queue → Active Sprint** at the start of each sprint planning session.
- When a task is completed, add a brief completion note and date, then archive it (move to a `## Completed` section at the bottom of this file).
- Any architectural decision made while executing a task must be recorded in `technical-design.md` in the same session (per the maintenance contract).
- Scope changes, re-prioritizations, and new tasks discovered mid-sprint go here first, then into ClickUp.
