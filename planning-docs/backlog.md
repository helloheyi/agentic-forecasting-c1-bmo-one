# Agentic Forecasting — Development Backlog

This file is a plain-text complement to ClickUp. It captures the current set of development tasks with enough detail to hand off to a new team member. Tasks are grouped into the active sprint first, then the holding queue. Update this file when tasks are started, completed, re-scoped, or reprioritized.

**Primary deliverable:** Bootcamp readiness. All sprint decisions should be made against this target first.

**Kaggle note:** Gemma 4 Good Hackathon final submission deadline is May 18, 2026. This is a "nice to have" that must not disrupt the bootcamp critical path. The fine-tunable LLMP / Kaggle submission task is in the holding queue; it is explicitly lower priority than active sprint work.

---

## Active Sprint

Five people are active this sprint, each with a single focused area.

| Person | Focus |
|--------|-------|
| Ethan | CFPR use case + backtest/eval/live testing engine |
| Ali | First LLMP → agentic forecaster |
| Behnoosh | S&P500 reference use case |
| Franklin | Code quality & Coder bootcamp environment |
| Ahmad | Call for Participation presentation |

---

### Ethan — CFPR Use Case & Testing Engine

Simultaneously develop the CFPR (Canada's Food Price Report) reference experiment and evolve the backtest/eval/live testing engine. These run concurrently: the CFPR task is familiar territory (Ethan is a many-time contributor to the real report) and grounds the more uncertain infrastructure work.

**CFPR use case:** ✅ **IN PROGRESS (Apr 16, 2026)** — Food price experiment is live at `implementations/experiments/food_price_forecasting/`. Multi-target backtest/eval infrastructure added to framework. `DartsAutoARIMAPredictor` moved to `implementations/methods/darts_arima.py` with covariate support. `FREDAdapter` implemented. 9-category food CPI specs in `reference_specs/food_cpi/`. Notebooks: data exploration, 18m CFPR-replica, 3m denser-eval variant. **Remaining:** run the notebooks with actual data to validate end-to-end, collect first backtest results.

**Testing engine:** The core backtest and eval infrastructure exists. What remains is the harder design question: what does "live testing" look like, and how do we handle it honestly for agentic forecasters? The central open question — to be explored with Ali — is how realistically we can retrieve internet context with effective information cutoffs for backtesting agentic forecasters. We may find that backtest results won't reliably generalize to live performance for agents that search the web; that's fine. Get the plumbing working, document the problem honestly, and chart the course toward agent skills that can interact with the backtest/eval engines and with baseline/numerical forecasters.

---

### Ali — First LLMP → Agentic Forecaster

Ali is the long-haul engineer for the agentic forecasting work. This sprint: implement the first LLMP (LLM Process), then graduate quickly to a first properly agentic forecaster.

**Start with research reading** before writing any code. Key starting point: Gruver et al. 2024, "Large Language Models Are Zero-Shot Time Series Forecasters." Also review the LiteLLM docs and Google ADK docs to understand both options before committing to a design.

**Base LLMP:** Implement `BaseLLMPredictor(Predictor)` in `implementations/methods/base_llmp.py`. This is a minimal LLM-based predictor — an LLMFunction, not a full agent. It takes serialized historical observations and a task description, and produces a `ContinuousForecast` via Pydantic structured output, with no hidden state or framework side-effects. Key design decision to document: LiteLLM directly (preferred for simplicity and transparency) vs. Google ADK in non-agentic mode. Run a backtest on the CPI reference spec and compare CRPS vs. the ARIMA baseline.

**Agentic forecaster:** Once the base LLMP is running end-to-end, graduate to a first properly agentic forecaster: an ADK-based coding agent that can retrieve data via tools, write and execute code to produce numerical forecasts, and optionally search for context. The agent backbone (ADK setup, tool definitions, prompt scaffolding) is reusable infrastructure and belongs in `aieng/forecasting/agents/`; task-specific configuration lives in `implementations/`. Timebox aggressively — a working demo with documented decisions is more valuable than completeness. Start with StatCan CPI; apply to S&P500 once Behnoosh's use case is ready. Coordinate with Ethan on what the agentic forecaster will need from the testing engine.

---

### Behnoosh — S&P500 Reference Use Case

Behnoosh owns the S&P500 reference use case end to end. This is an evolving task: it begins with task framing and grows as new methods land.

**This sprint:** Frame the forecasting task — this is a real design decision (30-day return distribution? directional binary? something else?) and the choice should be documented with rationale. Stand up the yfinance data adapter; register S&P500 series in the data service; write a `BacktestSpec` YAML; produce a demo notebook under `implementations/experiments/sp500/` with `DartsAutoARIMAPredictor` as the first baseline; write a `README.md` documenting data provenance, task framing decisions, and licence.

**Ongoing:** As additional methods become available (LLMP from Ali, additional numerical methods), apply them to the SP500 task and extend the comparison table. Think carefully about what a well-designed backtesting regime looks like for financial data (non-overlapping test windows, look-ahead risk, etc.), what "eval" testing means for this domain, and how live testing might eventually work — especially once agentic forecasters are involved. Coordinate with Ethan on these questions; financial data may surface requirements the existing testing engine doesn't yet handle.

---

### Franklin — Code Quality & Bootcamp Infrastructure

Franklin brings software engineering expertise and has limited time this sprint — timebox everything, and ensure anything he can't finish has a clear handoff.

**Code quality:** Review `aieng-forecasting/` and `implementations/methods/` for engineering quality: type coverage, docstring completeness, API clarity, test coverage gaps. Identify and address improvements that increase the likelihood of the package being used beyond the bootcamp. Note: the Methods/implementations separation Franklin suggested was completed on Apr 9 (see planning notes) — he should review what was done before assuming it's still to do, and build on top of it or identify what additional refactoring he has in mind. `make lint` must stay clean; mypy coverage must not regress.

**Coder platform:** Assess what environment configuration is needed for the bootcamp (Coder workspace images, dependencies, GPU access). Set up or prototype a Coder workspace that a participant could use to run the reference notebooks end-to-end. Where he can't complete the setup himself, document exactly what's needed and identify who should step in to ensure a smooth bootcamp.

---

### Ahmad — Call for Participation Presentation

Ahmad is joining for this sprint only to produce the presentation for the Call for Participation meeting next month. **Deadline:** review-ready at least 1 week before the meeting.

**First:** Read `planning-docs/bootcamp-project-charter.md`, `planning-docs/technical-design.md`, and the most recent entries in `planning-docs/planning-notes.md`.

**Then produce a presentation** (format TBD — slides or structured document) covering: bootcamp motivations and goals; the four forecasting paradigms (numerical, LLMP, agentic, hybrid); the reference use cases and datasets (CFPR, S&P500, BoC, ForecastBench); a walkthrough of the technical components being built so far (data service, evaluation harness, implementations layer, agentic forecaster design); and what participation in the bootcamp will look like. Share a draft with Ethan for review before the meeting.

---

## Holding Queue

These tasks are scoped and understood but not yet assigned. Reorder priorities freely.

---

### Numerical Forecaster Expansion & Foundation Models

**Owner:** TBD (good onboarding task — Darts/ML background helpful)
**Dependencies:** None

The current predictor library has one variant: `DartsAutoARIMAPredictor`. Before the bootcamp, we want a richer numerical forecaster leaderboard: a trivial baseline, a broader Darts model, and at least one time series foundation model. This gives participants clear reference points to beat and demonstrates the breadth of the numerical forecasting paradigm.

- ✅ Move `DartsAutoARIMAPredictor` from inline notebook definition to `implementations/methods/darts_arima.py`; update the CPI demo notebook to import it (completed Apr 16, 2026)
- `SeasonalNaivePredictor` in `implementations/methods/naive.py`
- A second Darts model predictor (ETS or N-BEATS)
- `ChronosPredictor` or `TimesFMPredictor` — one time series foundation model via HuggingFace, zero-shot
- Apply all to `cpi_allitems_12m`; extend the comparison table in `cpi_backtest_demo.ipynb`
- Remaining notebook polish: focus plot on the last 10 years; add a multi-series panel showing Food, Shelter, and Water/fuel/electricity alongside All-items

---

### Binary Forecasting + BoC Reference Experiment

**Owner:** TBD (economics interest helpful)
**Dependencies:** None (can work independently of active sprint tasks)

The current evaluation harness only supports `ContinuousForecast`. This adds the second paradigm: discrete event / binary forecasting. The Bank of Canada interest rate decision is the ideal first reference task — well-defined, sparsely-resolved, publicly available historical data, and directly relevant to bootcamp sponsors. This also lays the groundwork for ForecastBench integration.

- `BinaryForecast` Pydantic model (probability estimate, follows Metaculus conventions)
- `BinaryPredictor` ABC and binary evaluation loop with Brier score (reuse `run_eval_loop` where possible)
- BoC interest rate decisions: source historical decisions, ingest, define `ForecastingTask`, write `BacktestSpec` YAML, demo notebook under `implementations/experiments/boc_rate_decisions/`
- Document the ForecastBench integration point (no integration required yet)
- Update `technical-design.md` with `BinaryForecast` type and binary evaluation loop

---

### ForecastBench Integration *(next priority after binary forecasting)*

**Owner:** TBD
**Dependencies:** Binary forecasting task above

Integrate ForecastBench as the primary source of discrete event forecasting questions and resolutions. ForecastBench provides direct download access under CC-BY-SA-4.0 — no outreach or API key required. Data includes historical questions, resolutions, and published community predictions from Metaculus, FRED, Yahoo Finance, and Rand Forecasting. ForecastBench data is not a time series and does not flow through the `ProviderAdapter` / `SeriesStore` path — integration will be a separate loader into the binary evaluation infrastructure. Direct Metaculus API integration remains a future option but is no longer needed for a reference experiment.

**Decision date:** Apr 10, 2026.

---

### Fine-Tunable LLMP + Kaggle Submission *(nice to have)*

**Owner:** TBD (requires deep project context; Kaggle submission narrative needs it)
**Dependencies:** CFPR use case (Ethan), base LLMP (Ali)
**Deadline:** Gemma 4 Good Hackathon final submission May 18, 2026

The core research question: does fine-tuning a small open model (Gemma 4 via Unsloth) on historical forecasting I/O examples improve predictive performance relative to a zero-shot base LLMP, and in what conditions? This task must not block bootcamp readiness. Only activate if CFPR and base LLMP are complete with margin to spare before May 18.

- I/O example extraction: generate (prompt, `ContinuousForecast`) pairs for all backtest origins up to a cutoff
- Unsloth integration: fine-tune Gemma 4; wrap as a new `Predictor` variant
- Apply to CFPR task; compare CRPS vs. base LLMP and ARIMA
- Kaggle submission notebook/writeup

---

### NYISO Reference Experiment *(Behnoosh, after S&P500)*

**Owner:** Behnoosh
**Dependencies:** S&P500 use case (pattern established)

NYISO (New York Independent System Operator) replaces IESO (Ontario electricity) as the energy dataset. Behnoosh identified it as a better fit for classical multivariate forecasting. Define hourly demand/price `ForecastingTask` variants, NYISO data adapter, reference spec, demo notebook. By the time this is tackled, the use-case scaffolding pattern will be well-established and most of the effort is data ingestion + task framing.

---

### Per-User Eval Tracking

**Theme:** Infrastructure
**Dependencies:** Binary forecasting task (or later)

Wire `EvalTracker` to per-participant identity for the bootcamp leaderboard. The hook (`EvalTracker` path is caller-supplied) is already in place; this task decides on the identity mechanism and writes the wiring. Deferred until bootcamp infrastructure is more defined.

---

### BoC Rate Decisions: Discrete Event Framing *(may merge with binary forecasting task)*

If the binary forecasting task defines the BoC task as a continuous series (next rate value), this adds the discrete event reframing ("Will BoC cut at the next announcement?"). May be in-scope for the binary forecasting task itself — defer the decision until that task is started.

---

### ForecastBench Historical Predictions: ICL / Fine-Tuning Research *(future)*

**Dependencies:** ForecastBench integration, binary evaluation harness

ForecastBench publishes historical community predictions alongside questions and resolutions. This opens several research directions not needed for the bootcamp but worth recording:

- **ICL:** Can a discrete event forecasting agent use published historical predictions and resolutions as few-shot examples to improve calibration?
- **Fine-tuning:** Can fine-tuning on ForecastBench historical prediction data improve base model performance on discrete event tasks?
- **Hypothesis formation and resolution feedback:** Can an agent learn to form, test, and revise hypotheses by observing past resolution outcomes — simulating the superforecaster update loop?
- **Backtest-based strategy evaluation:** Can we replay different agent strategies against historical ForecastBench questions to identify which approaches generalize to live questions?

Not in scope for Phase 1. Documented here as a future research agenda.

**Decision date:** Apr 10, 2026.

---

## Completed

*(Tasks move here when done, with a brief note and date.)*

---

## Conventions

- Tasks move from **Holding Queue → Active Sprint** at the start of each sprint planning session.
- When a task is completed, add a brief completion note and date, then archive it to `## Completed`.
- Any architectural decision made while executing a task must be recorded in `technical-design.md` in the same session (per the maintenance contract).
- Scope changes, re-prioritizations, and new tasks discovered mid-sprint go here first, then into ClickUp.
