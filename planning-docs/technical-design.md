# Agentic Forecasting — Technical Design

## Purpose

This document is the **technical source of truth** for the agentic forecasting repository. It captures all significant architectural decisions, library selections, interface designs, and build plans.

> **Maintenance contract:** This document MUST be kept up to date. Whenever an architectural decision is made, revised, or reversed — in a coding session, a planning conversation, or a commit — this document should be updated in the same session. Do not let decisions live only in chat logs or planning notes. Planning notes are for exploration and quick logging; this document is for what we have decided and are building toward.

---

## Library & Tooling Decisions

### Forecasting: Darts (over sktime)

**Decision date:** Mar 31, 2026

**Darts** is the primary numerical forecasting library.

Key reasons:
- Consistent `fit()`/`predict()` API across all model types — one mental model to debug
- Better developer experience for a mixed-skill bootcamp audience
- Built-in `historical_forecasts()` and `backtest()` utilities are first-class
- Modular install (`pip install darts` vs `darts[torch]`) lets us stage complexity incrementally
- Lower support burden for the bootcamp instructor

sktime remains a valid reference for specific use cases (AutoARIMA, panel forecasting) but is not the primary interface we support or teach.

### Agent Framework: Google ADK

**Google ADK** is the default framework for building forecasting agents. Additional dependencies are introduced only when blocked by ADK's native capabilities.

### Tracing & Logging: Langfuse

**Langfuse** is selected for tracing. The integration point is at the **Predictor level** — reasoning traces are linked to prediction outcomes via `predictor_id` + `question_id`. This is separate from the evaluation harness's own prediction/resolution/score logging. Implementation details are deferred.

### Structured Outputs: Pydantic

All prediction payloads and data interfaces use **Pydantic** models with mypy-compatible typing throughout.

---

## Evaluation Architecture

### Core Insight

Backtesting and live evaluation are the same loop — they differ only in whether ground truth is already known. A single unified architecture handles both.

### Unified Loop

```
Predictor → Prediction → Resolution → Score
```

- **Predictor** — model-agnostic; produces a `Prediction` given a question/task and an as-of date
- **Prediction** — paradigm-specific payload, but shares common metadata: `question_id`, `predictor_id`, `issued_at`, `horizon`
- **ResolutionStore** — pre-populated in backtest mode; fills in asynchronously in live mode
- **Scorer** — swappable: CRPS for continuous forecasts, Brier score for discrete event

### ForecastingTask

A `ForecastingTask` is a Pydantic model that parameterizes the evaluation loop for a specific prediction problem. It is the bridge between the data service and the evaluation harness, and the place where series relationships are declared.

Fields:
- `task_id` — unique identifier
- `target_series_id` — the series being forecast (key into `SeriesStore`)
- `horizon` — number of steps ahead
- `frequency` — temporal resolution (e.g., `"MS"` for month-start, `"h"` for hourly)
- `past_covariate_ids` — list of `series_id`s available up to the forecast origin (e.g., related economic indicators)
- `future_covariate_ids` — list of `series_id`s known into the forecast horizon (e.g., calendar features)
- `gap_fill_strategy` — how to fill irregular gaps before handing to a numerical model (`"ffill"`, `"interpolate"`, `"none"`); defaults to `"ffill"`
- `resolution_fn` — how to look up ground truth from `ResolutionStore`; defaults to "observed value at the resolution timestamp"
- `description` — human-readable description of the task

For backtesting, the harness iterates over historical origins defined by the task. For live evaluation, it waits for the resolution date. The loop is identical.

**Series relationships** — `past_covariate_ids` and `future_covariate_ids` are the mechanism for declaring that a set of series are meaningfully related for a given task. This is intentionally task-scoped rather than stored globally: the same series can be a covariate in one task and irrelevant in another. A covariate registry (global declarations of which series are "related") is deferred as an open question.

### Prediction Payload Types

Two concrete payload types:

- **`ContinuousForecast`** — point values + quantiles, for economic/time series tasks
- **`BinaryForecast`** — probability estimate, for Metaculus-style discrete event questions

We follow existing standards rather than inventing new ones. For discrete event forecasting, we follow Metaculus conventions.

---

## Data Service

### Design Philosophy

Two categories of data are treated very differently:

| Category | Examples | How it's delivered | Live calls during sessions? |
| :--- | :--- | :--- | :--- |
| **Deterministic** | historical series, resolution targets | local data service, pre-populated | No |
| **Stochastic context** | news, web search, live indicators | live API calls, agentic tools | Yes — logged via Langfuse |

No outbound calls for historical or resolution data occur during bootcamp sessions or backtests. Adapters are run offline to populate the local store ahead of time.

### Architecture

```
DataService
├── SeriesStore          # historical time series, keyed by series_id
├── SeriesMetadataStore  # units, description, source, frequency hint per series_id
├── ResolutionStore      # ground truth values at resolution timestamps
├── CutoffEnforcer       # enforces information cutoff discipline (see below)
└── ProviderAdapters
    ├── BaseAdapter          # protocol / ABC all adapters must implement
    ├── LocalCSVAdapter      # first-class path for custom datasets
    ├── StatCanAdapter
    ├── FREDAdapter
    └── yfinanceAdapter
```

### Canonical Internal Format

Each series in `SeriesStore` is stored as a DataFrame with the following columns:

| Column | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `timestamp` | `datetime` | ✅ | Observation time |
| `value` | `float` | ✅ | The observed quantity |
| `released_at` | `datetime` | — | When this data point became publicly available; defaults to `timestamp` if absent |

**`series_id` is the store key, not a column.** One DataFrame per registered series.

**One value column per series.** Multivariate data (e.g., CPI + employment) is registered as separate series. Relationships between series are declared in `ForecastingTask` (via `past_covariate_ids`), not in the data format itself.

This format handles regular time series, irregular event sequences, and sparse data uniformly — missing values are absent rows, not NaN sentinels. No frequency needs to be declared at registration time.

### Adapter Protocol

`BaseAdapter` defines one required method:

```python
def fetch() -> pd.DataFrame:
    ...  # returns DataFrame with (timestamp, value) columns; released_at optional
```

`LocalCSVAdapter` implements this with a column-mapping config (`timestamp_col`, `value_col`, optional `released_at_col`). This is the intended path for participants bringing their own datasets — no subclassing required.

### Gap-Filling at the Darts Conversion Boundary

The `SeriesStore` representation makes no guarantees about regularity. When a numerical predictor needs a `darts.TimeSeries`, the `ForecastingTask.gap_fill_strategy` is applied at conversion time via `TimeSeries.from_dataframe()`. This is an explicit, documented step — not silent behaviour. LLM-based predictors do not go through this conversion.

### Information Cutoff Discipline

The `CutoffEnforcer` enforces a critical principle: **no model or agent may access data that would not have been available at the time the forecast was issued**. It filters series data by `released_at <= as_of_date`. For custom datasets where `released_at` is absent, the filter falls back to `timestamp <= as_of_date`, which is correct for most real-time or custom data.

This is the unifying concept across both time series backtesting and discrete event evaluation, and is a core teaching objective of the bootcamp.

### Open Questions

- **Data service update pipeline**: How are updates handled as new data releases come in (e.g., monthly StatCan drops)? Important for the live benchmark extension; needs to be resolved before live evaluation infrastructure is built.
- **Global covariate / series relationship registry**: `ForecastingTask` handles task-scoped relationships. A global registry declaring which series are structurally related (e.g., CPI sub-components, equity sector groupings) may be needed for discovery and documentation, but is deferred.

---

## Build Plan

### Principle: Two Concrete Passes Before Abstracting

Shared abstractions are extracted after both passes are working — not designed in advance.

1. **Pass 1 — Economic forecasting** (StatCan, continuous series, `ContinuousForecast` payloads)
2. **Pass 2 — Metaculus predictions** (binary/categorical, discrete event, `BinaryForecast` payloads)

### Long-Term Vision

This project is designed to support two related but distinct purposes:

1. **Bootcamp learning platform** — a structured environment for participants to experiment with forecasting methods on reference datasets, with backtesting, evaluation, and leaderboard infrastructure.
2. **Ongoing forecasting benchmark and competition** — an open platform where forecasting agents (human-designed or autonomous) submit predictions against live questions, resolutions are published as they occur, and performance is tracked longitudinally.

The data service, evaluation harness, and prediction/resolution/score architecture should be designed with both purposes in mind. The key design property that serves both: **the evaluation loop is identical for backtesting and live forecasting** — the same `ForecastingTask`, `Predictor`, and `Scorer` interfaces work in both modes. The data service's offline-first approach (deterministic data pre-populated locally, `released_at` discipline enforced) is also what makes the benchmark trustworthy at scale.

This long-term framing should inform decisions about interface stability, documentation quality, and extensibility — even during the Phase 1 bootcamp build.

### Connection to Project Charter Deliverables

- The **evaluation harness + data service** together constitute the *forecast resolution service* in Phase 2 of the project proposal.
- The **live experiment leaderboard** is the data service update pipeline made visible.
- The **information cutoff discipline** is the unifying teaching concept across both forecasting paradigms.
