# Agentic Forecasting Bootcamp — Project Charter

## Purpose

This document defines the scope, methods, datasets, and design principles for the Agentic Forecasting Bootcamp. It is intended as the central technical reference for the bootcamp. The design choices reflected in this document were all informed by feedback from our industry sponsors via QSM (Quarterly Sponsor Meeting) and IAP (Industry Advisory Panel).

The bootcamp is organized around three distinct approaches to forecasting applied to a focused set of Canadian finance, economics, and energy datasets. These paradigms are not mutually exclusive — the most interesting implementations may combine elements of several — but they represent meaningfully different philosophies about what a forecasting system is and how it works. A central learning objective is to compare these approaches empirically on shared, standardized datasets.

---

## Domain Focus

The bootcamp concentrates on three interconnected domains of applied forecasting, all with strong Canadian data availability and real-world relevance to sponsor organizations:

* **Finance** — equities, earnings, and foreign exchange (e.g., yfinance, SEDAR+)
* **Economics** — macroeconomic indicator forecasting (e.g., StatCan, Bank of Canada)
* **Energy** — electricity demand and price forecasting (e.g., Ontario grid via IESO)

Focusing on these domains allows participants to go deeper on techniques rather than wider on coverage. The target datasets are standardized; the forecasting tasks and questions framed against them are not — there are many meaningful ways to frame prediction problems in each domain, and the diversity of task framings is itself a learning objective. Additional data sources (e.g. datasets, APIs, or agentic tools) may be used to provide covariates or context.

---

## Cross-Cutting Design Principles

Two design principles apply across all methods and datasets in the bootcamp. They are not evaluation criteria bolted on at the end — they are considerations that should inform every implementation decision.

### LLM-Assisted Coding and Optimization

Using an LLM to help write, debug, tune, and iterate on any of the three forecasting approaches is not a fourth paradigm — it is a design practice that applies on top of all of them. A well-implemented, AI-assisted statistical or ML model may be a considerably stronger baseline than a hastily-coded one, and exploring this systematically is a legitimate direction in its own right. We will consult with AI Engineering to determine what tools could be available during a bootcamp for AI assisted coding. For example, simply supporting the use of GitHub Copilot via the Coder/VSCode interface or supporing Cursor (which is enabled/possible using the Coder platform), may be good enough.

### Transparency, Interpretability, and Explainability

Three related but distinct lenses apply across all methods. Rather than treating these as afterthoughts or evaluation criteria, we treat them as design considerations — questions a practitioner should ask at every stage of building a forecasting system.

**Transparency** operates at the level of the research process. It asks: *can others see how this forecast was produced?* Transparent practice includes publishing code, sharing datasets, logging experiments (including failures and dead ends), and versioning pipelines. Transparency is largely method-agnostic — it is a commitment to open science practice that applies regardless of whether the underlying model is a linear regression or an LLM agent.

**Interpretability** operates at the level of model internals. It asks: *why did this model produce this output?* The answer depends heavily on method family. Classical statistical models (ARIMA, VAR, ETS) are natively interpretable — their parameters carry direct semantic meaning. Gradient boosted trees are partially interpretable via feature importances and tools like SHAP. Deep learning models are generally opaque, though architectures like the Temporal Fusion Transformer were designed with inspectable attention mechanisms. LLM Processes occupy an interesting position: the model itself is a black box, but the natural language conditioning — the prompt — is inherently human-readable and constitutes part of the explanation. LLM reasoning traces may also be analyzed.

**Explainability** operates at the level of the forecast consumer. It asks: *can a decision-maker understand and appropriately trust this forecast?* This goes beyond model internals to include written rationales, well-communicated uncertainty, consistency across related predictions, and — for agentic discrete event forecasters — explicit evidence chains and cited sources. Explainability is where forecasting connects to decision-making, and it is the dimension most visible to sponsors and stakeholders.

These three lenses will be applied throughout the bootcamp as we evaluate and compare methods. We do not require that every implementation excel on all three dimensions — the tradeoffs between them are part of what makes the comparison interesting.

---

## Forecasting Methods

### 1\. Numerical Forecasters

The established paradigm: a model is trained on historical data and produces predictions from learned statistical or structural patterns. This family spans a wide range of complexity, from interpretable classical models to large pre-trained foundation models, but shares a common assumption: that the signal needed to forecast is latent in the historical data itself.

* **Classical statistical models** — ARIMA, ETS, VAR. Interpretable, well-understood, strong on stationary and seasonal series.
* **Machine learning models** — gradient boosted trees (LightGBM, XGBoost), random forests. Strong on tabular data with engineered features; handle non-linearity and exogenous inputs well.
* **Deep learning models** — LSTM, Temporal Fusion Transformer (Lim et al., 2021), N-BEATS (Oreshkin et al., 2019). Better at learning complex temporal dependencies from large datasets.
* **Time series foundation models** — pre-trained models such as TimesFM and Chronos that generalize across domains in a zero-shot or few-shot setting. A rapidly developing area as of 2024–2025.

Numerical forecasters serve as the baseline for all tracks. They are the standard against which LLM-based approaches are evaluated, and they are often surprisingly hard to beat — particularly when built and tuned carefully.

---

### 2\. LLM Processes

Where numerical forecasters process numbers, LLM Processes treat language itself as the computational substrate for prediction — a qualitative shift, not an incremental one. Introduced by Requeima, Bronskill, Choi, Turner, and Duvenaud (NeurIPS 2024), **LLM Processes** (LLMPs) treat a large language model as the probabilistic forecasting engine itself. Rather than training a model on historical data, an LLMP elicits joint predictive distributions directly from an LLM by conditioning it on both numerical observations and natural language descriptions of the problem setting.

The key insight is that LLMs encode rich prior knowledge about the world — domain-specific constraints, qualitative structure, expert intuitions — that is difficult to express in closed-form statistical models. LLMPs make this latent knowledge accessible: a user can describe the forecasting problem in plain language ("this is a financial time series"; "prices rarely go below zero"; "the company goes out of business on day 30") and receive a calibrated predictive distribution in return.

This paradigm is particularly well-suited to domains where current events, policy announcements, and qualitative context materially affect outcomes — precisely the conditions that characterize energy markets, macroeconomic indicators, and equities. LLMPs have been shown to be competitive with Gaussian Processes and other probabilistic regressors in zero-shot settings, with well-calibrated uncertainty, and apply naturally to settings with structural breaks or regime changes.

Requeima, J., Bronskill, J., Choi, D., Turner, R. E., & Duvenaud, D. (2024). LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language. *NeurIPS 2024*. arXiv:2405.12856.


### 2.1\. Hybrid Approaches

We can consider different ways to combine numerical methods with LLM processes. For instance, we can explore the idea that numerical forecasts could be used as an *input* into an LLM process, along with other sources of information. This way a forecasting agent could go through a workflow such as: "What would it look like if I used ARIMA / ETS to generate a forecast? Is there anything happening in the news that would make me want to modify this? After doing some investigation and analysis, which could include writing and executing code, what is my final prediction? How confident am I about it?"

In other words, this kind of approach could look at numerical forecasting as just another type of activity that a forecasting agent is able to perform before issuing its "final" prediction. This opens the door to a wide variety of toolsets, agent skills, code gen/exec. We also open the door to agent search techniques like ADAS, DGM, or the new Hyperagents. We'll have to think of what the right way to organize this is. A "hybrid" forecaster could be, at the end of the day, a forecasting agent that has abilities above and beyond a more contrained LLMP pipeline.

### 3\. Discrete Event Forecasters

A fundamentally different framing: rather than predicting the future value of a continuous series, the task is to estimate the **probability that a specific event will occur**. This is the paradigm of prediction markets and structured forecasting platforms like Metaculus and ForecastBench.

Discrete event forecasters are not (necessarily or typically) time-series models. They are more naturally described as **information retrieval and reasoning agents**: given a question with well-defined resolution criteria ("Will X happen by date Y?"), the agent gathers evidence from multiple sources — news, policy documents, historical base rates, expert commentary, market signals — and produces a calibrated probability estimate.

LLMs are a natural fit for this paradigm. Recent work has shown that LLM ensembles can approach human superforecaster accuracy on real-world questions (Schoenegger et al., 2024), and dedicated frameworks like the Metaculus `forecasting-tools` library provide scaffolding for building and evaluating such agents at scale.

News and current events are not optional context in this paradigm — they are core inputs to the evidence-gathering loop. This makes discrete event forecasting a particularly direct expression of the bootcamp's focus on economically and socially consequential prediction tasks. It is also the paradigm with the most natural support for explainability: an agent's retrieved sources, reasoning chain, and cited evidence can be fully logged and inspected for properties such as consistency or groundedness.

This approach applies wherever the forecasting task can be expressed as a binary or categorical outcome: earnings surprises, rate decisions, energy price thresholds, trade policy announcements. It is the primary paradigm for the Metaculus track and a natural framing for the finance and economics tracks.

Schoenegger, P., Tuminauskaite, I., Park, P. S., Bastos, R. V. S., & Tetlock, P. E. (2024). Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy. *Science Advances*, 10(45), eadp1528.

---

## Datasets

The bootcamp will use a small set of focused, standardized datasets drawn from Canadian and Canadian-relevant sources. Standardization supports rigorous cross-method comparison; the diversity of forecasting tasks that can be framed against each dataset ensures that standardized data does not mean standardized problems. This more prescriptive approach to datasets was encouraged by IAP panelists.

The following datasets are currently under consideration. This document will be updated as they are finalized.

### IESO — Ontario Electricity

Ontario's electricity grid, operated by the Independent Electricity System Operator (IESO), provides publicly available hourly demand and price data. This is a well-structured time series domain with strong seasonal patterns, meaningful exogenous variables (weather, day-of-week, grid events), and consequential real-world applications in energy planning and procurement.

Note: I've looked into this dataset and it's not all that interesting. The data are highly aggregated and even the units aren't super clear. That said, the data are regularly published and up to date, so it does provide a viable real-world prediction target.

### Canadian Economic Vitals — StatCan \+ FRED

Macroeconomic indicators from Statistics Canada (CPI, employment, trade) alongside FRED data. This track foregrounds the interaction between quantitative data and policy context — a natural domain for both LLM Processes and discrete event forecasters.

### Equities / Earnings — yfinance

Canadian-listed equities and earnings data sourced via yfinance. This track supports the full range of forecasting paradigms and naturally accommodates both continuous forecasting (price direction, return distribution) and discrete event framing (earnings beats, threshold crossings).

### Metaculus — World Events (Canadian lens)

Binary and categorical forecasting questions from Metaculus, curated with a focus on Canadian economic, energy, and policy outcomes. This track is the primary venue for discrete event forecasting and serves as the connection point to the broader superforecasting literature and ForecastBench evaluation infrastructure.

Note: We still need to reach out to Metaculus to see what historical data (including resolutions) could be available for our bootcamp and other projects. I think we will explore this type of prediction target and data source as a fast-follow to getting started with economic and markets forecasting.

---

## Dataset × Method Applicability

Not every method applies equally well to every dataset. The table below indicates which reference implementations are applicable to each dataset, and serves as a guide for participants choosing their track and approach.

| Dataset | Numerical Forecasters | LLM Processes | Discrete Event Forecasters |
| :---- | :---: | :---: | :---: |
| **IESO** (Ontario electricity) | ✅ | ✅ | ◑ |
| **Canadian Economic Vitals** (StatCan \+ FRED) | ✅ | ✅ | ◑ |
| **Equities / Earnings** (yfinance) | ✅ | ✅ | ✅ |
| **Metaculus** (Canadian-lens world events) | — | — | ✅ |

**Key**
✅ Applies naturally to the canonical task
◑  Applies with task reframing
—  Not as applicable

*Note: LLM-assisted coding and optimization is a cross-cutting practice applicable to all rows above. See Cross-Cutting Design Principles.*

---

## Example Tasks by Dataset and Method

**IESO — Ontario electricity**

* *Numerical:* Forecast hourly Ontario electricity demand or day-ahead price for the next 24 hours
* *LLM Processes:* Forecast demand conditioned on weather narrative, day-of-week context, and known grid events
* *Discrete Event (◑):* "Will Ontario peak demand exceed 18,000 MW tomorrow?" or "Will day-ahead prices exceed $100/MWh during the 5–7pm window?"

**Canadian Economic Vitals — StatCan \+ FRED**

* *Numerical:* Forecast next StatCan CPI release value using VAR model on basket series; forecast next-day CAD/USD using ARIMA
* *LLM Processes:* Forecast CPI conditioned on recent policy announcements, commodity price movements, and trade context in natural language
* *Discrete Event (◑):* "Will the next food CPI release show an increase greater than 0.3%?" or "Will CAD/USD close above 0.72 on Friday?"

**Equities / Earnings — yfinance**

* *Numerical:* Forecast 30-day price direction or return distribution using historical OHLCV data
* *LLM Processes:* Forecast price conditioned on earnings transcript sentiment, analyst consensus, and macro context
* *Discrete Event (✅):* "Will this company beat earnings consensus next quarter?" or "Will this stock be up more than 5% in the 30 days following the earnings release?"

**Metaculus — world events (Canadian lens)**

* *Discrete Event (✅):* Forecast the probability of binary questions resolving positively — e.g. "Will the Bank of Canada cut rates at its next announcement?" or "Will Canada's unemployment rate exceed 7% by end of Q2?"

---

## Long-Term Vision

The bootcamp is Phase 1 of a broader initiative. The platform is designed from the outset to support two complementary purposes:

* **Bootcamp learning platform** — a structured environment for participants to experiment with and compare forecasting methods against shared reference datasets, with backtesting, evaluation, and leaderboard infrastructure.
* **Ongoing forecasting benchmark and competition** — an open platform where forecasting agents submit predictions against live questions, resolutions are published as they occur, and performance is tracked longitudinally across participants and methods.

These purposes share the same evaluation infrastructure: the same interfaces for submitting predictions, resolving outcomes, and computing scores work in both backtesting and live modes. Building for both from the start avoids costly architectural rewrites later.

---

## Out of Scope (Phase 1\)

The following are documented here for continuity but are explicitly deferred beyond Phase 1:

* **Live open benchmark** — opening the platform to external participants as a public forecasting competition. The Phase 1 infrastructure is designed to support this; activation is deferred.
* **Self-adaptive agent research** (ALMA, ADAS/GEPA, LLM Processes evolution) — Phase 2 research agenda, documented separately in the full proposal.
