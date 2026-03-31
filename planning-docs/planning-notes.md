## Mar 31, 2026 — Data service design + long-term vision (session 3)

Key decisions and design refinements; full details in `technical-design.md`.

- **Canonical internal format**: each series stored as a `(timestamp, value, released_at?)` DataFrame; `series_id` is the store key, not a column. `released_at` is optional, defaults to `timestamp` — this handles both official datasets with known release lags and custom bring-your-own datasets with no lag.
- **Single-value-column convention**: one quantity per series object. Multivariate data = multiple registered series. Series relationships (covariates) are declared in `ForecastingTask`, not in the data format.
- **`ForecastingTask`**: new Pydantic model that parameterizes the evaluation loop — binds `target_series_id`, `horizon`, `frequency`, `past_covariate_ids`, `future_covariate_ids`, `gap_fill_strategy`, and `resolution_fn`. This is how series relationships and covariate structure are captured.
- **Gap-filling at conversion boundary**: `SeriesStore` makes no regularity guarantees. Gap-filling (ffill, interpolate, etc.) is an explicit step when converting to `darts.TimeSeries`, governed by `ForecastingTask.gap_fill_strategy`. LLM predictors skip this entirely.
- **Adapter protocol**: `BaseAdapter` requires one method — `fetch() -> pd.DataFrame`. `LocalCSVAdapter` is the first-class path for custom datasets, requiring only column-name mappings.
- **Series relationships open question**: task-scoped covariate declarations via `ForecastingTask` handle the immediate need. A global covariate/series relationship registry (for discovery across tasks) is deferred.
- **Long-term vision confirmed**: the project should serve both the bootcamp (learning + experimentation) and an ongoing forecasting benchmark/competition. The evaluation loop's backtest/live symmetry is the architectural property that makes this feasible.

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

