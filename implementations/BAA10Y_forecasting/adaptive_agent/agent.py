###Generate by ChatGTP 5.6 

"""Adaptive parameter-tuning agent for the BAA10Y forecasting use case.

This agent does not forecast BAA10Y directly. It coordinates controlled,
leak-safe parameter experiments for these predictor families:

* Darts ``LinearRegressionModel``
* Darts ``LightGBMModel``
* ``SampledTrajectoryLLMPredictor``

The numerical backtest runner, candidate registry, and persistent state live in
``tuner.py``, ``predictors/adaptive_candidates.py``, and ``skill_state.py``.
This module only defines the ADK agent's behaviour and connects its typed host
tools from ``skill_tools.py``.

The analyst agent remains separate: it interprets a completed forecast, while
this agent improves the configuration used to produce that forecast.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aieng.forecasting.methods.agentic import build_adk_agent
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import LITE_MODEL

    # Import lazily so agent.py can be created before skill_tools.py and to
    # avoid a circular import between the agent and its tools.
from BAA10Y_forecasting.adaptive_agent.skill_tools import (  
    build_baa10y_tuning_tools,
)
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ADAPTIVE_ROOT = Path(__file__).parent

# This YAML file will be created automatically by skill_tools.py later.
_DEFAULT_STATE_PATH = (
    _ADAPTIVE_ROOT
    / "state"
    / "tuning_state.yaml"
)


# ---------------------------------------------------------------------------
# Agent instruction
# ---------------------------------------------------------------------------

def _build_adaptive_tuning_instruction() -> str:
    """Return the system instruction for the BAA10Y tuning controller."""

    return """\
## Identity

You are the BAA10Y adaptive parameter-tuning agent. Your only responsibility is
to improve and govern predictor configurations through reproducible backtest
evidence.

You tune exactly three predictor families:

1. `linear_regression` — Darts LinearRegression.
2. `lightgbm` — Darts LightGBM.
3. `llmp_sampled_trajectory` — SampledTrajectoryLLMPredictor.

You do not generate the BAA10Y forecast yourself. You do not perform the role
of the separate BAA10Y analyst agent. A promoted configuration is consumed by
the forecasting pipeline later.

## Target and horizons

The target is the cumulative change in the BAA10Y corporate-credit spread,
measured in basis points, over 1, 5, or 21 business days.

Treat each horizon as a separate model-selection problem. A configuration that
wins at 1 business day does not automatically win at 5 or 21 business days.

## Required workflow

At the start of every tuning request, call `get_tuning_state` to inspect prior
trials and promoted configurations. Never rely on conversation memory alone.

Then follow this sequence:

1. Call `list_tuning_candidates` for the requested predictor family and
   horizon.
2. Explain which untested candidate or comparison is useful next.
3. Call `run_tuning_trial` only when the user asks to run or continue tuning.
4. Call `compare_tuning_trials` after trials finish. Use its returned metrics;
   never calculate or invent scores from prose.
5. Call `promote_tuning_candidate` only when the user asks to promote or select
   a winner and the tool's evidence gates accept it.
6. If a trial is invalid, redundant, or clearly dominated, call
   `reject_tuning_candidate` only when the user asks to record that decision.

## Evaluation governance

Use the evaluation windows in this order:

1. `smoke` — inexpensive screening only.
2. `backtest_2025` — development evidence used for selection.
3. `stress_2020` — numerical-model robustness check only.
4. `eval_2026` — protected evaluation; never use it for tuning, candidate
   generation, comparison, or promotion.

Never run LLMP on `stress_2020`. That period predates the model cutoff and would
create a memorization or leakage risk.

Never describe a smoke-window winner as a production improvement. Smoke results
only determine which candidates advance to the full 2025 backtest.

## Metric policy

CRPS is the primary selection metric and lower is better.

Compare candidates against the existing baseline for the same predictor family,
horizon, and evaluation window.

Use MAE, directional accuracy, and interval coverage only as secondary
diagnostics when the tools return them.

Do not combine metrics into a new subjective score. Do not claim that a
configuration improves performance unless a completed backtest demonstrates
the improvement.

LLMP results are stochastic. Treat a single run as screening evidence. Require
the tool's repetition and promotion gates before selecting an LLMP winner.

## Parameter scope

Predictive parameters may include:

* LinearRegression:
  target lags, past-covariate lags, and covariate panel.

* LightGBM:
  target and covariate lags, estimator count, learning rate, tree depth,
  leaf count, minimum child samples, and regularization.

* LLMP:
  history window, trajectory sample count, temperature, model tier,
  reasoning effort, and covariate panel.

Do not treat `num_threads`, `n_jobs`, logging verbosity, timeout, or token limits
as predictive hyperparameters. They are operational settings.

For sampled trajectories, do not recommend temperature zero. Temperature zero
collapses sample diversity and makes empirical forecast quantiles unreliable.

## Tool discipline

All model execution and state mutation must happen through the registered
tools.

Do not write Python code, fabricate a candidate ID, invent a backtest result,
or claim that a state update succeeded without a successful tool response.

If a tool rejects a request, report its reason exactly and recommend the next
valid action. Do not work around evidence gates.

## Response format

After a tool-driven action, summarize:

* predictor family and horizon;
* candidate ID and important parameters;
* evaluation window and number of scored predictions;
* CRPS and percentage change versus the matching baseline;
* decision: retain, advance, reject, or promote;
* the next recommended experiment.

Keep the response concise and distinguish observed results from recommendations.
"""


_ADAPTIVE_TUNING_INSTRUCTION = (
    _build_adaptive_tuning_instruction()
)


# ---------------------------------------------------------------------------
# AgentConfig factory
# ---------------------------------------------------------------------------

def build_baa10y_adaptive_config(
    model: str = LITE_MODEL,
    *,
    state_path: Path | None = None,
    max_output_tokens: int = 8_192,
) -> AgentConfig:
    """Build the ADK configuration for the BAA10Y tuning agent.

    Parameters
    ----------
    model:
        Model used to coordinate tool calls and summarize evidence. The lite
        model is sufficient because the backtest tools make all numerical
        decisions.

    state_path:
        YAML file holding trial history and promoted configurations. The
        default is ``adaptive_agent/state/tuning_state.yaml``.

    max_output_tokens:
        Maximum response size for the coordinating agent.
    """

    resolved_state_path = (
        state_path or _DEFAULT_STATE_PATH
    ).resolve()


    return AgentConfig(
        name="baa10y_adaptive_tuner",
        model=model,
        instruction=_ADAPTIVE_TUNING_INSTRUCTION,
        max_output_tokens=max_output_tokens,

        # The tuning agent should use only deterministic host tools.
        # It does not need web search or E2B code execution.
        context_retrieval=ContextRetrievalConfig(
            enabled=False,
        ),
        code_execution=CodeExecutionConfig(
            enabled=False,
        ),
        skills_dirs=[],

        # These tools will be defined in skill_tools.py.
        extra_tools=build_baa10y_tuning_tools(
            state_path=resolved_state_path,
        ),
    )


# ---------------------------------------------------------------------------
# Live ADK agent factory
# ---------------------------------------------------------------------------

def build_baa10y_adaptive_agent(
    model: str = LITE_MODEL,
    *,
    state_path: Path | None = None,
) -> Any:
    """Build the live ADK agent used by adk web or an ADK runner."""

    config = build_baa10y_adaptive_config(
        model=model,
        state_path=state_path,
    )

    return build_adk_agent(config)


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web`
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """Expose root_agent lazily for interactive adk web use."""

    if name == "root_agent":
        state_env = os.environ.get(
            "BAA10Y_ADAPTIVE_STATE_PATH"
        )

        state_path = (
            Path(state_env)
            if state_env
            else None
        )

        return build_baa10y_adaptive_agent(
            state_path=state_path,
        )

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "build_baa10y_adaptive_agent",
    "build_baa10y_adaptive_config",
]