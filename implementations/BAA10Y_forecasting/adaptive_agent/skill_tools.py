"""Simple tools for the BAA10Y adaptive tuning agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from BAA10Y_forecasting.adaptive_agent.skill_state import (
    TuningStateStore,
)
from BAA10Y_forecasting.adaptive_agent.tuner import (
    BAA10YTuner,
)


def _json(data) -> str:
    """Return readable JSON for the agent."""

    return json.dumps(
        data,
        indent=2,
        default=str,
    )


def build_baa10y_tuning_tools(
    *,
    state_path: Path,
) -> list[Callable[..., str]]:
    """Create the tools used by the adaptive tuning agent."""

    tuner = BAA10YTuner()
    store = TuningStateStore(state_path)

    # ------------------------------------------------------------------
    # Tool 1: Read previous results
    # ------------------------------------------------------------------

    def get_tuning_state() -> str:
        """Return all saved tuning trials."""

        state = store.load()

        return _json(
            {
                "trial_count": len(state.trials),
                "trials": [
                    trial.model_dump(
                        mode="json"
                    )
                    for trial in state.trials
                ],
            }
        )

    # ------------------------------------------------------------------
    # Tool 2: List available configurations
    # ------------------------------------------------------------------

    def list_tuning_candidates(
        method: str = "",
        horizon: int = 0,
    ) -> str:
        """List available parameter candidates.

        Use an empty method for all methods.
        Use horizon=0 for all horizons.
        """

        candidates = tuner.list_candidates(
            method=method or None,
            horizon=(
                horizon
                if horizon != 0
                else None
            ),
        )

        return _json(candidates)

    # ------------------------------------------------------------------
    # Tool 3: Run one backtest
    # ------------------------------------------------------------------

    def run_tuning_trial(
        candidate_id: str,
        horizon: int,
        experiment: str = "smoke",
        force_refresh: bool = False,
    ) -> str:
        """Run one tuning candidate and save the result."""

        try:
            result = tuner.run_trial(
                candidate_id=candidate_id,
                horizon=horizon,
                experiment=experiment,
                force_refresh=force_refresh,
            )

            trial, added = store.add_trial(
                result
            )

            return _json(
                {
                    "status": (
                        "saved"
                        if added
                        else "already_saved"
                    ),
                    "trial": trial.model_dump(
                        mode="json"
                    ),
                }
            )

        except Exception as exc:
            return (
                "Tuning trial failed: "
                f"{exc}"
            )

    # ------------------------------------------------------------------
    # Tool 4: Compare saved results
    # ------------------------------------------------------------------

    def compare_tuning_trials(
        method: str,
        horizon: int,
        experiment: str = "smoke",
    ) -> str:
        """Compare saved trials by CRPS. Lower CRPS is better."""

        state = store.load()

        trials = state.find_trials(
            method=method,
            horizon=horizon,
            experiment=experiment,
        )

        if not trials:
            return (
                "No matching tuning trials found."
            )

        results = [
            {
                "candidate_id": (
                    trial.candidate_id
                ),
                "mean_crps": (
                    trial.mean_crps
                ),
                "is_baseline": (
                    trial.is_baseline
                ),
                "parameters": (
                    trial.parameters
                ),
                "ran_at": (
                    trial.ran_at
                ),
            }
            for trial in trials
        ]

        results.sort(
            key=lambda item: item["mean_crps"]
        )

        return _json(
            {
                "method": method,
                "horizon": horizon,
                "experiment": experiment,
                "best_candidate": (
                    results[0]["candidate_id"]
                ),
                "results": results,
            }
        )

    return [
        get_tuning_state,
        list_tuning_candidates,
        run_tuning_trial,
        compare_tuning_trials,
    ]


__all__ = [
    "build_baa10y_tuning_tools",
]