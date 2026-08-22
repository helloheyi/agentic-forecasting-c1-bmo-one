"""Tools for the BAA10Y adaptive tuning agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from BAA10Y_forecasting.adaptive_agent.optimizer import (
    BAA10YAdaptiveOptimizer,
)
from BAA10Y_forecasting.adaptive_agent.skill_state import (
    PromotedConfiguration,
    RejectedCandidate,
    TuningStateStore,
)
from BAA10Y_forecasting.adaptive_agent.tuner import (
    BAA10YTuner,
    crps_improvement_pct,
)


MIN_PROMOTION_IMPROVEMENT_PCT = 1.0
MIN_PROMOTION_PREDICTIONS = 20


def _json(data) -> str:
    """Return readable JSON for the agent."""

    return json.dumps(
        data,
        indent=2,
        default=str,
    )


def _error(
    message: str,
) -> str:
    """Return a structured tool error."""

    return _json({
        "status": "error",
        "message": message,
    })


def build_baa10y_tuning_tools(
    *,
    state_path: Path,
) -> list[Callable[..., str]]:
    """Create the tools used by the adaptive tuning agent."""

    resolved_state_path = Path(
        state_path
    ).resolve()

    tuner = BAA10YTuner()

    store = TuningStateStore(
        resolved_state_path
    )

    optimizer = BAA10YAdaptiveOptimizer(
        state_path=resolved_state_path,
        optuna_storage_path=(
            resolved_state_path.parent
            / "optuna.db"
        ),
    )

    # ------------------------------------------------------------------
    # Tool 1: Read previous state
    # ------------------------------------------------------------------

    def get_tuning_state() -> str:
        """Return saved trials, studies and model decisions."""

        try:
            state = store.load()

            return _json({
                "status": "success",
                "schema_version": (
                    state.schema_version
                ),
                "trial_count": len(
                    state.trials
                ),
                "study_count": len(
                    state.studies
                ),
                "promoted_count": len(
                    state.promoted_configurations
                ),
                "rejected_count": len(
                    state.rejected_candidates
                ),
                "trials": [
                    trial.model_dump(
                        mode="json"
                    )
                    for trial in state.trials
                ],
                "studies": [
                    study.model_dump(
                        mode="json"
                    )
                    for study in state.studies
                ],
                "promoted_configurations": [
                    configuration.model_dump(
                        mode="json"
                    )
                    for configuration
                    in state.promoted_configurations
                ],
                "rejected_candidates": [
                    rejection.model_dump(
                        mode="json"
                    )
                    for rejection
                    in state.rejected_candidates
                ],
            })

        except Exception as exc:
            return _error(
                f"Unable to load tuning state: {exc}"
            )

    # ------------------------------------------------------------------
    # Tool 2: List the legacy fixed candidates
    # ------------------------------------------------------------------

    def list_tuning_candidates(
        method: str = "",
        horizon: int = 0,
    ) -> str:
        """List the existing fixed parameter candidates."""

        try:
            candidates = (
                tuner.list_candidates(
                    method=(
                        method or None
                    ),
                    horizon=(
                        horizon
                        if horizon != 0
                        else None
                    ),
                )
            )

            return _json({
                "status": "success",
                "candidate_count": len(
                    candidates
                ),
                "candidates": candidates,
            })

        except Exception as exc:
            return _error(
                "Unable to list candidates: "
                f"{exc}"
            )

    # ------------------------------------------------------------------
    # Tool 3: Describe the approved search space
    # ------------------------------------------------------------------

    def get_search_space(
        method: str = "lightgbm",
    ) -> str:
        """Return the approved adaptive parameter search space."""

        if method != "lightgbm":
            return _error(
                "The first adaptive search implementation "
                "supports LightGBM only."
            )

        return _json({
            "status": "success",
            "method": "lightgbm",
            "strategy": "optuna_tpe",
            "objective": (
                "minimize mean CRPS"
            ),
            "parameter_space": {
                "lags": [
                    3,
                    5,
                    10,
                    21,
                ],
                "lags_past_covariates": [
                    3,
                    5,
                    10,
                    21,
                ],
                "n_estimators": {
                    "minimum": 50,
                    "maximum": 400,
                    "step": 50,
                },
                "learning_rate": {
                    "minimum": 0.02,
                    "maximum": 0.15,
                    "scale": "log",
                },
                "tree_shapes": [
                    {
                        "max_depth": 3,
                        "num_leaves": 7,
                    },
                    {
                        "max_depth": 4,
                        "num_leaves": 15,
                    },
                    {
                        "max_depth": 6,
                        "num_leaves": 31,
                    },
                    {
                        "max_depth": -1,
                        "num_leaves": 15,
                    },
                    {
                        "max_depth": -1,
                        "num_leaves": 31,
                    },
                ],
                "min_child_samples": [
                    10,
                    20,
                    40,
                ],
                "reg_alpha_l1": {
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
                "reg_lambda_l2": {
                    "minimum": 0.0,
                    "maximum": 5.0,
                },
                "subsample": [
                    0.7,
                    0.85,
                    1.0,
                ],
                "colsample_bytree": [
                    0.7,
                    0.85,
                    1.0,
                ],
            },
            "fixed_execution_settings": {
                "num_threads": 1,
                "n_jobs": 1,
                "verbosity": -1,
                "random_state": 42,
            },
            "governance": {
                "smoke_is_screening_only": True,
                "smoke_can_promote": False,
                "protected_eval_can_tune": False,
                "maximum_trials": 50,
            },
        })

    # ------------------------------------------------------------------
    # Tool 4: Run one legacy fixed-candidate trial
    # ------------------------------------------------------------------

    def run_tuning_trial(
        candidate_id: str,
        horizon: int,
        experiment: str = "smoke",
        force_refresh: bool = False,
    ) -> str:
        """Run one fixed tuning candidate and save the result."""

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

            return _json({
                "status": (
                    "saved"
                    if added
                    else "already_saved"
                ),
                "trial": trial.model_dump(
                    mode="json"
                ),
            })

        except Exception as exc:
            return _error(
                f"Tuning trial failed: {exc}"
            )

    # ------------------------------------------------------------------
    # Tool 5: Run or resume the Optuna adaptive search
    # ------------------------------------------------------------------

    def run_adaptive_search(
        method: str = "lightgbm",
        horizon: int = 5,
        covariate_panel: str = "default",
        max_trials: int = 12,
        force_refresh: bool = False,
    ) -> str:
        """Run or resume an Optuna adaptive parameter search."""

        try:
            result = optimizer.run_search(
                method=method,
                horizon=horizon,
                covariate_panel=(
                    covariate_panel
                ),
                max_trials=max_trials,
                force_refresh=force_refresh,
            )

            return _json({
                "status": "success",
                **result,
            })

        except Exception as exc:
            return _error(
                "Adaptive search failed: "
                f"{exc}"
            )

    # ------------------------------------------------------------------
    # Tool 6: Compare saved trials
    # ------------------------------------------------------------------

    def compare_tuning_trials(
        method: str,
        horizon: int,
        experiment: str = "smoke",
        covariate_panel: str = "",
        study_name: str = "",
    ) -> str:
        """Compare saved trials using mean CRPS."""

        try:
            state = store.load()

            trials = state.find_trials(
                method=method,
                horizon=horizon,
                experiment=experiment,
                covariate_panel=(
                    covariate_panel or None
                ),
                study_name=(
                    study_name or None
                ),
            )

            if not trials:
                return _error(
                    "No matching tuning trials found."
                )

            grouped = {}

            for trial in trials:
                grouped.setdefault(
                    trial.candidate_id,
                    [],
                ).append(
                    trial
                )

            results = []

            for (
                candidate_id,
                candidate_trials,
            ) in grouped.items():
                mean_crps = (
                    sum(
                        trial.mean_crps
                        for trial
                        in candidate_trials
                    )
                    / len(candidate_trials)
                )

                latest_trial = sorted(
                    candidate_trials,
                    key=lambda item: (
                        item.ran_at
                    ),
                )[-1]

                results.append({
                    "candidate_id": (
                        candidate_id
                    ),
                    "is_baseline": any(
                        trial.is_baseline
                        for trial
                        in candidate_trials
                    ),
                    "mean_crps": (
                        mean_crps
                    ),
                    "runs": len(
                        candidate_trials
                    ),
                    "covariate_panel": (
                        latest_trial
                        .covariate_panel
                    ),
                    "parameters": (
                        latest_trial.parameters
                    ),
                })

            baseline_results = [
                result
                for result in results
                if result["is_baseline"]
            ]

            baseline_crps = None

            if baseline_results:
                baseline_crps = min(
                    result["mean_crps"]
                    for result
                    in baseline_results
                )

            for result in results:
                if baseline_crps is None:
                    result[
                        "improvement_pct"
                    ] = None
                else:
                    result[
                        "improvement_pct"
                    ] = crps_improvement_pct(
                        baseline_crps=(
                            baseline_crps
                        ),
                        candidate_crps=(
                            result["mean_crps"]
                        ),
                    )

            results.sort(
                key=lambda item: (
                    item["mean_crps"]
                )
            )

            return _json({
                "status": "success",
                "method": method,
                "horizon": horizon,
                "experiment": experiment,
                "covariate_panel": (
                    covariate_panel or "all"
                ),
                "study_name": (
                    study_name or "all"
                ),
                "baseline_crps": (
                    baseline_crps
                ),
                "best_candidate": (
                    results[0][
                        "candidate_id"
                    ]
                ),
                "results": results,
            })

        except Exception as exc:
            return _error(
                "Unable to compare trials: "
                f"{exc}"
            )

    # ------------------------------------------------------------------
    # Tool 7: Promote a validated configuration
    # ------------------------------------------------------------------

    def promote_tuning_candidate(
        method: str,
        horizon: int,
        covariate_panel: str,
        candidate_id: str,
        reason: str = "",
    ) -> str:
        """Promote a candidate only with sufficient backtest evidence."""

        try:
            evidence_experiment = (
                "backtest_2025"
            )

            state = store.load()

            candidate_trials = (
                state.find_trials(
                    method=method,
                    horizon=horizon,
                    experiment=(
                        evidence_experiment
                    ),
                    candidate_id=(
                        candidate_id
                    ),
                    covariate_panel=(
                        covariate_panel
                    ),
                )
            )

            if not candidate_trials:
                return _json({
                    "status": (
                        "promotion_blocked"
                    ),
                    "reason": (
                        "Candidate has no "
                        "backtest_2025 evidence."
                    ),
                })

            baseline_trials = [
                trial
                for trial in state.find_trials(
                    method=method,
                    horizon=horizon,
                    experiment=(
                        evidence_experiment
                    ),
                    covariate_panel=(
                        covariate_panel
                    ),
                )
                if trial.is_baseline
            ]

            if not baseline_trials:
                return _json({
                    "status": (
                        "promotion_blocked"
                    ),
                    "reason": (
                        "No matching baseline "
                        "backtest was found."
                    ),
                })

            baseline_crps = (
                sum(
                    trial.mean_crps
                    for trial
                    in baseline_trials
                )
                / len(baseline_trials)
            )

            candidate_crps = (
                sum(
                    trial.mean_crps
                    for trial
                    in candidate_trials
                )
                / len(candidate_trials)
            )

            improvement_pct = (
                crps_improvement_pct(
                    baseline_crps=(
                        baseline_crps
                    ),
                    candidate_crps=(
                        candidate_crps
                    ),
                )
            )

            minimum_predictions = min(
                trial.n_predictions
                for trial
                in candidate_trials
            )

            if (
                improvement_pct is None
                or improvement_pct
                < MIN_PROMOTION_IMPROVEMENT_PCT
            ):
                return _json({
                    "status": (
                        "promotion_blocked"
                    ),
                    "reason": (
                        "Candidate did not improve "
                        "CRPS by at least "
                        f"{MIN_PROMOTION_IMPROVEMENT_PCT:.1f}%."
                    ),
                    "baseline_crps": (
                        baseline_crps
                    ),
                    "candidate_crps": (
                        candidate_crps
                    ),
                    "improvement_pct": (
                        improvement_pct
                    ),
                })

            if (
                minimum_predictions
                < MIN_PROMOTION_PREDICTIONS
            ):
                return _json({
                    "status": (
                        "promotion_blocked"
                    ),
                    "reason": (
                        "Candidate has insufficient "
                        "scored predictions."
                    ),
                    "n_predictions": (
                        minimum_predictions
                    ),
                    "required_predictions": (
                        MIN_PROMOTION_PREDICTIONS
                    ),
                })

            latest_candidate = sorted(
                candidate_trials,
                key=lambda item: (
                    item.ran_at
                ),
            )[-1]

            origin_win_rate = None

            latest_baseline = sorted(
                baseline_trials,
                key=lambda item: (
                    item.ran_at
                ),
            )[-1]

            common_origins = (
                set(
                    latest_candidate
                    .score_by_origin
                )
                & set(
                    latest_baseline
                    .score_by_origin
                )
            )

            if common_origins:
                origin_win_rate = (
                    sum(
                        latest_candidate
                        .score_by_origin[origin]
                        < latest_baseline
                        .score_by_origin[origin]
                        for origin
                        in common_origins
                    )
                    / len(common_origins)
                )

            configuration = (
                PromotedConfiguration(
                    method=method,
                    horizon=horizon,
                    covariate_panel=(
                        covariate_panel
                    ),
                    candidate_id=(
                        candidate_id
                    ),
                    parameters=(
                        latest_candidate
                        .parameters
                    ),
                    baseline_crps=(
                        baseline_crps
                    ),
                    candidate_crps=(
                        candidate_crps
                    ),
                    improvement_pct=(
                        improvement_pct
                    ),
                    origin_win_rate=(
                        origin_win_rate
                    ),
                    evidence_experiment=(
                        evidence_experiment
                    ),
                    reason=reason,
                )
            )

            store.promote(
                configuration
            )

            return _json({
                "status": "promoted",
                "configuration": (
                    configuration.model_dump(
                        mode="json"
                    )
                ),
            })

        except Exception as exc:
            return _error(
                "Promotion failed: "
                f"{exc}"
            )

    # ------------------------------------------------------------------
    # Tool 8: Record a rejected candidate
    # ------------------------------------------------------------------

    def reject_tuning_candidate(
        method: str,
        horizon: int,
        covariate_panel: str,
        candidate_id: str,
        reason: str,
    ) -> str:
        """Record an explicit candidate rejection."""

        try:
            state = store.load()

            matching_trials = (
                state.find_trials(
                    method=method,
                    horizon=horizon,
                    candidate_id=(
                        candidate_id
                    ),
                    covariate_panel=(
                        covariate_panel
                    ),
                )
            )

            if not matching_trials:
                return _error(
                    "Cannot reject an unknown "
                    "candidate."
                )

            rejection = RejectedCandidate(
                candidate_id=candidate_id,
                method=method,
                horizon=horizon,
                covariate_panel=(
                    covariate_panel
                ),
                reason=reason,
            )

            store.reject(
                rejection
            )

            return _json({
                "status": "rejected",
                "rejection": (
                    rejection.model_dump(
                        mode="json"
                    )
                ),
            })

        except Exception as exc:
            return _error(
                "Candidate rejection failed: "
                f"{exc}"
            )

    return [
        get_tuning_state,
        list_tuning_candidates,
        get_search_space,
        run_tuning_trial,
        run_adaptive_search,
        compare_tuning_trials,
        promote_tuning_candidate,
        reject_tuning_candidate,
    ]


__all__ = [
    "build_baa10y_tuning_tools",
]