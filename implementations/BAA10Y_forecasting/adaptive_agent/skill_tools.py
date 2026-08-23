"""Tools for the BAA10Y adaptive tuning agent."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable

from BAA10Y_forecasting.adaptive_agent.optimizer import (
    BAA10YAdaptiveOptimizer,
    LLMP_SEARCH_SPACE,
)

from BAA10Y_forecasting.predictors.benchmark_configs import (
    get_benchmark_config,
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
MIN_PROMOTION_PREDICTIONS = 12

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
        """Return the approved search space for one forecasting method."""

        if method == "lightgbm":
            return _json({
                "status": "success",
                "method": "lightgbm",
                "strategy": "optuna_tpe",
                "covariate_panels": [
                    "target_only",
                    "default",
                    "default_plus_hyoas",
                ],
                "objective": (
                    "Minimize inner-validation "
                    "mean CRPS"
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
                "focus_actions": [
                    "broad_search",
                    "regularize_more",
                    "reduce_complexity",
                    "stabilize_boosting",
                    "continue_tpe",
                ],
                "evaluation": {
                    "development_experiment": (
                        "tune_development_2025"
                    ),
                    "inner_validation_experiment": (
                        "tune_inner_validation_2025"
                    ),
                    "outer_validation_experiment": (
                        "validate_2025"
                    ),
                },
                "governance": {
                    "maximum_trials": 18,
                    "tuning_can_promote": False,
                    "outer_validation_required": True,
                    "protected_eval_can_tune": False,
                },
            })

        if (
            method
            == "llmp_sampled_trajectory"
        ):
            return _json({
                "status": "success",
                "method": (
                    "llmp_sampled_trajectory"
                ),
                "strategy": (
                    "optuna_grid_llmp"
                ),
                "covariate_panels": [
                    "target_only",
                    "default",
                    "default_plus_hyoas",
                ],
                "objective": (
                    "Minimize inner-validation "
                    "mean CRPS"
                ),
                "parameter_space": {
                    "n_samples": (
                        LLMP_SEARCH_SPACE[
                            "n_samples"
                        ]
                    ),
                    "history_window": (
                        LLMP_SEARCH_SPACE[
                            "history_window"
                        ]
                    ),
                },
                "fixed_execution_settings": {
                    "model": (
                        "notebook_01_benchmark_model"
                    ),
                    "reasoning_effort": None,
                    "max_tokens": 16384,
                },
                "focus_actions": [
                    "broad_search",
                    "increase_samples",
                    "shorter_history",
                    "longer_history",
                    "continue_grid",
                ],
                "evaluation": {
                    "development_experiment": (
                        "tune_development_2025"
                    ),
                    "inner_validation_experiment": (
                        "tune_inner_validation_2025"
                    ),
                    "finalist_repeats": 3,
                    "outer_validation_experiment": (
                        "validate_2025"
                    ),
                },
                "governance": {
                    "maximum_trials": 12,
                    "tuning_can_promote": False,
                    "outer_validation_required": True,
                    "protected_eval_can_tune": False,
                    "stress_2020_allowed": False,
                },
            })

        return _error(
            "Unsupported adaptive-search method: "
            f"{method}. Expected 'lightgbm' or "
            "'llmp_sampled_trajectory'."
        )

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
        max_trials: int = 6,
        focus_action: str = "broad_search",
        reason: str = "",
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
                focus_action=focus_action,
                reason=reason,
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
    # Tool 6: Freeze one robust paired-tuning finalist
    # ------------------------------------------------------------------

    def freeze_search_candidate(
        study_name: str,
        candidate_id: str,
        reason: str,
    ) -> str:
        """Freeze a robust candidate before independent validation."""

        try:
            result = (
                optimizer.freeze_search_candidate(
                    study_name=study_name,
                    candidate_id=candidate_id,
                    reason=reason,
                )
            )

            return _json({
                "status": "success",
                **result,
            })

        except Exception as exc:
            return _error(
                "Unable to freeze search "
                f"candidate: {exc}"
            )



            # ------------------------------------------------------------------
    # Tool 7: Validate the frozen candidate against its benchmark
    # ------------------------------------------------------------------

    def run_frozen_validation(
        study_name: str,
        force_refresh: bool = False,
    ) -> str:
        """Run independent validation for a frozen search finalist.

        LightGBM uses one deterministic validation run. LLMP uses three
        independent repeated runs for both the benchmark and finalist.
        This tool never modifies the frozen parameter configuration.
        """

        try:
            state = store.load()

            study_record = state.get_study(
                study_name
            )

            if study_record is None:
                return _error(
                    f"Unknown study: {study_name}"
                )

            if not study_record.search_frozen:
                return _json({
                    "status": (
                        "validation_blocked"
                    ),
                    "reason": (
                        "The search finalist must "
                        "be frozen before accessing "
                        "validate_2025."
                    ),
                })

            candidate_id = (
                study_record.best_candidate_id
            )

            if not candidate_id:
                return _json({
                    "status": (
                        "validation_blocked"
                    ),
                    "reason": (
                        "The frozen study does not "
                        "contain a finalist."
                    ),
                })

            method = study_record.method
            horizon = study_record.horizon
            covariate_panel = (
                study_record.covariate_panel
            )

            baseline_parameters = (
                get_benchmark_config(
                    method,
                    covariate_panel,
                )
            )

            candidate_parameters = dict(
                study_record.best_parameters
            )

            required_repeats = (
                3
                if method
                == "llmp_sampled_trajectory"
                else 1
            )

            existing_validation_trials = (
                state.find_trials(
                    method=method,
                    horizon=horizon,
                    experiment=(
                        "validate_2025"
                    ),
                    covariate_panel=(
                        covariate_panel
                    ),
                    study_name=study_name,
                )
            )

            baseline_trials = [
                trial
                for trial
                in existing_validation_trials
                if trial.is_baseline
            ]

            candidate_trials = [
                trial
                for trial
                in existing_validation_trials
                if (
                    not trial.is_baseline
                    and trial.candidate_id
                    == candidate_id
                )
            ]

            def run_and_save(
                *,
                parameters: dict[str, Any],
                is_baseline: bool,
            ):
                """Run and save one outer-validation repetition."""

                # LLMP repetitions must bypass the prediction cache.
                trial_force_refresh = (
                    True
                    if method
                    == "llmp_sampled_trajectory"
                    else force_refresh
                )

                result = (
                    tuner.run_parameter_trial(
                        method=method,
                        horizon=horizon,
                        covariate_panel=(
                            covariate_panel
                        ),
                        parameters=parameters,
                        experiment=(
                            "validate_2025"
                        ),
                        force_refresh=(
                            trial_force_refresh
                        ),
                        is_baseline=(
                            is_baseline
                        ),
                    )
                )

                result.update({
                    "study_name": study_name,
                    "trial_number": None,
                    "parameter_hash": (
                        result[
                            "candidate_id"
                        ].rsplit(
                            "_",
                            1,
                        )[-1]
                    ),
                    "search_strategy": (
                        study_record
                        .search_strategy
                    ),
                })

                saved_trial, _ = (
                    store.add_trial(
                        result
                    )
                )

                return saved_trial

            missing_baseline_runs = max(
                0,
                required_repeats
                - len(baseline_trials),
            )

            missing_candidate_runs = max(
                0,
                required_repeats
                - len(candidate_trials),
            )

            for _ in range(
                missing_baseline_runs
            ):
                run_and_save(
                    parameters=(
                        baseline_parameters
                    ),
                    is_baseline=True,
                )

            for _ in range(
                missing_candidate_runs
            ):
                run_and_save(
                    parameters=(
                        candidate_parameters
                    ),
                    is_baseline=False,
                )

            final_state = store.load()

            validation_trials = (
                final_state.find_trials(
                    method=method,
                    horizon=horizon,
                    experiment=(
                        "validate_2025"
                    ),
                    covariate_panel=(
                        covariate_panel
                    ),
                    study_name=study_name,
                )
            )

            baseline_trials = sorted(
                [
                    trial
                    for trial
                    in validation_trials
                    if trial.is_baseline
                ],
                key=lambda item: item.ran_at,
            )[-required_repeats:]

            candidate_trials = sorted(
                [
                    trial
                    for trial
                    in validation_trials
                    if (
                        not trial.is_baseline
                        and trial.candidate_id
                        == candidate_id
                    )
                ],
                key=lambda item: item.ran_at,
            )[-required_repeats:]

            if (
                len(baseline_trials)
                < required_repeats
                or len(candidate_trials)
                < required_repeats
            ):
                raise RuntimeError(
                    "Independent validation did "
                    "not complete the required "
                    "number of repetitions."
                )

            baseline_scores = [
                trial.mean_crps
                for trial in baseline_trials
            ]

            candidate_scores = [
                trial.mean_crps
                for trial in candidate_trials
            ]

            baseline_mean_crps = fmean(
                baseline_scores
            )

            candidate_mean_crps = fmean(
                candidate_scores
            )

            baseline_crps_std = (
                pstdev(baseline_scores)
                if len(baseline_scores) > 1
                else 0.0
            )

            candidate_crps_std = (
                pstdev(candidate_scores)
                if len(candidate_scores) > 1
                else 0.0
            )

            improvement_pct = (
                crps_improvement_pct(
                    baseline_crps=(
                        baseline_mean_crps
                    ),
                    candidate_crps=(
                        candidate_mean_crps
                    ),
                )
            )

            minimum_predictions = min(
                trial.n_predictions
                for trial in [
                    *baseline_trials,
                    *candidate_trials,
                ]
            )

            if (
                improvement_pct is not None
                and improvement_pct
                >= MIN_PROMOTION_IMPROVEMENT_PCT
                and minimum_predictions
                >= MIN_PROMOTION_PREDICTIONS
            ):
                decision = "promote_tuned"
            else:
                decision = "retain_baseline"

            updated_study = (
                final_state.get_study(
                    study_name
                )
            )

            if updated_study is None:
                raise RuntimeError(
                    "Study disappeared from "
                    "persistent state."
                )

            updated_study.validation_decision = (
                decision
            )

            updated_study.agent_actions.append({
                "action_type": (
                    "outer_validation_review"
                ),
                "candidate_id": candidate_id,
                "required_repeats": (
                    required_repeats
                ),
                "baseline_mean_crps": (
                    baseline_mean_crps
                ),
                "candidate_mean_crps": (
                    candidate_mean_crps
                ),
                "improvement_pct": (
                    improvement_pct
                ),
                "minimum_predictions": (
                    minimum_predictions
                ),
                "decision": decision,
            })

            store.upsert_study(
                updated_study
            )

            return _json({
                "status": "success",
                "study_name": study_name,
                "method": method,
                "horizon": horizon,
                "covariate_panel": (
                    covariate_panel
                ),
                "experiment": (
                    "validate_2025"
                ),
                "required_repeats": (
                    required_repeats
                ),
                "baseline": {
                    "candidate_id": (
                        baseline_trials[-1]
                        .candidate_id
                    ),
                    "mean_crps": (
                        baseline_mean_crps
                    ),
                    "crps_std": (
                        baseline_crps_std
                    ),
                    "scores": baseline_scores,
                    "parameters": (
                        baseline_parameters
                    ),
                },
                "candidate": {
                    "candidate_id": (
                        candidate_id
                    ),
                    "mean_crps": (
                        candidate_mean_crps
                    ),
                    "crps_std": (
                        candidate_crps_std
                    ),
                    "scores": candidate_scores,
                    "parameters": (
                        candidate_parameters
                    ),
                },
                "improvement_pct": (
                    improvement_pct
                ),
                "minimum_predictions": (
                    minimum_predictions
                ),
                "decision": decision,
                "promotion_allowed": (
                    decision
                    == "promote_tuned"
                ),
            })

        except Exception as exc:
            return _error(
                "Frozen-candidate validation "
                f"failed: {exc}"
            )
    # ------------------------------------------------------------------
    # Tool 8: Compare saved trials
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
    # Tool 10: Promote an independently validated configuration
    # ------------------------------------------------------------------

    def promote_tuning_candidate(
        method: str,
        horizon: int,
        covariate_panel: str,
        candidate_id: str,
        reason: str = "",
    ) -> str:
        """Promote only a frozen, independently validated candidate."""

        try:
            state = store.load()

            matching_studies = [
                study
                for study in state.studies
                if (
                    study.method == method
                    and study.horizon == horizon
                    and study.covariate_panel
                    == covariate_panel
                    and study.best_candidate_id
                    == candidate_id
                )
            ]

            if not matching_studies:
                return _json({
                    "status": (
                        "promotion_blocked"
                    ),
                    "reason": (
                        "No matching adaptive "
                        "search study was found."
                    ),
                })

            study_record = sorted(
                matching_studies,
                key=lambda item: item.updated_at,
            )[-1]

            if not study_record.search_frozen:
                return _json({
                    "status": (
                        "promotion_blocked"
                    ),
                    "reason": (
                        "The selected candidate "
                        "has not been frozen."
                    ),
                })

            if (
                study_record.validation_decision
                != "promote_tuned"
            ):
                return _json({
                    "status": (
                        "promotion_blocked"
                    ),
                    "reason": (
                        "Independent validation did "
                        "not authorize promotion."
                    ),
                    "validation_decision": (
                        study_record
                        .validation_decision
                    ),
                })

            evidence_experiment = (
                "validate_2025"
            )

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
                    study_name=(
                        study_record.study_name
                    ),
                )
            )

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
                    study_name=(
                        study_record.study_name
                    ),
                )
                if trial.is_baseline
            ]

            required_repeats = (
                3
                if method
                == "llmp_sampled_trajectory"
                else 1
            )

            if (
                len(candidate_trials)
                < required_repeats
                or len(baseline_trials)
                < required_repeats
            ):
                return _json({
                    "status": (
                        "promotion_blocked"
                    ),
                    "reason": (
                        "Candidate does not have "
                        "the required independent "
                        "validation repetitions."
                    ),
                    "required_repeats": (
                        required_repeats
                    ),
                    "candidate_runs": len(
                        candidate_trials
                    ),
                    "baseline_runs": len(
                        baseline_trials
                    ),
                })

            candidate_trials = sorted(
                candidate_trials,
                key=lambda item: item.ran_at,
            )[-required_repeats:]

            baseline_trials = sorted(
                baseline_trials,
                key=lambda item: item.ran_at,
            )[-required_repeats:]

            baseline_crps = fmean(
                trial.mean_crps
                for trial in baseline_trials
            )

            candidate_crps = fmean(
                trial.mean_crps
                for trial in candidate_trials
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
                for trial in candidate_trials
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
                        "independent-validation "
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
                        "scored validation "
                        "predictions."
                    ),
                    "n_predictions": (
                        minimum_predictions
                    ),
                    "required_predictions": (
                        MIN_PROMOTION_PREDICTIONS
                    ),
                })

            latest_candidate = (
                candidate_trials[-1]
            )

            latest_baseline = (
                baseline_trials[-1]
            )

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

            origin_win_rate = None

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
                        latest_candidate.parameters
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
    def get_search_diagnostics(
        study_name: str,
    ) -> str:
        """Return paired development-versus-inner-validation diagnostics."""
        try:
            result = (
                optimizer.get_search_diagnostics(
                    study_name=study_name
                )
            )

            return _json({
                "status": "success",
                **result,
            })

        except Exception as exc:
            return _error(
                "Unable to create search "
                f"diagnostics: {exc}"
            )
        
    return [
        get_tuning_state,
        list_tuning_candidates,
        get_search_space,
        run_tuning_trial,
        run_adaptive_search,
        get_search_diagnostics,
        freeze_search_candidate,
        run_frozen_validation,
        compare_tuning_trials,
        promote_tuning_candidate,
        reject_tuning_candidate,
    ]


__all__ = [
    "build_baa10y_tuning_tools",
]