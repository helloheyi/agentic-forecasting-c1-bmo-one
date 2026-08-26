"""Optuna search controller for BAA10Y adaptive tuning."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from BAA10Y_forecasting.adaptive_agent.skill_state import (
    SearchStudyRecord,
    TuningStateStore,
)
from BAA10Y_forecasting.adaptive_agent.tuner import (
    BAA10YTuner,
    crps_improvement_pct,

)
from BAA10Y_forecasting.predictors.benchmark_configs import (
    get_benchmark_config,
)


TREE_SHAPES = {
    "depth3_leaves7": {
        "max_depth": 3,
        "num_leaves": 7,
    },
    "depth4_leaves15": {
        "max_depth": 4,
        "num_leaves": 15,
    },
    "depth6_leaves31": {
        "max_depth": 6,
        "num_leaves": 31,
    },
    "unlimited_leaves15": {
        "max_depth": -1,
        "num_leaves": 15,
    },
    "unlimited_leaves31": {
        "max_depth": -1,
        "num_leaves": 31,
    },
}

MAX_LIGHTGBM_TRIALS = 18
MAX_LLMP_TRIALS = 12


LIGHTGBM_FOCUS_TEMPLATES = {
    "broad_search": [],

    "continue_tpe": [],

    "reduce_complexity": [
        {
            "tree_shape": "depth3_leaves7",
        },
        {
            "tree_shape": "depth4_leaves15",
        },
    ],

    "regularize_more": [
        {
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
        },
        {
            "reg_alpha": 1.0,
            "reg_lambda": 4.0,
        },
    ],

    "stabilize_boosting": [
        {
            "learning_rate": 0.03,
            "n_estimators": 300,
        },
        {
            "learning_rate": 0.05,
            "n_estimators": 200,
        },
    ],
}
def suggest_lightgbm_parameters(
    trial: optuna.Trial,
    *,
    covariate_panel: str,
) -> dict[str, Any]:
    """Ask Optuna to suggest one LightGBM configuration."""

    parameters = get_benchmark_config(
        "lightgbm",
        covariate_panel,
    )

    parameters["lags"] = (
        trial.suggest_categorical(
            "lags",
            [3, 5, 10, 21],
        )
    )

    if covariate_panel != "target_only":
        parameters[
            "lags_past_covariates"
        ] = trial.suggest_categorical(
            "lags_past_covariates",
            [3, 5, 10, 21],
        )

    tree_shape_name = (
        trial.suggest_categorical(
            "tree_shape",
            list(TREE_SHAPES),
        )
    )

    tree_shape = TREE_SHAPES[
        tree_shape_name
    ]

    subsample = (
        trial.suggest_categorical(
            "subsample",
            [0.7, 0.85, 1.0],
        )
    )

    lgbm_kwargs = copy.deepcopy(
        parameters["lgbm_kwargs"]
    )

    lgbm_kwargs.update({
        "n_estimators": (
            trial.suggest_int(
                "n_estimators",
                50,
                400,
                step=50,
            )
        ),
        "learning_rate": (
            trial.suggest_float(
                "learning_rate",
                0.02,
                0.15,
                log=True,
            )
        ),
        "max_depth": tree_shape[
            "max_depth"
        ],
        "num_leaves": tree_shape[
            "num_leaves"
        ],
        "min_child_samples": (
            trial.suggest_categorical(
                "min_child_samples",
                [10, 20, 40],
            )
        ),
        "reg_alpha": (
            trial.suggest_float(
                "reg_alpha",
                0.0,
                2.0,
            )
        ),
        "reg_lambda": (
            trial.suggest_float(
                "reg_lambda",
                0.0,
                5.0,
            )
        ),
        "subsample": subsample,
        "subsample_freq": (
            1
            if subsample < 1.0
            else 0
        ),
        "colsample_bytree": (
            trial.suggest_categorical(
                "colsample_bytree",
                [0.7, 0.85, 1.0],
            )
        ),
    })

    parameters[
        "lgbm_kwargs"
    ] = lgbm_kwargs

    return parameters

LLMP_SEARCH_SPACE = {
    "n_samples": [
        8,
        12,
        16,
    ],
    "history_window": [
        32,
        48,
        64,
        96,
    ],
}

LLMP_FOCUS_TEMPLATES = {
    "broad_search": [],

    "continue_grid": [],

    "increase_samples": [
        {
            "n_samples": 12,
            "history_window": 48,
        },
        {
            "n_samples": 16,
            "history_window": 48,
        },
    ],

    "shorter_history": [
        {
            "n_samples": 8,
            "history_window": 32,
        },
        {
            "n_samples": 12,
            "history_window": 32,
        },
    ],

    "longer_history": [
        {
            "n_samples": 8,
            "history_window": 64,
        },
        {
            "n_samples": 8,
            "history_window": 96,
        },
    ],
}


def suggest_llmp_parameters(
    trial: optuna.Trial,
    *,
    covariate_panel: str,
) -> dict[str, Any]:
    """Ask Optuna to suggest one sampled-trajectory LLMP configuration."""

    parameters = get_benchmark_config(
        "llmp_sampled_trajectory",
        covariate_panel,
    )

    parameters["n_samples"] = (
        trial.suggest_categorical(
            "n_samples",
            LLMP_SEARCH_SPACE[
                "n_samples"
            ],
        )
    )

    parameters["history_window"] = (
        trial.suggest_categorical(
            "history_window",
            LLMP_SEARCH_SPACE[
                "history_window"
            ],
        )
    )

    # These are fixed execution settings, not tuned parameters.
    parameters.setdefault(
        "reasoning_effort",
        None,
    )

    parameters.setdefault(
        "max_tokens",
        16384,
    )

    return parameters

class BAA10YAdaptiveOptimizer:
    """Run persistent Optuna searches for BAA10Y models."""

    def __init__(
        self,
        *,
        state_path: Path,
        optuna_storage_path: Path | None = None,
    ) -> None:
        self.state_path = Path(
            state_path
        )

        self.store = TuningStateStore(
            self.state_path
        )

        self.tuner = BAA10YTuner()

        self.optuna_storage_path = (
            Path(optuna_storage_path)
            if optuna_storage_path
            is not None
            else (
                self.state_path.parent
                / "optuna.db"
            )
        )

        self.optuna_storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def storage_url(self) -> str:
        """Return the SQLite URL used by Optuna."""

        return (
            "sqlite:///"
            f"{self.optuna_storage_path.resolve()}"
        )

    def _study_name(
        self,
        *,
        method: str,
        horizon: int,
        covariate_panel: str,
        experiment: str,
    ) -> str:
        """Return a stable study name."""

        return (
            "baa10y_"
            f"{method}_"
            f"{covariate_panel}_"
            f"h{horizon}_"
            f"{experiment}"
        )
    def run_search(
        self,
        *,
        method: str,
        horizon: int,
        covariate_panel: str,
        max_trials: int = 6,
        focus_action: str = "broad_search",
        reason: str = "",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Run or resume a paired adaptive parameter search.

        LightGBM uses Optuna's TPE sampler. The sampled-trajectory
        LLMP uses a bounded Optuna grid. Every candidate is evaluated
        on both the development and inner-validation windows.

        This method never accesses the independent validate_2025
        experiment and never promotes a model.
        """

        supported_methods = {
            "lightgbm",
            "llmp_sampled_trajectory",
        }

        supported_covariate_panels = {
            "target_only",
            "default",
            "default_plus_hyoas",
        }

        if method not in supported_methods:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Expected one of {sorted(supported_methods)}."
            )

        if horizon not in {1, 5, 21}:
            raise ValueError(
                "horizon must be 1, 5, or 21."
            )

        if (
            covariate_panel
            not in supported_covariate_panels
        ):
            raise ValueError(
                "Unsupported covariate panel: "
                f"{covariate_panel}. Expected one of "
                f"{sorted(supported_covariate_panels)}."
            )

        if method == "lightgbm":
            maximum_allowed_trials = (
                MAX_LIGHTGBM_TRIALS
            )

            focus_templates = (
                LIGHTGBM_FOCUS_TEMPLATES
            )

            sampler = optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=5,
            )

            search_strategy = "optuna_tpe"

        else:
            maximum_allowed_trials = (
                MAX_LLMP_TRIALS
            )

            focus_templates = (
                LLMP_FOCUS_TEMPLATES
            )

            sampler = optuna.samplers.GridSampler(
                search_space=LLMP_SEARCH_SPACE,
                seed=42,
            )

            search_strategy = (
                "optuna_grid_llmp"
            )

        if not 1 <= max_trials <= maximum_allowed_trials:
            raise ValueError(
                f"{method} max_trials must be between "
                f"1 and {maximum_allowed_trials}."
            )

        if focus_action not in focus_templates:
            raise ValueError(
                f"Unsupported {method} focus_action: "
                f"{focus_action}. Expected one of "
                f"{sorted(focus_templates)}."
            )

        experiment = "tune_paired_2025"

        study_name = self._study_name(
            method=method,
            horizon=horizon,
            covariate_panel=covariate_panel,
            experiment=experiment,
        )

        # Do not continue changing parameters after the agent
        # freezes the selected finalist.
        state = self.store.load()

        study_record = state.get_study(
            study_name
        )

        if (
            study_record is not None
            and study_record.search_frozen
        ):
            raise RuntimeError(
                f"Study {study_name} is frozen. "
                "It cannot run additional tuning trials."
            )

        # Run the matching benchmark once on the same paired
        # development and inner-validation periods.
        baseline_parameters = (
            get_benchmark_config(
                method,
                covariate_panel,
            )
        )

        baseline_result = (
            self.tuner.run_paired_parameter_trial(
                method=method,
                horizon=horizon,
                covariate_panel=covariate_panel,
                parameters=baseline_parameters,
                force_refresh=force_refresh,
                is_baseline=True,
            )
        )

        baseline_result.update({
            "study_name": study_name,
            "trial_number": None,
            "parameter_hash": (
                baseline_result[
                    "candidate_id"
                ].rsplit(
                    "_",
                    1,
                )[-1]
            ),
            "search_strategy": search_strategy,
        })

        baseline_trial, _ = (
            self.store.add_trial(
                baseline_result
            )
        )

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            storage=self.storage_url,
            load_if_exists=True,
        )

        if study_record is None:
            study_record = SearchStudyRecord(
                study_name=study_name,
                method=method,
                horizon=horizon,
                covariate_panel=covariate_panel,
                search_strategy=search_strategy,
                objective_experiment=experiment,
                validation_experiment=(
                    "validate_2025"
                ),
                max_trials=max_trials,
                status="created",
            )

        study_record.search_strategy = (
            search_strategy
        )
        study_record.status = "running"
        study_record.max_trials = max_trials
        study_record.baseline_candidate_id = (
            baseline_trial.candidate_id
        )
        study_record.baseline_mean_crps = (
            baseline_trial.mean_crps
        )

        self.store.upsert_study(
            study_record
        )

        def objective(
            trial: optuna.Trial,
        ) -> float:
            """Evaluate one Optuna parameter suggestion."""

            if method == "lightgbm":
                parameters = (
                    suggest_lightgbm_parameters(
                        trial,
                        covariate_panel=(
                            covariate_panel
                        ),
                    )
                )
            else:
                parameters = (
                    suggest_llmp_parameters(
                        trial,
                        covariate_panel=(
                            covariate_panel
                        ),
                    )
                )

            result = (
                self.tuner.run_paired_parameter_trial(
                    method=method,
                    horizon=horizon,
                    covariate_panel=(
                        covariate_panel
                    ),
                    parameters=parameters,
                    force_refresh=force_refresh,
                    is_baseline=False,
                )
            )

            result.update({
                "study_name": study_name,
                "trial_number": trial.number,
                "parameter_hash": (
                    result[
                        "candidate_id"
                    ].rsplit(
                        "_",
                        1,
                    )[-1]
                ),
                "search_strategy": (
                    search_strategy
                ),
            })

            saved_trial, _ = (
                self.store.add_trial(
                    result
                )
            )

            trial.set_user_attr(
                "candidate_id",
                saved_trial.candidate_id,
            )
            trial.set_user_attr(
                "predictor_id",
                saved_trial.predictor_id,
            )
            trial.set_user_attr(
                "development_mean_crps",
                saved_trial.development_mean_crps,
            )
            trial.set_user_attr(
                "inner_validation_mean_crps",
                saved_trial.inner_validation_mean_crps,
            )
            trial.set_user_attr(
                "generalization_gap_pct",
                saved_trial.generalization_gap_pct,
            )

            # mean_crps is the inner-validation CRPS for
            # a tune_paired_2025 TrialRecord.
            return float(
                saved_trial.mean_crps
            )

        completed_before = sum(
            trial.state == TrialState.COMPLETE
            for trial in study.trials
        )

        remaining_trials = max(
            0,
            max_trials - completed_before,
        )

        # The agent's focus action affects the order of the
        # remaining parameter evaluations.
        if remaining_trials > 0:
            templates = copy.deepcopy(
                focus_templates[
                    focus_action
                ]
            )

            for template in templates:
                if (
                    method == "lightgbm"
                    and covariate_panel
                    == "target_only"
                ):
                    template.pop(
                        "lags_past_covariates",
                        None,
                    )

                study.enqueue_trial(
                    template,
                    skip_if_exists=True,
                )

            study.optimize(
                objective,
                n_trials=remaining_trials,
            )

        completed_trials = [
            trial
            for trial in study.trials
            if trial.state
            == TrialState.COMPLETE
        ]

        if remaining_trials > 0:
            study_record.agent_actions.append({
                "action_type": (
                    "parameter_search"
                ),
                "focus_action": focus_action,
                "reason": (
                    reason
                    or "Run the bounded adaptive search."
                ),
                "completed_before": (
                    completed_before
                ),
                "completed_after": (
                    len(completed_trials)
                ),
                "requested_total_trials": (
                    max_trials
                ),
            })

        if not completed_trials:
            study_record.status = "failed"

            self.store.upsert_study(
                study_record
            )

            raise RuntimeError(
                "Optuna did not complete any "
                "candidate trials."
            )

        best_optuna_trial = min(
            completed_trials,
            key=lambda item: float(
                item.value
            ),
        )

        best_candidate_id = (
            best_optuna_trial.user_attrs.get(
                "candidate_id"
            )
        )

        if not best_candidate_id:
            raise RuntimeError(
                "The best Optuna trial does not "
                "contain a candidate ID."
            )

        updated_state = self.store.load()

        best_records = (
            updated_state.find_trials(
                method=method,
                horizon=horizon,
                experiment=experiment,
                candidate_id=best_candidate_id,
                covariate_panel=(
                    covariate_panel
                ),
                study_name=study_name,
            )
        )

        if not best_records:
            raise RuntimeError(
                "The best Optuna result was not "
                "found in tuning state."
            )

        best_trial = best_records[-1]

        improvement_pct = (
            crps_improvement_pct(
                baseline_crps=(
                    baseline_trial.mean_crps
                ),
                candidate_crps=(
                    best_trial.mean_crps
                ),
            )
        )

        if (
            improvement_pct is not None
            and improvement_pct > 0
        ):
            decision = (
                "review_paired_diagnostics"
            )
        else:
            decision = (
                "retain_baseline_or_continue_search"
            )

        study_record.status = "completed"
        study_record.completed_trials = len(
            completed_trials
        )
        study_record.best_candidate_id = (
            best_trial.candidate_id
        )
        study_record.best_mean_crps = (
            best_trial.mean_crps
        )
        study_record.best_parameters = (
            copy.deepcopy(
                best_trial.parameters
            )
        )

        self.store.upsert_study(
            study_record
        )

        final_state = self.store.load()

        history = final_state.find_trials(
            method=method,
            horizon=horizon,
            experiment=experiment,
            covariate_panel=(
                covariate_panel
            ),
            study_name=study_name,
        )

        history_rows = [
            {
                "trial_number": (
                    trial.trial_number
                ),
                "candidate_id": (
                    trial.candidate_id
                ),
                "is_baseline": (
                    trial.is_baseline
                ),
                "development_crps": (
                    trial.development_mean_crps
                ),
                "inner_validation_crps": (
                    trial.inner_validation_mean_crps
                ),
                "generalization_gap_pct": (
                    trial.generalization_gap_pct
                ),
                "mean_crps": trial.mean_crps,
                "parameters": trial.parameters,
            }
            for trial in sorted(
                history,
                key=lambda item: (
                    0 if item.is_baseline else 1,
                    (
                        item.trial_number
                        if item.trial_number
                        is not None
                        else -1
                    ),
                ),
            )
        ]

        return {
            "study_name": study_name,
            "method": method,
            "horizon": horizon,
            "covariate_panel": (
                covariate_panel
            ),
            "search_strategy": (
                search_strategy
            ),
            "experiment": experiment,
            "candidate_trials_completed": (
                len(completed_trials)
            ),
            "baseline": {
                "candidate_id": (
                    baseline_trial.candidate_id
                ),
                "mean_crps": (
                    baseline_trial.mean_crps
                ),
                "parameters": (
                    baseline_trial.parameters
                ),
            },
            "best_trial": {
                "trial_number": (
                    best_trial.trial_number
                ),
                "candidate_id": (
                    best_trial.candidate_id
                ),
                "mean_crps": (
                    best_trial.mean_crps
                ),
                "parameters": (
                    best_trial.parameters
                ),
            },
            "improvement_pct": (
                improvement_pct
            ),
            "decision": decision,
            "search_frozen": (
                study_record.search_frozen
            ),
            "promotion_allowed": False,
            "promotion_block_reason": (
                "Paired tuning evidence cannot "
                "promote a model. The agent must "
                "review diagnostics, freeze a "
                "finalist, and then evaluate it on "
                "validate_2025."
            ),
            "trial_history": history_rows,
        }

    def freeze_search_candidate(
        self,
        *,
        study_name: str,
        candidate_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Freeze one robust candidate before outer validation.

        A candidate cannot be frozen when paired diagnostics mark
        it as possible overfitting or when it fails to improve the
        matching baseline on the inner-validation period.
        """

        diagnostics = self.get_search_diagnostics(
            study_name=study_name
        )

        selected_diagnostic = next(
            (
                row
                for row
                in diagnostics[
                    "trial_diagnostics"
                ]
                if row["candidate_id"]
                == candidate_id
            ),
            None,
        )

        if selected_diagnostic is None:
            raise ValueError(
                "Candidate does not belong to "
                f"study {study_name}: {candidate_id}"
            )

        if not selected_diagnostic[
            "robust_candidate"
        ]:
            raise ValueError(
                "Candidate cannot be frozen because "
                "it either failed to improve inner "
                "validation or shows possible "
                "overfitting."
            )

        state = self.store.load()

        study_record = state.get_study(
            study_name
        )

        if study_record is None:
            raise ValueError(
                f"Unknown study: {study_name}"
            )

        matching_trials = state.find_trials(
            method=study_record.method,
            horizon=study_record.horizon,
            experiment="tune_paired_2025",
            candidate_id=candidate_id,
            covariate_panel=(
                study_record.covariate_panel
            ),
            study_name=study_name,
        )

        if not matching_trials:
            raise RuntimeError(
                "The selected trial was not found "
                "in tuning state."
            )

        selected_trial = matching_trials[-1]

        study_record.best_candidate_id = (
            candidate_id
        )
        study_record.best_mean_crps = (
            selected_trial.mean_crps
        )
        study_record.best_parameters = (
            copy.deepcopy(
                selected_trial.parameters
            )
        )
        study_record.search_frozen = True
        study_record.validation_decision = (
            "pending_validation"
        )

        study_record.agent_actions.append({
            "action_type": "freeze_candidate",
            "candidate_id": candidate_id,
            "reason": reason,
            "development_crps": (
                selected_trial
                .development_mean_crps
            ),
            "inner_validation_crps": (
                selected_trial
                .inner_validation_mean_crps
            ),
            "generalization_gap_pct": (
                selected_trial
                .generalization_gap_pct
            ),
        })

        self.store.upsert_study(
            study_record
        )

        return {
            "status": "frozen",
            "study_name": study_name,
            "candidate_id": candidate_id,
            "parameters": (
                selected_trial.parameters
            ),
            "validation_experiment": (
                "validate_2025"
            ),
            "promotion_allowed": False,
        }
    def get_search_diagnostics(
        self,
        *,
        study_name: str,
    ) -> dict[str, Any]:
        """Compare development and inner-validation evidence."""

        state = self.store.load()

        study_record = state.get_study(
            study_name
        )

        if study_record is None:
            raise ValueError(
                f"Unknown study: {study_name}"
            )

        if study_record.method == "lightgbm":
            focus_templates = (
                LIGHTGBM_FOCUS_TEMPLATES
            )
        else:
            focus_templates = (
                LLMP_FOCUS_TEMPLATES
            )
        trials = state.find_trials(
            method=study_record.method,
            horizon=study_record.horizon,
            experiment=(
                "tune_paired_2025"
            ),
            covariate_panel=(
                study_record.covariate_panel
            ),
            study_name=study_name,
        )

        paired_trials = [
            trial
            for trial in trials
            if (
                trial.development_mean_crps
                is not None
                and trial
                .inner_validation_mean_crps
                is not None
            )
        ]

        baseline_trials = [
            trial
            for trial in paired_trials
            if trial.is_baseline
        ]

        if not baseline_trials:
            raise RuntimeError(
                "No paired benchmark trial found."
            )

        baseline = baseline_trials[-1]

        diagnostics = []

        for trial in paired_trials:
            if trial.is_baseline:
                continue

            development_improvement = (
                crps_improvement_pct(
                    baseline_crps=(
                        baseline
                        .development_mean_crps
                    ),
                    candidate_crps=(
                        trial
                        .development_mean_crps
                    ),
                )
            )

            validation_improvement = (
                crps_improvement_pct(
                    baseline_crps=(
                        baseline
                        .inner_validation_mean_crps
                    ),
                    candidate_crps=(
                        trial
                        .inner_validation_mean_crps
                    ),
                )
            )

            gap_deterioration = False

            if (
                trial.generalization_gap_pct
                is not None
                and baseline
                .generalization_gap_pct
                is not None
            ):
                gap_deterioration = (
                    trial
                    .generalization_gap_pct
                    > baseline
                    .generalization_gap_pct
                    + 10.0
                )

            possible_overfitting = (
                (
                    development_improvement
                    is not None
                    and development_improvement
                    > 0
                    and validation_improvement
                    is not None
                    and validation_improvement
                    <= 0
                )
                or gap_deterioration
            )

            robust_candidate = (
                development_improvement
                is not None
                and development_improvement > 0
                and validation_improvement
                is not None
                and validation_improvement > 0
                and not possible_overfitting
            )

            diagnostics.append({
                "trial_number": (
                    trial.trial_number
                ),
                "candidate_id": (
                    trial.candidate_id
                ),
                "development_crps": (
                    trial
                    .development_mean_crps
                ),
                "inner_validation_crps": (
                    trial
                    .inner_validation_mean_crps
                ),
                "development_improvement_pct": (
                    development_improvement
                ),
                "inner_validation_improvement_pct": (
                    validation_improvement
                ),
                "generalization_gap_pct": (
                    trial
                    .generalization_gap_pct
                ),
                "possible_overfitting": (
                    possible_overfitting
                ),

                "robust_candidate": (
                    robust_candidate
                ),

                "parameters": (
                    trial.parameters
                ),
            })

        diagnostics.sort(
            key=lambda row: (
                row[
                    "inner_validation_crps"
                ]
            )
        )

        return {
            "study_name": study_name,
            "benchmark": {
                "development_crps": (
                    baseline
                    .development_mean_crps
                ),
                "inner_validation_crps": (
                    baseline
                    .inner_validation_mean_crps
                ),
                "generalization_gap_pct": (
                    baseline
                    .generalization_gap_pct
                ),
            },
            "trial_diagnostics": (
                diagnostics
            ),
            "agent_actions": (
                study_record.agent_actions
            ),
            "allowed_focus_actions": (
                list(focus_templates)
            ),
            "search_frozen": (
                study_record.search_frozen
            ),
            "validation_decision": (
                study_record.validation_decision
            ),
        }


__all__ = [
    "BAA10YAdaptiveOptimizer",
    "LIGHTGBM_FOCUS_TEMPLATES",
    "LLMP_FOCUS_TEMPLATES",
    "LLMP_SEARCH_SPACE",
    "suggest_lightgbm_parameters",
    "suggest_llmp_parameters",
]