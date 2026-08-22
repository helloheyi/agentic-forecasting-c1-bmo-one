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
        max_trials: int = 12,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Run or resume one adaptive Optuna smoke search.

        This first implementation supports LightGBM only. Smoke results
        screen candidates but never authorize promotion.
        """

        if method != "lightgbm":
            raise ValueError(
                "The first Optuna implementation "
                "supports method='lightgbm' only."
            )

        if horizon not in {
            1,
            5,
            21,
        }:
            raise ValueError(
                "horizon must be 1, 5, or 21"
            )

        if covariate_panel not in {
            "target_only",
            "default",
        }:
            raise ValueError(
                "covariate_panel must be "
                "'target_only' or 'default'"
            )

        if not 1 <= max_trials <= 50:
            raise ValueError(
                "max_trials must be between "
                "1 and 50"
            )

        experiment = "tune_2025"

        study_name = self._study_name(
            method=method,
            horizon=horizon,
            covariate_panel=covariate_panel,
            experiment=experiment,
        )

        # ---------------------------------------------------------------
        # Run and save the notebook 01 benchmark
        # ---------------------------------------------------------------

        baseline_parameters = (
            get_benchmark_config(
                method,
                covariate_panel,
            )
        )

        baseline_result = (
            self.tuner.run_parameter_trial(
                method=method,
                horizon=horizon,
                covariate_panel=(
                    covariate_panel
                ),
                parameters=(
                    baseline_parameters
                ),
                experiment=experiment,
                force_refresh=force_refresh,
                is_baseline=True,
            )
        )

        baseline_result.update({
            "study_name": study_name,
            "trial_number": -1,
            "parameter_hash": (
                baseline_result[
                    "candidate_id"
                ].rsplit(
                    "_",
                    1,
                )[-1]
            ),
            "search_strategy": (
                "optuna_tpe"
            ),
        })

        baseline_trial, _ = (
            self.store.add_trial(
                baseline_result
            )
        )

        # ---------------------------------------------------------------
        # Create or resume the Optuna study
        # ---------------------------------------------------------------

        sampler = (
            optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=5,
            )
        )

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            storage=self.storage_url,
            load_if_exists=True,
        )

        state = self.store.load()

        study_record = state.get_study(
            study_name
        )

        if study_record is None:
            study_record = SearchStudyRecord(
                study_name=study_name,
                method=method,
                horizon=horizon,
                covariate_panel=(
                    covariate_panel
                ),
                search_strategy=(
                    "optuna_tpe"
                ),
                objective_experiment=(
                    experiment
                ),
                validation_experiment=(
                    "validate_2025"
                ),
                max_trials=max_trials,
                status="created",
            )

        study_record.status = "running"
        study_record.max_trials = (
            max_trials
        )
        study_record.baseline_candidate_id = (
            baseline_trial.candidate_id
        )
        study_record.baseline_mean_crps = (
            baseline_trial.mean_crps
        )

        self.store.upsert_study(
            study_record
        )

        # ---------------------------------------------------------------
        # Optuna objective
        # ---------------------------------------------------------------

        def objective(
            trial: optuna.Trial,
        ) -> float:
            parameters = (
                suggest_lightgbm_parameters(
                    trial,
                    covariate_panel=(
                        covariate_panel
                    ),
                )
            )

            result = (
                self.tuner.run_parameter_trial(
                    method=method,
                    horizon=horizon,
                    covariate_panel=(
                        covariate_panel
                    ),
                    parameters=parameters,
                    experiment=experiment,
                    force_refresh=(
                        force_refresh
                    ),
                    is_baseline=False,
                )
            )

            result.update({
                "study_name": study_name,
                "trial_number": (
                    trial.number
                ),
                "parameter_hash": (
                    result[
                        "candidate_id"
                    ].rsplit(
                        "_",
                        1,
                    )[-1]
                ),
                "search_strategy": (
                    "optuna_tpe"
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
                "mean_crps",
                saved_trial.mean_crps,
            )

            return float(
                saved_trial.mean_crps
            )

        # Run only the remaining number of trials when resuming.
        completed_before = sum(
            trial.state
            == TrialState.COMPLETE
            for trial in study.trials
        )

        remaining_trials = max(
            0,
            max_trials
            - completed_before,
        )

        if remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=remaining_trials,
            )

        # ---------------------------------------------------------------
        # Collect the best completed trial
        # ---------------------------------------------------------------

        completed_trials = [
            trial
            for trial in study.trials
            if (
                trial.state
                == TrialState.COMPLETE
            )
        ]

        if not completed_trials:
            study_record.status = "failed"
            self.store.upsert_study(
                study_record
            )

            raise RuntimeError(
                "Optuna did not complete "
                "any candidate trials."
            )

        best_optuna_trial = min(
            completed_trials,
            key=lambda trial: float(
                trial.value
            ),
        )

        best_candidate_id = (
            best_optuna_trial.user_attrs.get(
                "candidate_id"
            )
        )

        if not best_candidate_id:
            raise RuntimeError(
                "The best Optuna trial does "
                "not contain a candidate ID."
            )

        updated_state = self.store.load()

        best_records = (
            updated_state.find_trials(
                method=method,
                horizon=horizon,
                experiment=experiment,
                candidate_id=(
                    best_candidate_id
                ),
                covariate_panel=(
                    covariate_panel
                ),
                study_name=study_name,
            )
        )

        if not best_records:
            raise RuntimeError(
                "The best Optuna result was "
                "not found in tuning state."
            )

        best_trial = best_records[-1]

        improvement_pct = (
            100.0
            * (
                baseline_trial.mean_crps
                - best_trial.mean_crps
            )
            / baseline_trial.mean_crps
        )

        if improvement_pct > 0:
            decision = (
                "advance_best_candidate_"
                "to_validate_2025"
            )
        else:
            decision = (
                "retain_baseline_after_"
                "tuning"
            )

        # ---------------------------------------------------------------
        # Save the completed study summary
        # ---------------------------------------------------------------

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

        # ---------------------------------------------------------------
        # Return the search history
        # ---------------------------------------------------------------

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
                "mean_crps": (
                    trial.mean_crps
                ),
                "parameters": (
                    trial.parameters
                ),
            }
            for trial in sorted(
                history,
                key=lambda item: (
                    item.trial_number
                    if item.trial_number
                    is not None
                    else 999999
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
                "optuna_tpe"
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
            "promotion_allowed": False,
            "promotion_block_reason": (
                 "Tune-period evidence cannot "
                 "promote a model. Confirm the "
                "candidate on backtest_2025."
            ),
            "trial_history": history_rows,
        }


__all__ = [
    "BAA10YAdaptiveOptimizer",
    "suggest_lightgbm_parameters",
]