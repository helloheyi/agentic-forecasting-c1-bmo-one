"""BAA10Y parameter-tuning backtest runner."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from aieng.forecasting.evaluation import (
    MultiTargetBacktestSpec,
    cached_multi_backtest,
)

from BAA10Y_forecasting import (
    DEFAULT_COVARIATE_SERIES_IDS,
    build_baa10y_multivariate_service,
)
from BAA10Y_forecasting.predictors.adaptive_candidates import (
    build_adaptive_predictor,
    get_adaptive_candidate,
    list_adaptive_candidates,
)


ROOT = Path(__file__).resolve().parents[3]

SPECS_DIR = (
    ROOT
    / "implementations"
    / "BAA10Y_forecasting"
    / "specs"
)

PREDICTIONS_DIR = (
    ROOT
    / "data"
    / "predictions"
    / "baa10y_adaptive_tuning"
)

SPEC_FILES = {
    "smoke": "baa10y_smoke.yaml",
    "tune_2025": "baa10y_tune_2025.yaml",
    "validate_2025": "baa10y_validate_2025.yaml",
    "backtest_2025": "baa10y_backtest_2025.yaml",
    "stress_2020": "baa10y_stress_2020.yaml",
}

VALID_HORIZONS = {
    1,
    5,
    21,
}

VALID_METHODS = {
    "linear_regression",
    "lightgbm",
    "llmp_sampled_trajectory",
}

# HYOAS is not included yet because the current dynamic tuner only
# builds target-only and default-covariate data services.
VALID_COVARIATE_PANELS = {
    "target_only",
    "default",
}


def dynamic_candidate_id(
    *,
    method: str,
    covariate_panel: str,
    parameters: dict[str, Any],
) -> str:
    """Create a reproducible ID from dynamically supplied parameters."""

    payload = {
        "method": method,
        "covariate_panel": covariate_panel,
        "parameters": parameters,
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    parameter_hash = hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"{method}_"
        f"{covariate_panel}_"
        f"{parameter_hash}"
    )


def crps_improvement_pct(
    *,
    baseline_crps: float,
    candidate_crps: float,
) -> float | None:
    """Return positive improvement when candidate CRPS is lower."""

    if baseline_crps <= 0:
        return None

    return (
        100.0
        * (baseline_crps - candidate_crps)
        / baseline_crps
    )


class BAA10YTuner:
    """Run parameter configurations against one BAA10Y horizon."""

    def list_candidates(
        self,
        method: str | None = None,
        horizon: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the existing fixed parameter candidates."""

        return list_adaptive_candidates(
            method=method,
            horizon=horizon,
        )

    def run_trial(
        self,
        candidate_id: str,
        horizon: int,
        experiment: str = "smoke",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Run one fixed candidate from adaptive_candidates.py."""

        candidate = get_adaptive_candidate(
            candidate_id
        )

        return self._run_candidate(
            candidate_id=candidate_id,
            candidate=candidate,
            horizon=horizon,
            experiment=experiment,
            force_refresh=force_refresh,
        )

    def run_parameter_trial(
        self,
        *,
        method: str,
        horizon: int,
        covariate_panel: str,
        parameters: dict[str, Any],
        experiment: str = "smoke",
        force_refresh: bool = False,
        is_baseline: bool = False,
    ) -> dict[str, Any]:
        """Run a dynamically supplied parameter configuration."""

        if method not in VALID_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Expected one of "
                f"{sorted(VALID_METHODS)}."
            )

        if (
            covariate_panel
            not in VALID_COVARIATE_PANELS
        ):
            raise ValueError(
                "Unsupported covariate panel: "
                f"{covariate_panel}. Expected one of "
                f"{sorted(VALID_COVARIATE_PANELS)}."
            )

        resolved_parameters = copy.deepcopy(
            parameters
        )

        candidate_id = dynamic_candidate_id(
            method=method,
            covariate_panel=covariate_panel,
            parameters=resolved_parameters,
        )

        candidate = {
            "candidate_id": candidate_id,
            "method": method,
            "description": (
                "Dynamically generated parameter trial"
            ),
            "params": resolved_parameters,
            "covariate_panel": covariate_panel,
            "allowed_horizons": [horizon],
            "is_baseline": is_baseline,
        }

        return self._run_candidate(
            candidate_id=candidate_id,
            candidate=candidate,
            horizon=horizon,
            experiment=experiment,
            force_refresh=force_refresh,
        )

    def _run_candidate(
        self,
        *,
        candidate_id: str,
        candidate: dict[str, Any],
        horizon: int,
        experiment: str,
        force_refresh: bool,
    ) -> dict[str, Any]:
        """Execute one resolved candidate configuration."""

        # ---------------------------------------------------------------
        # Validate the request
        # ---------------------------------------------------------------

        if horizon not in VALID_HORIZONS:
            raise ValueError(
                "horizon must be 1, 5, or 21"
            )

        if experiment not in SPEC_FILES:
            raise ValueError(
                "Unsupported experiment: "
                f"{experiment}. Expected one of "
                f"{sorted(SPEC_FILES)}."
            )

        if (
            candidate["method"]
            not in VALID_METHODS
        ):
            raise ValueError(
                "Unsupported candidate method: "
                f"{candidate['method']}"
            )

        if (
            candidate["covariate_panel"]
            not in VALID_COVARIATE_PANELS
        ):
            raise ValueError(
                "Unsupported candidate covariate panel: "
                f"{candidate['covariate_panel']}"
            )

        if (
            horizon
            not in candidate["allowed_horizons"]
        ):
            raise ValueError(
                f"{candidate_id} does not support "
                f"horizon {horizon}"
            )

        # LLMP cannot be run on the pre-cutoff 2020 period.
        if (
            candidate["method"]
            == "llmp_sampled_trajectory"
            and experiment == "stress_2020"
        ):
            raise ValueError(
                "LLMP cannot be tested on "
                "stress_2020 because it is "
                "before the model cutoff."
            )

        # ---------------------------------------------------------------
        # Load the backtest specification
        # ---------------------------------------------------------------

        spec_path = (
            SPECS_DIR
            / SPEC_FILES[experiment]
        )

        with spec_path.open(
            encoding="utf-8"
        ) as file:
            full_spec = (
                MultiTargetBacktestSpec.model_validate(
                    yaml.safe_load(file)
                )
            )

        task_id = (
            f"baa10y_change_{horizon}b"
        )

        selected_tasks = [
            task
            for task in full_spec.tasks
            if task.task_id == task_id
        ]

        if not selected_tasks:
            raise ValueError(
                f"Task {task_id} not found "
                f"in {spec_path}"
            )

        single_horizon_spec = (
            full_spec.model_copy(
                update={
                    "spec_id": (
                        f"{full_spec.spec_id}"
                        f"_h{horizon}"
                    ),
                    "tasks": selected_tasks,
                }
            )
        )

        # ---------------------------------------------------------------
        # Build the target and covariate data service
        # ---------------------------------------------------------------

        use_covariates = (
            candidate["covariate_panel"]
            == "default"
        )

        requested_covariates = (
            list(
                DEFAULT_COVARIATE_SERIES_IDS
            )
            if use_covariates
            else []
        )

        data_service = (
            build_baa10y_multivariate_service(
                windows=(1, 5, 21),
                include_covariates=use_covariates,
                covariate_series_ids=(
                    requested_covariates
                    if requested_covariates
                    else None
                ),
                start="2016-01-01",
                refresh=False,
            )
        )

        registered_series = set(
            data_service.series_ids
        )

        available_covariates = [
            series_id
            for series_id
            in requested_covariates
            if series_id in registered_series
        ]

        # ---------------------------------------------------------------
        # Build the predictor
        # ---------------------------------------------------------------

        predictor = build_adaptive_predictor(
            candidate=candidate,
            covariate_series_ids=(
                available_covariates
                if available_covariates
                else None
            ),
        )

        # ---------------------------------------------------------------
        # Run or load the backtest
        # ---------------------------------------------------------------

        results = cached_multi_backtest(
            predictor=predictor,
            spec=single_horizon_spec,
            data_service=data_service,
            store_dir=(
                PREDICTIONS_DIR
                / experiment
                / f"h{horizon}"
            ),
            force_refresh=force_refresh,
        )

        result = results.get(
            task_id
        )

        if result is None:
            raise RuntimeError(
                "No backtest result returned "
                f"for {task_id}"
            )

        # ---------------------------------------------------------------
        # Return a JSON-compatible result
        # ---------------------------------------------------------------

        return {
            "candidate_id": candidate_id,
            "predictor_id": (
                predictor.predictor_id
            ),
            "method": candidate["method"],
            "horizon": horizon,
            "experiment": experiment,
            "parameters": copy.deepcopy(
                candidate["params"]
            ),
            "covariate_panel": candidate[
                "covariate_panel"
            ],
            "is_baseline": candidate[
                "is_baseline"
            ],
            "mean_crps": float(
                result.mean_score
            ),
            "n_predictions": len(
                result.predictions
            ),
            "skipped_origins": int(
                result.skipped_origins
            ),
            "ran_at": (
                result.ran_at.isoformat()
            ),
        }


__all__ = [
    "BAA10YTuner",
    "crps_improvement_pct",
    "dynamic_candidate_id",
]