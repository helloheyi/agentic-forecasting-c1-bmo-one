""" BAA10Y parameter-tuning backtest runner."""

from __future__ import annotations

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
    list_adaptive_candidates,
    build_adaptive_predictor,
    get_adaptive_candidate,
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
    "backtest_2025": "baa10y_backtest_2025.yaml",
    "stress_2020": "baa10y_stress_2020.yaml",
}

VALID_HORIZONS = {1, 5, 21}

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
    """Run one parameter candidate against one BAA10Y horizon."""

    def list_candidates(
        self,
        method: str | None = None,
        horizon: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return available parameter candidates."""

        # Imported lazily because adaptive_candidates.py
        # will be created later.

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
        """Run one tuning candidate and return its backtest result."""

        if horizon not in VALID_HORIZONS:
            raise ValueError(
                "horizon must be 1, 5, or 21"
            )

        if experiment not in SPEC_FILES:
            raise ValueError(
                "experiment must be smoke, "
                "backtest_2025, or stress_2020"
            )


        candidate = get_adaptive_candidate(
            candidate_id
        )

        if horizon not in candidate["allowed_horizons"]:
            raise ValueError(
                f"{candidate_id} does not support "
                f"horizon {horizon}"
            )

        # Do not use LLMP on the pre-cutoff 2020 period.
        if (
            candidate["method"]
            == "llmp_sampled_trajectory"
            and experiment == "stress_2020"
        ):
            raise ValueError(
                "LLMP cannot be tested on stress_2020 "
                "because it is before the model cutoff."
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

        task_id = f"baa10y_change_{horizon}b"

        selected_tasks = [
            task
            for task in full_spec.tasks
            if task.task_id == task_id
        ]

        if not selected_tasks:
            raise ValueError(
                f"Task {task_id} not found in {spec_path}"
            )

        single_horizon_spec = full_spec.model_copy(
            update={
                "spec_id": (
                    f"{full_spec.spec_id}_h{horizon}"
                ),
                "tasks": selected_tasks,
            }
        )

        # ---------------------------------------------------------------
        # Build target and covariate data
        # ---------------------------------------------------------------

        use_covariates = (
            candidate["covariate_panel"]
            == "default"
        )

        requested_covariates = (
            list(DEFAULT_COVARIATE_SERIES_IDS)
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

        registered = set(
            data_service.series_ids
        )

        available_covariates = [
            series_id
            for series_id in requested_covariates
            if series_id in registered
        ]

        # ---------------------------------------------------------------
        # Build the configured predictor
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

        result = results.get(task_id)

        if result is None:
            raise RuntimeError(
                f"No backtest result returned for {task_id}"
            )

        # Return a simple JSON-compatible dictionary.
        return {
            "candidate_id": candidate_id,
            "predictor_id": predictor.predictor_id,
            "method": candidate["method"],
            "horizon": horizon,
            "experiment": experiment,
            "parameters": candidate["params"],
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
            "ran_at": result.ran_at.isoformat(),
        }


__all__ = ["BAA10YTuner"]