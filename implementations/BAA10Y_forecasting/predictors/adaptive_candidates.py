"""Parameter candidates for BAA10Y adaptive tuning."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from aieng.forecasting.methods import (
    DartsLightGBMPredictor,
    DartsLinearRegressionPredictor,
)

from BAA10Y_forecasting.predictors.llmp_sampled_trajectory import (
    build_baa10y_llmp_sampled_trajectory,
)


# ---------------------------------------------------------------------------
# Candidate configurations
# ---------------------------------------------------------------------------

ADAPTIVE_CANDIDATES: list[dict[str, Any]] = [
    # ===================================================================
    # Linear Regression
    # ===================================================================
    {
        "candidate_id": "linreg_l5_target",
        "method": "linear_regression",
        "description": "Baseline: 5 target lags",
        "params": {
            "lags": 5,
            "num_samples": 100,
        },
        "covariate_panel": "target_only",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": True,
    },
    {
        "candidate_id": "linreg_l10_target",
        "method": "linear_regression",
        "description": "10 target lags",
        "params": {
            "lags": 10,
            "num_samples": 100,
        },
        "covariate_panel": "target_only",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": False,
    },
    {
        "candidate_id": "linreg_l5_cov",
        "method": "linear_regression",
        "description": "5 target and covariate lags",
        "params": {
            "lags": 5,
            "lags_past_covariates": 5,
            "num_samples": 100,
        },
        "covariate_panel": "default",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": False,
    },

    # ===================================================================
    # LightGBM
    # ===================================================================
    {
        "candidate_id": "lightgbm_l5_target",
        "method": "lightgbm",
        "description": "Baseline: 5 target lags",
        "params": {
            "lags": 5,
            "num_samples": 100,
            "lgbm_kwargs": {
                "num_threads": 1,
                "n_jobs": 1,
                "random_state": 42,
            },
        },
        "covariate_panel": "target_only",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": True,
    },
    {
        "candidate_id": "lightgbm_l5_cov",
        "method": "lightgbm",
        "description": "5 target and covariate lags",
        "params": {
            "lags": 5,
            "lags_past_covariates": 5,
            "num_samples": 100,
            "lgbm_kwargs": {
                "num_threads": 1,
                "n_jobs": 1,
                "random_state": 42,
            },
        },
        "covariate_panel": "default",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": False,
    },
    {
        "candidate_id": "lightgbm_l10_cov_tuned",
        "method": "lightgbm",
        "description": "Tuned LightGBM with 10 lags and covariates",
        "params": {
            "lags": 10,
            "lags_past_covariates": 10,
            "num_samples": 100,
            "lgbm_kwargs": {
                "num_threads": 1,
                "n_jobs": 1,
                "random_state": 42,
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 4,
                "num_leaves": 15,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            },
        },
        "covariate_panel": "default",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": False,
    },

    # ===================================================================
    # Sampled-Trajectory LLMP
    # ===================================================================
    {
        "candidate_id": "llmp_n8_h48_target",
        "method": "llmp_sampled_trajectory",
        "description": "Baseline: 8 samples and 48-day history",
        "params": {
            "n_samples": 8,
            "history_window": 48,
        },
        "covariate_panel": "target_only",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": True,
    },
    {
        "candidate_id": "llmp_n12_h48_target",
        "method": "llmp_sampled_trajectory",
        "description": "Increase trajectory samples from 8 to 12",
        "params": {
            "n_samples": 12,
            "history_window": 48,
        },
        "covariate_panel": "target_only",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": False,
    },
    {
        "candidate_id": "llmp_n8_h64_target",
        "method": "llmp_sampled_trajectory",
        "description": "Increase history window from 48 to 64",
        "params": {
            "n_samples": 8,
            "history_window": 64,
        },
        "covariate_panel": "target_only",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": False,
    },
    {
        "candidate_id": "llmp_n8_h48_cov",
        "method": "llmp_sampled_trajectory",
        "description": "Baseline LLMP parameters with covariates",
        "params": {
            "n_samples": 8,
            "history_window": 48,
        },
        "covariate_panel": "default",
        "allowed_horizons": [1, 5, 21],
        "is_baseline": False,
    },
]


# ---------------------------------------------------------------------------
# Candidate access
# ---------------------------------------------------------------------------

def list_adaptive_candidates(
    method: str | None = None,
    horizon: int | None = None,
) -> list[dict[str, Any]]:
    """List candidates, optionally filtered by method and horizon."""

    candidates = ADAPTIVE_CANDIDATES

    if method is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["method"] == method
        ]

    if horizon is not None:
        candidates = [
            candidate
            for candidate in candidates
            if horizon
            in candidate["allowed_horizons"]
        ]

    return copy.deepcopy(candidates)


def get_adaptive_candidate(
    candidate_id: str,
) -> dict[str, Any]:
    """Return one candidate by ID."""

    for candidate in ADAPTIVE_CANDIDATES:
        if (
            candidate["candidate_id"]
            == candidate_id
        ):
            return copy.deepcopy(candidate)

    raise ValueError(
        f"Unknown candidate_id: {candidate_id}"
    )


# ---------------------------------------------------------------------------
# Unique predictor IDs
# ---------------------------------------------------------------------------

def _candidate_tag(
    candidate: dict[str, Any],
    covariate_series_ids: list[str] | None,
) -> str:
    """Create a cache-safe fingerprint for the configuration."""

    payload = {
        "candidate_id": candidate[
            "candidate_id"
        ],
        "params": candidate["params"],
        "covariates": (
            covariate_series_ids or []
        ),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()[:10]


class _AdaptiveLinearRegressionPredictor(
    DartsLinearRegressionPredictor
):
    def __init__(
        self,
        *,
        trial_tag: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._trial_tag = trial_tag

    @property
    def predictor_id(self) -> str:
        return (
            f"baa10y_linreg_"
            f"{self._trial_tag}"
        )


class _AdaptiveLightGBMPredictor(
    DartsLightGBMPredictor
):
    def __init__(
        self,
        *,
        trial_tag: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._trial_tag = trial_tag

    @property
    def predictor_id(self) -> str:
        return (
            f"baa10y_lightgbm_"
            f"{self._trial_tag}"
        )


# ---------------------------------------------------------------------------
# Predictor construction
# ---------------------------------------------------------------------------

def build_adaptive_predictor(
    *,
    candidate: dict[str, Any],
    covariate_series_ids: list[str] | None,
):
    """Build the predictor represented by one candidate."""

    if (
        candidate["covariate_panel"]
        == "target_only"
    ):
        resolved_covariates = None
    else:
        resolved_covariates = (
            covariate_series_ids
        )

    tag = _candidate_tag(
        candidate,
        resolved_covariates,
    )

    params = copy.deepcopy(
        candidate["params"]
    )

    method = candidate["method"]

    if method == "linear_regression":
        return _AdaptiveLinearRegressionPredictor(
            trial_tag=tag,
            covariate_series_ids=(
                resolved_covariates
            ),
            **params,
        )

    if method == "lightgbm":
        return _AdaptiveLightGBMPredictor(
            trial_tag=tag,
            covariate_series_ids=(
                resolved_covariates
            ),
            **params,
        )

    if method == "llmp_sampled_trajectory":
        return build_baa10y_llmp_sampled_trajectory(
            covariate_series_ids=(
                resolved_covariates
            ),
            variant_tag=(
                f"baa10y_adaptive_{tag}"
            ),
            **params,
        )

    raise ValueError(
        f"Unsupported method: {method}"
    )


__all__ = [
    "ADAPTIVE_CANDIDATES",
    "build_adaptive_predictor",
    "get_adaptive_candidate",
    "list_adaptive_candidates",
]