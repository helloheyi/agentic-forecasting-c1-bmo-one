"""Shared BAA10Y benchmark configurations.

The target-only and default-covariate configurations reproduce the
model settings used by notebook 01. The default_plus_hyoas tracks use
the same model settings with the expanded HYOAS covariate panel.

Notebook 04 uses these configurations as the fixed baselines for
adaptive parameter tuning.
"""

from copy import deepcopy


NOTEBOOK_01_BENCHMARKS = {
    (
        "linear_regression",
        "target_only",
    ): {
        "lags": 5,
        "num_samples": 100,
    },

    (
        "linear_regression",
        "default",
    ): {
        "lags": 5,
        "lags_past_covariates": 5,
        "num_samples": 100,
    },

    (
        "lightgbm",
        "target_only",
    ): {
        "lags": 5,
        "num_samples": 100,
        "lgbm_kwargs": {
            "num_threads": 1,
            "n_jobs": 1,
            "verbosity": -1,
        },
    },

    (
        "lightgbm",
        "default",
    ): {
        "lags": 5,
        "lags_past_covariates": 5,
        "num_samples": 100,
        "lgbm_kwargs": {
            "num_threads": 1,
            "n_jobs": 1,
            "verbosity": -1,
        },
    },
    (
        "lightgbm",
        "default_plus_hyoas",
    ): {
        "lags": 5,
        "lags_past_covariates": 5,
        "num_samples": 100,
        "lgbm_kwargs": {
            "num_threads": 1,
            "n_jobs": 1,
            "verbosity": -1,
        },
    },
    (
        "llmp_sampled_trajectory",
        "target_only",
    ): {
        "n_samples": 8,
        "history_window": 48,
    },

    (
        "llmp_sampled_trajectory",
        "default",
    ): {
        "n_samples": 8,
        "history_window": 48,
    },
        (
        "llmp_sampled_trajectory",
        "default_plus_hyoas",
    ): {
        "n_samples": 8,
        "history_window": 48,
    },
}


def get_benchmark_config(
    method,
    covariate_panel,
):
    """Return an independent copy of one benchmark configuration."""

    key = (
        method,
        covariate_panel,
    )

    if key not in NOTEBOOK_01_BENCHMARKS:
        available_tracks = sorted(
            "/".join(key)
            for key in NOTEBOOK_01_BENCHMARKS
        )

        raise ValueError(
            "No BAA10Y benchmark for "
            f"{method}/{covariate_panel}. "
            f"Available tracks: {available_tracks}"
        )

    config = deepcopy(
        NOTEBOOK_01_BENCHMARKS[key]
    )

    if method == "lightgbm":
        config["lgbm_kwargs"][
            "random_state"
        ] = 42

    return config


__all__ = [
    "NOTEBOOK_01_BENCHMARKS",
    "get_benchmark_config",
]