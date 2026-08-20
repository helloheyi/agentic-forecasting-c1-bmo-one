"""Optuna-based per-quantile hyperparameter tuning for :class:`DartsLightGBMPredictor`.

:class:`~aieng.forecasting.methods.numerical.darts_regression.DartsLightGBMPredictor`
fits 13 independent quantile-regression boosters per forecast (one per level in
``_TRAINING_QUANTILES``), and can now take a ``per_quantile_kwargs`` dict giving
each of those 13 boosters its own LightGBM config. This module is what builds
that dict: it searches, via Optuna, for the per-quantile config that minimizes
mean CRPS on a trailing (already-elapsed, leakage-safe) validation window.

Two entry points:

- :func:`tune_lightgbm_quantile_config` — tune one predictor variant
  (univariate or covariate, depending on ``covariate_series_ids``).
- :func:`tune_lightgbm_configs` — tune both variants used in this project
  (univariate and covariate), with a ``separate`` flag choosing between one
  shared config reused for both, or two independently-tuned configs.

See ``docs/lightgbm-quantile-tuning-guide.md`` for the full design rationale
(why per-quantile, the interpolation trick, the no-leakage rule, cost caveats,
and worked usage examples) — this module's docstrings cover API mechanics
only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestSpec, backtest
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical.darts_regression import _TRAINING_QUANTILES, DartsLightGBMPredictor
from pydantic import BaseModel, Field


# LightGBM params whose values must be integers.
_INT_PARAMS: frozenset[str] = frozenset({"num_leaves", "max_depth", "min_data_in_leaf", "n_estimators"})

# Lower bound each interpolated param value is clamped to, so a large negative
# slope can never produce an invalid LightGBM config (e.g. num_leaves <= 1).
_PARAM_MINIMUMS: dict[str, float] = {
    "num_leaves": 2,
    "max_depth": 1,
    "min_data_in_leaf": 1,
    "n_estimators": 1,
    "learning_rate": 1e-4,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
}


def _tail_distance(quantile: float) -> float:
    """Distance of ``quantile`` from the median, normalized to ``[0, 1]``.

    0.0 at the median (``q=0.5``), 1.0 at the extreme tails
    (``q=0.025``/``q=0.975``, the outermost levels in ``_TRAINING_QUANTILES``).
    Symmetric by construction: ``0.025`` and ``0.975`` always get the same
    value, so a single slope coefficient moves both tails together rather
    than letting them diverge independently.
    """
    return abs(quantile - 0.5) / 0.475


def _expand_to_per_quantile(
    coefficients: dict[str, tuple[float, float]],
    quantiles: Sequence[float] = _TRAINING_QUANTILES,
) -> dict[float, dict[str, Any]]:
    """Expand ``{param: (base, slope)}`` into ``{quantile: {param: value}}``.

    Each param's value at quantile ``q`` is
    ``base + slope * _tail_distance(q)``, clamped to that param's floor in
    :data:`_PARAM_MINIMUMS`, then rounded to ``int`` for params in
    :data:`_INT_PARAMS`. This is the interpolation trick that keeps the
    search space at ``2 * len(coefficients)`` dimensions instead of
    ``len(quantiles) * len(coefficients)`` raw per-quantile values.
    """
    per_quantile: dict[float, dict[str, Any]] = {q: {} for q in quantiles}
    for name, (base, slope) in coefficients.items():
        floor = _PARAM_MINIMUMS.get(name, float("-inf"))
        for q in quantiles:
            value = max(base + slope * _tail_distance(q), floor)
            per_quantile[q][name] = int(round(value)) if name in _INT_PARAMS else value
    return per_quantile


@dataclass(frozen=True)
class ParamRange:
    """Optuna sampling bounds for one tunable param's ``(base, slope)`` pair.

    ``base`` is sampled from ``[base_low, base_high]`` and ``slope`` from
    ``[slope_low, slope_high]``; ``log=True`` samples ``base`` on a log
    scale (appropriate for ``learning_rate``, which spans orders of
    magnitude).
    """

    base_low: float
    base_high: float
    slope_low: float
    slope_high: float
    log: bool = False


#: Starting search-space bounds for the 7 tunable LightGBM params. These are
#: strawman defaults, not calibrated against real BAA10Y data yet — expect to
#: retune after the first real run (see docs/lightgbm-quantile-tuning-guide.md
#: section 9 for the checklist to follow when adjusting these).
_DEFAULT_PARAM_RANGES: dict[str, ParamRange] = {
    "num_leaves": ParamRange(8, 128, -64, 64),
    "max_depth": ParamRange(3, 12, -6, 6),
    "min_data_in_leaf": ParamRange(5, 100, -50, 50),
    "lambda_l1": ParamRange(0.0, 5.0, -2.0, 2.0),
    "lambda_l2": ParamRange(0.0, 5.0, -2.0, 2.0),
    "learning_rate": ParamRange(0.01, 0.3, -0.1, 0.1, log=True),
    "n_estimators": ParamRange(50, 500, -200, 200),
}


class TuningResult(BaseModel):
    """Outcome of one Optuna study tuning one LightGBM predictor variant.

    Parameters
    ----------
    predictor_variant : {"univariate", "covariate"}
        Which :class:`DartsLightGBMPredictor` feature-space variant this
        result was tuned for.
    task_id : str
        The :class:`~aieng.forecasting.evaluation.task.ForecastingTask.task_id`
        the tuning validation backtest ran against.
    coefficients : dict[str, tuple[float, float]]
        The winning trial's raw ``{param: (base, slope)}`` values.
    per_quantile_kwargs : dict[float, dict[str, Any]]
        ``coefficients`` already expanded via :func:`_expand_to_per_quantile`
        — pass this directly to ``DartsLightGBMPredictor(per_quantile_kwargs=...)``.
    best_score : float
        Mean CRPS on the validation window for the winning trial. Lower is
        better.
    n_trials : int
        Number of Optuna trials run.
    validation_start, validation_end : datetime
        The trailing validation window's bounds.
    ran_at : datetime
        UTC wall-clock time the study completed.
    """

    predictor_variant: str
    task_id: str
    coefficients: dict[str, tuple[float, float]]
    per_quantile_kwargs: dict[float, dict[str, Any]]
    best_score: float = Field(description="Mean CRPS on the validation window (lower is better).")
    n_trials: int
    validation_start: datetime
    validation_end: datetime
    ran_at: datetime


def tune_lightgbm_quantile_config(
    *,
    task: ForecastingTask,
    data_service: DataService,
    validation_end: datetime,
    validation_window: int = 60,
    covariate_series_ids: list[str] | None = None,
    lags: int = 12,
    lags_past_covariates: int | None = 12,
    base_lgbm_kwargs: dict[str, Any] | None = None,
    param_ranges: dict[str, ParamRange] | None = None,
    n_trials: int = 30,
    num_samples: int = 200,
    stride: int = 1,
    warmup: int = 0,
    seed: int | None = None,
    cutoff: datetime | None = None,
) -> TuningResult:
    """Run one Optuna study tuning per-quantile LightGBM coefficients.

    Builds a validation :class:`~aieng.forecasting.evaluation.backtest.BacktestSpec`
    over a trailing window of ``validation_window`` origins ending at
    ``validation_end``. Each trial samples ``(base, slope)`` coefficients for
    every param in ``param_ranges``, expands them to per-quantile kwargs via
    :func:`_expand_to_per_quantile`, builds a
    ``DartsLightGBMPredictor(per_quantile_kwargs=...)``, runs the existing
    :func:`~aieng.forecasting.evaluation.backtest.backtest` against the
    validation spec, and reports its ``mean_score`` (mean CRPS) to Optuna
    (``direction="minimize"``).

    No-leakage guard
    -----------------
    CRPS requires the realized outcome, which is not known at the origin
    being forecast — so tuning must never validate against origins at or
    after the live forecast this tuning run is in service of. If ``cutoff``
    is given (typically the live forecast's ``as_of``), this raises
    ``ValueError`` when ``validation_end > cutoff``. ``cutoff`` is optional
    only because standalone/exploratory tuning runs (not feeding a specific
    live prediction) have no such cutoff to check against.

    Parameters
    ----------
    task : ForecastingTask
        The forecasting task to tune against.
    data_service : DataService
        Pre-populated data service; must have the target (and, if
        ``covariate_series_ids`` is set, covariate) series registered.
    validation_end : datetime
        Last candidate validation origin (inclusive).
    validation_window : int, default=60
        Number of trailing origins (in task-frequency units) the validation
        window spans, ending at ``validation_end``.
    covariate_series_ids : list[str] or None
        ``None`` tunes the univariate variant; a non-empty list tunes the
        covariate variant using those series.
    lags, lags_past_covariates : int, int or None
        Forwarded to every trial's ``DartsLightGBMPredictor``.
    base_lgbm_kwargs : dict[str, Any] or None
        Forwarded as ``lgbm_kwargs`` to every trial's predictor — fixed
        settings (e.g. ``objective``) that are not part of the search.
    param_ranges : dict[str, ParamRange] or None
        Search-space bounds per tunable param. Defaults to
        :data:`_DEFAULT_PARAM_RANGES`.
    n_trials : int, default=30
        Number of Optuna trials.
    num_samples : int, default=200
        Monte Carlo samples per trial's predictor. Deliberately smaller than
        a production predictor's ``num_samples`` (e.g. 500) — tuning only
        needs CRPS *ranking* to be stable, not final-answer calibration
        precision, and this keeps each trial's fit cheap.
    stride, warmup : int, int
        Forwarded to the validation ``BacktestSpec``.
    seed : int or None
        Optuna sampler seed, for reproducible studies.
    cutoff : datetime or None
        See "No-leakage guard" above.

    Returns
    -------
    TuningResult
        The winning trial's config and score.

    Raises
    ------
    ValueError
        If ``cutoff`` is given and ``validation_end > cutoff``.
    """
    if cutoff is not None and validation_end > cutoff:
        raise ValueError(
            f"validation_end ({validation_end}) must not be after cutoff ({cutoff}); "
            "tuning must validate only against already-elapsed origins."
        )

    import optuna  # noqa: PLC0415

    offset = pd.tseries.frequencies.to_offset(task.frequency)
    validation_start = (pd.Timestamp(validation_end) - offset * validation_window).to_pydatetime()
    validation_spec = BacktestSpec(task=task, start=validation_start, end=validation_end, stride=stride, warmup=warmup)
    ranges = param_ranges or _DEFAULT_PARAM_RANGES

    def _objective(trial: "optuna.Trial") -> float:
        coefficients = {
            name: (
                trial.suggest_float(f"{name}_base", r.base_low, r.base_high, log=r.log),
                trial.suggest_float(f"{name}_slope", r.slope_low, r.slope_high),
            )
            for name, r in ranges.items()
        }
        per_quantile_kwargs = _expand_to_per_quantile(coefficients)
        trial.set_user_attr("per_quantile_kwargs", per_quantile_kwargs)

        predictor = DartsLightGBMPredictor(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            covariate_series_ids=covariate_series_ids,
            num_samples=num_samples,
            lgbm_kwargs=base_lgbm_kwargs,
            per_quantile_kwargs=per_quantile_kwargs,
        )
        try:
            result = backtest(predictor=predictor, spec=validation_spec, data_service=data_service, max_retries=0)
        except Exception:  # noqa: BLE001 — a bad sampled config must not abort the whole study.
            return float("inf")
        return result.mean_score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials)

    best_coefficients = {
        name: (study.best_trial.params[f"{name}_base"], study.best_trial.params[f"{name}_slope"])
        for name in ranges
    }
    return TuningResult(
        predictor_variant="covariate" if covariate_series_ids else "univariate",
        task_id=task.task_id,
        coefficients=best_coefficients,
        per_quantile_kwargs=study.best_trial.user_attrs["per_quantile_kwargs"],
        best_score=study.best_value,
        n_trials=n_trials,
        validation_start=validation_start,
        validation_end=validation_end,
        ran_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
    )


def tune_lightgbm_configs(
    *,
    task: ForecastingTask,
    data_service: DataService,
    validation_end: datetime,
    covariate_series_ids: list[str],
    lags: int = 12,
    lags_past_covariates: int | None = 12,
    base_lgbm_kwargs: dict[str, Any] | None = None,
    separate: bool = True,
    n_trials: int = 30,
    validation_window: int = 60,
    num_samples: int = 200,
    param_ranges: dict[str, ParamRange] | None = None,
    seed: int | None = None,
    cutoff: datetime | None = None,
) -> dict[str, TuningResult]:
    """Tune per-quantile LightGBM config(s) for the univariate + covariate variants.

    This is the entry point for the shared-vs-separate tuning choice: the
    univariate and covariate :class:`DartsLightGBMPredictor` variants see
    different feature counts (``lags`` vs.
    ``lags + lags_past_covariates * len(covariate_series_ids)``), so they can
    plausibly want different hyperparameters — but tuning both independently
    costs roughly twice the Optuna trials.

    Parameters
    ----------
    separate : bool, default=True
        ``True`` runs two independent Optuna studies (one per variant) via
        two calls to :func:`tune_lightgbm_quantile_config` — recommended,
        since the feature spaces differ, and especially once a new
        covariate is added (e.g. a RAG-derived regime-change signal) that
        widens that gap further. ``False`` runs a single study against the
        univariate variant and reuses its ``per_quantile_kwargs`` for the
        covariate variant too — cheaper, at the cost of the covariate
        variant not getting a config suited to its larger feature space.
    (all other parameters are forwarded to :func:`tune_lightgbm_quantile_config`)

    Returns
    -------
    dict[str, TuningResult]
        Always has both keys, ``"univariate"`` and ``"covariate"``. Under
        ``separate=False`` both entries carry identical
        ``per_quantile_kwargs`` (only ``predictor_variant`` differs).
    """
    shared_kwargs: dict[str, Any] = {
        "task": task,
        "data_service": data_service,
        "validation_end": validation_end,
        "lags": lags,
        "lags_past_covariates": lags_past_covariates,
        "base_lgbm_kwargs": base_lgbm_kwargs,
        "param_ranges": param_ranges,
        "n_trials": n_trials,
        "validation_window": validation_window,
        "num_samples": num_samples,
        "seed": seed,
        "cutoff": cutoff,
    }
    univariate_result = tune_lightgbm_quantile_config(covariate_series_ids=None, **shared_kwargs)
    if not separate:
        return {
            "univariate": univariate_result,
            "covariate": univariate_result.model_copy(update={"predictor_variant": "covariate"}),
        }
    covariate_result = tune_lightgbm_quantile_config(covariate_series_ids=covariate_series_ids, **shared_kwargs)
    return {"univariate": univariate_result, "covariate": covariate_result}
