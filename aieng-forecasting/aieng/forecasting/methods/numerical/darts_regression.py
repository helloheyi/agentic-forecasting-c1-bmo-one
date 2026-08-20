"""Darts regression-model predictors — LinearRegression and LightGBM.

Two concrete :class:`~aieng.forecasting.evaluation.predictor.Predictor`
subclasses, built on Darts' sklearn-style regression forecasters:

- :class:`DartsLinearRegressionPredictor` — thin wrapper around
  :class:`darts.models.LinearRegressionModel`.
- :class:`DartsLightGBMPredictor` — thin wrapper around
  :class:`darts.models.LightGBMModel`.

Both are **per-target** models — one independent fit per :class:`ForecastingTask`.
Both optionally accept a list of ``covariate_series_ids`` to use as *past*
covariates; covariates are fetched from the :class:`ForecastContext` (so the
information cutoff is enforced by the harness, not by the predictor).

Probabilistic forecasts are produced via Darts' ``likelihood="quantile"``
configuration: the model fits one quantile regression per requested level and
draws ``num_samples`` from the implied predictive distribution at predict time.
The point forecast is the sample median; quantiles at
:data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES` are read
off the sample distribution.

Multi-horizon support
---------------------
Both predictors honour ``task.horizons``.  The model is fitted once to
``n = max(task.horizons)`` and samples are extracted at each requested horizon
index from the resulting trajectory array.  This means a 12-step trajectory
costs the same as a single step in terms of fitting time — only the sample
extraction loop changes.

LightGBM notes
--------------
On macOS the LightGBM wheel requires an OpenMP runtime (``brew install libomp``).
If you see ``Library not loaded: @rpath/libomp.dylib`` when instantiating the
predictor, install ``libomp`` and retry.

Usage
-----
::

    from aieng.forecasting.methods.darts_regression import (
        DartsLinearRegressionPredictor,
        DartsLightGBMPredictor,
    )
    from aieng.forecasting.evaluation import backtest

    # Univariate (target only)
    pred = DartsLinearRegressionPredictor(lags=12)
    result = backtest(predictor=pred, spec=spec, data_service=svc)

    # With past covariates (e.g. FRED macro series)
    pred = DartsLinearRegressionPredictor(
        lags=12,
        lags_past_covariates=12,
        covariate_series_ids=[
            "fred_canada_us_exchange_rate",
            "fred_canada_10yr_bond_yield",
        ],
    )
"""

from __future__ import annotations

import functools
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


# Quantile levels Darts fits internally.  A denser grid than STANDARD_QUANTILES
# so sample-based quantile recovery at the reporting levels is stable.
_TRAINING_QUANTILES: list[float] = [
    0.025,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.975,
]


class _PerQuantileLightGBMModel:
    """Mixin providing per-quantile kwarg overrides for Darts' ``LightGBMModel``.

    Darts' ``LightGBMModel.fit()`` loops over ``self.quantiles``, calling
    ``self._create_model(**self.kwargs)`` fresh for each quantile — identical
    kwargs every call except ``alpha`` (confirmed against
    ``darts/models/forecasting/lgbm.py``). This overrides ``_create_model`` to
    merge ``per_quantile_kwargs[alpha]`` on top of the base kwargs before
    delegating, so each quantile's booster can use a different
    ``num_leaves``/``max_depth``/regularisation/``learning_rate``/``n_estimators``.

    Takes a plain ``dict[float, dict[str, Any]]`` in and passes plain kwargs
    through — this class has no notion of "tuning"; that lives entirely in
    :mod:`aieng.forecasting.methods.numerical.lgbm_quantile_tuning`, the only
    place that builds such a dict. See ``docs/lightgbm-quantile-tuning-guide.md``
    for the full design rationale.

    Declared as a plain mixin (not subclassing ``LightGBMModel`` directly) so
    it can be composed with the real class only where imported, keeping Darts
    out of this module's import graph at load time — consistent with the
    deferred-import convention used throughout this file.
    """

    def __init__(self, *args: Any, per_quantile_kwargs: dict[float, dict[str, Any]] | None = None, **kwargs: Any) -> None:
        # Set before super().__init__(): LightGBMModel.__init__ calls
        # self._create_model(**self.kwargs) synchronously, which dispatches to
        # our override below — the attribute must already exist by then.
        self._per_quantile_kwargs = per_quantile_kwargs or {}
        super().__init__(*args, **kwargs)

    def _create_model(self, **kwargs: Any) -> Any:
        overrides = self._per_quantile_kwargs.get(kwargs.get("alpha"), {})
        return super()._create_model(**{**kwargs, **overrides})


@functools.cache
def _get_per_quantile_lightgbm_model_cls() -> type:
    """Build (once) and cache ``LightGBMModel`` composed with the per-quantile mixin.

    Deferred so importing this module never triggers a Darts import; only
    calling :meth:`DartsLightGBMPredictor.predict` with ``per_quantile_kwargs``
    set does.
    """
    from darts.models import LightGBMModel  # noqa: PLC0415

    return type("_PerQuantileLightGBMModel", (_PerQuantileLightGBMModel, LightGBMModel), {})


class _DartsRegressionModel(Protocol):
    """Structural protocol for the Darts sklearn-style forecasters we support.

    Declared as a Protocol so static typing works without importing Darts at
    module load time (Darts pulls in LightGBM, which needs libomp on macOS).
    """

    def fit(self, series: Any, past_covariates: Any | None = ...) -> Any: ...
    def predict(self, n: int, num_samples: int = ...) -> Any: ...


def _to_timeseries(df: pd.DataFrame, frequency: str) -> Any:
    """Convert a ``(timestamp, value)`` DataFrame to a Darts ``TimeSeries``.

    Gaps are backfilled with ``fill_missing_dates=True`` so the regression
    models — which need regularly spaced observations — can consume the result.
    Pandas' ``B`` frequency counts US trading holidays as business days even
    though daily market series have no observation on those days, which would
    otherwise inject NaN rows that the sklearn-style fitters reject. We forward-
    fill those gaps so the model sees a contiguous series (a no-op for inputs
    with no missing days).
    """
    from darts import TimeSeries  # noqa: PLC0415
    from darts.utils.missing_values import fill_missing_values  # noqa: PLC0415

    ts = TimeSeries.from_dataframe(
        df,
        time_col="timestamp",
        value_cols="value",
        fill_missing_dates=True,
        freq=frequency,
    )
    return fill_missing_values(ts, fill="auto")


def _build_past_covariates(context: ForecastContext, series_ids: list[str], frequency: str) -> Any:
    """Build a single multivariate ``TimeSeries`` of past covariates.

    Each covariate is converted to a Darts ``TimeSeries`` and stacked into one
    multivariate series on the intersection of their time indices.  Callers
    must supply at least one covariate id.
    """
    from darts import concatenate  # noqa: PLC0415

    pieces = []
    for cov_id in series_ids:
        cov_df = context.get_series(cov_id)
        cov_ts = _to_timeseries(cov_df, frequency)
        cov_ts = cov_ts.with_columns_renamed(["value"], [cov_id])
        pieces.append(cov_ts)

    # Intersect time indices so stacking is well-defined.
    start = max(p.start_time() for p in pieces)
    end = min(p.end_time() for p in pieces)
    pieces = [p.slice(start, end) for p in pieces]
    return concatenate(pieces, axis=1)


def _compute_forecast_payload(samples: np.ndarray) -> ContinuousForecast:
    """Derive point forecast and STANDARD_QUANTILES from a 1-D sample vector."""
    point_forecast = float(np.median(samples))
    quantiles = {q: float(np.quantile(samples, q)) for q in STANDARD_QUANTILES}
    return ContinuousForecast(point_forecast=point_forecast, quantiles=quantiles)


def _fit_and_sample(
    *,
    model: _DartsRegressionModel,
    task: ForecastingTask,
    context: ForecastContext,
    covariate_series_ids: list[str] | None,
    num_samples: int,
) -> dict[int, np.ndarray]:
    """Fit a Darts regression model and return horizon-indexed sample arrays.

    Parameters
    ----------
    model :
        A Darts regression model already configured with
        ``likelihood="quantile"`` and appropriate lag parameters.
    task :
        The forecasting task; supplies ``target_series_id``, ``horizons`` and
        ``frequency``.  The model is fitted to ``n = task.horizon``
        (i.e. ``max(task.horizons)``) so every requested step is available in
        the trajectory.
    context :
        Cutoff-scoped data view.  All series returned respect
        ``context.as_of``.
    covariate_series_ids :
        Optional list of series to use as past covariates.  ``None`` is
        equivalent to an empty list (univariate fit).
    num_samples :
        Monte Carlo samples drawn from the predictive distribution.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from horizon step ``h`` → 1-D array of ``num_samples`` draws
        from the distribution at that step.  Only the steps listed in
        ``task.horizons`` are included.
    """
    target_df = context.get_series(task.target_series_id)
    target_ts = _to_timeseries(target_df, task.frequency)

    past_covariates: Any | None = None
    if covariate_series_ids:
        past_covariates = _build_past_covariates(context, covariate_series_ids, task.frequency)

    model.fit(target_ts, past_covariates=past_covariates)
    # Fit once to the outermost horizon; all steps 1..horizon are available.
    forecast_ts = model.predict(n=task.horizon, num_samples=num_samples)

    # all_values() shape: (n_steps, n_components, n_samples), 0-indexed.
    return {h: np.asarray(forecast_ts.all_values()[h - 1, 0, :]) for h in task.horizons}


def _build_predictions(
    *,
    predictor_id: str,
    task: ForecastingTask,
    context: ForecastContext,
    samples_by_horizon: dict[int, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> list[Prediction]:
    """Assemble one ``Prediction`` per horizon from per-horizon sample arrays."""
    offset = pd.tseries.frequencies.to_offset(task.frequency)
    issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    return [
        Prediction(
            predictor_id=predictor_id,
            task_id=task.task_id,
            issued_at=issued_at,
            as_of=context.as_of,
            forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
            payload=_compute_forecast_payload(samples),
            metadata=metadata or {},
        )
        for h, samples in samples_by_horizon.items()
    ]


class DartsLinearRegressionPredictor(Predictor):
    """Probabilistic predictor wrapping Darts :class:`LinearRegressionModel`.

    Fits a per-target quantile regression on lagged target values (and,
    optionally, lagged covariate values) at every forecast origin, then draws
    ``num_samples`` from the implied predictive distribution at predict time.

    Returns one :class:`~aieng.forecasting.evaluation.prediction.Prediction`
    per horizon step in ``task.horizons``.  The model is fitted once to the
    outermost horizon so the cost is the same regardless of how many horizon
    steps are requested.

    Parameters
    ----------
    lags : int
        Number of lagged target observations used as features.  Defaults to 12
        — sufficient to capture one annual cycle in monthly data.
    lags_past_covariates : int or None
        Number of lagged covariate observations used as features when
        ``covariate_series_ids`` is non-empty.  Ignored otherwise.
        Defaults to 12.
    covariate_series_ids : list[str] or None
        Series ids to fetch from the :class:`ForecastContext` and stack as
        past covariates.  ``None`` means univariate (no covariates).
    num_samples : int
        Monte Carlo samples drawn at predict time to compute quantiles.
        Defaults to 500.
    """

    def __init__(
        self,
        lags: int = 12,
        lags_past_covariates: int | None = 12,
        covariate_series_ids: list[str] | None = None,
        num_samples: int = 500,
    ) -> None:
        self._lags = lags
        self._lags_past_covariates = lags_past_covariates
        self._covariate_series_ids = list(covariate_series_ids) if covariate_series_ids else None
        self._num_samples = num_samples

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier, suffixed ``_cov`` when covariates are used."""
        suffix = "_cov" if self._covariate_series_ids else ""
        return f"darts_linreg{suffix}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Probabilistic linear-regression forecasts for each task horizon."""
        from darts.models import LinearRegressionModel  # noqa: PLC0415

        model = LinearRegressionModel(
            lags=self._lags,
            lags_past_covariates=(self._lags_past_covariates if self._covariate_series_ids else None),
            output_chunk_length=task.horizon,
            likelihood="quantile",
            quantiles=_TRAINING_QUANTILES,
        )

        samples_by_horizon = _fit_and_sample(
            model=model,
            task=task,
            context=context,
            covariate_series_ids=self._covariate_series_ids,
            num_samples=self._num_samples,
        )

        return _build_predictions(
            predictor_id=self.predictor_id,
            task=task,
            context=context,
            samples_by_horizon=samples_by_horizon,
            metadata={"covariates": self._covariate_series_ids or []},
        )


class DartsLightGBMPredictor(Predictor):
    """Probabilistic predictor wrapping Darts :class:`LightGBMModel`.

    Fits a per-target quantile-regression gradient booster on lagged target
    and covariate values.  Predicted distributions are drawn via Darts'
    Monte Carlo sampling over the fitted quantile regressors.

    Returns one :class:`~aieng.forecasting.evaluation.prediction.Prediction`
    per horizon step in ``task.horizons``.

    Parameters
    ----------
    lags : int
        Number of lagged target observations used as features.  Defaults to 12.
    lags_past_covariates : int or None
        Number of lagged covariate observations used as features when
        ``covariate_series_ids`` is non-empty.  Ignored otherwise.
        Defaults to 12.
    covariate_series_ids : list[str] or None
        Series ids to fetch from the :class:`ForecastContext` and stack as
        past covariates.  ``None`` means univariate (no covariates).
    num_samples : int
        Monte Carlo samples drawn at predict time to compute quantiles.
        Defaults to 500.
    lgbm_kwargs : dict[str, Any] or None
        Extra keyword arguments passed through to
        :class:`darts.models.LightGBMModel`.  Use this to tune tree depth,
        leaf count, regularisation, etc.  ``verbose=-1`` is always injected
        unless the caller overrides it.  Applied to every quantile's booster
        as a base config, before any ``per_quantile_kwargs`` override.
    per_quantile_kwargs : dict[float, dict[str, Any]] or None
        Optional per-quantile overrides, keyed by quantile level (must match
        entries in :data:`_TRAINING_QUANTILES`).  Each quantile's booster is
        built from ``lgbm_kwargs`` with that quantile's dict merged on top,
        letting e.g. tail quantiles (0.025/0.975) use a different
        ``num_leaves``/``min_data_in_leaf``/regularisation than the median.
        ``None`` (the default) preserves the single-shared-config behaviour
        every existing caller relies on.  Typically produced by
        :func:`aieng.forecasting.methods.numerical.lgbm_quantile_tuning.tune_lightgbm_quantile_config`
        rather than written by hand — see ``docs/lightgbm-quantile-tuning-guide.md``.

    Notes
    -----
    On macOS you must have ``libomp`` installed (``brew install libomp``) for
    LightGBM to load.  The import is deferred until :meth:`predict` so that
    users without libomp can still use other predictors in this module.
    """

    def __init__(
        self,
        lags: int = 12,
        lags_past_covariates: int | None = 12,
        covariate_series_ids: list[str] | None = None,
        num_samples: int = 500,
        lgbm_kwargs: dict[str, Any] | None = None,
        per_quantile_kwargs: dict[float, dict[str, Any]] | None = None,
    ) -> None:
        self._lags = lags
        self._lags_past_covariates = lags_past_covariates
        self._covariate_series_ids = list(covariate_series_ids) if covariate_series_ids else None
        self._num_samples = num_samples
        kwargs = dict(lgbm_kwargs or {})
        kwargs.setdefault("verbose", -1)
        self._lgbm_kwargs = kwargs
        self._per_quantile_kwargs = per_quantile_kwargs or None

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier, suffixed ``_cov``/``_tuned`` as applicable."""
        cov_suffix = "_cov" if self._covariate_series_ids else ""
        tuned_suffix = "_tuned" if self._per_quantile_kwargs else ""
        return f"darts_lightgbm{cov_suffix}{tuned_suffix}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic LightGBM forecasts for every horizon in the task."""
        from darts.models import LightGBMModel  # noqa: PLC0415

        if self._per_quantile_kwargs:
            model_cls = _get_per_quantile_lightgbm_model_cls()
            extra_kwargs: dict[str, Any] = {"per_quantile_kwargs": self._per_quantile_kwargs}
        else:
            model_cls = LightGBMModel
            extra_kwargs = {}

        model = model_cls(
            lags=self._lags,
            lags_past_covariates=(self._lags_past_covariates if self._covariate_series_ids else None),
            output_chunk_length=task.horizon,
            likelihood="quantile",
            quantiles=_TRAINING_QUANTILES,
            **self._lgbm_kwargs,
            **extra_kwargs,
        )

        samples_by_horizon = _fit_and_sample(
            model=model,
            task=task,
            context=context,
            covariate_series_ids=self._covariate_series_ids,
            num_samples=self._num_samples,
        )

        return _build_predictions(
            predictor_id=self.predictor_id,
            task=task,
            context=context,
            samples_by_horizon=samples_by_horizon,
            metadata={"covariates": self._covariate_series_ids or []},
        )
