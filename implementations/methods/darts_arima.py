"""Darts AutoARIMA predictor — probabilistic forecast via Monte Carlo sampling.

``DartsAutoARIMAPredictor`` wraps Darts ``AutoARIMA`` with two modes:

1. **Univariate** (default) — fits on the target series only.
2. **With covariates** — additionally fetches a list of covariate series from
   the forecast context and passes them to ``AutoARIMA`` as ``past_covariates``.
   Useful for exogenous indicators such as FRED economic series.

The probabilistic forecast is produced via Monte Carlo sampling (``num_samples``
draws from the predictive distribution).  Point forecast is the median;
quantiles are computed at :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`
levels.

Usage::

    from methods.darts_arima import DartsAutoARIMAPredictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    # Univariate
    predictor = DartsAutoARIMAPredictor()

    # With FRED covariates
    predictor_with_cov = DartsAutoARIMAPredictor(
        covariate_series_ids=[
            "fred_us_cpi_food_at_home",
            "fred_canada_us_exchange_rate",
        ]
    )

    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Mean CRPS: {result.mean_crps:.4f}")
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class DartsAutoARIMAPredictor(Predictor):
    """Probabilistic predictor wrapping Darts AutoARIMA.

    Fits AutoARIMA on the target series history available at the forecast
    origin, then generates a probabilistic forecast via Monte Carlo sampling.

    Optionally fetches one or more exogenous covariate series from the forecast
    context and passes them as ``past_covariates`` to the Darts model.  When
    covariates are requested but a series cannot be fetched or aligned (e.g.
    it is not registered), that covariate is silently dropped with a warning.

    Parameters
    ----------
    num_samples : int
        Number of Monte Carlo samples used to build the predictive distribution.
        Higher values give smoother quantile estimates at the cost of compute.
        Default: 500.
    covariate_series_ids : list[str] or None
        Series IDs to fetch from the forecast context and use as past
        covariates.  The covariate time grid is aligned to the target series
        using forward-fill.  If ``None`` or empty, the model is univariate.

    Notes
    -----
    - **Darts AutoARIMA** requires ``statsforecast`` (already a project
      dependency).  No additional install is needed.
    - Covariate time ranges must overlap with the target series.  If a
      covariate's available history is shorter than the target's, it is dropped
      with a warning rather than raising an error.
    - The ``predictor_id`` encodes whether covariates are in use so that results
      are clearly distinguished in :class:`~aieng.forecasting.evaluation.backtest.BacktestResult`.
    """

    def __init__(
        self,
        num_samples: int = 500,
        covariate_series_ids: list[str] | None = None,
    ) -> None:
        self._num_samples = num_samples
        self._covariate_series_ids: list[str] = covariate_series_ids or []

    @property
    def predictor_id(self) -> str:
        """Return a stable string identifier for this predictor."""
        if self._covariate_series_ids:
            return "darts_autoarima_with_covariates"
        return "darts_autoarima"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> Prediction:
        """Produce a probabilistic AutoARIMA forecast.

        Parameters
        ----------
        task : ForecastingTask
            Defines the target series, horizon, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.  All series returned respect
            ``context.as_of``.

        Returns
        -------
        Prediction
            A ``ContinuousForecast`` with ``point_forecast`` equal to the
            median of the predictive sample, and quantiles at
            :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`.
        """
        from darts import TimeSeries  # noqa: PLC0415
        from darts.models import AutoARIMA  # noqa: PLC0415  # type: ignore[import-untyped]

        series_df = context.get_series(task.target_series_id)

        ts = TimeSeries.from_dataframe(
            series_df,
            time_col="timestamp",
            value_cols="value",
            fill_missing_dates=True,
            freq=task.frequency,
        )

        past_covariates: Any | None = None
        if self._covariate_series_ids:
            past_covariates = self._build_covariates(
                series_ids=self._covariate_series_ids,
                context=context,
                target_ts=ts,
                frequency=task.frequency,
            )

        model = AutoARIMA()
        model.fit(ts, past_covariates=past_covariates)

        forecast_ts: Any = model.predict(
            n=task.horizon,
            num_samples=self._num_samples,
            past_covariates=past_covariates,
        )

        # all_values() shape: (horizon, n_components, n_samples).
        # Take the final step for the single-step-ahead horizon target.
        samples: np.ndarray = forecast_ts.all_values()[-1, 0, :]

        point_forecast = float(np.median(samples))
        quantiles = {q: float(np.quantile(samples, q)) for q in STANDARD_QUANTILES}

        forecast_date: datetime = (
            pd.Timestamp(context.as_of) + pd.tseries.frequencies.to_offset(task.frequency) * task.horizon
        ).to_pydatetime()

        payload = ContinuousForecast(point_forecast=point_forecast, quantiles=quantiles)

        return Prediction(
            predictor_id=self.predictor_id,
            task_id=task.task_id,
            issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
            as_of=context.as_of,
            forecast_date=forecast_date,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_covariates(
        series_ids: list[str],
        context: ForecastContext,
        target_ts: Any,
        frequency: str,
    ) -> Any | None:
        """Fetch and stack covariate series aligned to the target time index.

        Each covariate is reindexed to the target's time grid using
        forward-fill.  Covariates that cannot be fetched or have insufficient
        overlap are dropped with a warning.

        Parameters
        ----------
        series_ids : list[str]
            Covariate series IDs to retrieve from ``context``.
        context : ForecastContext
            Cutoff-scoped data view.
        target_ts : TimeSeries
            Target Darts TimeSeries; used to define the alignment grid.
        frequency : str
            Pandas offset alias for the shared frequency.

        Returns
        -------
        TimeSeries or None
            Stacked covariate ``TimeSeries``, or ``None`` if no covariates
            could be loaded.
        """
        from darts import TimeSeries  # noqa: PLC0415

        target_index: pd.DatetimeIndex = target_ts.time_index  # type: ignore[union-attr]

        cov_arrays: list[pd.Series] = []
        component_names: list[str] = []

        for sid in series_ids:
            try:
                df = context.get_series(sid)
            except KeyError:
                warnings.warn(f"Covariate series '{sid}' not registered; skipping.", UserWarning, stacklevel=3)
                continue

            s = (
                df.set_index("timestamp")["value"]
                .reindex(target_index, method="ffill")
            )
            if s.isna().any():
                warnings.warn(
                    f"Covariate '{sid}' has gaps after alignment; skipping.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            cov_arrays.append(s.rename(sid))
            component_names.append(sid)

        if not cov_arrays:
            return None

        cov_df = pd.concat(cov_arrays, axis=1)
        cov_df.index.freq = pd.tseries.frequencies.to_offset(frequency)

        return TimeSeries.from_dataframe(cov_df)
