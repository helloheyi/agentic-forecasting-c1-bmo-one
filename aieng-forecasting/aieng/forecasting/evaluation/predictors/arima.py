"""ARIMAPredictor — AutoARIMA baseline using the Darts library."""

from __future__ import annotations

from datetime import datetime

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


class ARIMAPredictor(Predictor):
    """Probabilistic predictor using Darts AutoARIMA.

    Fits a univariate ``AutoARIMA`` model on the target series history
    available as of the forecast origin, then generates a probabilistic
    forecast at the task horizon via Monte Carlo sampling.

    This is intended as a reference baseline — a reasonable statistical model
    that participants can compare against and build on.

    Parameters
    ----------
    num_samples : int
        Number of Monte Carlo samples to draw for the probabilistic forecast.
        More samples give a better-calibrated quantile estimate at the cost of
        slower prediction. Default is 500.
    predictor_id : str, optional
        Override the default identifier ``"arima_auto"``.

    Notes
    -----
    AutoARIMA automatically selects the ARIMA order (p, d, q) for each
    training window using AIC. This means the model order may vary between
    backtest origins, which is realistic but makes the predictor slower than
    a fixed-order ARIMA. For faster backtests at a slight accuracy cost, use a
    fixed-order ``ARIMA(p, d, q)`` from Darts instead.

    Gap-filling: Darts requires a regular time series. The CPI series from
    StatCan is already monthly and gap-free in practice, but this predictor
    forward-fills any missing observations before fitting.

    Examples
    --------
    >>> predictor = ARIMAPredictor(num_samples=200)
    >>> prediction = predictor.predict(task, context)
    >>> prediction.payload.point_forecast
    162.3
    """

    def __init__(self, num_samples: int = 500, predictor_id: str = "arima_auto") -> None:
        self._num_samples = num_samples
        self._predictor_id = predictor_id

    @property
    def predictor_id(self) -> str:
        """Return the predictor identifier."""
        return self._predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> Prediction:
        """Fit AutoARIMA on available history and return a probabilistic forecast.

        Parameters
        ----------
        task : ForecastingTask
            Must use ``frequency="MS"`` (month-start) or another frequency
            compatible with Darts ``TimeSeries``. ``horizon`` determines how
            many steps ahead to forecast; the prediction target is the
            observation at step ``task.horizon``.
        context : ForecastContext
            Data context scoped to the forecast origin. The target series is
            retrieved and converted to a Darts ``TimeSeries``.

        Returns
        -------
        Prediction
            Probabilistic forecast at ``as_of + horizon`` with quantiles at
            :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`.
        """
        # Lazy import: darts is heavy; only load when predicting.
        from darts import TimeSeries  # noqa: PLC0415
        from darts.models import AutoARIMA  # noqa: PLC0415

        series_df = context.get_series(task.target_series_id)

        # Convert to Darts TimeSeries — forward-fill to ensure regularity.
        ts = TimeSeries.from_dataframe(
            series_df,
            time_col="timestamp",
            value_cols="value",
            fill_missing_dates=True,
            freq=task.frequency,
        )

        model = AutoARIMA()
        model.fit(ts)

        # Generate probabilistic forecast via Monte Carlo sampling.
        forecast_ts = model.predict(n=task.horizon, num_samples=self._num_samples)

        # Extract samples at the final step (the horizon target).
        # forecast_ts.all_values() has shape (horizon, n_components, n_samples).
        samples: np.ndarray = forecast_ts.all_values()[-1, 0, :]

        point_forecast = float(np.median(samples))
        quantiles = {q: float(np.quantile(samples, q)) for q in STANDARD_QUANTILES}

        # Compute the forecast date: origin + horizon steps at task frequency.
        forecast_date_ts = pd.Timestamp(context.as_of) + pd.tseries.frequencies.to_offset(task.frequency) * task.horizon
        forecast_date: datetime = forecast_date_ts.to_pydatetime()

        payload = ContinuousForecast(point_forecast=point_forecast, quantiles=quantiles)

        return Prediction(
            predictor_id=self.predictor_id,
            task_id=task.task_id,
            issued_at=datetime.utcnow(),
            as_of=context.as_of,
            forecast_date=forecast_date,
            payload=payload,
        )
