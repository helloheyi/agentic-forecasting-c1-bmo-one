"""Predictor template — your starting point for writing a custom predictor.

Copy this file, rename the class, and replace the forecast logic inside
``predict()`` with your own model or approach. Everything else (the
interface, the Prediction construction, the forecast_date arithmetic) can
stay exactly as it is.

The ``Predictor`` ABC requires two things:
  - a ``predictor_id`` property that returns a unique string identifier
  - a ``predict(task, context)`` method that returns a ``Prediction``

Once your class implements those two things, it plugs directly into
``backtest()`` and ``evaluate()`` with no other changes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class LastValuePredictor(Predictor):
    """Naive baseline: forecast the most recently observed value at all quantiles.

    This is the simplest possible predictor — it predicts that the next
    observation will equal the last known value, with no uncertainty spread.
    Use it to establish a floor for CRPS before trying real models.

    It also serves as an annotated skeleton. Follow the comments to
    understand each required step, then replace the forecast logic with
    your own approach.
    """

    # ------------------------------------------------------------------
    # Step 1: give your predictor a stable string ID.
    # This appears in BacktestResult and every Prediction record,
    # so changing it mid-experiment will break comparisons.
    # ------------------------------------------------------------------
    @property
    def predictor_id(self) -> str:
        return "last_value_naive"

    # ------------------------------------------------------------------
    # Step 2: implement predict().
    #
    # Arguments:
    #   task    — ForecastingTask: defines the problem (target series,
    #             horizon, frequency). Read-only; do not modify it.
    #   context — ForecastContext: your data access object. All series
    #             returned by context.get_series() are already filtered
    #             to context.as_of — you cannot accidentally access
    #             future data.
    #
    # Return:
    #   A fully constructed Prediction object (see below).
    # ------------------------------------------------------------------
    def predict(self, task: ForecastingTask, context: ForecastContext) -> Prediction:
        # ------------------------------------------------------------------
        # Step 3: fetch the target series.
        # Returns a DataFrame with columns: timestamp, value, released_at.
        # Rows are already cut off at context.as_of.
        # ------------------------------------------------------------------
        series_df = context.get_series(task.target_series_id)

        # ------------------------------------------------------------------
        # Step 4: produce a forecast.
        # Replace everything below with your model logic.
        # Here we just take the last observed value as the point forecast.
        # ------------------------------------------------------------------
        last_value = float(series_df["value"].iloc[-1])

        # ------------------------------------------------------------------
        # Step 5: build the ContinuousForecast payload.
        # point_forecast: your central estimate (typically median).
        # quantiles: a dict mapping quantile level → forecast value.
        #   STANDARD_QUANTILES = [0.05, 0.10, ..., 0.90, 0.95]
        #   The evaluation engine uses these to compute CRPS.
        #   A naive predictor with no uncertainty puts the same value
        #   at every quantile — real models spread them out.
        # ------------------------------------------------------------------
        payload = ContinuousForecast(
            point_forecast=last_value,
            quantiles=dict.fromkeys(STANDARD_QUANTILES, last_value),
        )

        # ------------------------------------------------------------------
        # Step 6: compute the forecast date.
        # This is the future timestamp being predicted:
        #   context.as_of + task.horizon steps at task.frequency.
        # The harness uses this to look up the ground-truth observation
        # when scoring.
        # ------------------------------------------------------------------
        forecast_date: datetime = (
            pd.Timestamp(context.as_of) + pd.tseries.frequencies.to_offset(task.frequency) * task.horizon
        ).to_pydatetime()

        # ------------------------------------------------------------------
        # Step 7: wrap everything in a Prediction and return it.
        # All fields except metadata are required.
        # Use metadata to attach side-channel data (model stats, sources,
        # trace IDs, etc.) — the harness ignores it but passes it through.
        # ------------------------------------------------------------------
        return Prediction(
            predictor_id=self.predictor_id,
            task_id=task.task_id,
            issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
            as_of=context.as_of,
            forecast_date=forecast_date,
            payload=payload,
        )
