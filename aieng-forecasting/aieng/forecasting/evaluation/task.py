"""ForecastingTask: defines a prediction problem against the data service."""

from pydantic import BaseModel, Field


class ForecastingTask(BaseModel):
    """Defines a prediction problem, independent of how it is solved.

    A ``ForecastingTask`` specifies *what* to forecast: the target series,
    the horizon, the temporal resolution, and how to determine ground truth.
    It says nothing about *how* a predictor should solve the problem —
    covariate selection, gap-filling, and model choice are all predictor
    concerns.

    This separation means any two predictors (a vanilla ARIMA and a
    multi-step LLM agent) can be evaluated against the same task without
    the task needing to know anything about either of them.

    Parameters
    ----------
    task_id : str
        Unique identifier for this forecasting task.
    target_series_id : str
        The ``series_id`` (key in ``SeriesStore``) of the series to forecast.
    horizon : int
        Number of steps ahead to forecast.
    frequency : str
        Pandas offset alias for the forecast frequency (e.g. ``"MS"`` for
        month-start, ``"h"`` for hourly, ``"D"`` for daily). Combined with
        ``horizon``, this determines the forecast window.
    description : str
        Human-readable description of the prediction problem.
    resolution_fn : str
        How ground truth is determined. Defaults to
        ``"observed_value_at_resolution_timestamp"``, meaning the resolution
        is the actual observed value of ``target_series_id`` at the target
        timestamp.

        .. note::
            **This field is currently a placeholder.** The evaluation harness
            (``backtest()`` and ``evaluate()``) always uses the
            ``"observed_value_at_resolution_timestamp"`` strategy regardless
            of this value. Dispatch on ``resolution_fn`` is deferred; the
            field is defined now so that task specs and result records carry
            the intent, and so that alternative strategies (e.g. for derived
            quantities or aggregated targets) can be added without a breaking
            change to the model.

    Notes
    -----
    The evaluation loop is identical for backtesting and live forecasting:

    .. code-block:: text

        ForecastingTask  →  defines the question
        Predictor        →  decides how to answer it
        Prediction       →  the answer
        Resolution       →  ground truth
        Score            →  how well the answer matched

    In backtest mode, the harness iterates over historical forecast origins.
    In live mode, it waits for the resolution date. The task definition does
    not change between modes.

    Examples
    --------
    >>> task = ForecastingTask(
    ...     task_id="cpi_all_items_1m_ahead",
    ...     target_series_id="cpi_all_items_canada",
    ...     horizon=1,
    ...     frequency="MS",
    ...     description="Forecast Canada All-items CPI one month ahead.",
    ... )
    """

    task_id: str = Field(description="Unique identifier for this forecasting task.")
    target_series_id: str = Field(description="The series_id (key in SeriesStore) of the series to forecast.")
    horizon: int = Field(ge=1, description="Number of steps ahead to forecast.")
    frequency: str = Field(description="Pandas offset alias for the forecast frequency, e.g. 'MS', 'h', 'D'.")
    description: str = Field(description="Human-readable description of the prediction problem.")
    resolution_fn: str = Field(
        default="observed_value_at_resolution_timestamp",
        description=(
            "How ground truth is determined. Placeholder — harness currently always uses "
            "'observed_value_at_resolution_timestamp' regardless of this value. "
            "Dispatch on alternative strategies is deferred."
        ),
    )
