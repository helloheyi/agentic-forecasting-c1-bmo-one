"""Evaluation harness: forecasting tasks, prediction payloads, and scoring."""

from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec, backtest
from aieng.forecasting.evaluation.eval import (
    EvalBudgetExceededError,
    EvalResult,
    EvalSpec,
    EvalTracker,
    evaluate,
)
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


__all__ = [
    "BacktestResult",
    "BacktestSpec",
    "ContinuousForecast",
    "EvalBudgetExceededError",
    "EvalResult",
    "EvalSpec",
    "EvalTracker",
    "ForecastingTask",
    "Prediction",
    "Predictor",
    "STANDARD_QUANTILES",
    "backtest",
    "evaluate",
]
