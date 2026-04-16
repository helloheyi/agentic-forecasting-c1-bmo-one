"""Evaluation harness: forecasting tasks, prediction payloads, and scoring."""

from aieng.forecasting.evaluation.backtest import (
    BacktestResult,
    BacktestSpec,
    MultiTargetBacktestSpec,
    backtest,
    multi_backtest,
)
from aieng.forecasting.evaluation.eval import (
    EvalBudgetExceededError,
    EvalResult,
    EvalSpec,
    EvalTracker,
    MultiTargetEvalSpec,
    evaluate,
    multi_evaluate,
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
    "MultiTargetBacktestSpec",
    "MultiTargetEvalSpec",
    "Prediction",
    "Predictor",
    "STANDARD_QUANTILES",
    "backtest",
    "evaluate",
    "multi_backtest",
    "multi_evaluate",
]
