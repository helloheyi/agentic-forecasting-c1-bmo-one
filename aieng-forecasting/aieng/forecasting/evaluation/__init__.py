"""Evaluation harness: forecasting tasks, prediction payloads, and scoring."""

from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec, backtest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.predictors import ARIMAPredictor
from aieng.forecasting.evaluation.task import ForecastingTask


__all__ = [
    "ARIMAPredictor",
    "BacktestResult",
    "BacktestSpec",
    "ContinuousForecast",
    "ForecastingTask",
    "Prediction",
    "Predictor",
    "STANDARD_QUANTILES",
    "backtest",
]
