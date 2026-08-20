"""Numerical forecasting predictor implementations.

These predictors wrap classical or machine-learning time-series models behind
the shared :class:`~aieng.forecasting.evaluation.predictor.Predictor`
interface.
"""

from .darts_arima import DartsAutoARIMAPredictor
from .darts_classical import DartsExponentialSmoothingPredictor, DartsKalmanForecasterPredictor
from .darts_regression import DartsLightGBMPredictor, DartsLinearRegressionPredictor
from .lgbm_quantile_tuning import ParamRange, TuningResult, tune_lightgbm_configs, tune_lightgbm_quantile_config


__all__ = [
    "DartsAutoARIMAPredictor",
    "DartsExponentialSmoothingPredictor",
    "DartsKalmanForecasterPredictor",
    "DartsLightGBMPredictor",
    "DartsLinearRegressionPredictor",
    "ParamRange",
    "TuningResult",
    "tune_lightgbm_configs",
    "tune_lightgbm_quantile_config",
]
