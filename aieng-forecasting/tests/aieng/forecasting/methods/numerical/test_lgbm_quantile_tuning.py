"""Tests for ``aieng.forecasting.methods.numerical.lgbm_quantile_tuning``.

Pure-function tests cover the interpolation math and the
``_PerQuantileLightGBMModel`` override directly (no Optuna study, no Darts
import required). The shared-vs-separate orchestration is tested with
``tune_lightgbm_quantile_config`` mocked out. One smoke test runs a real,
tiny (``n_trials=1``) end-to-end Optuna study against synthetic data — the
only place the expensive path actually executes.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical.darts_regression import (
    _TRAINING_QUANTILES,
    DartsLightGBMPredictor,
    _PerQuantileLightGBMModel,
)
from aieng.forecasting.methods.numerical.lgbm_quantile_tuning import (
    TuningResult,
    _expand_to_per_quantile,
    _resolve_lgbm_kwargs,
    _tail_distance,
    tune_lightgbm_configs,
    tune_lightgbm_quantile_config,
)


AS_OF = datetime(2020, 12, 1)


class _InMemoryAdapter(BaseAdapter):
    """Adapter that returns a supplied DataFrame unchanged."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        """Return the supplied DataFrame."""
        return self._df.copy()


def _synthetic_series(seed: int, amplitude: float = 10.0) -> pd.DataFrame:
    """Build a 240-month trend+seasonal+noise series (deterministic via seed)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    t = np.arange(240, dtype=float)
    values = 100.0 + 0.5 * t + amplitude * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.0, 240)
    return pd.DataFrame({"timestamp": dates, "value": values})


@pytest.fixture
def svc() -> DataService:
    """Build a DataService with one target series."""
    service = DataService()
    service.register(
        "target",
        _InMemoryAdapter(_synthetic_series(seed=1)),
        SeriesMetadata(series_id="target", description="Synthetic target", source="test", units="index", frequency="MS"),
    )
    return service


@pytest.fixture
def task() -> ForecastingTask:
    """Build a 1-month horizon task against the synthetic target."""
    return ForecastingTask(
        task_id="synthetic_1m",
        target_series_id="target",
        horizons=[1],
        frequency="MS",
        description="Synthetic 1-month forecast for tuning tests.",
    )


def _canned_result(variant: str) -> TuningResult:
    """Build a minimal, valid TuningResult for mocking the single-variant tuner."""
    return TuningResult(
        predictor_variant=variant,
        task_id="synthetic_1m",
        coefficients={"num_leaves": (31.0, 0.0)},
        per_quantile_kwargs={q: {"num_leaves": 31} for q in _TRAINING_QUANTILES},
        best_score=1.23,
        n_trials=1,
        validation_start=datetime(2019, 1, 1),
        validation_end=datetime(2020, 1, 1),
        ran_at=datetime(2020, 1, 2),
    )


# --- Interpolation -----------------------------------------------------------


def test_tail_distance() -> None:
    """0.0 at the median, 1.0 at the extreme tails."""
    assert _tail_distance(0.5) == pytest.approx(0.0)
    assert _tail_distance(0.025) == pytest.approx(1.0)
    assert _tail_distance(0.975) == pytest.approx(1.0)
    assert 0.0 < _tail_distance(0.3) < 1.0


def test_expand_to_per_quantile_zero_slope_returns_base() -> None:
    """A zero slope gives every quantile exactly the base value."""
    per_quantile = _expand_to_per_quantile({"learning_rate": (0.1, 0.0)})
    assert all(kwargs["learning_rate"] == pytest.approx(0.1) for kwargs in per_quantile.values())


def test_expand_to_per_quantile_tails_are_symmetric() -> None:
    """0.025 and 0.975 always receive the same interpolated value."""
    per_quantile = _expand_to_per_quantile({"num_leaves": (32.0, 16.0)})
    assert per_quantile[0.025]["num_leaves"] == per_quantile[0.975]["num_leaves"]


def test_expand_to_per_quantile_int_rounding_and_floor() -> None:
    """Integer params round to int and clamp at their floor."""
    per_quantile = _expand_to_per_quantile({"num_leaves": (2.0, -100.0)})
    for kwargs in per_quantile.values():
        assert isinstance(kwargs["num_leaves"], int)
        assert kwargs["num_leaves"] >= 2  # _PARAM_MINIMUMS floor


# --- num_threads / n_jobs interaction -----------------------------------------


def test_resolve_lgbm_kwargs_caps_threads_when_parallel() -> None:
    """n_jobs != 1 with no explicit num_threads gets capped to 1."""
    assert _resolve_lgbm_kwargs(None, n_jobs=4) == {"num_threads": 1}


def test_resolve_lgbm_kwargs_leaves_sequential_untouched() -> None:
    """n_jobs=1 (sequential) does not inject num_threads at all."""
    assert _resolve_lgbm_kwargs({"objective": "quantile"}, n_jobs=1) == {"objective": "quantile"}


def test_resolve_lgbm_kwargs_respects_explicit_num_threads() -> None:
    """A caller-supplied num_threads is never overridden."""
    assert _resolve_lgbm_kwargs({"num_threads": 4}, n_jobs=8) == {"num_threads": 4}


# --- _PerQuantileLightGBMModel mixin ------------------------------------------


class _DummyBaseModel:
    """Stand-in for LightGBMModel exposing only what the mixin needs."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        pass

    def _create_model(self, **kwargs):  # noqa: ANN003
        return kwargs


class _TestPerQuantileModel(_PerQuantileLightGBMModel, _DummyBaseModel):
    """Composes the mixin with a dummy base for a Darts-free unit test."""


def test_per_quantile_lightgbm_model_applies_override_for_matching_alpha() -> None:
    """An override for the requested alpha wins over the base kwargs."""
    model = _TestPerQuantileModel(per_quantile_kwargs={0.025: {"num_leaves": 5}})
    result = model._create_model(alpha=0.025, num_leaves=31, other=1)
    assert result == {"alpha": 0.025, "num_leaves": 5, "other": 1}


def test_per_quantile_lightgbm_model_no_override_for_unlisted_alpha() -> None:
    """An alpha with no override passes the base kwargs through unchanged."""
    model = _TestPerQuantileModel(per_quantile_kwargs={0.025: {"num_leaves": 5}})
    result = model._create_model(alpha=0.5, num_leaves=31)
    assert result == {"alpha": 0.5, "num_leaves": 31}


# --- DartsLightGBMPredictor(per_quantile_kwargs=...) smoke test ---------------


def test_darts_lightgbm_predictor_with_per_quantile_kwargs_smoke(svc: DataService, task: ForecastingTask) -> None:
    """A predictor built with per_quantile_kwargs yields a valid forecast."""
    per_quantile_kwargs = {q: {"num_leaves": 8} for q in _TRAINING_QUANTILES}
    preds = DartsLightGBMPredictor(
        lags=12,
        num_samples=200,
        per_quantile_kwargs=per_quantile_kwargs,
    ).predict(task, svc.context(AS_OF))

    assert len(preds) == 1
    assert preds[0].predictor_id == "darts_lightgbm_tuned"
    quantiles = preds[0].payload.quantiles
    assert set(STANDARD_QUANTILES).issubset(quantiles)
    values = [quantiles[q] for q in sorted(quantiles)]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:])), "Quantiles not monotonic."


# --- tune_lightgbm_configs orchestration (mocked) -----------------------------


def test_tune_lightgbm_configs_shared_reuses_single_study(mocker, task: ForecastingTask, svc: DataService) -> None:
    """separate=False runs exactly one Optuna study and reuses it for both variants."""
    mock_tune = mocker.patch(
        "aieng.forecasting.methods.numerical.lgbm_quantile_tuning.tune_lightgbm_quantile_config",
        return_value=_canned_result("univariate"),
    )
    result = tune_lightgbm_configs(
        task=task, data_service=svc, validation_end=datetime(2015, 1, 1),
        covariate_series_ids=["cov"], separate=False,
    )
    assert mock_tune.call_count == 1
    assert mock_tune.call_args.kwargs["covariate_series_ids"] is None
    assert result["univariate"].per_quantile_kwargs == result["covariate"].per_quantile_kwargs
    assert result["covariate"].predictor_variant == "covariate"


def test_tune_lightgbm_configs_separate_runs_two_studies(mocker, task: ForecastingTask, svc: DataService) -> None:
    """separate=True runs two independent Optuna studies, one per variant."""
    mock_tune = mocker.patch(
        "aieng.forecasting.methods.numerical.lgbm_quantile_tuning.tune_lightgbm_quantile_config",
        side_effect=[_canned_result("univariate"), _canned_result("covariate")],
    )
    result = tune_lightgbm_configs(
        task=task, data_service=svc, validation_end=datetime(2015, 1, 1),
        covariate_series_ids=["cov"], separate=True,
    )
    assert mock_tune.call_count == 2
    covariate_series_ids_seen = [call.kwargs["covariate_series_ids"] for call in mock_tune.call_args_list]
    assert covariate_series_ids_seen == [None, ["cov"]]
    assert result["univariate"].predictor_variant == "univariate"
    assert result["covariate"].predictor_variant == "covariate"


def test_tune_lightgbm_configs_forwards_stride_and_warmup(mocker, task: ForecastingTask, svc: DataService) -> None:
    """stride/warmup reach the inner tuner (dropped when n_jobs was added)."""
    mock_tune = mocker.patch(
        "aieng.forecasting.methods.numerical.lgbm_quantile_tuning.tune_lightgbm_quantile_config",
        return_value=_canned_result("univariate"),
    )
    tune_lightgbm_configs(
        task=task, data_service=svc, validation_end=datetime(2015, 1, 1),
        covariate_series_ids=["cov"], separate=False, stride=5, warmup=100,
    )
    assert mock_tune.call_args.kwargs["stride"] == 5
    assert mock_tune.call_args.kwargs["warmup"] == 100


# --- No-leakage guard ----------------------------------------------------------


def test_tune_lightgbm_quantile_config_rejects_leakage(task: ForecastingTask, svc: DataService) -> None:
    """validation_end after cutoff raises ValueError before any tuning runs."""
    with pytest.raises(ValueError, match="cutoff"):
        tune_lightgbm_quantile_config(
            task=task, data_service=svc,
            validation_end=datetime(2015, 6, 1),
            cutoff=datetime(2015, 1, 1),
        )


# --- End-to-end smoke test (real Optuna study) --------------------------------


def test_tune_lightgbm_quantile_config_smoke(task: ForecastingTask, svc: DataService) -> None:
    """A tiny real study runs end-to-end and returns a config covering all quantiles."""
    pytest.importorskip("optuna")

    result = tune_lightgbm_quantile_config(
        task=task,
        data_service=svc,
        validation_end=datetime(2015, 1, 1),
        validation_window=12,
        n_trials=1,
        num_samples=20,
        seed=0,
    )

    assert set(result.per_quantile_kwargs) == set(_TRAINING_QUANTILES)
    assert np.isfinite(result.best_score)
