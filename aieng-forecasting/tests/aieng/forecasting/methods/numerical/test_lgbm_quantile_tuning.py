"""Tests for ``aieng.forecasting.methods.numerical.lgbm_quantile_tuning``.

Pure-function tests cover the interpolation math and the
``_PerQuantileLightGBMModel`` override directly (no Optuna study, no Darts
import required). The shared-vs-separate orchestration is tested with
``tune_lightgbm_quantile_config`` mocked out. One smoke test runs a real,
tiny (``n_trials=1``) end-to-end Optuna study against synthetic data — the
only place the expensive path actually executes. A further section covers
the save/resume (scratch/reuse/resume) modes against real, tiny SQLite-backed
studies using pytest's ``tmp_path`` fixture.
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


# --- Save/resume across sessions (persistent SQLite storage) ------------------


def test_tune_lightgbm_quantile_config_scratch_persists_study(tmp_path, task: ForecastingTask, svc: DataService) -> None:
    """mode='scratch' with storage_path writes a study db and runs the full n_trials."""
    pytest.importorskip("optuna")
    db_path = tmp_path / "studies.db"

    result = tune_lightgbm_quantile_config(
        task=task, data_service=svc, validation_end=datetime(2015, 1, 1),
        validation_window=12, num_samples=20, seed=0,
        n_trials=2, storage_path=db_path, mode="scratch",
    )

    assert result.n_trials == 2
    assert db_path.exists()


def test_tune_lightgbm_quantile_config_scratch_wipes_prior_trials(tmp_path, task: ForecastingTask, svc: DataService) -> None:
    """A second scratch run replaces, rather than accumulates onto, the first."""
    pytest.importorskip("optuna")
    shared = {
        "task": task, "data_service": svc, "validation_end": datetime(2015, 1, 1),
        "validation_window": 12, "num_samples": 20, "seed": 0,
        "storage_path": tmp_path / "studies.db", "mode": "scratch",
    }

    tune_lightgbm_quantile_config(n_trials=3, **shared)
    result = tune_lightgbm_quantile_config(n_trials=2, **shared)

    assert result.n_trials == 2


def test_tune_lightgbm_quantile_config_resume_adds_only_remaining_trials(
    tmp_path, task: ForecastingTask, svc: DataService
) -> None:
    """Resume tops a study up to n_trials total, not n_trials more."""
    optuna = pytest.importorskip("optuna")
    db_path = tmp_path / "studies.db"
    shared = {
        "task": task, "data_service": svc, "validation_end": datetime(2015, 1, 1),
        "validation_window": 12, "num_samples": 20, "seed": 0, "storage_path": db_path,
    }

    tune_lightgbm_quantile_config(n_trials=2, mode="scratch", **shared)
    result = tune_lightgbm_quantile_config(n_trials=5, mode="resume", **shared)

    assert result.n_trials == 5
    storage_url = f"sqlite:///{db_path.resolve().as_posix()}"
    study = optuna.load_study(study_name=f"{task.task_id}_univariate", storage=storage_url)
    assert len(study.trials) == 5


def test_tune_lightgbm_quantile_config_resume_at_or_above_budget_runs_nothing(
    tmp_path, mocker, task: ForecastingTask, svc: DataService
) -> None:
    """Resume with a budget already met makes zero new backtest calls."""
    pytest.importorskip("optuna")
    shared = {
        "task": task, "data_service": svc, "validation_end": datetime(2015, 1, 1),
        "validation_window": 12, "num_samples": 20, "seed": 0,
        "storage_path": tmp_path / "studies.db",
    }

    tune_lightgbm_quantile_config(n_trials=3, mode="scratch", **shared)
    mock_backtest = mocker.patch("aieng.forecasting.methods.numerical.lgbm_quantile_tuning.backtest")
    result = tune_lightgbm_quantile_config(n_trials=3, mode="resume", **shared)

    assert mock_backtest.call_count == 0
    assert result.n_trials == 3


def test_tune_lightgbm_quantile_config_reuse_makes_no_new_calls(
    tmp_path, mocker, task: ForecastingTask, svc: DataService
) -> None:
    """Reuse loads the saved result and never calls backtest."""
    pytest.importorskip("optuna")
    shared = {
        "task": task, "data_service": svc, "validation_end": datetime(2015, 1, 1),
        "validation_window": 12, "num_samples": 20, "seed": 0,
        "storage_path": tmp_path / "studies.db",
    }

    first = tune_lightgbm_quantile_config(n_trials=2, mode="scratch", **shared)
    mock_backtest = mocker.patch("aieng.forecasting.methods.numerical.lgbm_quantile_tuning.backtest")
    result = tune_lightgbm_quantile_config(n_trials=2, mode="reuse", **shared)

    assert mock_backtest.call_count == 0
    assert result.n_trials == 2
    assert result.best_score == first.best_score


def test_tune_lightgbm_quantile_config_reuse_raises_when_no_saved_study(
    tmp_path, task: ForecastingTask, svc: DataService
) -> None:
    """Reuse fails fast rather than silently running a full study on a cold start."""
    pytest.importorskip("optuna")
    with pytest.raises(ValueError, match="reuse"):
        tune_lightgbm_quantile_config(
            task=task, data_service=svc, validation_end=datetime(2015, 1, 1),
            storage_path=tmp_path / "studies.db", mode="reuse",
        )


def test_tune_lightgbm_quantile_config_mode_requires_storage_path(task: ForecastingTask, svc: DataService) -> None:
    """resume/reuse without storage_path raise before optuna is even imported."""
    for mode in ("reuse", "resume"):
        with pytest.raises(ValueError, match="storage_path"):
            tune_lightgbm_quantile_config(
                task=task, data_service=svc, validation_end=datetime(2015, 1, 1), mode=mode,
            )


def test_tune_lightgbm_quantile_config_rejects_unknown_mode(task: ForecastingTask, svc: DataService) -> None:
    """An unrecognized mode string raises immediately."""
    with pytest.raises(ValueError, match="mode"):
        tune_lightgbm_quantile_config(
            task=task, data_service=svc, validation_end=datetime(2015, 1, 1), mode="bogus",
        )


def test_tune_lightgbm_configs_forwards_storage_path_and_mode(
    tmp_path, mocker, task: ForecastingTask, svc: DataService
) -> None:
    """storage_path/mode reach the inner tuner; study_name keeps its own default."""
    mock_tune = mocker.patch(
        "aieng.forecasting.methods.numerical.lgbm_quantile_tuning.tune_lightgbm_quantile_config",
        side_effect=[_canned_result("univariate"), _canned_result("covariate")],
    )
    db_path = tmp_path / "studies.db"

    tune_lightgbm_configs(
        task=task, data_service=svc, validation_end=datetime(2015, 1, 1),
        covariate_series_ids=["cov"], separate=True, storage_path=db_path, mode="resume",
    )

    for call in mock_tune.call_args_list:
        assert call.kwargs["storage_path"] == db_path
        assert call.kwargs["mode"] == "resume"
        assert "study_name" not in call.kwargs
