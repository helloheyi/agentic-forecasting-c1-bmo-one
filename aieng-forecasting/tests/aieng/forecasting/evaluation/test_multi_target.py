"""Tests for MultiTargetBacktestSpec, multi_backtest(), MultiTargetEvalSpec, multi_evaluate()."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestResult, MultiTargetBacktestSpec, multi_backtest
from aieng.forecasting.evaluation.eval import (
    EvalBudgetExceededError,
    EvalResult,
    EvalTracker,
    MultiTargetEvalSpec,
    multi_evaluate,
)
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "task_a", series_id: str = "series_a") -> ForecastingTask:
    return ForecastingTask(
        task_id=task_id,
        target_series_id=series_id,
        horizon=12,
        frequency="MS",
        description=f"Test task {task_id}",
    )


def _build_data_service(*series_ids: str, series_start: str = "2000-01-01", series_end: str = "2026-01-01") -> DataService:
    """Build a DataService with one synthetic monthly series per series_id."""
    dates = pd.date_range(start=series_start, end=series_end, freq="MS")
    svc = DataService()
    for sid in series_ids:
        df = pd.DataFrame({"timestamp": dates, "value": np.arange(len(dates), dtype=float)})
        adapter = MagicMock()
        adapter.fetch.return_value = df
        meta = SeriesMetadata(
            series_id=sid,
            description=f"Synthetic series {sid}",
            source="test",
            units="units",
            frequency="MS",
        )
        svc.register(sid, adapter, meta)
    return svc


class ConstantPredictor(Predictor):
    def __init__(self, value: float = 100.0) -> None:
        self._value = value

    @property
    def predictor_id(self) -> str:
        return "constant"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> Prediction:
        forecast_date = (pd.Timestamp(context.as_of) + pd.DateOffset(months=task.horizon)).to_pydatetime()
        point = self._value
        return Prediction(
            predictor_id=self.predictor_id,
            task_id=task.task_id,
            issued_at=datetime(2024, 1, 1),
            as_of=context.as_of,
            forecast_date=forecast_date,
            payload=ContinuousForecast(
                point_forecast=point,
                quantiles={q: point + (q - 0.5) * 5 for q in STANDARD_QUANTILES},
            ),
        )


# ---------------------------------------------------------------------------
# MultiTargetBacktestSpec tests
# ---------------------------------------------------------------------------


class TestMultiTargetBacktestSpec:
    def test_construction_two_tasks(self) -> None:
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            warmup=0,
        )
        assert len(spec.tasks) == 2

    def test_specs_returns_one_per_task(self) -> None:
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b"), _make_task("c", "s_c")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        individual = spec.specs()
        assert len(individual) == 3
        task_ids = [s.task.task_id for s in individual]
        assert "a" in task_ids and "b" in task_ids and "c" in task_ids

    def test_specs_share_window_parameters(self) -> None:
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2014, 1, 1),
            stride=3,
            warmup=12,
        )
        for s in spec.specs():
            assert s.start == spec.start
            assert s.end == spec.end
            assert s.stride == spec.stride
            assert s.warmup == spec.warmup

    def test_start_after_end_raises(self) -> None:
        with pytest.raises(ValueError, match="start.*must be before end"):
            MultiTargetBacktestSpec(
                tasks=[_make_task()],
                start=datetime(2021, 1, 1),
                end=datetime(2020, 1, 1),
            )

    def test_mixed_frequencies_raises(self) -> None:
        task_monthly = ForecastingTask(
            task_id="monthly",
            target_series_id="s1",
            horizon=12,
            frequency="MS",
            description="monthly",
        )
        task_quarterly = ForecastingTask(
            task_id="quarterly",
            target_series_id="s2",
            horizon=4,
            frequency="QS",
            description="quarterly",
        )
        with pytest.raises(ValueError, match="same frequency"):
            MultiTargetBacktestSpec(
                tasks=[task_monthly, task_quarterly],
                start=datetime(2010, 1, 1),
                end=datetime(2012, 1, 1),
            )

    def test_single_task_minimum(self) -> None:
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task()],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
        )
        assert len(spec.specs()) == 1

    def test_yaml_roundtrip(self) -> None:
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2014, 1, 1),
            stride=6,
            warmup=24,
        )
        dumped = spec.model_dump()
        restored = MultiTargetBacktestSpec.model_validate(dumped)
        assert len(restored.tasks) == 2
        assert restored.stride == 6
        assert restored.warmup == 24


# ---------------------------------------------------------------------------
# multi_backtest() tests
# ---------------------------------------------------------------------------


class TestMultiBacktest:
    def test_returns_dict_keyed_by_task_id(self) -> None:
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_backtest(ConstantPredictor(), spec, svc)
        assert set(results.keys()) == {"a", "b"}

    def test_each_result_is_backtest_result(self) -> None:
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_backtest(ConstantPredictor(), spec, svc)
        for result in results.values():
            assert isinstance(result, BacktestResult)

    def test_predictor_id_matches(self) -> None:
        svc = _build_data_service("s_a")
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_backtest(ConstantPredictor(), spec, svc)
        assert results["a"].predictor_id == "constant"

    def test_mean_crps_per_task(self) -> None:
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_backtest(ConstantPredictor(), spec, svc)
        for result in results.values():
            assert abs(result.mean_crps - float(np.mean(result.scores))) < 1e-10


# ---------------------------------------------------------------------------
# MultiTargetEvalSpec tests
# ---------------------------------------------------------------------------


class TestMultiTargetEvalSpec:
    def test_construction(self) -> None:
        spec = MultiTargetEvalSpec(
            spec_id="test_mt_eval",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            max_runs=5,
        )
        assert spec.spec_id == "test_mt_eval"
        assert spec.max_runs == 5

    def test_specs_share_spec_id(self) -> None:
        spec = MultiTargetEvalSpec(
            spec_id="shared_id",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
        )
        for s in spec.specs():
            assert s.spec_id == "shared_id"

    def test_mixed_frequencies_raises(self) -> None:
        task_m = ForecastingTask(task_id="m", target_series_id="s1", horizon=12, frequency="MS", description="m")
        task_q = ForecastingTask(task_id="q", target_series_id="s2", horizon=4, frequency="QS", description="q")
        with pytest.raises(ValueError, match="same frequency"):
            MultiTargetEvalSpec(
                spec_id="bad",
                tasks=[task_m, task_q],
                start=datetime(2010, 1, 1),
                end=datetime(2012, 1, 1),
            )

    def test_yaml_roundtrip(self) -> None:
        spec = MultiTargetEvalSpec(
            spec_id="food_18m_eval",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2022, 1, 1),
            end=datetime(2024, 1, 1),
            stride=6,
            warmup=24,
            max_runs=5,
        )
        dumped = spec.model_dump()
        restored = MultiTargetEvalSpec.model_validate(dumped)
        assert restored.spec_id == spec.spec_id
        assert restored.max_runs == spec.max_runs
        assert len(restored.tasks) == 2


# ---------------------------------------------------------------------------
# multi_evaluate() tests
# ---------------------------------------------------------------------------


class TestMultiEvaluate:
    def test_returns_dict_keyed_by_task_id(self) -> None:
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetEvalSpec(
            spec_id="mt_eval",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(ConstantPredictor(), spec, svc)
        assert set(results.keys()) == {"a", "b"}

    def test_each_result_is_eval_result(self) -> None:
        svc = _build_data_service("s_a")
        spec = MultiTargetEvalSpec(
            spec_id="mt_eval",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(ConstantPredictor(), spec, svc)
        assert isinstance(results["a"], EvalResult)

    def test_single_call_counts_as_one_budget_run(self, tmp_path: Path) -> None:
        """One multi_evaluate call should use exactly one run from the budget."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetEvalSpec(
            spec_id="budget_test",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            max_runs=3,
        )
        tracker = EvalTracker(tmp_path / "runs.yaml")
        multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        assert tracker.runs_for("budget_test") == 1

    def test_run_number_increments_across_calls(self, tmp_path: Path) -> None:
        svc = _build_data_service("s_a")
        spec = MultiTargetEvalSpec(
            spec_id="run_num_test",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            max_runs=5,
        )
        tracker = EvalTracker(tmp_path / "runs.yaml")
        r1 = multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        r2 = multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        assert r1["a"].run_number == 1
        assert r2["a"].run_number == 2

    def test_budget_enforced_across_multi_calls(self, tmp_path: Path) -> None:
        svc = _build_data_service("s_a")
        spec = MultiTargetEvalSpec(
            spec_id="budget_limit",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            max_runs=2,
        )
        tracker = EvalTracker(tmp_path / "runs.yaml")
        multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        with pytest.raises(EvalBudgetExceededError):
            multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)

    def test_no_tracker_runs_unconditionally(self) -> None:
        svc = _build_data_service("s_a")
        spec = MultiTargetEvalSpec(
            spec_id="no_tracker",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(ConstantPredictor(), spec, svc)
        assert results["a"].run_number == 1

    def test_run_number_is_one_without_tracker(self) -> None:
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetEvalSpec(
            spec_id="no_tracker_two_tasks",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(ConstantPredictor(), spec, svc)
        assert all(r.run_number == 1 for r in results.values())
