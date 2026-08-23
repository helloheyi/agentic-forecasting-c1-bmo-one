"""Persistent state for the BAA10Y adaptive tuning agent."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = 2

TuningMethod = Literal[
    "linear_regression",
    "lightgbm",
    "llmp_sampled_trajectory",
]

TuningExperiment = Literal[
    "smoke",
    "tune_2025",
    "tune_paired_2025",
    "validate_2025",
    "backtest_2025",
    "stress_2020",
]

CovariatePanel = Literal[
    "target_only",
    "default",
    "default_plus_hyoas",
]

SearchStrategy = Literal[
    "grid",
    "optuna_tpe",
    "successive_halving",
]

StudyStatus = Literal[
    "created",
    "running",
    "completed",
    "stopped",
    "failed",
]


def _utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


class TrialRecord(BaseModel):
    """One completed parameter-tuning backtest."""

    model_config = ConfigDict(
        extra="forbid"
    )

    trial_id: str = Field(
        default_factory=lambda: (
            f"trial-{uuid4().hex[:12]}"
        )
    )

    candidate_id: str
    predictor_id: str
    method: TuningMethod
    horizon: int
    experiment: TuningExperiment

    parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    covariate_panel: CovariatePanel
    is_baseline: bool = False

    # Optuna/search metadata. Defaults preserve compatibility
    # with trials created by the original fixed-candidate tuner.
    study_name: str = ""
    trial_number: int | None = None
    parameter_hash: str = ""
    search_strategy: str = ""

    # Primary and secondary performance evidence.
    mean_crps: float
    crps_std: float | None = None
    median_crps: float | None = None

    # Paired tuning evidence.
    development_mean_crps: float | None = None
    inner_validation_mean_crps: float | None = None
    generalization_gap_pct: float | None = None
    development_n_predictions: int | None = None
    inner_validation_n_predictions: int | None = None
    # Origin date -> CRPS. This will later allow a direct
    # baseline-versus-candidate origin win-rate calculation.
    score_by_origin: dict[str, float] = Field(
        default_factory=dict
    )

    n_predictions: int
    skipped_origins: int

    ran_at: str


class SearchStudyRecord(BaseModel):
    """Summary of one adaptive parameter-search study."""

    model_config = ConfigDict(
        extra="forbid"
    )

    study_name: str
    method: TuningMethod
    horizon: int
    covariate_panel: CovariatePanel
    search_strategy: SearchStrategy

    objective_experiment: TuningExperiment = (
        "tune_2025"
    )
    validation_experiment: TuningExperiment = (
        "validate_2025"
    )

    max_trials: int = Field(
        default=0,
        ge=0,
    )
    completed_trials: int = Field(
        default=0,
        ge=0,
    )

    status: StudyStatus = "created"

    baseline_candidate_id: str = ""
    baseline_mean_crps: float | None = None

    best_candidate_id: str | None = None
    best_mean_crps: float | None = None

    best_parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: str = Field(
        default_factory=_utc_now
    )
    updated_at: str = Field(
        default_factory=_utc_now
    )


class PromotedConfiguration(BaseModel):
    """The active configuration for one model track."""

    model_config = ConfigDict(
        extra="forbid"
    )

    method: TuningMethod
    horizon: int

    # Default preserves compatibility with older saved state.
    covariate_panel: CovariatePanel = (
        "target_only"
    )

    candidate_id: str
    parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    baseline_crps: float
    candidate_crps: float
    improvement_pct: float

    origin_win_rate: float | None = None

    evidence_experiment: TuningExperiment = (
        "validate_2025"
    )

    reason: str = ""

    promoted_on: str = Field(
        default_factory=_utc_now
    )


class RejectedCandidate(BaseModel):
    """A candidate intentionally removed from further testing."""

    model_config = ConfigDict(
        extra="forbid"
    )

    candidate_id: str
    method: TuningMethod
    horizon: int

    covariate_panel: CovariatePanel = (
        "target_only"
    )

    reason: str

    rejected_on: str = Field(
        default_factory=_utc_now
    )


class BAA10YTuningState(BaseModel):
    """Complete persistent state for adaptive tuning."""

    model_config = ConfigDict(
        extra="forbid"
    )

    schema_version: int = SCHEMA_VERSION

    created_at: str = Field(
        default_factory=_utc_now
    )

    updated_at: str = Field(
        default_factory=_utc_now
    )

    trials: list[TrialRecord] = Field(
        default_factory=list
    )

    studies: list[SearchStudyRecord] = Field(
        default_factory=list
    )

    promoted_configurations: list[
        PromotedConfiguration
    ] = Field(
        default_factory=list
    )

    rejected_candidates: list[
        RejectedCandidate
    ] = Field(
        default_factory=list
    )

    def find_trials(
        self,
        *,
        method: str | None = None,
        horizon: int | None = None,
        experiment: str | None = None,
        candidate_id: str | None = None,
        covariate_panel: str | None = None,
        study_name: str | None = None,
    ) -> list[TrialRecord]:
        """Return trials matching the supplied filters."""

        results = list(
            self.trials
        )

        if method is not None:
            results = [
                trial
                for trial in results
                if trial.method == method
            ]

        if horizon is not None:
            results = [
                trial
                for trial in results
                if trial.horizon == horizon
            ]

        if experiment is not None:
            results = [
                trial
                for trial in results
                if trial.experiment == experiment
            ]

        if candidate_id is not None:
            results = [
                trial
                for trial in results
                if (
                    trial.candidate_id
                    == candidate_id
                )
            ]

        if covariate_panel is not None:
            results = [
                trial
                for trial in results
                if (
                    trial.covariate_panel
                    == covariate_panel
                )
            ]

        if study_name is not None:
            results = [
                trial
                for trial in results
                if (
                    trial.study_name
                    == study_name
                )
            ]

        return results

    def get_study(
        self,
        study_name: str,
    ) -> SearchStudyRecord | None:
        """Return one stored search study."""

        for study in self.studies:
            if (
                study.study_name
                == study_name
            ):
                return study

        return None

    def get_promoted_configuration(
        self,
        *,
        method: str,
        horizon: int,
        covariate_panel: str,
    ) -> PromotedConfiguration | None:
        """Return the active configuration for one model track."""

        for configuration in (
            self.promoted_configurations
        ):
            if (
                configuration.method
                == method
                and configuration.horizon
                == horizon
                and configuration.covariate_panel
                == covariate_panel
            ):
                return configuration

        return None


class TuningStateStore:
    """Load and save BAA10Y adaptive-tuning state."""

    def __init__(
        self,
        state_path: Path,
    ) -> None:
        self.state_path = Path(
            state_path
        )

    def load(self) -> BAA10YTuningState:
        """Load state, creating an empty state when necessary."""

        if not self.state_path.exists():
            state = BAA10YTuningState()
            self.save(state)
            return state

        with self.state_path.open(
            encoding="utf-8"
        ) as file:
            raw = yaml.safe_load(
                file
            ) or {}

        state = (
            BAA10YTuningState.model_validate(
                raw
            )
        )

        # Existing version-1 files remain valid because all new
        # fields have defaults. Upgrade the version after loading.
        if (
            state.schema_version
            < SCHEMA_VERSION
        ):
            state.schema_version = (
                SCHEMA_VERSION
            )
            self.save(state)

        return state

    def save(
        self,
        state: BAA10YTuningState,
    ) -> None:
        """Save state atomically as YAML."""

        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state.updated_at = _utc_now()

        temporary_path = (
            self.state_path.with_suffix(
                self.state_path.suffix
                + ".tmp"
            )
        )

        temporary_path.write_text(
            yaml.safe_dump(
                state.model_dump(
                    mode="json"
                ),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            self.state_path
        )

    def add_trial(
        self,
        trial_data: dict[str, Any],
    ) -> tuple[TrialRecord, bool]:
        """Add a trial unless the same cached run already exists."""

        state = self.load()

        trial = TrialRecord.model_validate(
            trial_data
        )

        for existing in state.trials:
            same_run = (
                existing.candidate_id
                == trial.candidate_id
                and existing.horizon
                == trial.horizon
                and existing.experiment
                == trial.experiment
                and existing.ran_at
                == trial.ran_at
                and existing.study_name
                == trial.study_name
            )

            if same_run:
                return existing, False

        state.trials.append(
            trial
        )
        self.save(state)

        return trial, True

    def upsert_study(
        self,
        study: SearchStudyRecord,
    ) -> None:
        """Create or replace one search-study summary."""
        agent_actions: list[
            dict[str, Any]
        ] = Field(
            default_factory=list
        )
        search_frozen: bool = False
        validation_decision: str = ""
        
        state = self.load()

        study.updated_at = _utc_now()

        state.studies = [
            existing
            for existing in state.studies
            if (
                existing.study_name
                != study.study_name
            )
        ]

        state.studies.append(
            study
        )

        self.save(state)

    def promote(
        self,
        configuration: PromotedConfiguration,
    ) -> None:
        """Set the active configuration for one model track."""

        state = self.load()

        state.promoted_configurations = [
            existing
            for existing
            in state.promoted_configurations
            if not (
                existing.method
                == configuration.method
                and existing.horizon
                == configuration.horizon
                and existing.covariate_panel
                == configuration.covariate_panel
            )
        ]

        state.promoted_configurations.append(
            configuration
        )

        self.save(state)

    def reject(
        self,
        rejection: RejectedCandidate,
    ) -> None:
        """Record a rejected candidate."""

        state = self.load()

        state.rejected_candidates.append(
            rejection
        )

        self.save(state)


__all__ = [
    "BAA10YTuningState",
    "CovariatePanel",
    "PromotedConfiguration",
    "RejectedCandidate",
    "SCHEMA_VERSION",
    "SearchStudyRecord",
    "SearchStrategy",
    "TrialRecord",
    "TuningExperiment",
    "TuningMethod",
    "TuningStateStore",
]