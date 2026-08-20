"""Persistent state for the BAA10Y adaptive tuning agent."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field


TuningMethod = Literal[
    "linear_regression",
    "lightgbm",
    "llmp_sampled_trajectory",
]

TuningExperiment = Literal[
    "smoke",
    "backtest_2025",
    "stress_2020",
]

CovariatePanel = Literal[
    "target_only",
    "default",
    "default_plus_hyoas",
]


def _utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


# ---------------------------------------------------------------------------
# Individual tuning trial
# ---------------------------------------------------------------------------

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

    mean_crps: float
    n_predictions: int
    skipped_origins: int

    ran_at: str


# ---------------------------------------------------------------------------
# Promoted configuration
# ---------------------------------------------------------------------------

class PromotedConfiguration(BaseModel):
    """The active candidate for one model and horizon."""

    model_config = ConfigDict(
        extra="forbid"
    )

    method: TuningMethod
    horizon: int
    candidate_id: str

    baseline_crps: float
    candidate_crps: float
    improvement_pct: float

    evidence_experiment: str = (
        "backtest_2025"
    )

    reason: str = ""
    promoted_on: str = Field(
        default_factory=_utc_now
    )


# ---------------------------------------------------------------------------
# Rejected candidate
# ---------------------------------------------------------------------------

class RejectedCandidate(BaseModel):
    """A candidate intentionally removed from further testing."""

    model_config = ConfigDict(
        extra="forbid"
    )

    candidate_id: str
    method: TuningMethod
    horizon: int

    reason: str

    rejected_on: str = Field(
        default_factory=_utc_now
    )


# ---------------------------------------------------------------------------
# Complete agent state
# ---------------------------------------------------------------------------

class BAA10YTuningState(BaseModel):
    """Complete persistent state for adaptive tuning."""

    model_config = ConfigDict(
        extra="forbid"
    )

    schema_version: int = 1

    created_at: str = Field(
        default_factory=_utc_now
    )

    updated_at: str = Field(
        default_factory=_utc_now
    )

    trials: list[TrialRecord] = Field(
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
    ) -> list[TrialRecord]:
        """Return trials matching the supplied filters."""

        results = self.trials

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
                if trial.candidate_id
                == candidate_id
            ]

        return results

    def get_promoted_configuration(
        self,
        *,
        method: str,
        horizon: int,
    ) -> PromotedConfiguration | None:
        """Return the active configuration for a model and horizon."""

        for configuration in (
            self.promoted_configurations
        ):
            if (
                configuration.method == method
                and configuration.horizon
                == horizon
            ):
                return configuration

        return None


# ---------------------------------------------------------------------------
# YAML persistence
# ---------------------------------------------------------------------------

class TuningStateStore:
    """Load and save BAA10Y adaptive tuning state."""

    def __init__(
        self,
        state_path: Path,
    ) -> None:
        self.state_path = Path(
            state_path
        )

    def load(self) -> BAA10YTuningState:
        """Load state, creating an empty file when necessary."""

        if not self.state_path.exists():
            state = BAA10YTuningState()
            self.save(state)
            return state

        with self.state_path.open(
            encoding="utf-8"
        ) as file:
            raw = yaml.safe_load(file) or {}

        return BAA10YTuningState.model_validate(
            raw
        )

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
                self.state_path.suffix + ".tmp"
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
            )

            if same_run:
                return existing, False

        state.trials.append(trial)
        self.save(state)

        return trial, True

    def promote(
        self,
        configuration: PromotedConfiguration,
    ) -> None:
        """Set the active candidate for one model and horizon."""

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
    "PromotedConfiguration",
    "RejectedCandidate",
    "TrialRecord",
    "TuningStateStore",
]