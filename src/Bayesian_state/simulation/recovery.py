"""Recovery data contracts, schedule fingerprints, and synthetic trial tables."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")


ORDER_COLUMNS = ("iSession", "iBlock", "iTrial")


SCHEDULE_COLUMNS = (*ORDER_COLUMNS, *FEATURE_COLUMNS, "category")


MODEL_PARAMETER_NAMES = (
    "M",
    "chi",
    "gamma",
    "E_C",
    "delta_E",
    "g_0",
    "c_A",
    "c_G",
    "beta_0",
    "eta_plus",
    "eta_minus",
)


CELL_FREE_PARAMETERS = {
    "P": ("beta_0", "eta_plus", "eta_minus"),
    "PM": ("gamma", "beta_0", "eta_plus", "eta_minus"),
    "PH": (
        "M", "chi", "E_C", "delta_E", "g_0", "c_A", "c_G",
        "beta_0", "eta_plus", "eta_minus",
    ),
    "PMH": MODEL_PARAMETER_NAMES,
}


@dataclass(frozen=True)
class RecoveryDatasetSpec:
    dataset_id: str
    family: str
    subject_id: int
    trial_count: int
    replicate: int
    truth_cell: str
    truth_profile: str
    truth: dict[str, Any]
    generation_seed: int
    train_trial_count: int
    evaluation_trial_count: int


@dataclass(frozen=True)
class RecoveryDesign:
    analysis_id: str
    source_path: Path
    model_engine_config: Path
    parameter_space_path: Path
    base_simulation_config: Path
    output_root: Path
    subject_trial_counts: dict[int, int]
    module_datasets: tuple[RecoveryDatasetSpec, ...]
    parameter_datasets: tuple[RecoveryDatasetSpec, ...]
    config: dict[str, Any]

    @property
    def all_datasets(self) -> tuple[RecoveryDatasetSpec, ...]:
        return self.module_datasets + self.parameter_datasets


def schedule_fingerprint(schedule_frame: pd.DataFrame) -> str:
    """Hash only task order, stimulus, and category—not observed behavior."""

    missing = set(SCHEDULE_COLUMNS) - set(schedule_frame.columns)
    if missing:
        raise ValueError(f"schedule is missing columns: {sorted(missing)}")
    schedule = schedule_frame.loc[:, list(SCHEDULE_COLUMNS)].copy()
    if schedule.empty:
        raise ValueError("schedule cannot be empty")
    encoded = schedule.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def synthetic_dataset_frame(
    schedule_frame: pd.DataFrame,
    *,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Replace observed behavior with one autonomously generated trajectory."""

    generated_choice = np.asarray(choices, dtype=int).reshape(-1)
    generated_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if len(schedule_frame) != generated_choice.size or (
        generated_feedback.size != generated_choice.size
    ):
        raise ValueError("generated choices/feedback must align with the schedule")
    if not np.all(np.isin(generated_choice, [1, 2])):
        raise ValueError("generated choices must be encoded as 1 or 2")
    if not np.all(np.isfinite(generated_feedback)):
        raise ValueError("generated feedback must be finite")
    leading = [
        name
        for name in ("iSub", "condition", *SCHEDULE_COLUMNS)
        if name in schedule_frame.columns
    ]
    frame = schedule_frame.loc[:, leading].copy().reset_index(drop=True)
    frame["choice"] = generated_choice
    frame["feedback"] = generated_feedback
    frame.attrs["observed_choices_used"] = False
    frame.attrs["schedule_fingerprint"] = schedule_fingerprint(schedule_frame)
    return frame
