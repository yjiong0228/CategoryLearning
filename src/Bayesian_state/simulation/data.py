"""Trial-array validation, dataset loading, and subject slicing."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..utils.paths import PROCESSED_DATA_DIR, TASK2_PROCESSED_PATH


@dataclass
class TrialArrays:
    """Subject trial arrays with optional hard and probabilistic targets."""

    stimulus: np.ndarray
    choices: np.ndarray
    feedback: np.ndarray
    categories: Optional[np.ndarray] = None
    target_probs: Optional[np.ndarray] = None


# Trial-data preparation
def _coerce_trial_arrays(arrays: TrialArrays | tuple | list) -> TrialArrays:
    if isinstance(arrays, TrialArrays):
        return arrays
    if not isinstance(arrays, (tuple, list)) or len(arrays) < 3:
        raise ValueError("arrays must be a TrialArrays instance or a tuple/list with at least 3 entries")
    categories = arrays[3] if len(arrays) >= 4 else None
    target_probs = arrays[4] if len(arrays) >= 5 else None
    return TrialArrays(
        stimulus=np.asarray(arrays[0], dtype=float),
        choices=np.asarray(arrays[1], dtype=int),
        feedback=np.asarray(arrays[2], dtype=float),
        categories=None if categories is None else np.asarray(categories, dtype=int),
        target_probs=None if target_probs is None else np.asarray(target_probs, dtype=float),
    )


def _normalize_probability_rows(values: np.ndarray, *, context: str) -> np.ndarray:
    probs = np.asarray(values, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"{context} must be a 2-D matrix, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{context} contains non-finite values")
    if np.any(probs < 0):
        raise ValueError(f"{context} contains negative values")
    denom = probs.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError(f"{context} has rows that sum to zero")
    return probs / denom


def _probability_columns_from_frame(subject_frame: pd.DataFrame) -> list[str]:
    cols: list[tuple[int, str]] = []
    for col in subject_frame.columns:
        name = str(col)
        if not name.startswith("probCat"):
            continue
        suffix = name[len("probCat"):]
        if suffix.isdigit():
            cols.append((int(suffix), name))
    return [name for _, name in sorted(cols)]


class SubjectTrialDataLoader:
    """Load and validate subject-level trial arrays for model execution."""

    def __init__(
        self,
        engine_config: Dict[str, Any],
        processed_data_dir: Optional[Path | str] = None,
        n_jobs: int = 1,
        dataset_paths: Optional[Mapping[str, Path | str]] = None,
    ) -> None:
        self._engine_config_template = deepcopy(engine_config)
        self._processed_data_dir = (
            Path(processed_data_dir).resolve()
            if processed_data_dir is not None
            else PROCESSED_DATA_DIR
        )
        self._dataset_paths = dict(dataset_paths or {})
        self.learning_data: Optional[pd.DataFrame] = None
        self.n_jobs = n_jobs
        data_cfg = self._engine_config_template.get("data", {}) or {}
        self._feature_columns = list(
            data_cfg.get("feature_columns", ["feature1", "feature2", "feature3", "feature4"])
        )
        self._condition_column = str(data_cfg.get("condition_column", "condition"))
        self._subject_column = str(data_cfg.get("subject_column", "iSub"))
        self._category_column = str(data_cfg.get("category_column", "category"))
        self._target_type = str(data_cfg.get("target_type", "auto")).strip().lower()
        self._probability_columns = list(data_cfg.get("probability_columns", []))

    def prepare_data(self, data_path: Path | str = TASK2_PROCESSED_PATH) -> None:
        data_path = Path(data_path).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        self.learning_data = pd.read_csv(data_path, encoding="utf-8-sig")

    def _get_subject_frame(self, subject_id: int, stop_at: float) -> pd.DataFrame:
        if self.learning_data is None:
            self.prepare_data()
        assert self.learning_data is not None

        if self._subject_column not in self.learning_data.columns:
            raise ValueError(f"Subject column '{self._subject_column}' not found in dataset")

        subject_frame = self.learning_data[self.learning_data[self._subject_column] == subject_id]
        if subject_frame.empty:
            raise ValueError(f"Subject {subject_id} not found in dataset")

        stop_index = max(1, int(len(subject_frame) * stop_at + 0.5))
        return subject_frame.iloc[:stop_index].copy()

    def _extract_arrays(
        self,
        subject_frame: pd.DataFrame,
        max_trials: Optional[int],
    ) -> TrialArrays:
        missing_features = [col for col in self._feature_columns if col not in subject_frame.columns]
        if missing_features:
            raise ValueError(
                "Dataset is missing configured feature columns: "
                + ", ".join(missing_features)
            )
        stimulus = subject_frame[self._feature_columns].to_numpy(dtype=float)
        choices = subject_frame["choice"].to_numpy(dtype=int)
        feedback = subject_frame["feedback"].to_numpy(dtype=float)

        probabilistic_target_types = {"probabilistic", "probability", "soft", "soft_category"}
        categories: Optional[np.ndarray] = None
        target_probs: Optional[np.ndarray] = None

        prob_cols = list(self._probability_columns)
        if not prob_cols:
            prob_cols = _probability_columns_from_frame(subject_frame)

        if self._target_type in probabilistic_target_types:
            if not prob_cols:
                raise ValueError(
                    "data.target_type is probabilistic, but no probability columns were configured "
                    "and no probCat* columns were found."
                )
        elif self._target_type not in {"auto", "hard", "category", "categorical"}:
            raise ValueError(
                "data.target_type must be auto, hard/category/categorical, or probabilistic/probability/soft"
            )

        if prob_cols:
            missing_probs = [col for col in prob_cols if col not in subject_frame.columns]
            if missing_probs:
                raise ValueError(
                    "Dataset is missing configured probability columns: "
                    + ", ".join(missing_probs)
                )
            target_probs = _normalize_probability_rows(
                subject_frame[prob_cols].to_numpy(dtype=float),
                context="target probability columns",
            )

        if self._target_type not in probabilistic_target_types and self._category_column in subject_frame.columns:
            categories = subject_frame[self._category_column].to_numpy(dtype=int)
        elif self._target_type in {"hard", "category", "categorical"}:
            raise ValueError(f"Dataset is missing configured category column: {self._category_column}")

        if max_trials is not None:
            usable = min(max_trials, stimulus.shape[0])
            stimulus = stimulus[:usable]
            choices = choices[:usable]
            feedback = feedback[:usable]
            if categories is not None:
                categories = categories[:usable]
            if target_probs is not None:
                target_probs = target_probs[:usable]

        return TrialArrays(
            stimulus=stimulus,
            choices=choices,
            feedback=feedback,
            categories=categories,
            target_probs=target_probs,
        )

    def _get_condition_value(self, subject_frame: pd.DataFrame) -> int:
        if self._condition_column in subject_frame.columns:
            return int(subject_frame[self._condition_column].iloc[0])
        if "ruleID" in subject_frame.columns:
            return int(subject_frame["ruleID"].iloc[0])
        return 1


def prepare_trial_sequence(
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
) -> List[List[float]]:
    trials: List[List[float]] = []
    for stim, choice, fb in zip(stimulus, choices, feedback):
        trial: List[float] = [stim, int(choice), float(fb)]
        trials.append(trial)
    return trials



__all__ = ["SubjectTrialDataLoader", "TrialArrays", "prepare_trial_sequence"]
