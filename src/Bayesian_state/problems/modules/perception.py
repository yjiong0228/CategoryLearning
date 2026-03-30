"""
Module: Perception Mechanism
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from .base_module import BaseModule
from ...utils.paths import (
    PROCESSED_DATA_DIR,
    TASK1B_ERRORSUMMARY_PATH,
    TASK2_PROCESSED_PATH,
)

FEATURE_NAMES = ["neck", "head", "leg", "tail"]
SUMMARY_MEAN_COLUMNS = [
    "neck_length_error_mean",
    "head_length_error_mean",
    "leg_length_error_mean",
    "tail_length_error_mean",
]
SUMMARY_STD_COLUMNS = [
    "neck_length_error_sd",
    "head_length_error_sd",
    "leg_length_error_sd",
    "tail_length_error_sd",
]


@lru_cache(maxsize=None)
def _load_csv_cached(summary_path: str) -> pd.DataFrame:
    csv_path = Path(summary_path)
    if not csv_path.exists():
        raise ValueError(f"Required dataset is missing: {csv_path}")
    return pd.read_csv(csv_path)


def _compute_subject_stats_from_summary(
    summary_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {"iSub", *SUMMARY_MEAN_COLUMNS, *SUMMARY_STD_COLUMNS}
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(
            "Task1b error summary is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    grouped = summary_df.groupby("iSub", as_index=True)
    mean_df = grouped[SUMMARY_MEAN_COLUMNS].mean().rename(
        columns=dict(zip(SUMMARY_MEAN_COLUMNS, FEATURE_NAMES))
    )
    std_df = grouped[SUMMARY_STD_COLUMNS].mean().rename(
        columns=dict(zip(SUMMARY_STD_COLUMNS, FEATURE_NAMES))
    )

    mean_df = mean_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std_df = std_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if mean_df.empty:
        raise ValueError(
            "Task1b error summary does not contain valid subject statistics"
        )

    return mean_df, std_df


def _extract_feature_orders(task2_df: pd.DataFrame) -> Dict[int, list[str]]:
    required_cols = [
        "iSub",
        "feature1_name",
        "feature2_name",
        "feature3_name",
        "feature4_name",
    ]
    missing_cols = [col for col in required_cols if col not in task2_df.columns]
    if missing_cols:
        raise ValueError(
            "Task2 dataset is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    feature_orders: Dict[int, list[str]] = {}
    for sub_id, group in task2_df.groupby("iSub"):
        rows = group[
            ["feature1_name", "feature2_name", "feature3_name", "feature4_name"]
        ].drop_duplicates()
        if rows.empty:
            continue
        if len(rows) != 1:
            raise ValueError(
                f"Subject {sub_id} has inconsistent feature name order in Task2 data"
            )

        names = [str(v).strip().lower() for v in rows.iloc[0].tolist()]
        invalid = [name for name in names if name not in FEATURE_NAMES]
        if invalid:
            raise ValueError(
                f"Subject {sub_id} has unknown feature names: {invalid}"
            )
        feature_orders[int(sub_id)] = names

    if not feature_orders:
        raise ValueError("Task2 dataset does not contain valid subject feature orders")
    return feature_orders


def _get_perception_noise_stats(
    processed_data_dir: Path | str | None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    if processed_data_dir is None:
        summary_path = TASK1B_ERRORSUMMARY_PATH.resolve()
        task2_path = TASK2_PROCESSED_PATH.resolve()
    else:
        resolved_dir = Path(processed_data_dir).resolve()
        summary_path = (resolved_dir / TASK1B_ERRORSUMMARY_PATH.name).resolve()
        task2_path = (resolved_dir / TASK2_PROCESSED_PATH.name).resolve()

    summary_df = _load_csv_cached(str(summary_path))
    task2_df = _load_csv_cached(str(task2_path))
    mean_df, std_df = _compute_subject_stats_from_summary(summary_df)
    feature_orders = _extract_feature_orders(task2_df)

    mean_map: Dict[int, np.ndarray] = {}
    std_map: Dict[int, np.ndarray] = {}

    for sub_id in mean_df.index:
        i_sub = int(sub_id)
        if i_sub not in feature_orders:
            raise ValueError(f"Subject {i_sub} not found in Task2 feature order data")

        order = feature_orders[i_sub]
        mean_dict = mean_df.loc[sub_id, FEATURE_NAMES].to_dict()
        std_dict = std_df.loc[sub_id, FEATURE_NAMES].to_dict()

        subject_mean = np.array([mean_dict[name] for name in order], dtype=float)
        subject_std = np.array([std_dict[name] for name in order], dtype=float)
        mean_map[i_sub] = np.nan_to_num(subject_mean, nan=0.0)
        std_map[i_sub] = np.nan_to_num(subject_std, nan=0.0)

    if not mean_map:
        raise ValueError(
            "Failed to compute perception statistics for any subject"
        )

    return mean_map, std_map


################### NEW Perception Module ###################

class PerceptionModule(BaseModule):
    """Perception noise module with subject-specific parameters."""

    def __init__(self, engine, **kwargs):
        """Initialize the perception module.

        Parameters
        ----------
        engine: BaseEngine
            Hosting inference engine instance.
        features: int, optional
            Number of stimulus features (defaults to 4).
        mean: Sequence[float] | float, optional
            Mean offsets for each feature. Scalars are broadcast and iterables
            are coerced to numpy arrays.
        std: Sequence[float] | float, optional
            Standard deviations for each feature. Scalars are broadcast and
            iterables are coerced to numpy arrays.
        subject_id: int, optional
            Stored for debugging/logging; it does not affect computations.
        """

        super().__init__(engine, **kwargs)
        self.features = kwargs.pop("features", 4)
        self.subject_id = kwargs.pop("subject_id", getattr(engine, "subject_id", None))
        processed_data_dir = kwargs.pop(
            "processed_data_dir", getattr(engine, "processed_data_dir", PROCESSED_DATA_DIR)
        )

        mean_value = kwargs.pop("mean", None)
        std_value = kwargs.pop("std", None)

        if mean_value is None or std_value is None:
            if self.subject_id is None:
                raise ValueError(
                    "PerceptionModule requires 'subject_id' when mean/std are not provided."
                )
            auto_mean, auto_std = self._load_subject_stats(
                int(self.subject_id),
                processed_data_dir,
            )
            if mean_value is None:
                mean_value = auto_mean
            if std_value is None:
                std_value = auto_std

        self.mean = self._coerce_vector(mean_value, "mean")
        self.std = np.abs(self._coerce_vector(std_value, "std"))

    @staticmethod
    def _load_subject_stats(subject_id: int, processed_data_dir: Path | str | None):
        try:
            mean_map, std_map = _get_perception_noise_stats(processed_data_dir)
        except ValueError as exc:
            raise ValueError(
                f"Failed to load perception statistics from {processed_data_dir}"
            ) from exc

        if subject_id not in mean_map:
            raise ValueError(
                f"Subject {subject_id} does not exist in perception statistics"
            )
        return mean_map[subject_id], std_map[subject_id]

    def _coerce_vector(self, value, name: str) -> np.ndarray:
        """Convert incoming parameter to a feature-sized numpy array."""

        if isinstance(value, (float, int)):
            return np.full(self.features, float(value), dtype=float)

        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or array.shape[0] != self.features:
            raise ValueError(
                f"Parameter '{name}' must be a 1-D array of length {self.features}"
            )
        return np.nan_to_num(array, nan=0.0)

    def sample(self, stimu):
        """
        stimu for single trial, shape: (features, )
        """
        # stimu 的每个维度加上各个特征各自的mean和std采样的噪声
        noise = np.random.normal(loc=self.mean, scale=self.std, size=stimu.shape)
        return stimu + noise
    
    def process(self, **kwargs):
        """
        Process the stimulus with perception noise

        Args:
            **kwargs: Additional keyword arguments
                - stimu (np.ndarray): Stimulus to process, shape: (features, )
        """
        stimu = kwargs.get("stimu", self.engine.observation[0])
        sampled = self.sample(stimu)
        # Ensure the values are in the range of [0, 1]
        sampled = np.clip(sampled, 0, 1)
        self.engine.observation = (sampled, self.engine.observation[1], self.engine.observation[2])
        return sampled
