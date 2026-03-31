"""
Module: Perception Mechanism
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Iterable
import numpy as np
import pandas as pd
from .base_module import BaseModule
from ...utils.paths import (
    PROCESSED_DATA_DIR,
    TASK1B_ERRORSUMMARY_PATH,
    TASK1B_ERRORSUMMARY_72_PATH,
    TASK2_PROCESSED_PATH,
)

FEATURE_NAMES = ["neck", "head", "leg", "tail"]
SUMMARY_REQUIRED_COLUMNS = {"iSub", "feature_name", "feature_value", "error_mean", "error_std"}
SUMMARY72_REQUIRED_COLUMNS = {"iSub", "feature_name", "threshold_mean_mean"}

DEFAULT_NORMAL_SUBJECT_IDS = (
    list(range(125, 133))
    + list(range(225, 233))
    + list(range(325, 333))
)
DEFAULT_UNIFORM_SUBJECT_IDS = (
    list(range(101, 125))
    + list(range(201, 225))
    + list(range(301, 325))
)


@lru_cache(maxsize=None)
def _load_csv_cached(summary_path: str) -> pd.DataFrame:
    csv_path = Path(summary_path)
    if not csv_path.exists():
        raise ValueError(f"Required dataset is missing: {csv_path}")
    return pd.read_csv(csv_path)


def _compute_subject_stats_from_summary(
    summary_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing_cols = [col for col in SUMMARY_REQUIRED_COLUMNS if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(
            "Task1b error summary is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    data = summary_df.copy()
    data["feature_name"] = data["feature_name"].astype(str).str.strip().str.lower()
    invalid_names = sorted(set(data["feature_name"]) - set(FEATURE_NAMES))
    if invalid_names:
        raise ValueError(f"Task1b error summary has unknown feature_name values: {invalid_names}")

    grouped = (
        data.groupby(["iSub", "feature_name"], as_index=False)[["error_mean", "error_std"]]
        .mean()
    )

    for sub_id, sub_df in grouped.groupby("iSub"):
        missing_features = sorted(set(FEATURE_NAMES) - set(sub_df["feature_name"]))
        if missing_features:
            raise ValueError(
                f"Subject {int(sub_id)} is missing features in Task1b summary: {missing_features}"
            )

    mean_df = (
        grouped.pivot(index="iSub", columns="feature_name", values="error_mean")
        .reindex(columns=FEATURE_NAMES)
    )
    std_df = (
        grouped.pivot(index="iSub", columns="feature_name", values="error_std")
        .reindex(columns=FEATURE_NAMES)
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


def _resolve_data_paths(processed_data_dir: Path | str | None) -> tuple[Path, Path, Path]:
    if processed_data_dir is None:
        return (
            TASK1B_ERRORSUMMARY_PATH.resolve(),
            TASK1B_ERRORSUMMARY_72_PATH.resolve(),
            TASK2_PROCESSED_PATH.resolve(),
        )
    resolved_dir = Path(processed_data_dir).resolve()
    return (
        (resolved_dir / TASK1B_ERRORSUMMARY_PATH.name).resolve(),
        (resolved_dir / TASK1B_ERRORSUMMARY_72_PATH.name).resolve(),
        (resolved_dir / TASK2_PROCESSED_PATH.name).resolve(),
    )


def _get_perception_noise_stats(
    processed_data_dir: Path | str | None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    summary_path, _, task2_path = _resolve_data_paths(processed_data_dir)

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


def _get_uniform_threshold_stats(
    processed_data_dir: Path | str | None,
) -> Dict[int, np.ndarray]:
    _, summary72_path, task2_path = _resolve_data_paths(processed_data_dir)

    summary72_df = _load_csv_cached(str(summary72_path))
    task2_df = _load_csv_cached(str(task2_path))
    feature_orders = _extract_feature_orders(task2_df)

    missing_cols = [c for c in SUMMARY72_REQUIRED_COLUMNS if c not in summary72_df.columns]
    if missing_cols:
        raise ValueError(
            "Task1b 72-subject summary is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    data = summary72_df.copy()
    data["feature_name"] = data["feature_name"].astype(str).str.strip().str.lower()
    invalid_names = sorted(set(data["feature_name"]) - set(FEATURE_NAMES))
    if invalid_names:
        raise ValueError(f"Task1b 72-subject summary has unknown feature_name values: {invalid_names}")

    grouped = (
        data.groupby(["iSub", "feature_name"], as_index=False)["threshold_mean_mean"]
        .mean()
    )

    threshold_df = (
        grouped.pivot(index="iSub", columns="feature_name", values="threshold_mean_mean")
        .reindex(columns=FEATURE_NAMES)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    threshold_map: Dict[int, np.ndarray] = {}
    for sub_id in threshold_df.index:
        i_sub = int(sub_id)
        if i_sub not in feature_orders:
            raise ValueError(f"Subject {i_sub} not found in Task2 feature order data")
        order = feature_orders[i_sub]
        values = threshold_df.loc[sub_id, FEATURE_NAMES].to_dict()
        vec = np.array([values[name] for name in order], dtype=float)
        threshold_map[i_sub] = np.abs(np.nan_to_num(vec, nan=0.0))

    if not threshold_map:
        raise ValueError("Failed to compute uniform-threshold statistics for any subject")
    return threshold_map




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
        normal_subject_ids = self._normalize_subject_ids(
            kwargs.pop("normal_subject_ids", DEFAULT_NORMAL_SUBJECT_IDS)
        )
        uniform_subject_ids = self._normalize_subject_ids(
            kwargs.pop("uniform_subject_ids", DEFAULT_UNIFORM_SUBJECT_IDS)
        )

        mean_value = kwargs.pop("mean", None)
        std_value = kwargs.pop("std", None)
        self.noise_mode = "normal"
        self.uniform_half_range = None

        if mean_value is None or std_value is None:
            if self.subject_id is None:
                raise ValueError(
                    "PerceptionModule requires 'subject_id' when mean/std are not provided."
                )
            sid = int(self.subject_id)
            noise_mode = self._resolve_subject_noise_mode(
                sid,
                normal_subject_ids=normal_subject_ids,
                uniform_subject_ids=uniform_subject_ids,
            )
            if noise_mode == "uniform":
                half_range = self._load_uniform_half_range(sid, processed_data_dir)
                self.noise_mode = "uniform"
                self.uniform_half_range = np.abs(
                    self._coerce_vector(half_range, "uniform_half_range")
                )
                if mean_value is None:
                    mean_value = np.zeros(self.features, dtype=float)
                if std_value is None:
                    std_value = np.zeros(self.features, dtype=float)
            else:
                auto_mean, auto_std = self._load_subject_stats(sid, processed_data_dir)
                if mean_value is None:
                    mean_value = auto_mean
                if std_value is None:
                    std_value = auto_std

        self.mean = self._coerce_vector(mean_value, "mean")
        self.std = np.abs(self._coerce_vector(std_value, "std"))

    @staticmethod
    def _normalize_subject_ids(values: Iterable[int] | None) -> set[int]:
        if values is None:
            return set()
        return {int(v) for v in values}

    @staticmethod
    def _resolve_subject_noise_mode(
        subject_id: int,
        normal_subject_ids: set[int],
        uniform_subject_ids: set[int],
    ) -> str:
        if subject_id in uniform_subject_ids:
            return "uniform"
        if subject_id in normal_subject_ids:
            return "normal"
        return "normal"

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

    @staticmethod
    def _load_uniform_half_range(subject_id: int, processed_data_dir: Path | str | None):
        try:
            threshold_map = _get_uniform_threshold_stats(processed_data_dir)
        except ValueError as exc:
            raise ValueError(
                f"Failed to load uniform-threshold statistics from {processed_data_dir}"
            ) from exc

        if subject_id not in threshold_map:
            raise ValueError(
                f"Subject {subject_id} does not exist in Task1b_errorsummary_72 statistics"
            )
        return threshold_map[subject_id]

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
        if self.noise_mode == "uniform":
            assert self.uniform_half_range is not None
            low = -self.uniform_half_range
            high = self.uniform_half_range
            noise = np.random.uniform(low=low, high=high, size=stimu.shape)
        else:
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
