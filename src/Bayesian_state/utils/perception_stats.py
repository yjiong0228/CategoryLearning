"""Utility helpers to precompute perception noise statistics.

This module computes subject-specific mean and standard deviation values for the
perception noise model from the preprocessed behavioural datasets.  The logic
re-implements the legacy Bayesian pipeline so the new state-based code can
re-use the same statistics without relying on external inputs.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd  # type: ignore[import]

FEATURE_COLUMNS = ["neck_length", "head_length", "leg_length", "tail_length"]
FEATURE_NAMES = ["neck", "head", "leg", "tail"]


class PerceptionStatsError(RuntimeError):
    """Raised when perception statistics cannot be computed."""


def _resolve_processed_dir(processed_data_dir: Path | str | None) -> Path:
    path = Path(processed_data_dir) if processed_data_dir is not None else None
    if path is None:
        raise PerceptionStatsError("processed_data_dir must be provided")
    if not path.exists():
        raise PerceptionStatsError(f"Processed data directory not found: {path}")
    return path


@lru_cache(maxsize=None)
def _load_dataframe(processed_data_dir: Path, filename: str) -> pd.DataFrame:
    csv_path = processed_data_dir / filename
    if not csv_path.exists():
        raise PerceptionStatsError(f"Required dataset is missing: {csv_path}")
    return pd.read_csv(csv_path)


def _compute_error_dataframe(task1b_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-subject error deltas between adjustment and target trials."""
    errors = []
    # Ensure consistent column ordering
    feature_cols = FEATURE_COLUMNS

    for i_sub, group in task1b_df.groupby("iSub"):
        target = (
            group[group["type"] == "target"][feature_cols]
            .reset_index(drop=True)
        )
        adjust_after = (
            group[group["type"] == "adjust_after"][feature_cols]
            .reset_index(drop=True)
        )
        if target.empty or adjust_after.empty:
            # Skip malformed entries while keeping diagnostics useful.
            continue
        if len(target) != len(adjust_after):
            raise PerceptionStatsError(
                f"Mismatched trial counts for subject {i_sub} in Task1b data"
            )

        diff = adjust_after - target
        diff.columns = FEATURE_NAMES
        diff.insert(0, "iSub", i_sub)
        errors.append(diff)

    if not errors:
        raise PerceptionStatsError("Task1b dataset does not contain valid trials")

    return pd.concat(errors, ignore_index=True)


def _extract_structures(task2_df: pd.DataFrame) -> Dict[int, Tuple[int, int]]:
    """Map subject ids to their (structure1, structure2) pair."""
    structures: Dict[int, Tuple[int, int]] = {}
    for i_sub, group in task2_df.groupby("iSub"):
        values = group[["structure1", "structure2"]].drop_duplicates()
        if values.empty:
            continue
        if len(values) != 1:
            raise PerceptionStatsError(
                f"Subject {i_sub} has inconsistent structure annotations"
            )
        structure_pair = tuple(int(v) for v in values.iloc[0].tolist())
        structures[i_sub] = structure_pair  # type: ignore[assignment]
    if not structures:
        raise PerceptionStatsError("No subject structures found in Task2 data")
    return structures


def _convert_structure(structure: Tuple[int, int]) -> Iterable[str]:
    """Replicate the legacy feature ordering given a subject structure."""
    struct1, struct2 = structure

    if struct1 == 1:
        features = ["neck", "head", "leg", "tail"]
    elif struct1 == 2:
        features = ["neck", "head", "tail", "leg"]
    elif struct1 == 3:
        features = ["neck", "leg", "tail", "head"]
    elif struct1 == 4:
        features = ["head", "leg", "tail", "neck"]
    else:
        raise PerceptionStatsError(f"Unknown structure1 value: {struct1}")

    if struct2 == 1:
        features = features[:]
    elif struct2 == 2:
        features = [features[0], features[2], features[1], features[3]]
    elif struct2 == 3:
        features = [features[1], features[0], features[2], features[3]]
    elif struct2 == 4:
        features = [features[1], features[2], features[0], features[3]]
    elif struct2 == 5:
        features = [features[2], features[0], features[1], features[3]]
    elif struct2 == 6:
        features = [features[2], features[1], features[0], features[3]]
    else:
        raise PerceptionStatsError(f"Unknown structure2 value: {struct2}")

    # Final rearrangement copied from the legacy implementation.
    features = [features[0], features[2], features[1], features[3]]
    return features


def get_perception_noise_stats(
    processed_data_dir: Path | str | None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Compute subject-wise mean and std arrays for perception noise.

    Parameters
    ----------
    processed_data_dir: Path | str | None
        Directory that contains the "Task1b_processed.csv" and
        "Task2_processed.csv" files.

    Returns
    -------
    Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]
        Two dictionaries keyed by subject id that map to ordered numpy arrays of
        mean and standard deviation values respectively.
    """

    resolved_dir = _resolve_processed_dir(processed_data_dir)

    task1b_df = _load_dataframe(resolved_dir, "Task1b_processed.csv")
    task2_df = _load_dataframe(resolved_dir, "Task2_processed.csv")

    error_df = _compute_error_dataframe(task1b_df)
    mean_df = error_df.groupby("iSub")[FEATURE_NAMES].mean().fillna(0.0)
    std_df = error_df.groupby("iSub")[FEATURE_NAMES].std().fillna(0.0)

    structures = _extract_structures(task2_df)

    mean_map: Dict[int, np.ndarray] = {}
    std_map: Dict[int, np.ndarray] = {}

    for sub_id, structure in structures.items():
        if sub_id not in mean_df.index:
            # Skip subjects without Task1b stats – keep behaviour consistent with
            # the original implementation by omitting them silently.
            continue
        feature_order = list(_convert_structure(structure))
        subject_mean = mean_df.loc[sub_id].to_dict()
        subject_std = std_df.loc[sub_id].to_dict()

        mean_vector = np.array([subject_mean.get(name, 0.0) for name in feature_order], dtype=float)
        std_vector = np.array([subject_std.get(name, 0.0) for name in feature_order], dtype=float)

        mean_map[sub_id] = np.nan_to_num(mean_vector, nan=0.0)
        std_map[sub_id] = np.nan_to_num(std_vector, nan=0.0)

    if not mean_map:
        raise PerceptionStatsError(
            "Failed to compute perception statistics for any subject"
        )

    return mean_map, std_map
