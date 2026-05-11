"""Dataset path resolution for Bayesian_state configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .paths import (
    PROCESSED_DATA_DIR,
    TASK1B_ERRORSUMMARY_PATH,
    TASK1B_ERRORSUMMARY_72_PATH,
    TASK2_PROCESSED_PATH,
)

DATASET_FILE_KEYS = (
    "learning_data",
    "perception_summary",
    "perception_summary_72",
    "feature_order_data",
)


def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _resolve_dataset_file(processed_dir: Path, value: Any | None, default_name: str) -> Path:
    if value is None:
        return (processed_dir / default_name).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = (processed_dir / path).resolve()
    return path


def resolve_dataset_paths(cfg: Mapping[str, Any], yaml_dir: Path, default_learning_data: Path = TASK2_PROCESSED_PATH) -> dict[str, Path]:
    """Resolve dataset config into concrete paths.

    New style:
      dataset:
        processed_dir: ../../data_meg/processed
        learning_data: Task3b_processed.csv
        perception_summary: Task1b_errorsummary.csv
        perception_summary_72: Task1b_errorsummary_72.csv
        feature_order_data: Task3b_processed.csv

    Legacy style:
      data_path: ../../data/processed/Task2_processed_new.csv
    """
    dataset = cfg.get("dataset") or {}
    if dataset and not isinstance(dataset, Mapping):
        raise ValueError("dataset must be a mapping when provided")

    legacy_data_path = cfg.get("data_path")
    if dataset.get("processed_dir") is not None:
        processed_dir = _resolve_path(yaml_dir, dataset["processed_dir"])
    elif legacy_data_path is not None:
        processed_dir = _resolve_path(yaml_dir, legacy_data_path).parent
    else:
        processed_dir = PROCESSED_DATA_DIR.resolve()

    if dataset.get("learning_data") is not None:
        learning_data = _resolve_dataset_file(processed_dir, dataset.get("learning_data"), default_learning_data.name)
    elif legacy_data_path is not None:
        learning_data = _resolve_path(yaml_dir, legacy_data_path)
    else:
        learning_data = default_learning_data.resolve()

    return {
        "processed_dir": processed_dir,
        "learning_data": learning_data,
        "perception_summary": _resolve_dataset_file(
            processed_dir,
            dataset.get("perception_summary"),
            TASK1B_ERRORSUMMARY_PATH.name,
        ),
        "perception_summary_72": _resolve_dataset_file(
            processed_dir,
            dataset.get("perception_summary_72"),
            TASK1B_ERRORSUMMARY_72_PATH.name,
        ),
        "feature_order_data": _resolve_dataset_file(
            processed_dir,
            dataset.get("feature_order_data"),
            TASK2_PROCESSED_PATH.name,
        ),
    }
