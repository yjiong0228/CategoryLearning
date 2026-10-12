from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.model.modules.perception import (
    DEFAULT_NORMAL_SUBJECT_IDS,
    DEFAULT_UNIFORM_SUBJECT_IDS,
    PerceptionModule,
    _get_perception_noise_stats,
    _get_uniform_threshold_stats,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths


ROOT = Path(__file__).resolve().parents[3]
SIMULATION_CONFIG = (
    ROOT / "configs/exp123/simulation_cfg/model0826_cond1_recovery_base.yaml"
)


def test_perception_statistics_are_cached_as_read_only_maps() -> None:
    config = yaml.safe_load(SIMULATION_CONFIG.read_text(encoding="utf-8"))
    paths = resolve_dataset_paths(config, SIMULATION_CONFIG.parent)

    first_uniform = _get_uniform_threshold_stats(
        paths["processed_dir"], paths
    )
    second_uniform = _get_uniform_threshold_stats(
        paths["processed_dir"], paths
    )
    first_normal = _get_perception_noise_stats(paths["processed_dir"], paths)
    second_normal = _get_perception_noise_stats(paths["processed_dir"], paths)

    assert first_uniform is second_uniform
    assert first_normal is second_normal
    assert all(not values.flags.writeable for values in first_uniform.values())
    for statistic_map in first_normal:
        assert all(not values.flags.writeable for values in statistic_map.values())


@pytest.mark.parametrize(
    ("subject_id", "expected"),
    [
        (101, "uniform"), (125, "normal"), (332, "normal"),
        (333, "uniform"), (335, "uniform"), (336, "uniform"),
        (401, "uniform"), (425, "uniform"), (501, "uniform"),
    ],
)
def test_default_perception_noise_changes_at_subject_333(
    subject_id: int, expected: str
) -> None:
    assert PerceptionModule._resolve_subject_noise_mode(
        subject_id, set(DEFAULT_NORMAL_SUBJECT_IDS), set(DEFAULT_UNIFORM_SUBJECT_IDS)
    ) == expected


def test_explicit_subject_group_can_preserve_normal_noise() -> None:
    assert PerceptionModule._resolve_subject_noise_mode(
        333, {333}, set(DEFAULT_UNIFORM_SUBJECT_IDS)
    ) == "normal"


@pytest.mark.parametrize(
    ("experiment", "subject_id", "learning_file"),
    [
        ("meg", 333, "Task3b_processed.csv"),
        ("meg", 334, "Task3b_processed.csv"),
        ("meg", 335, "Task3b_processed.csv"),
        ("exp4", 401, "Task2_processed.csv"),
        ("exp4", 402, "Task2_processed.csv"),
        ("exp4", 403, "Task2_processed.csv"),
    ],
)
def test_auto_noise_loads_meg_and_exp4_threshold_half_widths(
    experiment: str, subject_id: int, learning_file: str
) -> None:
    processed_dir = ROOT / "data" / experiment / "processed"
    paths = resolve_dataset_paths(
        {
            "dataset": {
                "processed_dir": str(processed_dir),
                "learning_data": learning_file,
                "perception_summary_72": "Task1b_errorsummary.csv",
                "feature_order_data": learning_file,
            }
        },
        ROOT,
    )
    module = PerceptionModule(
        SimpleNamespace(
            subject_id=subject_id,
            processed_data_dir=processed_dir,
            dataset_paths=paths,
        ),
        module_seed=123,
    )

    summary = pd.read_csv(paths["perception_summary_72"])
    subject_summary = summary[summary["iSub"] == subject_id]
    feature_order = pd.read_csv(paths["feature_order_data"])
    order = feature_order.loc[
        feature_order["iSub"] == subject_id,
        [f"feature{i}_name" for i in range(1, 5)],
    ].iloc[0]
    expected = np.array([
        subject_summary.loc[
            subject_summary["feature_name"] == name, "threshold_mean_mean"
        ].mean()
        for name in order
    ])

    assert module.noise_mode == "uniform"
    np.testing.assert_allclose(module.uniform_half_range, expected)
    assert np.isfinite(expected).all() and np.all(expected > 0)
    stimulus = np.full(4, 0.5)
    noise = np.random.default_rng(123).uniform(-expected, expected)
    np.testing.assert_allclose(module.sample(stimulus), stimulus + noise)
    for _ in range(32):
        sampled_noise = module.sample(stimulus) - stimulus
        assert np.all(np.abs(sampled_noise) <= expected)
