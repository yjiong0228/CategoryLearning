from __future__ import annotations

from pathlib import Path

import yaml

from src.Bayesian_state.model.modules.perception import (
    _get_perception_noise_stats,
    _get_uniform_threshold_stats,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths


ROOT = Path(__file__).resolve().parents[3]
SIMULATION_CONFIG = (
    ROOT / "configs/simulation_cfg/model0826_cond1_recovery_base.yaml"
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
