from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.optimization.search.coordinate_descent import (
    HyperCDOptimizer,
)
from src.Bayesian_state.optimization.search import cd_v2


def _minimal_hyper_config(tmp_path: Path) -> tuple[dict, Path]:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump({"subjects": [101]}), encoding="utf-8")
    config_path = tmp_path / "hyper.yaml"
    return (
        {
            "search_schema_version": 2,
            "base_sim_config_path": str(base_path),
            "output_dir": str(tmp_path / "output"),
            "hyper_base_seed": 20260901,
            "objective_order": [{"path": "simulation.mean_error"}],
            "hyperparam_space": {"engine.value": {"values": [0, 1]}},
            "stages": {"coarse": {"simulation_overrides": {}}},
            "cd": {
                "n_restarts": 1,
                "max_outer_iters": 2,
                "patience": 1,
                "min_delta": 0.0,
                "coordinate_order": "fixed",
                "parallel_budget": 1,
                "checkpoint_every_coordinate": True,
            },
        },
        config_path,
    )


def test_schema_v2_requires_explicit_resume_mode(tmp_path: Path) -> None:
    """Catch schema-v2 searches that could silently inherit legacy overwrite behavior."""

    config, config_path = _minimal_hyper_config(tmp_path)

    with pytest.raises(ValueError, match="cd.resume_mode"):
        HyperCDOptimizer(config, config_path)


def test_fine_projection_keeps_mapping_exact_and_uses_nearest_numeric_value() -> None:
    """Catch fine initialization that interpolates or decomposes a joint structure."""

    point = {"joint": {"M": 3, "chi": 1}, "x": 0.49}
    space = {
        "joint": [{"M": 2, "chi": 0}, {"M": 3, "chi": 1}],
        "x": [0.25, 0.50, 0.75],
    }

    assert cd_v2.project_point_to_space(point, space) == {
        "joint": {"M": 3, "chi": 1},
        "x": 0.50,
    }
