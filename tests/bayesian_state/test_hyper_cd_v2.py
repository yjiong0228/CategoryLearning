from __future__ import annotations

from pathlib import Path
import json
import random

import pytest
import yaml

from src.Bayesian_state.optimization.search.coordinate_descent import (
    CombinationResult,
    HyperCDOptimizer,
)
from src.Bayesian_state.optimization.search import cd_v2
from src.Bayesian_state.optimization.objectives import resolve_objective_order


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


def _run_surface_search(
    tmp_path: Path,
    *,
    surface: dict[tuple[int, ...], float],
    initial_point: dict[str, int],
    space: dict[str, list[int]],
    min_delta: float,
) -> tuple[CombinationResult, list[dict], Path]:
    optimizer = object.__new__(HyperCDOptimizer)
    optimizer.n_restarts = 1
    optimizer.max_outer_iters = 3
    optimizer.coordinate_order = "fixed"
    optimizer.patience = 1
    optimizer.min_delta = min_delta
    optimizer.init_strategy = "anchor"
    optimizer.anchor = {}
    optimizer.initial_points = [initial_point]
    optimizer.objective_order = resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )
    optimizer.save_level = "compact"
    optimizer._combination_counter = 0
    optimizer._coordinate_parallel_plan = lambda *args: (1, 1)

    coordinate_names = list(space)

    def evaluate_entries(**kwargs):
        rows = []
        for entry in kwargs["missing_entries"]:
            point = dict(entry["point"])
            error = surface[tuple(int(point[name]) for name in coordinate_names)]
            rows.append(
                CombinationResult(
                    stage=str(kwargs["stage_name"]),
                    combination_index=int(entry["combination_index"]),
                    hyperparams=point,
                    aggregated_error=error,
                    objective_values={"simulation.mean_error": error},
                    subject_metrics={},
                    hyper_candidate_seed=1,
                    restart_id=int(kwargs["restart_id"]),
                    iter_id=int(kwargs["iter_id"]),
                    coordinate=str(kwargs["coordinate"]),
                )
            )
        return rows, {
            "flat_task_count": len(rows),
            "flat_jobs": 1,
            "parallel_backend": "test",
            "planned_total_jobs": 1,
        }

    optimizer._evaluate_missing_entries_flat = evaluate_entries
    trace_path = tmp_path / "coordinate_trace.jsonl"
    _, restarts, best = optimizer._coordinate_descent(
        stage_name="coarse",
        stage_sim_cfg={},
        subjects=[101],
        space=space,
        all_combinations_path=tmp_path / "all_combinations.jsonl",
        coordinate_trace_path=trace_path,
        rng=random.Random(7),
    )
    return best, restarts, trace_path


def test_schema_v2_requires_explicit_resume_mode(tmp_path: Path) -> None:
    """Catch schema-v2 searches that could silently inherit legacy overwrite behavior."""

    config, config_path = _minimal_hyper_config(tmp_path)

    with pytest.raises(ValueError, match="cd.resume_mode"):
        HyperCDOptimizer(config, config_path)


def test_schema_v2_rejects_negative_min_delta(tmp_path: Path) -> None:
    """Catch a threshold whose sign would make every ordered improvement movable."""

    config, config_path = _minimal_hyper_config(tmp_path)
    config["cd"]["resume_mode"] = "explicit"
    config["cd"]["min_delta"] = -0.1

    with pytest.raises(ValueError, match="cd.min_delta"):
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


def test_candidate_improvement_enforces_primary_min_delta() -> None:
    """Catch PF-noise-sized improvements being accepted as coordinate moves."""

    specs = resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )
    current = {"simulation.mean_error": 1.0}

    assert not cd_v2.candidate_improves(
        current,
        {"simulation.mean_error": 0.99995},
        specs,
        min_delta=0.0001,
    )
    assert cd_v2.candidate_improves(
        current,
        {"simulation.mean_error": 0.9998},
        specs,
        min_delta=0.0001,
    )


def test_coordinate_descent_does_not_move_for_subthreshold_improvement(
    tmp_path: Path,
) -> None:
    """Catch the optimizer bypassing its parsed min_delta during a real scan."""

    best, _, trace_path = _run_surface_search(
        tmp_path,
        surface={(0,): 1.0, (1,): 0.99995},
        initial_point={"x": 0},
        space={"x": [0, 1]},
        min_delta=0.0001,
    )

    assert best.hyperparams == {"x": 0}
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert trace[0]["min_delta_reject_count"] == 1


def test_coordinate_descent_revisits_earlier_coordinate_on_second_sweep(
    tmp_path: Path,
) -> None:
    """Protect the multi-sweep search needed when later coordinates unlock earlier ones."""

    best, restarts, _ = _run_surface_search(
        tmp_path,
        surface={(0, 0): 3.0, (1, 0): 3.0, (0, 1): 2.0, (1, 1): 1.0},
        initial_point={"x": 0, "y": 0},
        space={"x": [0, 1], "y": [0, 1]},
        min_delta=0.0,
    )

    assert best.hyperparams == {"x": 1, "y": 1}
    assert restarts[0]["num_improvements"] == 2
    assert restarts[0]["outer_iters_completed"] >= 2
