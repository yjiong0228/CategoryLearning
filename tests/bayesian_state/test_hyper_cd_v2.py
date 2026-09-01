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
    initial_points_override: list[dict[str, int]] | None = None,
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
    search_kwargs = {}
    if initial_points_override is not None:
        search_kwargs["initial_points"] = initial_points_override
    _, restarts, best = optimizer._coordinate_descent(
        stage_name="coarse",
        stage_sim_cfg={},
        subjects=[101],
        space=space,
        all_combinations_path=tmp_path / "all_combinations.jsonl",
        coordinate_trace_path=trace_path,
        rng=random.Random(7),
        **search_kwargs,
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


def test_schema_v2_rejects_unknown_fine_initialization(tmp_path: Path) -> None:
    """Catch schema-v2 fine search silently falling back to legacy starts."""

    config, config_path = _minimal_hyper_config(tmp_path)
    config["cd"]["resume_mode"] = "explicit"
    config["refine_policy"] = {"fine_initialization": "random"}

    with pytest.raises(ValueError, match="fine_initialization"):
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


def test_fine_initial_points_are_ranked_projected_and_deduplicated() -> None:
    """Catch fine search restarting randomly or duplicating projected coarse points."""

    optimizer = object.__new__(HyperCDOptimizer)
    optimizer.config = {"refine_policy": {"top_k": 3}}
    optimizer.objective_order = resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )

    def coarse(index, error, joint, x):
        return CombinationResult(
            stage="coarse",
            combination_index=index,
            hyperparams={"joint": joint, "x": x},
            aggregated_error=error,
            objective_values={"simulation.mean_error": error},
            subject_metrics={},
            hyper_candidate_seed=index,
            restart_id=0,
            iter_id=1,
            coordinate="x",
        )

    combinations = [
        coarse(1, 0.80, {"M": 3, "chi": 0}, 0.49),
        coarse(2, 0.90, {"M": 3, "chi": 1}, 0.74),
        coarse(3, 1.00, {"M": 3, "chi": 0}, 0.51),
    ]
    fine_space = {
        "joint": [{"M": 3, "chi": 0}, {"M": 3, "chi": 1}],
        "x": [0.50, 0.75],
    }

    assert optimizer._fine_initial_points(combinations, fine_space) == [
        {"joint": {"M": 3, "chi": 0}, "x": 0.50},
        {"joint": {"M": 3, "chi": 1}, "x": 0.75},
    ]


def test_stage_initial_points_override_global_restart_configuration(
    tmp_path: Path,
) -> None:
    """Catch fine restarts accidentally reusing the global coarse starts."""

    _, restarts, _ = _run_surface_search(
        tmp_path,
        surface={(0,): 2.0, (1,): 1.0},
        initial_point={"x": 0},
        space={"x": [0, 1]},
        min_delta=0.0,
        initial_points_override=[{"x": 1}, {"x": 0}],
    )

    assert len(restarts) == 2
    assert [row["initial_error"] for row in restarts] == [1.0, 2.0]


def test_all_stage_pipeline_passes_projected_coarse_shortlist_to_fine(
    tmp_path: Path,
) -> None:
    """Catch the pipeline computing fine starts but failing to use them."""

    class FineInvocationObserved(RuntimeError):
        pass

    optimizer = object.__new__(HyperCDOptimizer)
    optimizer.config = {
        "stages": {
            "coarse": {"hyperparam_space": {"x": {"values": [0.49, 0.74]}}},
            "fine": {"hyperparam_space": {"x": {"values": [0.50, 0.75]}}},
        },
        "refine_policy": {"top_k": 1},
    }
    optimizer.cd_v2 = cd_v2.CDV2Config(
        enabled=True,
        resume_mode="explicit",
        checkpoint_every_coordinate=True,
        fine_initialization="coarse_shortlist",
    )
    optimizer.hyper_base_seed = 20260901
    optimizer._prepare_stage_config = lambda stage_name: {}
    optimizer._combination_counter = 0
    optimizer.objective_order = resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )
    coarse = CombinationResult(
        stage="coarse",
        combination_index=0,
        hyperparams={"x": 0.49},
        aggregated_error=0.8,
        objective_values={"simulation.mean_error": 0.8},
        subject_metrics={},
        hyper_candidate_seed=1,
        restart_id=0,
        iter_id=1,
        coordinate="x",
    )

    def coordinate_descent(**kwargs):
        if kwargs["stage_name"] == "coarse":
            return [coarse], [], coarse
        assert kwargs["initial_points"] == [{"x": 0.50}]
        raise FineInvocationObserved

    optimizer._coordinate_descent = coordinate_descent

    with pytest.raises(FineInvocationObserved):
        optimizer._run_pipeline(
            subjects=[101],
            stage="all",
            output_dir=tmp_path,
        )
