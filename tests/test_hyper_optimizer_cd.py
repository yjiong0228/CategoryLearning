from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

import src.Bayesian_state.hyper_cd.optimizer as cd_optimizer
from src.Bayesian_state.hyper_cd.optimizer import CombinationResult, HyperCDOptimizer
from src.Bayesian_state.run_hyper_then_simulation import build_hyper_selector


@pytest.fixture
def tmp_path() -> Path:
    root = Path(tempfile.gettempdir()) / "catelearn_test_tmp"
    root.mkdir(exist_ok=True)
    path = root / f"hyper_cd_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _strategy_candidates_payload() -> dict:
    return {
        "cond23": [
            {
                "id": "small",
                "hypo_transitions_kwargs": {
                    "init_num": 6,
                    "max_active_hypotheses": 6,
                    "strategies": [
                        {"label": "retain", "amount": "confidence_4", "method": "epsilon_posterior", "pool": "active"}
                    ],
                },
            },
            {
                "id": "large",
                "hypo_transitions_kwargs": {
                    "init_num": 12,
                    "max_active_hypotheses": 10,
                    "strategies": [
                        {"label": "retain", "amount": "entropy_norm_8", "method": "temperature_posterior", "pool": "active"}
                    ],
                },
            },
        ]
    }


def _build_min_cd_config(tmp_path: Path) -> Path:
    sim_cfg = {
        "engine_config": {
            "agenda": ["memory_mod"],
            "modules": {"memory_mod": {"class": "x", "kwargs": {"gamma": 0.5, "w0": 0.1}}},
        },
        "subjects": [1],
        "window_size": 8,
        "loss_metric": "accuracy_curve_mse",
        "simulation_repeats": 1,
    }
    _write_yaml(tmp_path / "sim.yaml", sim_cfg)

    cd_cfg = {
        "base_sim_config_path": "sim.yaml",
        "subjects": [1],
        "output_dir": "./out_cd",
        "selection_metric": "mean_simulation_error",
        "save_level": "compact",
        "hyper_base_seed": 42,
        "cd": {
            "n_restarts": 2,
            "max_outer_iters": 2,
            "init_strategy": "anchor",
            "anchor": {"simulation.window_size": 8, "engine.modules.beta_mod.kwargs.beta_init": 1.0},
            "coordinate_order": "fixed",
            "patience": 1,
            "min_delta": 0.0,
        },
        "hyperparam_space": {
            "simulation.window_size": {"values": [6, 8]},
            "engine.modules.beta_mod.kwargs.beta_init": {"values": [1.0, 2.0]},
        },
        "stages": {
            "coarse": {
                "cd_parallel": {"max_repeat_jobs": 1},
                "simulation_overrides": {"simulation_repeats": 1},
            },
            "fine": {
                "cd_parallel": {"max_repeat_jobs": 1},
                "simulation_overrides": {"simulation_repeats": 1},
            },
        },
        "refine_policy": {"top_k": 2},
    }
    cd_path = tmp_path / "hyper_cd.yaml"
    _write_yaml(cd_path, cd_cfg)
    return cd_path


def test_cd_outputs_combination_schema(tmp_path: Path, monkeypatch) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    opt = HyperCDOptimizer(cfg, cd_path)

    def _fake_eval(stage_name, point, stage_sim_cfg, subjects, restart_id, iter_id, coordinate, combination_index):
        score = float(point.get("simulation.window_size", 0)) + float(
            point.get("engine.modules.beta_mod.kwargs.beta_init", 0)
        )
        return CombinationResult(
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=dict(point),
            aggregated_error=score,
            subject_metrics={1: {"mean_error": score, "best_error": score, "std_error": 0.0, "simulation_repeats": 1}},
            hyper_candidate_seed=100 + combination_index,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    monkeypatch.setattr(opt, "_evaluate_point_with_index", _fake_eval)

    result = opt.run(subjects=[1], stage="all")
    subject_out = result["per_subject_outputs"]["1"]
    all_combinations = Path(subject_out["all_combinations"])
    coordinate_trace = Path(subject_out["coordinate_trace"])
    stage_summary = json.loads(Path(subject_out["stage_summary"]).read_text(encoding="utf-8"))
    restart_summary = json.loads(Path(subject_out["restart_summary"]).read_text(encoding="utf-8"))
    subject_best = json.loads(Path(subject_out["best_hyperparams"]).read_text(encoding="utf-8"))
    root_best = json.loads(Path(result["best_hyperparams"]).read_text(encoding="utf-8"))

    first_line = json.loads(all_combinations.read_text(encoding="utf-8").splitlines()[0])
    assert "combination_index" in first_line
    assert "trial_index" not in first_line
    assert "top_combinations" in stage_summary["coarse"]
    assert "num_combinations" in stage_summary["coarse"]
    assert coordinate_trace.exists()
    first_trace = json.loads(coordinate_trace.read_text(encoding="utf-8").splitlines()[0])
    assert "coordinate" in first_trace
    assert "new_evaluations" in first_trace
    assert "cache_hits" in first_trace
    assert "value_jobs" in first_trace
    assert "repeat_jobs" in first_trace
    assert "best_combination_index" in restart_summary["coarse"][0]
    assert "initial_combination_index" in restart_summary["coarse"][0]
    assert "outer_iters_completed" in restart_summary["coarse"][0]
    assert "stopped_by" in restart_summary["coarse"][0]
    assert "num_improvements" in restart_summary["coarse"][0]
    assert "num_new_evaluations" in restart_summary["coarse"][0]
    assert "best_combination_index" in subject_best
    assert subject_best["hyper_backend"] == "hyper_cd"
    assert root_best["hyper_backend"] == "hyper_cd"
    assert "per_subject_best" in root_best


def test_cd_requires_stage_max_repeat_jobs(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    del cfg["stages"]["coarse"]["cd_parallel"]
    opt = HyperCDOptimizer(cfg, cd_path)

    with pytest.raises(ValueError, match=r"stages\.coarse\.cd_parallel\.max_repeat_jobs"):
        opt._coordinate_parallel_plan("coarse", {"simulation_repeats": 1}, 1)


def test_cd_coordinate_parallel_plan_prioritizes_repeat_jobs(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["cd"]["parallel_budget"] = 4
    cfg["stages"]["coarse"]["cd_parallel"]["max_repeat_jobs"] = 2
    opt = HyperCDOptimizer(cfg, cd_path)

    assert opt._coordinate_parallel_plan("coarse", {"simulation_repeats": 4}, 3) == (2, 2)
    assert opt._coordinate_parallel_plan("coarse", {"simulation_repeats": 1}, 3) == (3, 1)


def test_cd_coarse_and_fine_max_repeat_jobs_are_stage_specific(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["cd"]["parallel_budget"] = 8
    cfg["stages"]["coarse"]["cd_parallel"]["max_repeat_jobs"] = 2
    cfg["stages"]["fine"]["cd_parallel"]["max_repeat_jobs"] = 4
    opt = HyperCDOptimizer(cfg, cd_path)

    assert opt._coordinate_parallel_plan("coarse", {"simulation_repeats": 8}, 10) == (4, 2)
    assert opt._coordinate_parallel_plan("fine", {"simulation_repeats": 8}, 10) == (2, 4)


def test_cd_coordinate_values_parallelize_with_runtime_repeat_jobs(tmp_path: Path, monkeypatch) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["cd"].update(
        {
            "parallel_budget": 4,
            "n_restarts": 1,
            "max_outer_iters": 1,
            "init_strategy": "anchor",
            "anchor": {"x": 1},
            "patience": 1,
        }
    )
    cfg["stages"]["coarse"]["cd_parallel"]["max_repeat_jobs"] = 2
    opt = HyperCDOptimizer(cfg, cd_path)
    seen: list[tuple[int, int, int]] = []

    def _fake_eval(stage_name, point, stage_sim_cfg, subjects, restart_id, iter_id, coordinate, combination_index):
        seen.append((int(combination_index), int(stage_sim_cfg["n_jobs"]), int(point["x"])))
        score = float(point["x"])
        return CombinationResult(
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=dict(point),
            aggregated_error=score,
            subject_metrics={1: {"mean_error": score, "best_error": score, "std_error": 0.0, "simulation_repeats": 4}},
            hyper_candidate_seed=100 + combination_index,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    monkeypatch.setattr(cd_optimizer, "delayed", lambda fn: lambda *args, **kwargs: lambda: fn(*args, **kwargs))
    monkeypatch.setattr(cd_optimizer, "Parallel", lambda n_jobs: lambda tasks: [task() for task in tasks])
    monkeypatch.setattr(opt, "_evaluate_point_with_index", _fake_eval)
    combinations, _, _ = opt._coordinate_descent(
        stage_name="coarse",
        stage_sim_cfg={"simulation_repeats": 4},
        subjects=[1],
        space={"x": [1, 2, 3]},
        all_combinations_path=tmp_path / "all_combinations.jsonl",
        coordinate_trace_path=tmp_path / "coordinate_trace.jsonl",
    )

    assert [idx for idx, _, _ in seen] == [0, 1, 2]
    assert {n_jobs for _, n_jobs, _ in seen} == {2}
    assert [combo.combination_index for combo in combinations] == [0, 1, 2]
    trace = [json.loads(line) for line in (tmp_path / "coordinate_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert trace[0]["value_jobs"] == 2
    assert trace[0]["repeat_jobs"] == 2
    assert trace[0]["candidate_count"] == 3
    assert trace[0]["new_evaluations"] == 2
    assert trace[0]["cache_hits"] == 1


def test_cd_values_from_json_expands_kwargs_coordinate(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    _write_json(tmp_path / "strategy_candidates.json", _strategy_candidates_payload())
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["hyperparam_space"] = {
        "engine.modules.hypo_transitions_mod.kwargs": {
            "values_from_json": {
                "path": "strategy_candidates.json",
                "key": "cond23",
                "value_key": "hypo_transitions_kwargs",
            }
        },
        "simulation.window_size": {"values": [8]},
    }
    opt = HyperCDOptimizer(cfg, cd_path)

    specs = opt._param_specs_for_stage("coarse")
    space = {k: opt._hyperparam_values(v) for k, v in specs.items()}

    assert len(space["engine.modules.hypo_transitions_mod.kwargs"]) == 2
    first = space["engine.modules.hypo_transitions_mod.kwargs"][0]
    assert first["init_num"] == 6
    assert first["max_active_hypotheses"] == 6

    _, out_engine = opt._apply_hyperparams(
        {"engine.modules.hypo_transitions_mod.kwargs": space["engine.modules.hypo_transitions_mod.kwargs"][1]},
        {"window_size": 8},
        {"modules": {"hypo_transitions_mod": {"kwargs": {}}}},
    )
    assert out_engine["modules"]["hypo_transitions_mod"]["kwargs"]["init_num"] == 12


def test_cd_values_product_expands_grouped_memory_coordinate(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["hyperparam_space"] = {
        "engine.modules.memory_mod.kwargs": {
            "values_product": {
                "gamma": [0.10, 0.25, 0.40, 0.55, 0.70, 0.90],
                "w0": [0.005, 0.02, 0.06, 0.15, 0.35, 0.80],
            }
        }
    }
    cfg["cd"]["parallel_budget"] = 64
    cfg["stages"]["coarse"]["cd_parallel"]["max_repeat_jobs"] = 16
    opt = HyperCDOptimizer(cfg, cd_path)

    specs = opt._param_specs_for_stage("coarse")
    space = {k: opt._hyperparam_values(v) for k, v in specs.items()}
    memory_values = space["engine.modules.memory_mod.kwargs"]

    assert list(space.keys()) == ["engine.modules.memory_mod.kwargs"]
    assert len(memory_values) == 36
    assert memory_values[0] == {"gamma": 0.10, "w0": 0.005}
    assert memory_values[-1] == {"gamma": 0.90, "w0": 0.80}
    assert opt._coordinate_parallel_plan("coarse", {"simulation_repeats": 16}, len(memory_values)) == (4, 16)

    _, out_engine = opt._apply_hyperparams(
        {"engine.modules.memory_mod.kwargs": memory_values[10]},
        {"window_size": 8},
        {"modules": {"memory_mod": {"kwargs": {"gamma": 0.5, "w0": 0.1}}}},
    )
    assert out_engine["modules"]["memory_mod"]["kwargs"] == memory_values[10]


def test_cd_refine_expand_accepts_values_product(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["refine_policy"] = {
        "top_k": 1,
        "expand": {
            "engine.modules.memory_mod.kwargs": {
                "values_product": {"gamma": [0.1, 0.2], "w0": [0.01, 0.02]}
            }
        },
    }
    opt = HyperCDOptimizer(cfg, cd_path)
    combinations = [
        {
            "engine.modules.memory_mod.kwargs": {"gamma": 0.5, "w0": 0.1},
            "engine.modules.beta_mod.kwargs.beta_init": 1.0,
        }
    ]
    fallback_specs = {
        "engine.modules.memory_mod.kwargs": {
            "values_product": {"gamma": [0.5], "w0": [0.1]}
        },
        "engine.modules.beta_mod.kwargs.beta_init": {"values": [1.0, 2.0]},
    }

    space = opt._space_from_combinations(combinations, fallback_specs)

    assert space["engine.modules.memory_mod.kwargs"] == [
        {"gamma": 0.1, "w0": 0.01},
        {"gamma": 0.1, "w0": 0.02},
        {"gamma": 0.2, "w0": 0.01},
        {"gamma": 0.2, "w0": 0.02},
    ]
    assert space["engine.modules.beta_mod.kwargs.beta_init"] == [1.0]


def test_cd_shuffle_per_restart_reuses_coordinate_order_within_restart(tmp_path: Path, monkeypatch) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["cd"].update(
        {
            "coordinate_order": "shuffle_per_restart",
            "n_restarts": 2,
            "max_outer_iters": 2,
            "patience": 3,
        }
    )
    opt = HyperCDOptimizer(cfg, cd_path)

    def _fake_eval(stage_name, point, stage_sim_cfg, subjects, restart_id, iter_id, coordinate, combination_index):
        return CombinationResult(
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=dict(point),
            aggregated_error=1.0,
            subject_metrics={1: {"mean_error": 1.0, "best_error": 1.0, "std_error": 0.0, "simulation_repeats": 1}},
            hyper_candidate_seed=100 + combination_index,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    monkeypatch.setattr(opt, "_evaluate_point_with_index", _fake_eval)
    opt._coordinate_descent(
        stage_name="coarse",
        stage_sim_cfg={"simulation_repeats": 1},
        subjects=[1],
        space={"a": [1, 2], "b": [1, 2], "c": [1, 2]},
        all_combinations_path=tmp_path / "all_combinations.jsonl",
        coordinate_trace_path=tmp_path / "coordinate_trace.jsonl",
    )

    trace = [json.loads(line) for line in (tmp_path / "coordinate_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    orders_by_restart: dict[int, set[tuple[str, ...]]] = {}
    for row in trace:
        orders_by_restart.setdefault(int(row["restart_id"]), set()).add(tuple(row["coordinate_order"]))

    assert set(orders_by_restart) == {0, 1}
    assert all(len(orders) == 1 for orders in orders_by_restart.values())
    assert all(sorted(next(iter(orders))) == ["a", "b", "c"] for orders in orders_by_restart.values())


def test_cd_apply_hyperparams_deepcopies_composite_values(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    opt = HyperCDOptimizer(cfg, cd_path)
    kwargs_value = {
        "init_num": 6,
        "max_active_hypotheses": 6,
        "strategies": [{"label": "retain", "amount": "fixed", "value": 1, "method": "random", "pool": "active"}],
    }

    _, out_engine = opt._apply_hyperparams(
        {"engine.modules.hypo_transitions_mod.kwargs": kwargs_value},
        {"window_size": 8},
        {"modules": {"hypo_transitions_mod": {"kwargs": {}}}},
    )
    out_engine["modules"]["hypo_transitions_mod"]["kwargs"]["random_seed"] = 456
    out_engine["modules"]["hypo_transitions_mod"]["kwargs"]["strategies"][0]["label"] = "mutated"

    assert "random_seed" not in kwargs_value
    assert kwargs_value["strategies"][0]["label"] == "retain"


def test_build_hyper_selector_supports_grid_and_cd_backends(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    grid_cfg = dict(cfg)
    grid_cfg["output_dir"] = "./out_grid"
    grid_path = tmp_path / "hyper_grid.yaml"
    _write_yaml(grid_path, grid_cfg)

    assert isinstance(build_hyper_selector("hyper_cd", cd_path), HyperCDOptimizer)
    assert build_hyper_selector("hyper_cd", cd_path).hyper_base_seed == 42
    assert build_hyper_selector("hyper_grid", grid_path).hyper_base_seed == 42


def test_cd_missing_loss_metric_in_sim_config_raises(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["loss_metric"] = "accuracy_curve_mse"
    opt = HyperCDOptimizer(cfg, cd_path)
    bad_sim = dict(opt.base_sim_config)
    bad_sim.pop("loss_metric", None)
    try:
        _ = opt._resolve_sim_components(bad_sim, 1, [1])
        assert False, "Expected ValueError for missing loss_metric"
    except ValueError as e:
        assert "loss_metric" in str(e)


def test_cd_missing_loss_delta_with_berhu_in_sim_config_raises(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["loss_metric"] = "accuracy_curve_berhu"
    opt = HyperCDOptimizer(cfg, cd_path)
    bad_sim = dict(opt.base_sim_config)
    bad_sim["loss_metric"] = "accuracy_curve_berhu"
    bad_sim.pop("loss_delta", None)
    try:
        _ = opt._resolve_sim_components(bad_sim, 1, [1])
        assert False, "Expected ValueError for missing loss_delta with accuracy_curve_berhu"
    except ValueError as e:
        assert "loss_delta" in str(e)
