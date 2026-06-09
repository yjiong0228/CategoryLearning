from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

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
            "coarse": {"simulation_overrides": {"simulation_repeats": 1}},
            "fine": {"simulation_overrides": {"simulation_repeats": 1}},
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

    def _fake_eval(stage_name, point, stage_sim_cfg, subjects, restart_id, iter_id, coordinate):
        score = float(point.get("simulation.window_size", 0)) + float(
            point.get("engine.modules.beta_mod.kwargs.beta_init", 0)
        )
        return CombinationResult(
            stage=stage_name,
            combination_index=opt._combination_counter,
            hyperparams=dict(point),
            aggregated_error=score,
            subject_metrics={1: {"mean_error": score, "best_error": score, "std_error": 0.0, "simulation_repeats": 1}},
            hyper_candidate_seed=100 + opt._combination_counter,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    def _fake_eval_with_counter(*args, **kwargs):
        result = _fake_eval(*args, **kwargs)
        opt._combination_counter += 1
        return result

    monkeypatch.setattr(opt, "_evaluate_point", _fake_eval_with_counter)

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
