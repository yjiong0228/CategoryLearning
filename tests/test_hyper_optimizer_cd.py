from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.Bayesian_state.hyper_opt_cd.optimizer import CombinationResult, HyperOptimizerCD
from src.Bayesian_state.run_hyper_then_grid import (
    default_generated_grid_config_for_backend,
    default_grid_output_dir_for_backend,
    resolve_hyper_backend,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _build_min_cd_config(tmp_path: Path) -> Path:
    inner_cfg = {
        "engine_config": {
            "agenda": ["memory_mod"],
            "modules": {"memory_mod": {"class": "x", "kwargs": {"gamma": 0.5, "w0": 0.1}}},
        },
        "subjects": [1],
        "param_grid": {"gamma": [0.5], "w0": [0.1]},
        "window_size": 8,
    }
    _write_yaml(tmp_path / "inner.yaml", inner_cfg)

    cd_cfg = {
        "inner_optimizer": "grid",
        "inner_base_config_path": "inner.yaml",
        "subjects": [1],
        "output_dir": "./out_cd",
        "selection_metric": "min_inner_mean_error",
        "hyperparam_selection_mode": "per_subject",
        "save_level": "compact",
        "random_seed": 42,
        "cd": {
            "n_restarts": 2,
            "max_outer_iters": 2,
            "init_strategy": "anchor",
            "anchor": {"inner.window_size": 8, "engine.modules.beta_mod.kwargs.beta_init": 1.0},
            "coordinate_order": "fixed",
            "patience": 1,
            "min_delta": 0.0,
        },
        "hyperparam_space": {
            "inner.window_size": {"values": [6, 8]},
            "engine.modules.beta_mod.kwargs.beta_init": {"values": [1.0, 2.0]},
        },
        "stages": {
            "coarse": {"inner_overrides": {"n_repeats": 1, "refit_repeats": 0}},
            "fine": {"inner_overrides": {"n_repeats": 1, "refit_repeats": 0}},
        },
        "refine_policy": {"top_k": 2},
    }
    cd_path = tmp_path / "hyper_cd.yaml"
    _write_yaml(cd_path, cd_cfg)
    return cd_path


def test_cd_outputs_combination_schema(tmp_path: Path, monkeypatch) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    opt = HyperOptimizerCD(cfg, cd_path)

    def _fake_eval(stage_name, point, stage_inner_cfg, subjects, restart_id, iter_id, coordinate):
        score = float(point.get("inner.window_size", 0)) + float(
            point.get("engine.modules.beta_mod.kwargs.beta_init", 0)
        )
        return CombinationResult(
            stage=stage_name,
            combination_index=opt._combination_counter,
            hyperparams=dict(point),
            aggregated_error=score,
            subject_metrics={1: {"mean_error": score, "best_error": score}},
            random_seed=100 + opt._combination_counter,
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
    assert subject_best["hyper_backend"] == "cd"
    assert root_best["hyper_backend"] == "cd"
    assert "per_subject_best" in root_best


def test_cd_backend_auto_defaults_do_not_overlap_standard_hyper(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))

    assert resolve_hyper_backend(cd_path, cfg, "auto") == "cd"
    assert default_generated_grid_config_for_backend("cd").name.endswith("hyper_cd_best.yaml")
    assert default_grid_output_dir_for_backend("cd").name.endswith("hyper_cd_best")
    assert default_generated_grid_config_for_backend("hyper").name.endswith("hyper_best.yaml")
    assert default_grid_output_dir_for_backend("hyper").name.endswith("hyper_best")
