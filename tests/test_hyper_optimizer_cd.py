from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

import src.Bayesian_state.utils.hyper_cd_optimizer as cd_optimizer
from src.Bayesian_state.utils.hyper_cd_optimizer import CombinationResult, HyperCDOptimizer
from src.Bayesian_state.utils.hyper_objectives import (
    compare_objective_values,
    passes_anchor_guard,
    resolve_objective_order,
    select_best_by_objectives,
    update_anchor_values,
)
from src.Bayesian_state.run_hyper_then_simulation import build_hyper_selector
from src.Bayesian_state.utils.optimizer_common import SingleRunResult

OBJECTIVE_PATH = "simulation.mean_error"
SECONDARY_OBJECTIVE_PATH = "statistics.scores.history_kernel.value"


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


def _objective_order(path: str = OBJECTIVE_PATH) -> list[dict]:
    return [
        {
            "path": path,
            "rel_tolerance": 0.0,
            "abs_tolerance": 0.0,
            "scale_floor": 0.0,
            "anchor_guard": True,
        }
    ]


def _subject_metrics(score: float, extra_objectives: dict | None = None) -> dict:
    values = {OBJECTIVE_PATH: score}
    if extra_objectives:
        values.update(extra_objectives)
    return {
        "simulation": {
            "mean_error": score,
            "best_error": score,
            "std_error": 0.0,
            "simulation_repeats": 1,
        },
        "objectives": {"values": values},
    }


def _cd_result(
    *,
    stage: str,
    combination_index: int,
    point: dict,
    score: float,
    restart_id: int,
    iter_id: int,
    coordinate: str,
    extra_objectives: dict | None = None,
) -> CombinationResult:
    objective_values = {OBJECTIVE_PATH: score}
    if extra_objectives:
        objective_values.update(extra_objectives)
    return CombinationResult(
        stage=stage,
        combination_index=combination_index,
        hyperparams=dict(point),
        aggregated_error=score,
        objective_values=objective_values,
        subject_metrics={1: _subject_metrics(score, extra_objectives)},
        hyper_candidate_seed=100 + combination_index,
        restart_id=restart_id,
        iter_id=iter_id,
        coordinate=coordinate,
    )


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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("selection_metric", "simulation.mean_error"),
        ("secondary_selection", {"enabled": True}),
        ("simulation_statistics", {"enabled": True}),
        ("tie_break_metric", "statistics.loss.choice_brier.mean"),
        ("acceptance_selection", {"enabled": True}),
    ],
)
def test_objective_config_rejects_legacy_hyper_keys(tmp_path: Path, key: str, value: object) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg[key] = value
    with pytest.raises(ValueError, match="Legacy hyper config keys"):
        HyperCDOptimizer(cfg, cd_path)


def test_objective_comparator_uses_priority_and_raw_tiebreak() -> None:
    specs = resolve_objective_order(
        {
            "objective_order": [
                {
                    "path": OBJECTIVE_PATH,
                    "rel_tolerance": 0.03,
                    "abs_tolerance": 0.0,
                    "scale_floor": 0.0,
                    "anchor_guard": True,
                },
                {
                    "path": SECONDARY_OBJECTIVE_PATH,
                    "rel_tolerance": 0.0,
                    "abs_tolerance": 0.01,
                    "scale_floor": 0.0,
                    "anchor_guard": True,
                },
            ]
        }
    )

    assert compare_objective_values(
        {OBJECTIVE_PATH: 0.10, SECONDARY_OBJECTIVE_PATH: 10.0},
        {OBJECTIVE_PATH: 0.20, SECONDARY_OBJECTIVE_PATH: 0.0},
        specs,
    ) < 0
    assert compare_objective_values(
        {OBJECTIVE_PATH: 0.101, SECONDARY_OBJECTIVE_PATH: 0.10},
        {OBJECTIVE_PATH: 0.100, SECONDARY_OBJECTIVE_PATH: 0.20},
        specs,
    ) < 0
    assert compare_objective_values(
        {OBJECTIVE_PATH: 0.101, SECONDARY_OBJECTIVE_PATH: 0.199},
        {OBJECTIVE_PATH: 0.100, SECONDARY_OBJECTIVE_PATH: 0.200},
        specs,
    ) > 0


def test_objective_batch_selection_is_restart_order_independent() -> None:
    specs = resolve_objective_order(
        {
            "objective_order": [
                {
                    "path": OBJECTIVE_PATH,
                    "rel_tolerance": 0.03,
                    "abs_tolerance": 0.0,
                    "scale_floor": 0.0,
                    "anchor_guard": True,
                },
                {
                    "path": SECONDARY_OBJECTIVE_PATH,
                    "rel_tolerance": 0.0,
                    "abs_tolerance": 0.0,
                    "scale_floor": 0.0,
                    "anchor_guard": True,
                },
            ]
        }
    )
    rows = [
        {"id": 1, "values": {OBJECTIVE_PATH: 0.100, SECONDARY_OBJECTIVE_PATH: 0.40}},
        {"id": 2, "values": {OBJECTIVE_PATH: 0.102, SECONDARY_OBJECTIVE_PATH: 0.10}},
        {"id": 3, "values": {OBJECTIVE_PATH: 0.130, SECONDARY_OBJECTIVE_PATH: 0.00}},
    ]

    selected_a, _ = select_best_by_objectives(rows, lambda row: row["values"], specs, tie_breaker=lambda row: row["id"])
    selected_b, _ = select_best_by_objectives(list(reversed(rows)), lambda row: row["values"], specs, tie_breaker=lambda row: row["id"])

    assert selected_a["id"] == 2
    assert selected_b["id"] == 2


def test_objective_anchor_guard_rejects_drift_and_updates_only_accepted() -> None:
    specs = resolve_objective_order(
        {
            "objective_order": [
                {
                    "path": OBJECTIVE_PATH,
                    "rel_tolerance": 0.0,
                    "abs_tolerance": 0.01,
                    "scale_floor": 0.0,
                    "anchor_guard": True,
                },
                {
                    "path": SECONDARY_OBJECTIVE_PATH,
                    "rel_tolerance": 0.0,
                    "abs_tolerance": 0.02,
                    "scale_floor": 0.0,
                    "anchor_guard": True,
                },
            ]
        }
    )
    anchor = {OBJECTIVE_PATH: 0.10, SECONDARY_OBJECTIVE_PATH: 0.20}
    rejected = {OBJECTIVE_PATH: 0.105, SECONDARY_OBJECTIVE_PATH: 0.25}
    accepted = {OBJECTIVE_PATH: 0.095, SECONDARY_OBJECTIVE_PATH: 0.19}

    assert passes_anchor_guard(rejected, anchor, specs)[0] is False
    updated = update_anchor_values(anchor, accepted, specs)

    assert updated[OBJECTIVE_PATH] == pytest.approx(0.095)
    assert updated[SECONDARY_OBJECTIVE_PATH] == pytest.approx(0.19)


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
        "objective_order": _objective_order(),
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
        return _cd_result(
            stage=stage_name,
            combination_index=combination_index,
            point=point,
            score=score,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    def _fake_flat(**kwargs):
        results = [
            _fake_eval(
                kwargs["stage_name"],
                entry["point"],
                {**kwargs["stage_sim_cfg"], "n_jobs": kwargs["repeat_jobs"]},
                kwargs["subjects"],
                kwargs["restart_id"],
                kwargs["iter_id"],
                kwargs["coordinate"],
                int(entry["combination_index"]),
            )
            for entry in kwargs["missing_entries"]
        ]
        return results, {
            "flat_task_count": len(kwargs["missing_entries"]),
            "flat_jobs": len(kwargs["missing_entries"]),
            "parallel_backend": "flat_value_repeat_processes",
            "planned_total_jobs": kwargs["value_jobs"] * kwargs["repeat_jobs"],
        }

    monkeypatch.setattr(opt, "_evaluate_point_with_index", _fake_eval)
    monkeypatch.setattr(opt, "_evaluate_missing_entries_flat", _fake_flat)

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
    assert subject_best["schema_version"] == "hyper_result.v2"
    assert "provenance" in subject_best
    assert "combination_index" in subject_best["selection"]["candidate"]
    assert subject_best["selection"]["method"] == "objective_order"
    assert "objectives" in subject_best["selection"]
    assert subject_best["hyper"]["backend"] == "hyper_cd"
    assert first_line["schema_version"] == "hyper_result.v2"
    assert "aggregate" in first_line["metrics_summary"]
    assert "subjects" in first_line["metrics_summary"]
    assert root_best["hyper"]["backend"] == "hyper_cd"
    assert root_best["schema_version"] == "hyper_result.v2"
    assert "provenance" in root_best
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


def test_cd_coordinate_values_flatten_value_repeat_jobs(tmp_path: Path, monkeypatch) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["cd"].update(
        {
            "parallel_budget": 12,
            "n_restarts": 1,
            "max_outer_iters": 1,
            "init_strategy": "anchor",
            "anchor": {"x": 0},
            "patience": 1,
        }
    )
    cfg["stages"]["coarse"]["cd_parallel"]["max_repeat_jobs"] = 3
    opt = HyperCDOptimizer(cfg, cd_path)
    seen: list[tuple[int, int, int]] = []
    flat_calls: list[dict] = []

    def _fake_eval(stage_name, point, stage_sim_cfg, subjects, restart_id, iter_id, coordinate, combination_index):
        seen.append((int(combination_index), int(stage_sim_cfg["n_jobs"]), int(point["x"])))
        score = float(point["x"])
        return _cd_result(
            stage=stage_name,
            combination_index=combination_index,
            point=point,
            score=score,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    def _fake_flat(**kwargs):
        flat_calls.append(
            {
                "missing": len(kwargs["missing_entries"]),
                "value_jobs": kwargs["value_jobs"],
                "repeat_jobs": kwargs["repeat_jobs"],
            }
        )
        results = [
            _fake_eval(
                kwargs["stage_name"],
                entry["point"],
                {**kwargs["stage_sim_cfg"], "n_jobs": kwargs["repeat_jobs"]},
                kwargs["subjects"],
                kwargs["restart_id"],
                kwargs["iter_id"],
                kwargs["coordinate"],
                int(entry["combination_index"]),
            )
            for entry in kwargs["missing_entries"]
        ]
        return results, {
            "flat_task_count": len(kwargs["missing_entries"]) * 3,
            "flat_jobs": min(12, len(kwargs["missing_entries"]) * 3),
            "parallel_backend": "flat_value_repeat_processes",
            "planned_total_jobs": kwargs["value_jobs"] * kwargs["repeat_jobs"],
        }

    monkeypatch.setattr(opt, "_evaluate_point_with_index", _fake_eval)
    monkeypatch.setattr(opt, "_evaluate_missing_entries_flat", _fake_flat)
    combinations, _, _ = opt._coordinate_descent(
        stage_name="coarse",
        stage_sim_cfg={"simulation_repeats": 3},
        subjects=[1],
        space={"x": [1, 2, 3, 4]},
        all_combinations_path=tmp_path / "all_combinations.jsonl",
        coordinate_trace_path=tmp_path / "coordinate_trace.jsonl",
    )

    assert [idx for idx, _, _ in seen] == [0, 1, 2, 3, 4]
    assert {n_jobs for _, n_jobs, _ in seen} == {3}
    assert flat_calls == [{"missing": 4, "value_jobs": 4, "repeat_jobs": 3}]
    assert [combo.combination_index for combo in combinations] == [0, 1, 2, 3, 4]
    trace = [json.loads(line) for line in (tmp_path / "coordinate_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert trace[0]["value_jobs"] == 4
    assert trace[0]["repeat_jobs"] == 3
    assert trace[0]["missing_value_count"] == 4
    assert trace[0]["flat_task_count"] == 12
    assert trace[0]["flat_jobs"] == 12
    assert trace[0]["planned_total_jobs"] == 12
    assert trace[0]["parallel_backend"] == "flat_value_repeat_processes"
    assert trace[0]["candidate_count"] == 4
    assert trace[0]["new_evaluations"] == 4
    assert trace[0]["cache_hits"] == 0


def test_cd_flat_evaluator_groups_repeats_by_candidate_order(tmp_path: Path, monkeypatch) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["cd"]["parallel_budget"] = 12
    opt = HyperCDOptimizer(cfg, cd_path)
    parallel_jobs: list[int] = []
    task_order: list[tuple[int, int]] = []

    class _FakeRunner:
        _engine_config_template = {"modules": {}}
        _processed_data_dir = tmp_path
        _dataset_paths = {"processed_dir": tmp_path, "learning_data": tmp_path / "learning.csv"}

        def _get_subject_frame(self, subject_id, stop_at):
            return {"subject_id": subject_id, "stop_at": stop_at}

        def _get_condition_value(self, subject_frame):
            return 1

        def _extract_arrays(self, subject_frame, max_trials):
            return ("arrays", subject_frame, max_trials)

    def _fake_resolve(stage_sim_cfg, subject_id, subjects):
        return (
            {"simulation_repeats": 3, "loss_metric": "accuracy_curve_mse", "window_size": 8},
            {"modules": {}},
            "prior_t",
            "prior_t",
            "accuracy_curve_mse",
            None,
            8,
            1,
        )

    def _fake_apply(point, subject_cfg, engine_cfg):
        return dict(subject_cfg), dict(engine_cfg)

    def _fake_build(point_sim_cfg, point_engine_cfg):
        return _FakeRunner(), {"processed_dir": tmp_path, "learning_data": tmp_path / "learning.csv"}

    def _fake_task(task):
        position = int(task["position"])
        repeat_index = int(task["repeat_index"])
        task_order.append((position, repeat_index))
        x_val = float(task["params"]["x"])
        err = x_val + repeat_index / 10.0
        return {
            "position": position,
            "repeat_index": repeat_index,
            "run": SingleRunResult(
                params=dict(task["params"]),
                mean_error=err,
                metrics_by_mode={"prior_t": {"mean_error": err}},
                selection_prediction_mode="prior_t",
                loss_metric="accuracy_curve_mse",
                loss_delta=None,
                posterior_log=None,
                prior_log=None,
                beta_log=None,
                step_log=None,
                strategy_counts_log=None,
                simulation_point_seed=task["simulation_point_seed"],
                trajectory_seed=task["trajectory_seed"],
                seed_context=task["seed_context"],
            ),
        }

    def _fake_delayed(fn):
        return lambda *args, **kwargs: lambda: fn(*args, **kwargs)

    def _fake_parallel(n_jobs):
        parallel_jobs.append(int(n_jobs))

        def _run(tasks):
            task_list = list(tasks)
            return [task() for task in reversed(task_list)]

        return _run

    monkeypatch.setattr(opt, "_resolve_sim_components", _fake_resolve)
    monkeypatch.setattr(opt, "_apply_hyperparams", _fake_apply)
    monkeypatch.setattr(opt, "_build_runner", _fake_build)
    monkeypatch.setattr(cd_optimizer, "_evaluate_cd_flat_repeat_task", _fake_task)
    monkeypatch.setattr(cd_optimizer, "delayed", _fake_delayed)
    monkeypatch.setattr(cd_optimizer, "Parallel", _fake_parallel)

    entries = [
        {"position": 0, "point": {"x": 2}, "combination_index": 10},
        {"position": 1, "point": {"x": 3}, "combination_index": 11},
    ]
    results, diag = opt._evaluate_missing_entries_flat(
        stage_name="coarse",
        stage_sim_cfg={"simulation_repeats": 3},
        subjects=[1],
        restart_id=0,
        iter_id=1,
        coordinate="x",
        missing_entries=entries,
        value_jobs=4,
        repeat_jobs=3,
    )

    assert parallel_jobs == [6]
    assert diag["flat_task_count"] == 6
    assert diag["flat_jobs"] == 6
    assert diag["planned_total_jobs"] == 12
    assert diag["parallel_backend"] == "flat_value_repeat_processes"
    assert task_order == [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)]
    assert [result.combination_index for result in results] == [10, 11]
    assert [result.hyperparams["x"] for result in results] == [2, 3]
    assert results[0].aggregated_error == pytest.approx(2.1)
    assert results[1].aggregated_error == pytest.approx(3.1)
    assert results[0].subject_metrics[1]["simulation"]["sample_errors"] == [2.0, 2.1, 2.2]
    assert results[1].subject_metrics[1]["simulation"]["sample_errors"] == [3.0, 3.1, 3.2]


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


def test_cd_profile_candidate_injects_multiple_hyperparam_paths(tmp_path: Path) -> None:
    cd_path = _build_min_cd_config(tmp_path)
    payload = {
        "cond1_v11": [
            {
                "id": "profile_a",
                "model_kwargs": {
                    "engine.modules.hypo_transitions_mod.kwargs": {
                        "init_num": 3,
                        "strategy_controller": {
                            "method": "feedback_gated_softmax",
                            "profiles": [
                                {
                                    "id": "exploit",
                                    "strategies": [
                                        {
                                            "label": "retain",
                                            "amount": "fixed",
                                            "value": 1,
                                            "method": "random",
                                            "pool": "active",
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                    "engine.choice_readout.kwargs": {
                        "method": "stubborn_sticky",
                        "switch_probability": 0.1,
                    },
                },
            }
        ]
    }
    _write_json(tmp_path / "profile_candidates.json", payload)
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    cfg["hyperparam_space"] = {
        "__profile_candidate__": {
            "values_from_json": {
                "path": "profile_candidates.json",
                "key": "cond1_v11",
                "value_key": "model_kwargs",
            }
        }
    }
    opt = HyperCDOptimizer(cfg, cd_path)

    specs = opt._param_specs_for_stage("coarse")
    values = opt._hyperparam_values(specs["__profile_candidate__"])
    next_sim, out_engine = opt._apply_hyperparams(
        {"__profile_candidate__": values[0]},
        {"window_size": 8},
        {"modules": {"hypo_transitions_mod": {"kwargs": {}}}},
    )

    assert out_engine["modules"]["hypo_transitions_mod"]["kwargs"]["init_num"] == 3
    assert out_engine["choice_readout"]["kwargs"]["method"] == "stubborn_sticky"
    assert "__profile_candidate__" in next_sim["fixed_hyperparams"]


def test_cond1_v13_cd_config_optimizes_memory_and_profile_only() -> None:
    cd_path = Path(__file__).parents[1] / "configs" / "hyper_cd_cfg" / "pmh_cond1_hyper_cd_v13.yaml"
    cfg = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    opt = HyperCDOptimizer(cfg, cd_path)

    specs = opt._param_specs_for_stage("coarse")
    assert set(specs) == {
        "engine.modules.memory_mod.kwargs",
        "engine.modules.hypo_transitions_mod.kwargs",
        "engine.choice_readout.kwargs",
    }
    assert "beta_mod" not in yaml.safe_dump(specs)

    memory_values = opt._hyperparam_values(specs["engine.modules.memory_mod.kwargs"])
    profile_values = opt._hyperparam_values(specs["engine.modules.hypo_transitions_mod.kwargs"])
    readout_values = opt._hyperparam_values(specs["engine.choice_readout.kwargs"])

    memory_product = specs["engine.modules.memory_mod.kwargs"]["values_product"]
    expected_memory_values = 1
    for factor_values in memory_product.values():
        expected_memory_values *= len(factor_values)
    assert len(memory_values) == expected_memory_values
    assert profile_values
    assert {value["method"] for value in readout_values} == {"expectation", "map_hypothesis"}
    for value in profile_values:
        assert "strategy_controller" in value
        assert "choice_readout" not in yaml.safe_dump(value)


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
        return _cd_result(
            stage=stage_name,
            combination_index=combination_index,
            point=point,
            score=1.0,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    def _fake_flat(**kwargs):
        results = [
            _fake_eval(
                kwargs["stage_name"],
                entry["point"],
                {**kwargs["stage_sim_cfg"], "n_jobs": kwargs["repeat_jobs"]},
                kwargs["subjects"],
                kwargs["restart_id"],
                kwargs["iter_id"],
                kwargs["coordinate"],
                int(entry["combination_index"]),
            )
            for entry in kwargs["missing_entries"]
        ]
        return results, {
            "flat_task_count": len(kwargs["missing_entries"]),
            "flat_jobs": len(kwargs["missing_entries"]),
            "parallel_backend": "flat_value_repeat_processes",
            "planned_total_jobs": kwargs["value_jobs"] * kwargs["repeat_jobs"],
        }

    monkeypatch.setattr(opt, "_evaluate_point_with_index", _fake_eval)
    monkeypatch.setattr(opt, "_evaluate_missing_entries_flat", _fake_flat)
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
