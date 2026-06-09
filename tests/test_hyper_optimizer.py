from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.hyper_grid.optimizer import CombinationResult, HyperGridOptimizer


@pytest.fixture
def tmp_path() -> Path:
    root = Path(tempfile.gettempdir()) / "catelearn_test_tmp"
    root.mkdir(exist_ok=True)
    path = root / f"hyper_grid_{uuid.uuid4().hex}"
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
        "cond1": [
            {
                "id": "small",
                "description": "small memory",
                "hypo_transitions_kwargs": {
                    "init_num": 2,
                    "max_active_hypotheses": 3,
                    "strategies": [
                        {"label": "retain", "amount": "confidence_2", "method": "top_posterior", "pool": "active"}
                    ],
                },
            },
            {
                "id": "broad",
                "description": "broad memory",
                "hypo_transitions_kwargs": {
                    "init_num": 4,
                    "max_active_hypotheses": 5,
                    "strategies": [
                        {"label": "retain", "amount": "entropy_norm_4", "method": "epsilon_posterior", "pool": "active"}
                    ],
                },
            },
        ]
    }


def _build_min_configs(tmp_path: Path) -> tuple[Path, Path]:
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
    sim_path = tmp_path / "sim.yaml"
    _write_yaml(sim_path, sim_cfg)

    hyper_cfg = {
        "base_sim_config_path": "sim.yaml",
        "subjects": [1],
        "output_dir": "./out",
        "selection_metric": "mean_simulation_error",
        "hyper_base_seed": 100,
        "hyperparam_space": {
            "engine.modules.beta_mod.kwargs.beta_init": {"values": [0.5, 1.0, 2.0]},
            "simulation.window_size": {"values": [6, 8, 10]},
        },
        "stages": {
            "coarse": {"simulation_overrides": {"simulation_repeats": 1}},
            "fine": {"simulation_overrides": {"simulation_repeats": 1}},
        },
        "refine_policy": {"top_k": 2},
    }
    hyper_path = tmp_path / "hyper.yaml"
    _write_yaml(hyper_path, hyper_cfg)
    return sim_path, hyper_path


def test_apply_hyperparams_injection(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)

    inner = {"window_size": 8}
    engine = {"modules": {"beta_mod": {"kwargs": {"beta_init": 1.0}}}}
    combination = {
        "simulation.window_size": 12,
        "engine.modules.beta_mod.kwargs.beta_init": 2.5,
    }
    out_inner, out_engine = opt._apply_hyperparams(combination, inner, engine)

    assert out_inner["window_size"] == 12
    assert out_engine["modules"]["beta_mod"]["kwargs"]["beta_init"] == 2.5


def test_values_from_json_expands_kwargs_and_applies_to_engine(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    candidates_path = tmp_path / "strategy_candidates.json"
    _write_json(candidates_path, _strategy_candidates_payload())
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    cfg["hyperparam_space"] = {
        "engine.modules.hypo_transitions_mod.kwargs": {
            "values_from_json": {
                "path": "strategy_candidates.json",
                "key": "cond1",
                "value_key": "hypo_transitions_kwargs",
            }
        },
        "simulation.window_size": {"values": [8]},
    }
    opt = HyperGridOptimizer(cfg, hyper_path)

    specs = opt._param_specs_for_stage("coarse")
    combos = opt._expand_combinations(specs)

    assert len(combos) == 2
    first_kwargs = combos[0]["engine.modules.hypo_transitions_mod.kwargs"]
    assert first_kwargs["init_num"] == 2
    assert first_kwargs["max_active_hypotheses"] == 3
    assert first_kwargs["strategies"][0]["label"] == "retain"

    _, out_engine = opt._apply_hyperparams(
        combos[1],
        {"window_size": 8},
        {"modules": {"hypo_transitions_mod": {"kwargs": {}}}},
    )
    injected = out_engine["modules"]["hypo_transitions_mod"]["kwargs"]
    assert injected["init_num"] == 4
    assert injected["max_active_hypotheses"] == 5


def test_apply_hyperparams_deepcopies_composite_values(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)
    kwargs_value = {
        "init_num": 2,
        "max_active_hypotheses": 3,
        "strategies": [{"label": "retain", "amount": "fixed", "value": 1, "method": "random", "pool": "active"}],
    }
    combination = {"engine.modules.hypo_transitions_mod.kwargs": kwargs_value}

    _, out_engine = opt._apply_hyperparams(
        combination,
        {"window_size": 8},
        {"modules": {"hypo_transitions_mod": {"kwargs": {}}}},
    )
    out_engine["modules"]["hypo_transitions_mod"]["kwargs"]["random_seed"] = 123
    out_engine["modules"]["hypo_transitions_mod"]["kwargs"]["strategies"][0]["label"] = "mutated"

    assert "random_seed" not in kwargs_value
    assert kwargs_value["strategies"][0]["label"] == "retain"


def test_values_product_expands_memory_kwargs_and_applies_to_engine(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)

    values = opt._hyperparam_values(
        {
            "values_product": {
                "gamma": [0.10, 0.25, 0.40, 0.55, 0.70, 0.90],
                "w0": [0.005, 0.02, 0.06, 0.15, 0.35, 0.80],
            }
        }
    )

    assert len(values) == 36
    assert values[0] == {"gamma": 0.10, "w0": 0.005}
    assert values[-1] == {"gamma": 0.90, "w0": 0.80}

    _, out_engine = opt._apply_hyperparams(
        {"engine.modules.memory_mod.kwargs": values[3]},
        {"window_size": 8},
        {"modules": {"memory_mod": {"kwargs": {"gamma": 0.5, "w0": 0.1}}}},
    )
    assert out_engine["modules"]["memory_mod"]["kwargs"] == {"gamma": 0.10, "w0": 0.15}


def test_refine_expand_accepts_values_product(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    cfg["refine_policy"] = {
        "top_k": 1,
        "expand": {
            "engine.modules.memory_mod.kwargs": {
                "values_product": {"gamma": [0.1, 0.2], "w0": [0.01, 0.02]}
            }
        },
    }
    opt = HyperGridOptimizer(cfg, hyper_path)
    coarse_combinations = [
        CombinationResult(
            "coarse",
            0,
            {
                "engine.modules.memory_mod.kwargs": {"gamma": 0.5, "w0": 0.1},
                "engine.modules.beta_mod.kwargs.beta_init": 1.0,
            },
            0.1,
            {1: {"mean_error": 0.1, "std_error": 0.0}},
            1,
        )
    ]

    refined = opt._refine_combinations_from_coarse(coarse_combinations)

    assert len(refined) == 4
    assert refined[0]["engine.modules.memory_mod.kwargs"] == {"gamma": 0.1, "w0": 0.01}
    assert refined[-1]["engine.modules.memory_mod.kwargs"] == {"gamma": 0.2, "w0": 0.02}
    assert {point["engine.modules.beta_mod.kwargs.beta_init"] for point in refined} == {1.0}


def test_top_k_combinations_from_coarse(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)

    coarse_combinations = [
        CombinationResult("coarse", 0, {"simulation.window_size": 8, "engine.modules.beta_mod.kwargs.beta_init": 1.0}, 0.2, {1: {"mean_error": 0.2, "std_error": 0.0}}, 1),
        CombinationResult("coarse", 1, {"simulation.window_size": 10, "engine.modules.beta_mod.kwargs.beta_init": 2.0}, 0.3, {1: {"mean_error": 0.3, "std_error": 0.0}}, 2),
        CombinationResult("coarse", 2, {"simulation.window_size": 12, "engine.modules.beta_mod.kwargs.beta_init": 3.0}, 0.4, {1: {"mean_error": 0.4, "std_error": 0.0}}, 3),
    ]

    selected = opt._top_k_combinations_from_coarse(coarse_combinations)
    assert len(selected) == 2
    assert selected[0]["simulation.window_size"] == 8
    assert selected[1]["simulation.window_size"] == 10


def test_run_outputs_files_with_mocked_combinations(tmp_path: Path, monkeypatch) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)

    def _fake_eval(stage_name, combination_index, combination_params, stage_inner_cfg, subjects):
        err = float(combination_index + (0 if stage_name == "coarse" else 0.1))
        return CombinationResult(stage_name, combination_index, dict(combination_params), err, {1: {"mean_error": err, "best_error": err, "std_error": 0.0, "simulation_repeats": 1}}, 42 + combination_index)

    monkeypatch.setattr(opt, "_evaluate_combination", _fake_eval)

    result = opt.run(subjects=[1], stage="all")
    assert "per_subject_outputs" in result
    assert Path(result["per_subject_outputs"]["1"]["all_combinations"]).exists()
    assert Path(result["per_subject_outputs"]["1"]["stage_summary"]).exists()
    assert Path(result["best_hyperparams"]).exists()

    best = json.loads(Path(result["best_hyperparams"]).read_text(encoding="utf-8"))
    assert "per_subject_best" in best
    assert "1" in best["per_subject_best"]
    assert best["save_level"] == "compact"
    assert "subject_metrics" not in best

    first_line = Path(result["per_subject_outputs"]["1"]["all_combinations"]).read_text(encoding="utf-8").splitlines()[0]
    first_payload = json.loads(first_line)
    assert "subject_metrics" not in first_payload
    assert "combination_index" in first_payload


def test_group_mean_mode_is_rejected(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    cfg["hyperparam_selection_mode"] = "group_mean"
    with pytest.raises(ValueError, match="Only per_subject"):
        HyperGridOptimizer(cfg, hyper_path)


def test_simulation_overrides_are_used_for_runner_call(tmp_path: Path, monkeypatch) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)

    captured: dict = {}

    def _fake_resolve(sim_cfg, subject_id, subjects):
        return sim_cfg, {"modules": {}}, "posterior_t_minus_1", "posterior_t_minus_1", "accuracy_curve_mse", None, 8, 1

    class _FakeRunner:
        def __init__(self):
            self.n_jobs = 1

        def simulate_subject(self, **kwargs):
            captured["window_size"] = kwargs["window_size"]
            captured["prediction_mode"] = kwargs["prediction_mode"]
            captured["selection_prediction_mode"] = kwargs["selection_prediction_mode"]
            captured["loss_metric"] = kwargs["loss_metric"]
            captured["loss_delta"] = kwargs["loss_delta"]
            class _Best:
                mean_error = 0.1
                best_error = 0.1
                params = {"gamma": 0.9, "w0": 0.2}
            return {"best": _Best(), "condition": 1}

    def _fake_build(sim_cfg, engine_cfg):
        return _FakeRunner(), {"learning_data": "x", "processed_dir": "y"}

    monkeypatch.setattr(opt, "_resolve_sim_components", _fake_resolve)
    monkeypatch.setattr(opt, "_build_runner", _fake_build)

    combination = {
        "simulation.window_size": 12,
        "simulation.prediction_mode": "prior_t",
        "simulation.selection_prediction_mode": "prior_t",
    }
    _ = opt._evaluate_combination("coarse", 0, combination, opt.base_sim_config, [1])

    assert captured["window_size"] == 12
    assert captured["prediction_mode"] == "prior_t"
    assert captured["selection_prediction_mode"] == "prior_t"
    assert captured["loss_metric"] == "accuracy_curve_mse"
    assert captured["loss_delta"] is None


def test_fine_stage_runs_without_coarse_when_fine_hyperparam_space_exists(tmp_path: Path, monkeypatch) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    cfg["stages"]["fine"]["hyperparam_space"] = {
        "simulation.window_size": {"values": [8]},
    }
    opt = HyperGridOptimizer(cfg, hyper_path)

    def _fake_eval(stage_name, combination_index, combination_params, stage_inner_cfg, subjects):
        return CombinationResult(stage_name, combination_index, dict(combination_params), 0.2, {1: {"mean_error": 0.2, "best_error": 0.2, "std_error": 0.0, "simulation_repeats": 1}}, 999)

    monkeypatch.setattr(opt, "_evaluate_combination", _fake_eval)

    result = opt.run(subjects=[1], stage="fine")
    assert Path(result["per_subject_outputs"]["1"]["all_combinations"]).exists()
    assert Path(result["best_hyperparams"]).exists()


def test_fine_stage_without_hyperparam_space_requires_coarse_results(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)

    try:
        _ = opt.run(subjects=[1], stage="fine")
        assert False, "Expected ValueError for missing coarse results"
    except ValueError as e:
        assert "requires coarse stage results" in str(e)


def test_missing_loss_metric_in_sim_config_raises(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    cfg["loss_metric"] = "accuracy_curve_mse"
    opt = HyperGridOptimizer(cfg, hyper_path)
    bad_sim = dict(opt.base_sim_config)
    bad_sim.pop("loss_metric", None)
    try:
        _ = opt._resolve_sim_components(bad_sim, 1, [1])
        assert False, "Expected ValueError for missing loss_metric"
    except ValueError as e:
        assert "loss_metric" in str(e)


def test_missing_loss_delta_with_berhu_in_sim_config_raises(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    cfg["loss_metric"] = "accuracy_curve_berhu"
    opt = HyperGridOptimizer(cfg, hyper_path)
    bad_sim = dict(opt.base_sim_config)
    bad_sim["loss_metric"] = "accuracy_curve_berhu"
    bad_sim.pop("loss_delta", None)
    try:
        _ = opt._resolve_sim_components(bad_sim, 1, [1])
        assert False, "Expected ValueError for missing loss_delta with accuracy_curve_berhu"
    except ValueError as e:
        assert "loss_delta" in str(e)


def test_values_from_json_errors_are_clear(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    candidates_path = tmp_path / "strategy_candidates.json"
    _write_json(candidates_path, _strategy_candidates_payload())
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)

    bad_specs = [
        (
            {"values_from_json": {"path": "missing.json", "key": "cond1", "value_key": "hypo_transitions_kwargs"}},
            "does not exist",
        ),
        (
            {"values_from_json": {"path": "strategy_candidates.json", "key": "condX", "value_key": "hypo_transitions_kwargs"}},
            "not found",
        ),
        (
            {"values_from_json": {"path": "strategy_candidates.json", "key": "cond1", "value_key": "missing"}},
            "missing value_key",
        ),
    ]
    for spec, expected in bad_specs:
        try:
            _ = opt._hyperparam_values(spec)
            assert False, f"Expected ValueError containing {expected!r}"
        except ValueError as e:
            assert expected in str(e)

    empty_path = tmp_path / "empty_candidates.json"
    _write_json(empty_path, {"cond1": []})
    try:
        _ = opt._hyperparam_values(
            {"values_from_json": {"path": "empty_candidates.json", "key": "cond1", "value_key": "hypo_transitions_kwargs"}}
        )
        assert False, "Expected ValueError for empty JSON candidate list"
    except ValueError as e:
        assert "empty list" in str(e)

    scalar_path = tmp_path / "scalar_candidates.json"
    _write_json(scalar_path, {"cond1": {"not": "a list"}})
    try:
        _ = opt._hyperparam_values(
            {"values_from_json": {"path": "scalar_candidates.json", "key": "cond1", "value_key": "hypo_transitions_kwargs"}}
        )
        assert False, "Expected ValueError for non-list JSON key"
    except ValueError as e:
        assert "must contain a list" in str(e)


def test_values_product_errors_are_clear(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    opt = HyperGridOptimizer(cfg, hyper_path)

    bad_specs = [
        ({"values_product": {}}, "at least one factor"),
        ({"values_product": {"gamma": "not-a-list"}}, "non-empty list"),
        ({"values_product": {"gamma": []}}, "cannot be empty"),
        ({"values_product": {"": [1]}}, "non-empty strings"),
    ]
    for spec, expected in bad_specs:
        with pytest.raises(ValueError, match=expected):
            opt._hyperparam_values(spec)


def test_parent_and_child_hyperparam_paths_are_rejected(tmp_path: Path) -> None:
    _, hyper_path = _build_min_configs(tmp_path)
    cfg = yaml.safe_load(hyper_path.read_text(encoding="utf-8"))
    cfg["hyperparam_space"] = {
        "engine.modules.hypo_transitions_mod.kwargs": {"values": [{"init_num": 2}]},
        "engine.modules.hypo_transitions_mod.kwargs.strategies": {"values": [[]]},
    }
    opt = HyperGridOptimizer(cfg, hyper_path)

    try:
        _ = opt._param_specs_for_stage("coarse")
        assert False, "Expected ValueError for parent/child hyperparam paths"
    except ValueError as e:
        assert "parent path and its child path" in str(e)
