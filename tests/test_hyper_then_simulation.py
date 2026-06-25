from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.Bayesian_state.run_hyper_then_simulation import (
    aggregate_per_subject_best,
    build_subjectwise_simulation_config,
)
from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.optimization_config import resolve_engine_config


@pytest.fixture
def tmp_path() -> Path:
    root = Path(tempfile.gettempdir()) / "catelearn_test_tmp"
    root.mkdir(exist_ok=True)
    path = root / f"hyper_then_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def test_aggregate_per_subject_best_filters_to_requested_subjects(tmp_path: Path) -> None:
    output_dir = tmp_path / "hyper"
    _write_json(
        output_dir / "subject_101" / "best_hyperparams.json",
        {
            "selection": {"best_stage": "fine", "hyper_candidate_seed": 101},
            "selected": {"best_hyperparams": {"simulation.window_size": 8}},
        },
    )
    _write_json(
        output_dir / "subject_108" / "best_hyperparams.json",
        {
            "selection": {"best_stage": "fine", "hyper_candidate_seed": 108},
            "selected": {"best_hyperparams": {"simulation.window_size": 16}},
        },
    )
    optimizer = SimpleNamespace(
        tie_break_metric="simulation.mean_error",
        save_level="compact",
        base_sim_config_path=tmp_path / "sim.yaml",
        hyper_base_seed=42,
    )

    result = aggregate_per_subject_best(
        output_dir=output_dir,
        optimizer=optimizer,
        config_path=tmp_path / "hyper.yaml",
        backend="hyper_grid",
        subjects=[108],
        require_all=True,
    )

    payload = result["best"]
    assert payload["subjects"] == [108]
    assert payload["selection"]["metric"] == "simulation.mean_error"
    assert list(payload["per_subject_best"].keys()) == ["108"]
    root_payload = json.loads((output_dir / "best_hyperparams.json").read_text(encoding="utf-8"))
    assert root_payload["subjects"] == [108]
    assert list(root_payload["per_subject_best"].keys()) == ["108"]


def test_materialized_sim_config_preserves_base_subject_overrides(tmp_path: Path) -> None:
    base_sim_path = tmp_path / "simulation.yaml"
    _write_yaml(
        base_sim_path,
        {
            "engine_config": {
                "agenda": ["memory_mod", "beta_mod"],
                "modules": {
                    "memory_mod": {"class": "x", "kwargs": {"gamma": 0.2, "w0": 0.02}},
                    "beta_mod": {"class": "y", "kwargs": {"beta_init": 1.0, "beta_min": 0.1}},
                },
                "subject_overrides": {
                    108: {"modules": {"beta_mod": {"kwargs": {"beta_min": 0.2}}}},
                },
            },
            "subject_range": [101, 132],
            "hyper_base_seed": 42,
            "dataset": {"processed_dir": "data"},
            "output_dir": "old-out",
            "window_size": 16,
            "simulation_repeats": 1024,
            "loss_metric": "choice_brier",
            "prediction_mode": "prior_t",
            "selection_prediction_mode": "prior_t",
            "subject_overrides": {
                108: {
                    "window_size": 20,
                    "engine_config": {
                        "modules": {
                            "memory_mod": {"kwargs": {"gamma": 0.33}},
                        },
                    },
                },
            },
        },
    )
    hyper_best_payload = {
        "selection": {"metric": "simulation.mean_error"},
        "hyper": {
            "base_sim_config_path": str(base_sim_path),
            "hyper_base_seed": 42,
        },
        "per_subject_best": {
            "108": {
                "selection": {
                    "best_stage": "fine",
                    "hyper_candidate_seed": 123,
                },
                "selected": {
                    "best_hyperparams": {
                        "engine.modules.memory_mod.kwargs.gamma": 0.6,
                        "engine.modules.memory_mod.kwargs.w0": 0.5,
                    },
                },
            },
        },
    }

    generated = build_subjectwise_simulation_config(
        hyper_best_payload=hyper_best_payload,
        generated_sim_config_path=tmp_path / "generated" / "simulation.yaml",
        sim_output_dir=tmp_path / "sim-out",
        keep_logs=True,
    )

    override = generated["subject_overrides"][108]
    assert generated["subjects"] == [108]
    assert "subject_range" not in generated
    assert override["window_size"] == 20
    assert override["hyper_candidate_seed"] == 123
    assert override["engine_config"]["modules"]["beta_mod"]["kwargs"]["beta_min"] == 0.2
    assert override["engine_config"]["modules"]["memory_mod"]["kwargs"]["gamma"] == 0.6
    assert override["engine_config"]["modules"]["memory_mod"]["kwargs"]["w0"] == 0.5


def test_materialized_sim_config_replaces_whole_engine_kwargs_without_stale_base_keys(tmp_path: Path) -> None:
    base_sim_path = tmp_path / "simulation.yaml"
    selected_transition_kwargs = {
        "init_num": 4,
        "strategies": [
            {
                "label": "retain",
                "amount": "fixed",
                "value": 1,
                "method": "top_posterior",
                "pool": "active",
            }
        ],
    }
    _write_yaml(
        base_sim_path,
        {
            "engine_config": {
                "agenda": ["hypo_transitions_mod"],
                "modules": {
                    "hypo_transitions_mod": {
                        "class": "src.Bayesian_state.problems.modules.hypo_transitions.DynamicHypothesisModule",
                        "kwargs": {
                            "init_num": 2,
                            "max_active_hypotheses": 3,
                            "strategies": [
                                {
                                    "label": "base",
                                    "amount": "fixed",
                                    "value": 1,
                                    "method": "random",
                                    "pool": "inactive",
                                }
                            ],
                        },
                    }
                },
            },
            "subjects": [108],
            "hyper_base_seed": 42,
            "dataset": {"processed_dir": "data"},
            "output_dir": "old-out",
            "window_size": 16,
            "simulation_repeats": 1024,
            "loss_metric": "choice_brier",
            "prediction_mode": "prior_t",
            "selection_prediction_mode": "prior_t",
        },
    )
    hyper_best_payload = {
        "selection": {"metric": "simulation.mean_error"},
        "hyper": {
            "base_sim_config_path": str(base_sim_path),
            "hyper_base_seed": 42,
        },
        "per_subject_best": {
            "108": {
                "selection": {
                    "best_stage": "fine",
                    "hyper_candidate_seed": 123,
                },
                "selected": {
                    "best_hyperparams": {
                        "engine.modules.hypo_transitions_mod.kwargs": selected_transition_kwargs,
                    },
                },
            },
        },
    }

    generated = build_subjectwise_simulation_config(
        hyper_best_payload=hyper_best_payload,
        generated_sim_config_path=tmp_path / "generated" / "simulation.yaml",
        sim_output_dir=tmp_path / "sim-out",
        keep_logs=True,
    )

    base_kwargs = generated["engine_config"]["modules"]["hypo_transitions_mod"]["kwargs"]
    override_kwargs = generated["subject_overrides"][108]["engine_config"]["modules"]["hypo_transitions_mod"]["kwargs"]
    subject_cfg = resolve_subject_config(generated, 108)
    resolved_kwargs = resolve_engine_config(subject_cfg, tmp_path / "generated", subject_id=108)["modules"][
        "hypo_transitions_mod"
    ]["kwargs"]

    assert base_kwargs == {}
    assert override_kwargs == selected_transition_kwargs
    assert resolved_kwargs == selected_transition_kwargs
    assert "max_active_hypotheses" not in resolved_kwargs
