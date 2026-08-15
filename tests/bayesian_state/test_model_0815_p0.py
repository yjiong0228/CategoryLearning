from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.run_model_0815_p0_pf_convergence import (
    compare_successive_counts,
    summarize_repeat_panel,
)
from src.Bayesian_state.optimization import objectives
from src.Bayesian_state.optimization.search import coordinate_descent as cd_module
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
from src.Bayesian_state.simulation.execution import (
    compute_metrics_from_category_probabilities,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "configs/model_struct/pmh_model_cond1_0815_p0.yaml"
SIM_CONFIG = (
    ROOT
    / "configs/simulation_cfg/model0815_p0_cond1_beta_recalibration.yaml"
)
HYPER_CONFIG = (
    ROOT
    / "configs/hyper_cd_cfg/model0815_p0_cond1_beta_recalibration.yaml"
)


def test_p0_model_assembles_with_declared_simple_structure() -> None:
    engine = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    model = StateModel(engine, context=ModelContext(condition=1, subject_id=103))
    transition = model.engine.modules["hypo_transitions_mod"]

    assert model.observation_likelihood.distance_mode == "boundary"
    assert model.observation_likelihood.beta_source == "fixed"
    assert model.observation_likelihood.default_beta == 5.0
    assert transition.capacity == 3
    assert transition.persistent_execution_enabled is True
    assert "init_hypotheses" not in engine["modules"]["hypo_transitions_mod"]["kwargs"]
    assert engine["choice_readout"]["kwargs"] == {
        "method": "expectation",
        "power": 1.0,
        "strategy_confidence_gain": 0.0,
    }
    assert engine["provenance"]["label_mapping"]["mode"] == "fixed_task_labels"


def test_p0_calibration_keeps_capacity_fixed_and_scores_mean_probability() -> None:
    simulation = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
    hyper = yaml.safe_load(HYPER_CONFIG.read_text(encoding="utf-8"))

    assert simulation["repeat_aggregation"] == "mean_probability"
    assert hyper["hyperparam_selection_mode"] == "shared"
    assert hyper["objective_order"][0]["path"] == "simulation.mean_error"
    assert all(
        stage["simulation_overrides"]["repeat_aggregation"] == "mean_probability"
        for stage in hyper["stages"].values()
    )
    assert not any("capacity" in key for key in hyper["hyperparam_space"])
    assert set(hyper["hyperparam_space"]) == {
        "engine.likelihood.default_beta",
        "engine.modules.beta_mod.kwargs.beta_init",
        "engine.modules.beta_mod.kwargs.beta_max",
        "engine.modules.beta_mod.kwargs.decrease_rate",
        "engine.modules.beta_mod.kwargs.correct_additive",
    }


def test_shared_hyper_cd_routes_all_subjects_to_one_pipeline(monkeypatch) -> None:
    config = yaml.safe_load(HYPER_CONFIG.read_text(encoding="utf-8"))
    optimizer = HyperCDOptimizer(config, HYPER_CONFIG)
    captured = {}

    def fake_run_pipeline(*, subjects, stage, output_dir, resume_from_coarse):
        captured.update(
            subjects=subjects,
            stage=stage,
            output_dir=output_dir,
            resume_from_coarse=resume_from_coarse,
        )
        return {
            "output_dir": str(output_dir),
            "all_combinations": str(output_dir / "all_combinations.jsonl"),
            "stage_summary": str(output_dir / "stage_summary.json"),
            "restart_summary": str(output_dir / "restart_summary.json"),
            "coordinate_trace": str(output_dir / "coordinate_trace.jsonl"),
            "best_hyperparams": str(output_dir / "best_hyperparams.json"),
            "best": {"subject_id": -1},
        }

    monkeypatch.setattr(optimizer, "_run_pipeline", fake_run_pipeline)
    result = optimizer.run([103, 105], stage="coarse")

    assert captured["subjects"] == [103, 105]
    assert captured["stage"] == "coarse"
    assert result["hyperparam_selection_mode"] == "shared"
    assert result["subjects"] == [103, 105]
    assert result["best"]["subject_id"] == -1


def test_shared_flat_cd_averages_subject_objectives_equally(monkeypatch) -> None:
    class FakeRunner:
        _engine_config_template = {}
        _processed_data_dir = ROOT / "data/processed"

        @staticmethod
        def _get_subject_frame(subject_id, stop_at):
            return int(subject_id)

        @staticmethod
        def _get_condition_value(subject_frame):
            return 1

        @staticmethod
        def _extract_arrays(subject_frame, max_trials):
            return SimpleNamespace(feedback=np.ones(24, dtype=float))

    class SequentialParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        @staticmethod
        def __call__(tasks):
            return [task() for task in tasks]

    optimizer = object.__new__(HyperCDOptimizer)
    optimizer.parallel_budget = 8
    optimizer.statistics_config = {}
    optimizer.objective_order = objectives.resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )
    optimizer._hyper_candidate_seed = lambda *args, **kwargs: 123
    optimizer._resolve_sim_components = lambda *args, **kwargs: (
        {
            "simulation_repeats": 2,
            "repeat_aggregation": "mean_probability",
            "prediction_mode": "prior_t",
            "selection_prediction_mode": "prior_t",
            "loss_metric": "choice_nll",
            "window_size": 16,
            "keep_logs": False,
        },
        {},
        "prior_t",
        "prior_t",
        "choice_nll",
        None,
        16,
        1,
    )
    optimizer._apply_hyperparams = lambda point, sim_cfg, engine_cfg: (
        sim_cfg,
        engine_cfg,
    )
    optimizer._build_runner = lambda *args, **kwargs: (
        FakeRunner(),
        {"learning_data": ROOT / "data/processed/Task2_processed.csv"},
    )

    monkeypatch.setattr(cd_module, "Parallel", SequentialParallel)
    monkeypatch.setattr(cd_module, "delayed", lambda fn: lambda task: lambda: fn(task))
    monkeypatch.setattr(
        cd_module,
        "_evaluate_cd_flat_repeat_task",
        lambda task: {
            "position": task["position"],
            "subject_id": task["subject_id"],
            "repeat_index": task["repeat_index"],
            "run": object(),
        },
    )

    def fake_aggregate(*args, subject_id, **kwargs):
        error = {103: 0.2, 105: 0.6}[int(subject_id)]
        return SimpleNamespace(
            mean_error=error,
            best_error=error,
            sample_errors=[error, error],
            std_error=0.0,
            statistics_summary={},
            repeat_aggregation="mean_probability",
            aggregation_diagnostics={"method": "mean_probability"},
        )

    monkeypatch.setattr(cd_module, "aggregate_simulation_runs", fake_aggregate)
    combinations, diagnostics = optimizer._evaluate_missing_entries_flat(
        stage_name="coarse",
        stage_sim_cfg={},
        subjects=[103, 105],
        restart_id=0,
        iter_id=1,
        coordinate="engine.likelihood.default_beta",
        missing_entries=[
            {
                "position": 0,
                "point": {"engine.likelihood.default_beta": 5.0},
                "combination_index": 0,
            }
        ],
        value_jobs=1,
        repeat_jobs=2,
    )

    assert combinations[0].aggregated_error == pytest.approx(0.4)
    assert set(combinations[0].subject_metrics) == {103, 105}
    assert diagnostics["flat_task_count"] == 4
    assert diagnostics["flat_jobs"] == 4


def _raw_run(probability: float, particle_count: int) -> dict:
    choices = np.asarray([1, 1, 2, 2, 1, 2], dtype=int)
    probabilities = np.column_stack(
        [
            np.where(choices == 1, probability, 1.0 - probability),
            np.where(choices == 2, probability, 1.0 - probability),
        ]
    )
    metrics = compute_metrics_from_category_probabilities(
        probabilities,
        choices=choices,
        feedback=np.ones(choices.size),
        categories=choices,
        target_probs=np.eye(2)[choices - 1],
        window_size=2,
        loss_metric="choice_nll",
    )
    executed = np.tile(np.asarray([0.8, 0.2]), (choices.size, 1))
    return {
        "metrics_by_mode": {"prior_t": metrics},
        "state_log": {
            "filtered_executed_probability": executed,
            "post_choice_ess": np.full(choices.size, particle_count * 0.75),
        },
    }


def test_pf_convergence_gate_uses_probability_and_executed_state_stability() -> None:
    lower_row, lower_arrays = summarize_repeat_panel(
        [_raw_run(0.70, 64), _raw_run(0.72, 64)],
        prediction_mode="prior_t",
        particle_count=64,
    )
    upper_row, upper_arrays = summarize_repeat_panel(
        [_raw_run(0.705, 128), _raw_run(0.715, 128)],
        prediction_mode="prior_t",
        particle_count=128,
    )
    comparisons = compare_successive_counts(
        [lower_row, upper_row],
        {64: lower_arrays, 128: upper_arrays},
        gates={
            "maximum_successive_choice_nll_change": 0.01,
            "maximum_successive_choice_probability_rmse": 0.01,
            "maximum_successive_executed_posterior_js": 0.01,
            "maximum_split_half_choice_probability_rmse": 0.03,
            "minimum_median_post_choice_ess_fraction": 0.20,
        },
    )
    assert len(comparisons) == 1
    assert comparisons[0]["all_gates_passed"] is True
    assert lower_row["choice_nll"] == pytest.approx(-np.log(0.71))
