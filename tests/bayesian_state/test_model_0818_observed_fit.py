from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.optimization.observed_fit import (
    ACCUMULATOR_GAIN_PATH,
    CAPACITY_PATH,
    EXECUTION_PATH,
    GLOBAL_GAIN_PATH,
    build_model_0818_hyper_config,
    extract_model_0818_parameters,
    summarize_observed_choice_fit,
)
from src.Bayesian_state.optimization.parameter_space import load_parameter_space
from src.Bayesian_state.optimization.search.coordinate_descent import (
    HyperCDOptimizer,
)
from src.Bayesian_state.model.modules.memory import DualMemoryModule
from src.Bayesian_state.simulation.config import (
    expand_profile_candidate_hyperparams,
    load_yaml,
)


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PATH = (
    ROOT / "configs/exp123/specific_models/model_0818_exploratory_observed_fit.yaml"
)
FULL_ANALYSIS_PATH = (
    ROOT / "configs/exp123/specific_models/model_0818_cond1_full_observed_fit.yaml"
)


def _optimizer_and_space() -> tuple[HyperCDOptimizer, dict[str, list[object]]]:
    analysis = load_yaml(ANALYSIS_PATH)
    parameter_space = load_parameter_space(ROOT / analysis["parameter_space"])
    hyper = build_model_0818_hyper_config(
        analysis,
        parameter_space,
        root=ROOT,
        parallel_budget=8,
    )
    optimizer = HyperCDOptimizer(hyper, ANALYSIS_PATH)
    space = {
        name: optimizer._hyperparam_values(spec)
        for name, spec in hyper["hyperparam_space"].items()
    }
    return optimizer, space


def test_observed_fit_config_is_explicitly_exploratory_and_first64() -> None:
    analysis = load_yaml(ANALYSIS_PATH)
    assert analysis["subjects"] == list(range(101, 133))
    assert analysis["trial_scope"] == {
        "mode": "first_n_trials_per_subject",
        "max_trials": 64,
        "condition": 1,
        "include_ambiguous_trials": True,
    }
    authorization = analysis["authorization"]
    assert authorization["user_authorized_observed_fit"] is True
    assert authorization["observed_choices_used"] is True
    assert authorization["exploratory_only"] is True
    assert authorization["formal_parameter_inference_authorized"] is False
    assert authorization["formal_module_inference_authorized"] is False
    assert authorization["manuscript_result_authorized"] is False
    assert analysis["optimization"]["final_rescore"] == {
        "particle_count": 128,
        "filter_seed_count": 128,
        "seed_family": "model0818_observed_final_r128_b128_v2",
        "cache_namespace": "R128_B128_final_v2",
        "probability_aggregation": "mean_probability_then_metrics",
    }


def test_full_observed_fit_uses_every_available_condition1_trial() -> None:
    analysis = load_yaml(FULL_ANALYSIS_PATH)
    assert analysis["subjects"] == list(range(101, 133))
    assert analysis["trial_scope"] == {
        "mode": "all_trials_per_subject",
        "max_trials": None,
        "condition": 1,
        "include_ambiguous_trials": True,
    }
    assert analysis["output_dir"].endswith("/full_observed_fit_v1")
    parameter_space = load_parameter_space(ROOT / analysis["parameter_space"])
    hyper = build_model_0818_hyper_config(
        analysis,
        parameter_space,
        root=ROOT,
        parallel_budget=8,
    )
    assert hyper["stages"]["coarse"]["simulation_overrides"]["max_trials"] is None
    base = load_yaml(ROOT / analysis["base_simulation_config"])
    assert base["max_trials"] is None


def test_hyper_config_builds_joint_coordinates_and_two_exact_starts() -> None:
    optimizer, space = _optimizer_and_space()
    assert [len(values) for values in space.values()] == [9, 7, 30, 7, 42, 6, 7, 7]
    assert optimizer.common_random_numbers_within_candidate_comparisons is True
    assert optimizer.n_restarts == 2

    starts = [
        optimizer._init_point(space, random.Random(1), restart_id)
        for restart_id in range(2)
    ]
    expanded = [expand_profile_candidate_hyperparams(point) for point in starts]
    assert [point[CAPACITY_PATH] for point in expanded] == [3, 3]
    assert [point[EXECUTION_PATH] for point in expanded] == [False, True]
    assert all(point[ACCUMULATOR_GAIN_PATH] == 0.0 for point in expanded)
    assert all(point[GLOBAL_GAIN_PATH] == 0.0 for point in expanded)
    for point in starts:
        named = extract_model_0818_parameters(point)
        assert named["delta_E"] > 0.0
        assert named["c_A"] == 0.0
        assert named["c_G"] == 0.0


def test_named_profile_coordinates_expand_without_silent_overwrite() -> None:
    packed = {
        "__profile_candidate__:one": {"engine.a": 1, "engine.b": 2},
        "__profile_candidate__:two": {"engine.c": 3},
    }
    assert expand_profile_candidate_hyperparams(packed) == {
        "engine.a": 1,
        "engine.b": 2,
        "engine.c": 3,
    }
    with pytest.raises(ValueError, match="duplicate hyperparameter path"):
        expand_profile_candidate_hyperparams(
            {
                "__profile_candidate__:one": {"engine.a": 1},
                "engine.a": 2,
            }
        )
    with pytest.raises(ValueError, match="duplicate hyperparameter path"):
        expand_profile_candidate_hyperparams(
            {
                "engine.a": 2,
                "__profile_candidate__:one": {"engine.a": 1},
            }
        )


def test_common_random_numbers_pair_candidate_comparisons() -> None:
    optimizer, _ = _optimizer_and_space()
    seed_a = optimizer._simulation_point_seed(
        stage_name="coarse",
        hyper_candidate_seed=10,
        subject_id=103,
        point={"parameter": 1},
    )
    seed_b = optimizer._simulation_point_seed(
        stage_name="coarse",
        hyper_candidate_seed=99,
        subject_id=103,
        point={"parameter": 2},
    )
    assert seed_a == seed_b
    assert seed_a != optimizer._simulation_point_seed(
        stage_name="coarse",
        hyper_candidate_seed=99,
        subject_id=104,
        point={"parameter": 2},
    )


def test_observed_fit_metrics_use_mean_probability_and_causal_baseline() -> None:
    runs = np.asarray(
        [
            [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]],
            [[0.6, 0.4], [0.5, 0.5], [0.8, 0.2], [0.3, 0.7]],
        ],
        dtype=float,
    )
    choices = np.asarray([1, 2, 1, 2])
    categories = np.asarray([1, 2, 2, 2])
    summary, mean_probability = summarize_observed_choice_fit(
        runs, choices, categories, window_size=2
    )
    expected = runs.mean(axis=0)
    selected = expected[np.arange(4), choices - 1]
    assert np.allclose(mean_probability, expected)
    assert summary["mean_trial_nll"] == pytest.approx(-np.log(selected).mean())
    assert summary["random_mean_trial_nll"] == pytest.approx(np.log(2.0))
    assert summary["causal_bias_mean_trial_nll"] == pytest.approx(
        -np.mean(np.log([0.5, 0.25, 0.5, 0.375]))
    )
    assert np.isfinite(summary["seed_nll_sd"])
    assert np.isfinite(summary["maximum_trial_probability_mcse"])


def test_observed_fit_metrics_reject_invalid_seed_bank() -> None:
    with pytest.raises(ValueError, match="at least two filter seeds"):
        summarize_observed_choice_fit(
            np.asarray([[[0.5, 0.5], [0.5, 0.5]]]),
            [1, 2],
            [1, 2],
            window_size=2,
        )


def test_dual_memory_gamma_zero_is_an_exact_finite_boundary() -> None:
    engine = SimpleNamespace(
        state={},
        hypotheses_mask=np.asarray([1.0, 0.0]),
        set_size=2,
        prior=np.asarray([1.0, 0.0]),
    )
    memory = DualMemoryModule(engine, gamma=0.0, w0=0.0)
    with np.errstate(invalid="raise"):
        memory.state_update(np.asarray([0.75, 0.25]))
    assert memory.state["fade"][0] == pytest.approx(np.log(0.75))
    assert memory.state["fade"][1] == -np.inf
    assert np.isfinite(memory.baseline_state["fade"])
