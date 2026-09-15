"""Condition-3 PF integration, observation timing, and cognitive snapshots."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.inference.backends.particle_filter import (
    _particle_config,
    _restore,
    _snapshot,
    run_state_model_particle_filter,
)
from src.Bayesian_state.inference.dispatcher import run_inference_backend
from src.Bayesian_state.model import StateModel
from src.Bayesian_state.model.config import ModelContext
from src.Bayesian_state.model.modules.base_module import ModuleRole


@pytest.fixture
def case():
    config = yaml.safe_load(
        Path("configs/exp123/model_struct/pmh_model_cond3_0826.yaml").read_text()
    )
    config["inference"]["particle_count"] = 2
    frame = pd.read_csv("data/exp123/processed/Task2_processed.csv")
    frame = frame.loc[frame.iSub.eq(301)].iloc[:8]
    return dict(
        engine_config=config,
        subject_id=301,
        condition=3,
        stimulus=frame[[f"feature{i}" for i in range(1, 5)]].to_numpy(),
        choices=frame.choice.to_numpy(),
        feedback=frame.feedback.to_numpy(),
        inference_seed=20260915,
    )


def _direct_case(case):
    values = deepcopy(case)
    values["filter_seed"] = values.pop("inference_seed")
    values.update(particle_count=2, choice_readout_power=1.0)
    return values


@pytest.mark.parametrize("execution", [False, True])
def test_condition3_normalized_deterministic_pairing_and_choice_predictions(case, execution):
    config = case["engine_config"]
    config["modules"]["hypo_transitions_mod"]["kwargs"]["persistent_execution"]["enabled"] = execution
    result = run_inference_backend(**case)
    repeated = run_inference_backend(**case)
    assert result.marginal_probabilities.shape == (8, 4)
    for key in ("pairing_prior", "pairing_posterior"):
        probabilities = result.state_probabilities[key]
        assert probabilities.shape == (8, 3)
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_array_equal(probabilities, repeated.state_probabilities[key])
        entropy = result.diagnostics[f"{key}_entropy"]
        confidence = result.diagnostics[f"{key}_confidence"]
        assert np.all((entropy >= 0) & (entropy <= np.log(3) + 1e-12))
        np.testing.assert_allclose(confidence, probabilities.max(axis=1))
    np.testing.assert_allclose(result.marginal_probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(result.marginal_probabilities, repeated.marginal_probabilities)
    np.testing.assert_allclose(result.state_probabilities["pairing_prior"][0], 1 / 3)
    assert result.metadata["pairing_order"] == ["12|34", "13|24", "14|23"]
    assert result.metadata["probability_coordinate"] == "choice"


def test_current_feedback_changes_cognition_but_not_current_pf_weights(case):
    direct = _direct_case(case)
    direct["resample_threshold_fraction"] = 0.0
    original = run_state_model_particle_filter(**direct)
    changed = deepcopy(direct)
    changed["feedback"][-1] = 0.5 if direct["feedback"][-1] != 0.5 else 0.0
    alternate = run_state_model_particle_filter(**changed)
    np.testing.assert_array_equal(original.marginal_probabilities, alternate.marginal_probabilities)
    np.testing.assert_array_equal(original.final_weights, alternate.final_weights)
    np.testing.assert_array_equal(
        original.state_probabilities["pairing_prior"], alternate.state_probabilities["pairing_prior"]
    )
    np.testing.assert_array_equal(
        original.state_probabilities["pairing_posterior"][:-1],
        alternate.state_probabilities["pairing_posterior"][:-1],
    )
    assert not np.allclose(
        original.state_probabilities["pairing_posterior"][-1],
        alternate.state_probabilities["pairing_posterior"][-1],
    )


def test_current_choice_does_not_leak_into_current_predictions(case):
    original = run_inference_backend(**case)
    changed = deepcopy(case)
    changed["choices"][-1] = (changed["choices"][-1] % 4) + 1
    alternate = run_inference_backend(**changed)
    np.testing.assert_array_equal(original.marginal_probabilities, alternate.marginal_probabilities)
    np.testing.assert_array_equal(
        original.state_probabilities["pairing_prior"], alternate.state_probabilities["pairing_prior"]
    )


def test_pairing_summaries_use_choice_importance_weights(case):
    from src.Bayesian_state.model.readout import read_choice_probabilities_from_model

    direct = _direct_case(case)
    direct["resample_threshold_fraction"] = 0.0
    direct["stimulus"] = direct["stimulus"][:2]
    direct["choices"] = direct["choices"][:2]
    direct["feedback"] = direct["feedback"][:2]
    result = run_state_model_particle_filter(**direct)
    models = [
        StateModel(
            _particle_config(direct["engine_config"], direct["filter_seed"], index),
            context=ModelContext(condition=3, subject_id=301),
        )
        for index in range(2)
    ]
    weights = np.asarray([0.5, 0.5])
    for trial in range(2):
        priors, likelihoods = [], []
        for model in models:
            prepared = model.begin_trial(direct["stimulus"][trial])
            priors.append(model.engine.get_module(ModuleRole.MEMORY).pairing_marginal())
            likelihoods.append(read_choice_probabilities_from_model(
                model, prepared.perceived_stimulus, power=1.0, lapse=0.0
            )[int(direct["choices"][trial]) - 1])
        np.testing.assert_allclose(
            result.state_probabilities["pairing_prior"][trial], weights @ np.asarray(priors)
        )
        weights = weights * likelihoods
        weights /= weights.sum()
        posteriors = []
        for model in models:
            model.complete_trial(int(direct["choices"][trial]), float(direct["feedback"][trial]))
            posteriors.append(model.engine.get_module(ModuleRole.MEMORY).pairing_marginal())
        np.testing.assert_allclose(
            result.state_probabilities["pairing_posterior"][trial], weights @ np.asarray(posteriors)
        )
    assert not np.allclose(weights, 0.5)
    np.testing.assert_allclose(result.final_weights, weights)


def test_condition3_resampling_preserves_valid_pairing_state(case):
    case["engine_config"]["inference"]["resample_threshold_fraction"] = 1.0
    result = run_inference_backend(**case)
    assert result.resampled.any()
    np.testing.assert_allclose(result.state_probabilities["pairing_posterior"].sum(axis=1), 1.0)
    repeated = run_inference_backend(**case)
    np.testing.assert_array_equal(
        result.state_probabilities["pairing_posterior"], repeated.state_probabilities["pairing_posterior"]
    )


def test_condition3_snapshot_restores_joint_and_future_memory_updates(case):
    config = _particle_config(case["engine_config"], 123, 0)
    source = StateModel(config, context=ModelContext(condition=3, subject_id=301))
    target = StateModel(config, context=ModelContext(condition=3, subject_id=301))
    for trial in range(3):
        source.begin_trial(case["stimulus"][trial])
        source.complete_trial(int(case["choices"][trial]), float(case["feedback"][trial]))
    memory = source.engine.get_module(ModuleRole.MEMORY, required=True)
    expected_joint = memory.joint.copy()
    snapshot = _snapshot(source)
    source.begin_trial(case["stimulus"][3])
    source.complete_trial(int(case["choices"][3]), float(case["feedback"][3]))
    for model in (source, target):
        _restore(model, snapshot, filter_seed=456, trial_index=3, particle_index=0)
        np.testing.assert_array_equal(
            model.engine.get_module(ModuleRole.MEMORY, required=True).joint, expected_joint
        )
        model.begin_trial(case["stimulus"][4])
        model.complete_trial(int(case["choices"][4]), 0.5)
    np.testing.assert_array_equal(
        source.engine.get_module(ModuleRole.MEMORY, required=True).joint,
        target.engine.get_module(ModuleRole.MEMORY, required=True).joint,
    )


@pytest.mark.parametrize("invalid", [0.25, 0.75, float("nan")])
def test_condition3_rejects_non_task_feedback(case, invalid):
    case["feedback"][0] = invalid
    with pytest.raises(ValueError, match="0.*0.5.*1"):
        run_inference_backend(**case)


@pytest.mark.parametrize("invalid", [1.5, float("nan"), float("inf"), 0, 5])
@pytest.mark.parametrize("dispatch", [False, True])
def test_condition3_rejects_invalid_response_ids_before_integer_conversion(case, invalid, dispatch):
    case["choices"] = np.asarray(case["choices"], dtype=float)
    case["choices"][0] = invalid
    with pytest.raises(ValueError, match="choices.*finite integers.*1.*4"):
        if dispatch:
            run_inference_backend(**case)
        else:
            run_state_model_particle_filter(**_direct_case(case))


def test_condition3_binary_only_prefix_keeps_pairing_mode_and_uniform_lapse(case):
    case["feedback"] = np.zeros(8)
    result = run_inference_backend(**case, output_lapse=1.0)
    np.testing.assert_allclose(result.marginal_probabilities, 0.25)
    assert "pairing_posterior" in result.state_probabilities


def test_condition3_rejects_binary_transmission_audit(case):
    case["engine_config"]["inference"]["choice_transmission_audit"] = True
    with pytest.raises(ValueError, match="audit.*condition 1"):
        run_inference_backend(**case)


def test_simulation_exports_pairing_diagnostics_and_response_coordinates(case):
    from src.Bayesian_state.simulation.data import TrialArrays
    from src.Bayesian_state.simulation.execution import evaluate_state_model_run
    from src.Bayesian_state.simulation.runner import aggregate_simulation_runs

    mapping = {1: 4, 2: 2, 3: 3, 4: 1}
    frame = pd.read_csv("data/exp123/processed/Task2_processed.csv")
    categories = frame.loc[frame.iSub.eq(301)].iloc[:8].category.to_numpy()
    arrays = TrialArrays(
        stimulus=case["stimulus"], choices=case["choices"], feedback=case["feedback"],
        categories=categories,
        presskeys=np.asarray([mapping[int(choice)] for choice in case["choices"]]),
        choice_to_presskey=mapping,
        presskey_to_choice={key: choice for choice, key in mapping.items()},
    )
    run = evaluate_state_model_run(
        subject_id=301, condition=3, arrays=arrays, params={},
        engine_config_template=case["engine_config"],
        processed_data_dir=Path("data/exp123/processed"), window_size=2,
        prediction_mode="prior_t", selection_prediction_mode="prior_t",
        loss_metric="choice_nll", run_seed=case["inference_seed"], keep_logs=True,
    )
    metrics = run.metrics_by_mode["prior_t"]
    assert metrics["observed_task_metrics"] == pytest.approx({
        "species_accuracy": 0.0, "family_accuracy": 3 / 7, "mean_reward": 1.5 / 7,
    })
    np.testing.assert_allclose(metrics["particle_pairing_prior"].sum(axis=1), 1.0)
    np.testing.assert_array_equal(run.state_log["pairing_prior"], metrics["particle_pairing_prior"])
    assert metrics["response_key_mapping"]["choice_to_presskey"] == mapping
    assert metrics["probability_coordinate"] == "choice"
    np.testing.assert_array_equal(metrics["observed_presskey"], arrays.presskeys)
    # This also guards the no-logs probability aggregation used by fitting.
    summary = aggregate_simulation_runs(
        [run, run], params={}, subject_id=301, condition=3, window_size=2,
        selection_prediction_mode="prior_t", simulation_repeats=2,
        simulation_point_seed=123, keep_logs=False, compute_statistics=False,
        repeat_aggregation="mean_probability",
    )
    summary_metrics = summary.metrics_by_mode["prior_t"]
    assert summary_metrics["observed_task_metrics"] == metrics["observed_task_metrics"]
    assert summary_metrics["predicted_task_metrics"] == pytest.approx(metrics["predicted_task_metrics"])
    assert summary_metrics["response_key_mapping"] == metrics["response_key_mapping"]
    np.testing.assert_allclose(
        summary_metrics["particle_pairing_posterior"], metrics["particle_pairing_posterior"]
    )
    changed_arrays = deepcopy(arrays)
    changed_arrays.categories = categories % 4 + 1
    alternate = evaluate_state_model_run(
        subject_id=301, condition=3, arrays=changed_arrays, params={},
        engine_config_template=case["engine_config"],
        processed_data_dir=Path("data/exp123/processed"), window_size=2,
        prediction_mode="prior_t", selection_prediction_mode="prior_t",
        loss_metric="choice_nll", run_seed=case["inference_seed"], keep_logs=False,
    )
    np.testing.assert_array_equal(
        alternate.metrics_by_mode["prior_t"]["pred_category_probs"], metrics["pred_category_probs"]
    )


def test_repeat_pairing_entropy_is_computed_after_averaging_distributions():
    from src.Bayesian_state.simulation.execution import compute_metrics_from_category_probabilities
    from src.Bayesian_state.simulation.results import SingleRunResult
    from src.Bayesian_state.simulation.runner import aggregate_simulation_runs

    metrics = compute_metrics_from_category_probabilities(
        np.full((4, 4), 0.25), choices=np.ones(4, dtype=int),
        feedback=np.ones(4), categories=np.ones(4, dtype=int), target_probs=None,
        window_size=2, loss_metric="choice_nll",
    )
    runs = []
    for pairing in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
        run_metrics = deepcopy(metrics)
        run_metrics["particle_pairing_posterior"] = np.tile(pairing, (4, 1))
        run_metrics["particle_pairing_posterior_entropy"] = np.zeros(4)
        runs.append(SingleRunResult(
            params={}, mean_error=float(metrics["mean_error"]),
            metrics_by_mode={"prior_t": run_metrics}, selection_prediction_mode="prior_t",
            loss_metric="choice_nll", loss_delta=None,
        ))
    summary = aggregate_simulation_runs(
        runs, params={}, subject_id=301, condition=3, window_size=2,
        selection_prediction_mode="prior_t", simulation_repeats=2,
        simulation_point_seed=123, keep_logs=False, compute_statistics=False,
        repeat_aggregation="mean_probability",
    ).metrics_by_mode["prior_t"]
    np.testing.assert_allclose(summary["particle_pairing_posterior"], np.tile([0.5, 0.5, 0.0], (4, 1)))
    np.testing.assert_allclose(summary["particle_pairing_posterior_entropy"], np.log(2))
    np.testing.assert_allclose(summary["particle_pairing_posterior_confidence"], 0.5)


@pytest.mark.parametrize("with_categories", [False, True])
def test_task_summaries_distinguish_species_family_reward_and_preserve_scoring_mask(with_categories):
    from src.Bayesian_state.metrics.task import condition3_task_metric_summaries

    summary = condition3_task_metric_summaries(
        probabilities=np.asarray([
            [1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.4, 0.1, 0.3], [0.0, 0.0, 1.0, 0.0],
        ]),
        choices=np.asarray([1, 2, 4, 3]), feedback=np.asarray([1.0, 0.5, 0.0, 1.0]),
        categories=np.asarray([1, 1, 2, 3]) if with_categories else None,
        valid_trial_mask=np.asarray([False, True, True, False]),
    )
    assert summary["observed_task_metrics"] == pytest.approx({
        "species_accuracy": 0.0, "family_accuracy": 0.5, "mean_reward": 0.25,
    })
    assert summary["task_metrics_n_trials"] == 2
    if with_categories:
        assert summary["predicted_task_metrics"] == pytest.approx({
            "species_accuracy": 0.25, "family_accuracy": 0.45, "mean_reward": 0.35,
        })
    else:
        assert all(np.isnan(value) for value in summary["predicted_task_metrics"].values())
