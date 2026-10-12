"""Integrate the revised 0923 equations with prediction, snapshots and filtering."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
from src.Bayesian_state.inference.backends.particle_filter import (
    _particle_config,
    run_state_model_particle_filter,
)
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.readout.choice import (
    resolve_choice_readout_config,
    resolve_output_noise_config,
)
from src.Bayesian_state.simulation.data import TrialArrays
from src.Bayesian_state.simulation.execution import compute_prediction_metrics, evaluate_state_model_run
from src.Bayesian_state.simulation.parameters import infer_fixed_hyperparams_from_engine_config


ROOT = Path(__file__).resolve().parents[2]
STIMULI = np.array([
    [.8, .3, .2, .4], [.2, .7, .8, .6], [.6, .4, .3, .7], [.4, .6, .7, .3],
])
CHOICES = np.array([1, 2, 1, 2])


@pytest.fixture(autouse=True)
def temporary_similarity_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, "RUNTIME_CACHE_DIR", tmp_path / "similarity")


def _config(condition):
    path = ROOT / f"configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml"
    config = yaml.safe_load(path.read_text())
    config["modules"]["perception_mod"]["kwargs"] = {
        "features": 4, "mean": [0.] * 4, "std": [0.] * 4, "module_seed": 32,
    }
    transition = config["modules"]["hypo_transitions_mod"]["kwargs"]
    transition.update(relation_samples=512, module_seed=23, exit_rate=0.,
                      transfer_scale=.5, execution_inertia=np.log(3.))
    transition["search_controller"]["baseline_probability"] = .99
    config["provenance"]["hypothesis_similarity"]["sample_count"] = 512
    return config


def _feedback(condition):
    return np.array([0., .5 if condition == 3 else 0., 1., 0.])


def _model(config, condition):
    return StateModel(config, context=ModelContext(condition=condition))


def _choose(model, choices=(), feedback=()):
    return model.predict_choice(
        choices=choices, feedback=feedback,
        choice_readout_config=resolve_choice_readout_config(None, model.engine_config),
        output_noise_config=resolve_output_noise_config(None, model.engine_config),
        rng=np.random.default_rng(12), sticky_state={}, post_error_lapse_state=0.,
    )


def _assert_state_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_state_equal(actual_item, expected_item)
    else:
        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_live_choice_uses_continuous_inertial_weights(condition):
    config = _config(condition)
    model = _model(config, condition)
    transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
    feedback = _feedback(condition)
    previous_dominant = None
    saw_multiple_contributors = False
    for index, stimulus in enumerate(STIMULI):
        prepared = model.begin_trial(stimulus)
        decision = _choose(model, CHOICES[:index], feedback[:index])
        snapshot = decision.rule_snapshot
        assert snapshot is not None
        expected = prepared.prior.copy()
        if previous_dominant is not None:
            expected[previous_dominant] *= 3.
        expected /= expected.sum()
        np.testing.assert_allclose(snapshot.rule_weights, expected, atol=1e-14)
        np.testing.assert_allclose(decision.observed_probabilities,
                                   expected @ snapshot.category_probabilities, atol=1e-14)
        assert transition.continuous_choice_enabled
        assert not transition.persistent_execution_enabled
        dominant = transition.dominant_hypothesis
        assert expected[dominant] == pytest.approx(expected.max())
        np.testing.assert_allclose(prepared.log["choice_weights"], expected)
        assert prepared.log["dominant_hypothesis"] == dominant
        saw_multiple_contributors |= np.count_nonzero(expected) > 1
        previous_dominant = dominant

        posterior, _, _ = model.complete_trial(CHOICES[index], feedback[index])
        active = np.asarray(model.engine.hypotheses_mask) > 0
        assert np.isfinite(posterior).all() and np.all(posterior >= 0.)
        assert posterior.sum() == pytest.approx(1.)
        assert np.all(posterior[~active] == 0.)
        assert np.all((model.engine.beta[active] >= .1) & (model.engine.beta[active] <= 25.))
        if condition == 3:
            memory = model.engine.get_module(ModuleRole.MEMORY)
            np.testing.assert_allclose(memory.joint.sum(axis=1), posterior, atol=1e-14)
            np.testing.assert_allclose(memory.conditional_pairing()[active].sum(axis=1), 1.)
    assert saw_multiple_contributors


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_snapshot_restores_a_different_history_and_future_predictions(condition):
    config = _config(condition)
    source, restored = [_model(config, condition) for _ in range(2)]
    # Histories have equal length but different feedback. Restoration must replace
    # learned beliefs, precisions, search state and the incumbent rule together.
    for index in range(2):
        for model, feedback in ((source, 0.), (restored, 1.)):
            model.begin_trial(STIMULI[index])
            _choose(model)
            model.complete_trial(CHOICES[index], feedback)
    saved = deepcopy(source.engine.state_dict())
    restored.engine.load_state_dict(saved)
    _assert_state_equal(restored.engine.state_dict(), saved)

    for index in range(2, len(STIMULI)):
        decisions = []
        for model in (source, restored):
            model.begin_trial(STIMULI[index])
            decisions.append(_choose(model))
            model.complete_trial(CHOICES[index], _feedback(condition)[index])
        np.testing.assert_array_equal(decisions[0].observed_probabilities,
                                      decisions[1].observed_probabilities)
        np.testing.assert_array_equal(decisions[0].rule_snapshot.rule_weights,
                                      decisions[1].rule_snapshot.rule_weights)
        _assert_state_equal(restored.engine.state_dict(), source.engine.state_dict())


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_saved_trajectory_reproduces_live_continuous_predictions(condition):
    config = _config(condition)
    model = _model(config, condition)
    feedback = _feedback(condition)
    predictions, priors, posteriors, beta_log, step_log = [], [], [], [], []
    for trial, stimulus in enumerate(STIMULI):
        prepared = model.begin_trial(stimulus)
        predictions.append(_choose(model, CHOICES[:trial], feedback[:trial]).observed_probabilities)
        beta_log.append(prepared.beta)
        posterior, prior, log = model.complete_trial(CHOICES[trial], feedback[trial])
        priors.append(prior)
        posteriors.append(posterior)
        step_log.append(log)
    metrics = compute_prediction_metrics(
        model, posteriors, priors, step_log, STIMULI, CHOICES, feedback,
        categories=np.array([2, 1, 1, 1]) if condition == 1 else np.array([3, 1, 1, 4]),
        target_probs=None, window_size=1,
        prediction_mode="prior_t", loss_metric="choice_nll", beta_log=beta_log,
        choice_readout_config=resolve_choice_readout_config(None, config),
        output_noise_config=resolve_output_noise_config(None, config),
    )["prior_t"]
    # The existing trajectory metric path excludes its first alignment trial.
    valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    assert valid.any()
    np.testing.assert_allclose(metrics["pred_category_probs"][valid],
                               np.asarray(predictions)[valid], atol=1e-14)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_particle_filter_is_causal_normalized_and_reproducible(condition):
    config = _config(condition)
    arguments = dict(
        engine_config=config, subject_id=122, condition=condition,
        stimulus=STIMULI, choices=CHOICES, particle_count=4,
        choice_readout_power=1., filter_seed=89, resample_threshold_fraction=1.,
    )
    feedback = _feedback(condition)
    first = run_state_model_particle_filter(feedback=feedback, **arguments)
    repeated = run_state_model_particle_filter(feedback=feedback, **arguments)
    np.testing.assert_array_equal(first.marginal_probabilities, repeated.marginal_probabilities)
    np.testing.assert_array_equal(first.final_weights, repeated.final_weights)
    assert first.marginal_probabilities.shape == (4, 2 if condition == 1 else 4)
    assert np.isfinite(first.marginal_probabilities).all()
    np.testing.assert_allclose(first.marginal_probabilities.sum(axis=1), 1.)
    active_count = first.marginal_active_probability.sum(axis=1)
    assert np.all((active_count >= 1-1e-12) & (active_count <= 3+1e-12))
    assert first.resampled.any()
    for key in ("choice_weights", "dominant_probability",
                "filtered_choice_weights", "filtered_dominant_probability"):
        values = first.state_probabilities[key]
        assert np.isfinite(values).all() and np.all(values >= 0.)
        np.testing.assert_allclose(values.sum(axis=1), 1.)
        np.testing.assert_array_equal(values, repeated.state_probabilities[key])
    assert np.all(first.latent_summaries["effective_participation"] >= 1-1e-12)
    assert np.all(first.latent_summaries["effective_participation"] <= active_count+1e-12)

    altered_feedback = feedback.copy()
    altered_feedback[-1] = 1.
    altered = run_state_model_particle_filter(feedback=altered_feedback, **arguments)
    np.testing.assert_array_equal(first.marginal_probabilities, altered.marginal_probabilities)
    np.testing.assert_array_equal(first.final_weights, altered.final_weights)
    if condition == 3:
        for key in ("pairing_prior", "pairing_posterior"):
            values = first.state_probabilities[key]
            assert np.isfinite(values).all() and np.all(values >= 0.)
            np.testing.assert_allclose(values.sum(axis=1), 1.)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_filter_prediction_and_state_summaries_match_explicit_path_pooling(condition):
    config = _config(condition)
    filter_seed, particle_count = 61, 4
    feedback = _feedback(condition)
    result = run_state_model_particle_filter(
        engine_config=config, subject_id=122, condition=condition,
        stimulus=STIMULI, choices=CHOICES, feedback=feedback,
        particle_count=particle_count, choice_readout_power=1.,
        filter_seed=filter_seed, resample_threshold_fraction=0.,
    )
    models = [_model(_particle_config(config, filter_seed, index), condition)
              for index in range(particle_count)]
    weights = np.full(particle_count, 1/particle_count)
    for trial, stimulus in enumerate(STIMULI):
        predictions, contributions, dominant, participation = [], [], [], []
        for model in models:
            model.begin_trial(stimulus)
            decision = _choose(model, CHOICES[:trial], feedback[:trial])
            predictions.append(decision.observed_probabilities)
            contributions.append(decision.rule_snapshot.rule_weights)
            transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            one_hot = np.zeros(model.engine.set_size)
            one_hot[transition.dominant_hypothesis] = 1.
            dominant.append(one_hot)
            participation.append(1/np.sum(decision.rule_snapshot.rule_weights**2))
        predictions = np.asarray(predictions)
        np.testing.assert_allclose(result.marginal_probabilities[trial], weights @ predictions)
        np.testing.assert_allclose(result.state_probabilities["choice_weights"][trial],
                                   weights @ contributions)
        np.testing.assert_allclose(result.state_probabilities["dominant_probability"][trial],
                                   weights @ dominant)
        assert result.latent_summaries["effective_participation"][trial] == pytest.approx(weights @ participation)
        weights *= predictions[:, CHOICES[trial]-1]
        weights /= weights.sum()
        np.testing.assert_allclose(result.state_probabilities["filtered_choice_weights"][trial],
                                   weights @ contributions)
        np.testing.assert_allclose(result.state_probabilities["filtered_dominant_probability"][trial],
                                   weights @ dominant)
        assert result.latent_summaries["filtered_effective_participation"][trial] == pytest.approx(weights @ participation)
        for model in models:
            model.complete_trial(CHOICES[trial], feedback[trial])


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_simulation_preserves_continuous_choice_summaries_and_timing(condition):
    config = _config(condition)
    config["inference"]["particle_count"] = 4
    arrays = TrialArrays(
        stimulus=STIMULI, choices=CHOICES, feedback=_feedback(condition),
        categories=np.array([2, 1, 1, 1]) if condition == 1 else np.array([3, 1, 1, 4]),
    )
    run = evaluate_state_model_run(
        subject_id=122, condition=condition, arrays=arrays, params={},
        engine_config_template=config, processed_data_dir=ROOT / "data/exp123/processed",
        window_size=1, prediction_mode="prior_t", selection_prediction_mode="prior_t",
        loss_metric="choice_nll", run_seed=89, keep_logs=True,
    )
    metrics = run.metrics_by_mode["prior_t"]
    assert np.isfinite(run.mean_error)
    for key in ("choice_weights", "dominant_probability",
                "filtered_choice_weights", "filtered_dominant_probability",
                "effective_participation", "filtered_effective_participation"):
        np.testing.assert_array_equal(run.state_log[key], metrics[f"particle_{key}"])
        assert np.isfinite(run.state_log[key]).all()
    np.testing.assert_allclose(run.state_log["choice_weights"].sum(axis=1), 1.)
    np.testing.assert_allclose(run.state_log["dominant_probability"].sum(axis=1), 1.)
    metadata = metrics["continuous_choice_metadata"]
    assert metadata == run.state_log["continuous_choice_metadata"]
    assert metadata["timing"] == "pre_feedback_state"
    assert metadata["unfiltered_weights"] == "pre_choice_particle_weights"
    assert metadata["filtered_weights"] == "post_choice_particle_weights"
    assert metadata["effective_participation"] == "particle_mean_of_inverse_sum_squared_rule_weights"
    assert "particle_executed_probability" not in metrics


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_fixed_parameter_extraction_records_its_new_mechanisms(condition):
    config = _config(condition)
    parameters = infer_fixed_hyperparams_from_engine_config(config)
    assert parameters["engine.modules.hypo_transitions_mod.kwargs.mechanism_version"] == "0923-full-v3"
    assert parameters["engine.modules.beta_mod.kwargs.learning_rate"] == config["modules"]["beta_mod"]["kwargs"]["learning_rate"]
    assert parameters["engine.modules.memory_mod.kwargs.forgetting_mode"] == (
        "factorized" if condition == 3 else "exact_power"
    )
