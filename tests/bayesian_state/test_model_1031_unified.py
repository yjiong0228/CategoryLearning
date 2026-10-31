"""Causal, joint-probability and replay checks for the opt-in unified revision."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.readout.choice import (
    resolve_choice_readout_config, resolve_output_noise_config,
)
from src.Bayesian_state.hypothesis_space.observation_model.pairing import pairing_feedback_kernel
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.simulation.execution import compute_prediction_metrics


ROOT = Path(__file__).resolve().parents[2]


def config(condition=3, **transition):
    cfg = yaml.safe_load((ROOT / "configs/exp123/model_struct" /
                         f"model_1031_cond{condition}_unified_v2.yaml").read_text())
    cfg["modules"]["perception_mod"]["kwargs"] = dict(
        features=4, mean=[0.] * 4, std=[0.] * 4, module_seed=7)
    cfg["modules"]["hypo_transitions_mod"]["kwargs"].update(transition)
    return cfg


def predict(model):
    return model.predict_choice(choices=[], feedback=[],
        choice_readout_config=resolve_choice_readout_config(None, model.engine_config),
        output_noise_config=resolve_output_noise_config(None, model.engine_config),
        rng=np.random.default_rng(31), sticky_state={}, post_error_lapse_state=0.)


@pytest.mark.parametrize("feedback", [0., .5, 1.])
def test_first_conditional_example_retains_exact_task_information(feedback):
    m = StateModel(config(), context=ModelContext(condition=3))
    x = [.2, .3, .4, .5]
    m.begin_trial(x)
    np.testing.assert_allclose(predict(m).observed_probabilities, .25)
    m.complete_trial(1, feedback)
    m.begin_trial(x)
    theta = np.asarray(m.partition_model.nodes[0]["category_belief"])
    allowed = pairing_feedback_kernel(np.eye(4), 1, feedback).T
    np.testing.assert_allclose(theta, allowed / allowed.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(m.partition_model.pairing_weights[0], 1 / 3)
    kernel = m.engine.observation_likelihood.pairing_kernel((x, 1, feedback), [0], beta=[25.])
    assert kernel.min() > .99999


@pytest.mark.parametrize("feedback", [0., .5, 1.])
def test_conditional_precision_uses_pre_feedback_joint_likelihood(feedback):
    m = StateModel(config(), context=ModelContext(condition=3))
    m.begin_trial([.2, .3, .4, .5])
    m.complete_trial(1, .5)
    m.begin_trial([.22, .31, .42, .53])
    likelihood = m.engine.observation_likelihood
    obs = ([.25, .35, .4, .5], 2, feedback)
    q = np.array([[.15, .6, .25]])
    step, beta = 1e-5, 2.3
    plus = (likelihood.pairing_kernel(obs, [0], beta=[beta + step]) * q).sum()
    minus = (likelihood.pairing_kernel(obs, [0], beta=[beta - step]) * q).sum()
    expected = (np.log(plus) - np.log(minus)) / (2 * step)
    observed = likelihood.feedback_precision_gradient(obs, [0], [beta], pairing_weights=q)
    np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=1e-6)


def test_conditional_update_is_one_joint_update_and_survives_snapshot():
    cfg = config(baseline_probability=.9)
    m = StateModel(cfg, context=ModelContext(condition=3))
    x = [.2, .3, .4, .5]
    m.begin_trial(x)
    m.complete_trial(1, .5)
    m.begin_trial(x)
    memory = m.engine.get_module(ModuleRole.MEMORY)
    memory.gamma = 1.
    theta = np.asarray(m.partition_model.nodes[0]["category_belief"])
    prior = memory.joint[0, :, None] * theta
    gain = m.partition_model.template_gain(m.partition_model.nodes[0], [x], m.engine.beta[0])[0][0]
    allowed = pairing_feedback_kernel(np.eye(4), 2, 1.).T
    expected = prior * ((1 - gain) * allowed.mean(axis=1, keepdims=True) + gain * allowed)
    expected /= expected.sum()
    m.complete_trial(2, 1.)
    observed = memory.joint[0, :, None] * np.asarray(m.partition_model.nodes[0]["category_belief"])
    np.testing.assert_allclose(observed, expected, atol=1e-14)
    clone = StateModel(cfg, context=ModelContext(condition=3))
    clone.engine.get_module(ModuleRole.MEMORY).gamma = 1.
    clone.engine.load_state_dict(m.engine.state_dict())
    np.testing.assert_allclose(clone.partition_model.pairing_weights, m.partition_model.pairing_weights)
    for own in (m, clone):
        own.begin_trial([.4, .4, .4, .4])
    np.testing.assert_array_equal(predict(clone).observed_probabilities, predict(m).observed_probabilities)
    assert clone.partition_model.nodes == m.partition_model.nodes
    clone.partition_model.nodes[0]["category_belief"][0][0] = .123
    assert clone.partition_model.nodes != m.partition_model.nodes


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_marginalized_execution_matches_exact_content_sum(condition):
    cfg = config(condition, execution_inertia=0., marginalize_execution=True,
                 baseline_probability=.9, retrieval_mode="similarity")
    m = StateModel(cfg, context=ModelContext(condition=condition))
    for t in range(12):
        x = [.1 + .05 * t, .3, .4, .5]
        m.begin_trial(x)
        transition = m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        probabilities = np.asarray([m.partition_model.get_category_probabilities(
            slot, ([x], [], []), m.engine.beta[slot])[:, 0] for slot in range(m.engine.set_size)])
        expected = transition.execution_selection_probabilities @ probabilities
        np.testing.assert_allclose(predict(m).observed_probabilities, expected, atol=1e-14)
        m.complete_trial(1, .5 if condition == 3 and t % 2 else float(t % 2))


def test_cond3_generation_replay_inherits_conditional_pairing():
    cfg = config(baseline_probability=.9, global_search=.8)
    m = StateModel(cfg, context=ModelContext(condition=3))
    x = np.random.default_rng(1031).uniform(.05, .95, (24, 4))
    categories = 1 + 2 * (x[:, 0] > .5) + (x[:, 1] > .5)
    def feedback(t, choice):
        return (1. if choice == categories[t] else
                .5 if (choice - 1) // 2 == (categories[t] - 1) // 2 else 0.)
    generated = m.generate_step_by_step(x, feedback, choice_seed=107)
    saved = deepcopy(m.partition_model.nodes)
    kwargs = dict(categories=categories, target_probs=None, window_size=1,
        loss_metric="choice_nll", beta_log=generated.beta,
        choice_readout_config=resolve_choice_readout_config(None, m.engine_config),
        output_noise_config=resolve_output_noise_config(None, m.engine_config))
    result = compute_prediction_metrics(m, generated.posterior, generated.prior, generated.step_log,
        x, generated.choices, generated.feedback, prediction_mode="prior_t", **kwargs)["prior_t"]
    valid = result["valid_trial_mask"]
    np.testing.assert_allclose(result["pred_category_probs"][valid], generated.observed_probabilities[valid],
                               atol=1e-14)
    assert saved == m.partition_model.nodes
    for log in generated.step_log:
        np.testing.assert_allclose(log["conditional_pairing_pre"].sum(axis=1), 1.)
        np.testing.assert_allclose(log["conditional_pairing_post"].sum(axis=1), 1.)


def test_cond3_particle_filter_is_causal_and_exports_conditional_state():
    cfg = config(execution_inertia=0., marginalize_execution=True, baseline_probability=.9)
    x = np.random.default_rng(1041).uniform(.05, .95, (8, 4))
    observed = []
    states = {}
    def observer(trial, phase, values):
        states[trial, phase] = deepcopy(dict(values))
        if phase == "filtered":
            observed.append(values["template_conditional_category_belief"].copy())
    kwargs = dict(engine_config=cfg, subject_id=0, condition=3, stimulus=x,
        choices=np.ones(8, int), particle_count=4, choice_readout_power=1., filter_seed=47)
    feedback = np.array([1., .5, 0., .5, 1., 0., .5, 0.])
    result = run_state_model_particle_filter(**kwargs, feedback=feedback, state_observer=observer,
                                             observe_decision_basis=True)
    for trial in range(8):
        prior, filtered = states[trial, "predictive"], states[trial, "filtered"]
        expected = prior["choice_weights"] * prior["rule_choice_probabilities"][:, :, 0]
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(filtered["choice_weights"], expected, atol=1e-14)
        np.testing.assert_array_equal(prior["belief"], filtered["belief"])
    feedback[-1] = 1.
    altered = run_state_model_particle_filter(**kwargs, feedback=feedback)
    np.testing.assert_array_equal(result.marginal_probabilities, altered.marginal_probabilities)
    assert np.asarray(observed).shape == (8, 4, 4, 3, 4)
    np.testing.assert_allclose(result.marginal_probabilities.sum(axis=1), 1.)


def test_incompatible_marginalization_is_rejected():
    with pytest.raises(ValueError, match="execution policy"):
        StateModel(config(execution_inertia=2., marginalize_execution=True), context=ModelContext(condition=3))
