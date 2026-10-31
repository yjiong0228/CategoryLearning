"""Mechanism, timing and particle-identity checks for the experience candidate."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.readout.choice import (
    resolve_choice_readout_config, resolve_output_noise_config)
from src.Bayesian_state.inference.backends.particle_filter import _particle_config, run_state_model_particle_filter
from src.Bayesian_state.simulation.execution import compute_prediction_metrics
from src.Bayesian_state.hypothesis_space.observation_model.experience_workspace import (
    ExperienceWorkspacePartition, ExperienceWorkspaceLikelihood)


ROOT = Path(__file__).resolve().parents[2]


def config(condition=1, **transition):
    value = yaml.safe_load((ROOT / f"configs/exp123/model_struct/model_1031_cond{condition}_experience_v1.yaml").read_text())
    value["modules"]["perception_mod"]["kwargs"] = dict(features=4, mean=[0.] * 4, std=[0.] * 4, module_seed=7)
    value["modules"]["hypo_transitions_mod"]["kwargs"].update(transition)
    return value


def make_model(condition=1, **transition):
    return StateModel(config(condition, **transition), context=ModelContext(condition=condition))


def predict(model):
    return model.predict_choice(choices=[], feedback=[],
        choice_readout_config=resolve_choice_readout_config(None, model.engine_config),
        output_noise_config=resolve_output_noise_config(None, model.engine_config),
        rng=np.random.default_rng(1), sticky_state={}, post_error_lapse_state=0.)


def template(theta, mask=(True, True, True, True), uid=1):
    return dict(kind="template", anchor=[.2, .3, .4, .5], mask=list(mask),
                category_belief=list(theta), uid=uid, parent_uid=-1, born_trial=1, source_trial=0)


@pytest.mark.parametrize("condition", [1, 2])
def test_first_choice_is_uniform_and_first_memory_uses_only_completed_feedback(condition):
    m = make_model(condition)
    m.begin_trial([.2, .3, .4, .5])
    np.testing.assert_allclose(predict(m).observed_probabilities, 1 / m.n_cats)
    assert m.partition_model.nodes[0]["kind"] == "uncommitted"
    m.complete_trial(1, 0.)
    m.begin_trial([.9, .9, .9, .9])
    node = m.partition_model.nodes[0]
    assert node["anchor"] == [.2, .3, .4, .5]
    np.testing.assert_allclose(node["category_belief"], [0.] + [1 / (m.n_cats - 1)] * (m.n_cats - 1))
    assert node["mask"] == [True] * 4


@pytest.mark.parametrize("n_cats", [2, 4])
@pytest.mark.parametrize("feedback", [0., 1.])
def test_template_precision_derivative_matches_finite_difference(n_cats, feedback):
    p = ExperienceWorkspacePartition(n_cats=n_cats)
    p.nodes[0] = template(np.arange(1, n_cats + 1) / np.arange(1, n_cats + 1).sum())
    likelihood = ExperienceWorkspaceLikelihood(p, distance_mode="boundary")
    observation = ([.25, .28, .38, .55], 1, feedback)
    beta, step = 1.7, 1e-5
    numerical = (np.log(likelihood.binary_feedback_likelihood(observation, [0], [beta + step]))
                 - np.log(likelihood.binary_feedback_likelihood(observation, [0], [beta - step]))) / (2 * step)
    np.testing.assert_allclose(likelihood.feedback_precision_gradient(observation, [0], [beta]), numerical,
                               atol=1e-10, rtol=1e-6)
    q = p.get_category_probabilities(0, ([observation[0]], [], []), beta)
    np.testing.assert_allclose(q.sum(axis=0), 1.)
    assert likelihood.binary_feedback_likelihood(observation, [0], [beta])[0] == pytest.approx(q[0, 0] if feedback else q[1:, 0].sum())


def test_removing_irrelevant_dimension_generalizes_and_counterexample_penalizes_it():
    m = make_model()
    p = m.partition_model
    parent = template([1., 0.])
    child = template([1., 0.], (True, True, True, False), uid=2)
    p.nodes[:2] = [parent, child]
    x = [.2, .3, .4, 1.]
    likelihood = m.engine.observation_likelihood
    success = likelihood.binary_feedback_likelihood((x, 1, 1.), [0, 1], [5., 5.])
    counterexample = likelihood.binary_feedback_likelihood((x, 1, 0.), [0, 1], [5., 5.])
    assert success[1] > success[0]
    assert counterexample[1] < counterexample[0]
    m.engine.prior = np.array([.5, .5, 0, 0])
    m.engine.hypotheses_mask = np.array([1., 1., 0, 0])
    m.engine.likelihood = np.r_[counterexample, 0., 0.]
    m.engine.observation = (x, 1, 0.)
    m.engine.get_module(ModuleRole.MEMORY).process()
    assert m.engine.posterior[1] < m.engine.posterior[0]
    assert parent["mask"] == [True] * 4


def test_ambiguous_four_way_error_stays_uncertain_then_uses_future_feedback():
    p = ExperienceWorkspacePartition(n_cats=4)
    p.nodes[0] = template([0., 1 / 3, 1 / 3, 1 / 3])
    p.update_template_labels(([.2, .3, .4, .5], 2, 1.), np.array([5., 0, 0, 0]))
    theta = p.nodes[0]["category_belief"]
    assert theta[1] > .98 and theta[0] == 0.
    assert theta[2] == theta[3]


def test_generalization_can_grow_after_success_and_old_example_can_remain():
    m = make_model(baseline_probability=.99, global_search=0., module_seed=13)
    seen = False
    for t in range(24):
        m.begin_trial([.2, .3, .4, .5 + .001 * t])
        nodes = [n for n in m.partition_model.nodes if n is not None and n["kind"] == "template"]
        seen |= any(sum(n["mask"]) < 4 for n in nodes) and any(sum(n["mask"]) == 4 for n in nodes)
        m.complete_trial(1, 1.)
    assert seen
    assert all(e["active_count"] <= 4 for e in m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION).transition_log)


def test_catalogue_rules_enter_through_a_generalization():
    m = make_model(baseline_probability=.99, global_search=1., module_seed=17)
    events = []
    for t in range(40):
        m.begin_trial([.2, .3 + .001 * t, .4, .5])
        events.append(deepcopy(m.partition_model.nodes))
        m.complete_trial(1, 1.)
    rules = [n for nodes in events for n in nodes if n is not None and n["kind"] == "rule"]
    assert rules
    all_nodes = {n["uid"]: n for nodes in events for n in nodes if n is not None}
    for rule in rules:
        parent = all_nodes[rule["parent_uid"]]
        assert parent["kind"] == "template" and sum(parent["mask"]) < 4
        assert rule["rule_id"] in m.partition_model.compatible_rule_candidates(parent)


@pytest.mark.parametrize("n_cats", [2, 4])
def test_local_generalization_retains_positive_support_for_complete_rule_catalogue(n_cats):
    p = ExperienceWorkspacePartition(n_cats=n_cats)
    parent = template([1.] + [0.] * (n_cats - 1), (True, False, False, False))
    rules = p.compatible_rule_candidates(parent)
    weights = p.rule_proposal_weights(parent, rules, beta=5.)
    assert len(rules) == (29 if n_cats == 2 else 152)
    assert np.all(weights > 0.) and weights.sum() == pytest.approx(1.)
    assert weights.max() > weights.min()


@pytest.mark.parametrize("condition", [1, 2])
def test_snapshot_restores_private_contents_and_exact_future(condition):
    original = make_model(condition, baseline_probability=.9)
    for t in range(8):
        original.begin_trial([.2, .3, .4, .5 + .01 * t])
        original.complete_trial(1, float(t % 2))
    clone = StateModel(config(condition, baseline_probability=.9), context=ModelContext(condition=condition),
                       partition=original.partition_model, hypotheses_set=original.hypotheses_set)
    assert clone.partition_model is not original.partition_model
    assert clone.partition_model.catalogue is original.partition_model.catalogue
    saved = deepcopy(original.engine.state_dict())
    clone.engine.load_state_dict(saved)
    assert clone.partition_model.nodes == original.partition_model.nodes
    assert clone.partition_model.nodes is not original.partition_model.nodes
    for t in range(8):
        probabilities = []
        for model in (original, clone):
            model.begin_trial([.7, .3, .4, .5 + .01 * t])
            probabilities.append(predict(model).observed_probabilities)
            model.complete_trial(1, float(t % 2))
        np.testing.assert_array_equal(*probabilities)
        assert original.partition_model.nodes == clone.partition_model.nodes


def test_reused_slot_identity_does_not_receive_execution_stay_bonus():
    cfg = config(baseline_probability=.99, execution_inertia=100.)
    cfg["partition"]["kwargs"]["capacity"] = 1
    m = StateModel(cfg, context=ModelContext(condition=1))
    m.begin_trial([.2, .3, .4, .5]); m.complete_trial(1, 1.)
    m.begin_trial([.2, .3, .4, .5])
    transition = m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
    assert transition.executed_hypothesis == 0 and transition.executed_uid == 1
    assert transition.current_execution_switch_event
    assert transition.current_execution_switch_probability == 1.


@pytest.mark.parametrize("condition", [1, 2])
def test_generation_replays_choices_with_identical_learning_and_probabilities(condition):
    cfg = config(condition, baseline_probability=.8)
    stimulus = np.random.default_rng(3).uniform(.05, .95, (24, 4))
    model = StateModel(cfg, context=ModelContext(condition=condition))
    target = 1 + (stimulus[:, 0] > .5).astype(int)
    generated = model.generate_step_by_step(stimulus, lambda t, choice: float(choice == target[t]), choice_seed=2)
    replay = StateModel(cfg, context=ModelContext(condition=condition))
    for t, x in enumerate(stimulus):
        replay.begin_trial(x)
        np.testing.assert_array_equal(predict(replay).observed_probabilities, generated.observed_probabilities[t])
        replay.complete_trial(int(generated.choices[t]), float(generated.feedback[t]))
    assert replay.partition_model.nodes == model.partition_model.nodes


@pytest.mark.parametrize("condition", [1, 2])
def test_filter_clones_and_observes_representation_contents(condition):
    cfg = config(condition, baseline_probability=.9)
    stimulus = np.random.default_rng(11).uniform(.1, .9, (24, 4))
    snapshots = []
    def observer(t, phase, values):
        assert not values["template_anchor"].flags.writeable
        snapshots.append((t, phase, {key: value.copy() for key, value in values.items()}))
    kwargs = dict(engine_config=cfg, condition=condition, subject_id=0, stimulus=stimulus,
                  choices=np.ones(24, int), feedback=np.arange(24) % 2,
                  particle_count=8, choice_readout_power=1., resample_threshold_fraction=1., filter_seed=37)
    result = run_state_model_particle_filter(**kwargs, state_observer=observer, observe_decision_basis=True)
    repeated = run_state_model_particle_filter(**kwargs)
    for key in ("representation_belief", "representation_execution", "filtered_representation_belief"):
        values = result.state_probabilities[key]
        np.testing.assert_allclose(values.sum(axis=1), 1.)
        np.testing.assert_array_equal(values, repeated.state_probabilities[key])
    assert len(snapshots) == 48
    assert result.metadata["representation_coordinates"]["uid_scope"] == "within_ancestral_path_only"
    assert any(np.unique(item[2]["template_anchor"][:, :, 0], axis=0).shape[0] > 1 for item in snapshots[4:])


def test_skipped_learning_does_not_encode_current_feedback():
    m = make_model()
    m.begin_trial([.2, .3, .4, .5]); m.complete_trial(1, 1., update_state=False)
    m.begin_trial([.8, .7, .6, .5])
    assert m.partition_model.nodes[0]["kind"] == "uncommitted"


def test_half_feedback_and_condition_three_are_rejected():
    with pytest.raises(ValueError, match="condition"):
        StateModel(config(), context=ModelContext(condition=3))
    m = make_model()
    m.begin_trial([.2, .3, .4, .5])
    with pytest.raises(ValueError, match="binary"):
        m.complete_trial(1, .5)


def test_saved_prediction_replays_each_trials_contents_without_changing_final_state():
    m = make_model(baseline_probability=.9)
    x = np.random.default_rng(5).uniform(.05, .95, (20, 4))
    generated = m.generate_step_by_step(x, lambda t, c: float(c == 1 + (x[t, 0] > .5)), choice_seed=3)
    final_contents = m.partition_model.export_representations()
    kwargs = dict(categories=1 + (x[:, 0] > .5), target_probs=None, window_size=1,
        loss_metric="choice_nll", beta_log=generated.beta,
        choice_readout_config=resolve_choice_readout_config(None, m.engine_config),
        output_noise_config=resolve_output_noise_config(None, m.engine_config))
    args = (m, generated.posterior, generated.prior, generated.step_log, x,
            generated.choices, generated.feedback)
    result = compute_prediction_metrics(*args, prediction_mode="prior_t", **kwargs)["prior_t"]
    valid = result["valid_trial_mask"]
    np.testing.assert_allclose(result["pred_category_probs"][valid], generated.observed_probabilities[valid], atol=1e-14)
    assert final_contents == m.partition_model.nodes
    with pytest.raises(ValueError, match="prior_t"):
        compute_prediction_metrics(*args, prediction_mode="posterior_t_minus_1", **kwargs)


def test_filter_prediction_matches_explicit_pool_and_is_causal():
    cfg = config(baseline_probability=.9)
    x = np.random.default_rng(8).uniform(.05, .95, (12, 4))
    feedback = np.arange(12) % 2
    kwargs = dict(engine_config=cfg, subject_id=0, condition=1, stimulus=x,
                  choices=np.ones(12, int), particle_count=4, choice_readout_power=1.,
                  resample_threshold_fraction=0., filter_seed=71)
    result = run_state_model_particle_filter(**kwargs, feedback=feedback)
    models = [StateModel(_particle_config(cfg, 71, i), context=ModelContext(condition=1)) for i in range(4)]
    weights = np.full(4, .25)
    for t, stimulus in enumerate(x):
        probabilities = []
        for model in models:
            model.begin_trial(stimulus)
            probabilities.append(predict(model).observed_probabilities)
        probabilities = np.asarray(probabilities)
        np.testing.assert_allclose(result.marginal_probabilities[t], weights @ probabilities, atol=1e-14)
        weights *= probabilities[:, 0]
        weights /= weights.sum()
        for model in models:
            model.complete_trial(1, float(feedback[t]))
    altered = feedback.copy(); altered[-1] = 1 - altered[-1]
    causal = run_state_model_particle_filter(**kwargs, feedback=altered)
    np.testing.assert_array_equal(causal.marginal_probabilities, result.marginal_probabilities)
    np.testing.assert_array_equal(causal.state_probabilities["filtered_representation_belief"],
                                  result.state_probabilities["filtered_representation_belief"])


@pytest.mark.parametrize("malformation", ["oral", "agenda", "memory"])
def test_mixed_incompatible_profiles_are_rejected(malformation):
    cfg = config()
    if malformation == "oral":
        cfg["oral_readout"] = {"method": "rule_content_v1", "kwargs": {}}
    elif malformation == "agenda":
        cfg["agenda"][-2:] = cfg["agenda"][-2:][::-1]
    else:
        cfg["modules"]["memory_mod"]["class"] = "src.Bayesian_state.model.modules.memory.DualMemoryModule"
    with pytest.raises(ValueError):
        StateModel(cfg, context=ModelContext(condition=1))


def test_shared_partition_cannot_override_declared_similarity_width():
    source = make_model()
    cfg = config()
    cfg["partition"]["kwargs"]["similarity_width"] = .5
    with pytest.raises(ValueError, match="supplied partition"):
        StateModel(cfg, context=ModelContext(condition=1), partition=source.partition_model)


def test_particle_blocks_fail_before_launching_unvalidated_workers():
    with pytest.raises(ValueError, match="particle_blocks=1"):
        run_state_model_particle_filter(engine_config=config(), subject_id=0, condition=1,
            stimulus=[[.2, .3, .4, .5]], choices=[1], feedback=[1.], particle_count=4,
            particle_blocks=2, choice_readout_power=1.)
