"""Mechanistic 0923: causal search, semantic edits, support transport and live readout."""

from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from src.Bayesian_state.hypothesis_space.rule_programs import build_rule_edit_costs
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.config import ModelConfig
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule,
)
from src.Bayesian_state.model.readout import resolve_choice_readout_config, resolve_output_noise_config
from src.Bayesian_state.simulation.parameters import infer_fixed_hyperparams_from_engine_config
from src.Bayesian_state.simulation.provenance import build_model_provenance


ROOT = Path(__file__).resolve().parents[2]
COSTS = {"feature": 1., "relation": 1.5, "reference": 1., "branch": 2.}
X = np.array([.8, .3, .2, .4])


def _config(condition=1):
    config = yaml.safe_load((ROOT / f"configs/exp123/model_struct/model_0923_cond{condition}_full.yaml").read_text())
    config["modules"]["perception_mod"]["kwargs"] = {
        "features": 4, "mean": [0.] * 4, "std": [0.] * 4, "module_seed": 32,
    }
    config["modules"]["hypo_transitions_mod"]["kwargs"].update(relation_samples=512, module_seed=23)
    config["provenance"]["hypothesis_similarity"]["sample_count"] = 512
    return config


def _model(condition=1, **kwargs):
    config = _config(condition)
    config["modules"]["hypo_transitions_mod"]["kwargs"].update(kwargs)
    return StateModel(config, context=ModelContext(condition=condition))


def _module(model):
    return model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)


def _install(model, active, mass, executed=0):
    transition = _module(model)
    transition.active = np.asarray(active)
    transition.old_active = transition.active.copy()
    transition.executed_hypothesis = executed
    transition.trial_index = 1
    transition._apply_mask()
    prior = np.zeros(model.engine.set_size)
    prior[active] = mass
    model.engine.prior = prior.copy()
    model.engine.posterior = prior.copy()
    transition.execution_probabilities = prior.copy()


class _ProposalRng:
    def __init__(self, newcomer, *, exit=False):
        self.newcomer, self.exit = newcomer, exit

    def random(self, size=None):
        if size is not None:
            return np.zeros(size) if self.exit else np.ones(size)
        return 0.0

    def choice(self, values, p=None):
        if isinstance(values, int):
            return 0
        if p is not None and self.newcomer in values:
            return self.newcomer
        return values[0]


def _choose(model):
    return model.predict_choice(
        choices=[], feedback=[], choice_readout_config=resolve_choice_readout_config(None, model.engine_config),
        output_noise_config=resolve_output_noise_config(None, model.engine_config),
        rng=np.random.default_rng(12), sticky_state={}, post_error_lapse_state=0.,
    )


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_profiles_are_declared_and_keep_the_adopted_catalogue(condition):
    config = _config(condition)
    before = deepcopy(config)
    ModelConfig.from_mapping(config)
    assert config == before
    model = _model(condition)
    assert model.engine.set_size == (29 if condition == 1 else 128)
    assert isinstance(_module(model), MechanisticRuleSearchHypothesisTransitionModule)
    assert len(_module(model).active) == 1


@pytest.mark.parametrize("condition", [1, 2])
def test_programs_preserve_geometry_and_labels_for_every_rule(condition):
    model = _model(condition)
    partition = model.partition_model
    transition = _module(model)
    points = np.r_[np.random.default_rng(61).random((31, 4)),
                   np.array([[.5]*4, [.2, .2, .8, .8]])]
    for h, program in enumerate(transition.rule_programs):
        # Representation is checked independently against shared geometry.
        np.testing.assert_array_equal(program.evaluate(points),
            partition.get_category_assignments(h, points, distance_mode="boundary"))
    assert np.isfinite(transition.edit_distances).sum() > model.engine.set_size
    assert np.any(np.isfinite(transition.edit_distances).sum(axis=1) == 1)


def test_feature_edit_can_be_functionally_far_and_comparison_cost_is_explicit():
    model = _model()
    transition = _module(model)
    assert transition.edit_distances[0, 1] == COSTS["feature"]
    assert transition.edit_distances[4, 10] == COSTS["relation"]
    assert abs(transition.functional_similarity[0, 1] - .5) < .1
    expensive = {**COSTS, "relation": 3.}
    _, edits = build_rule_edit_costs(model.partition_model.hypothesis_space, expensive)
    assert edits[4, 10] == 3.


def test_confusion_records_class_imbalance_and_never_writes_legacy_resources(tmp_path):
    partition = ContinuousPartition(4, 2, similarity_cache_dir=tmp_path)
    threshold = np.array([[a, b, .23, .77] for a, b in product((.2, .8), repeat=2)])
    tensor = partition.similarity.get_confusion_tensor(distance_mode="boundary", stimuli=threshold)
    np.testing.assert_array_equal(tensor[0, 1], np.full((2, 2), .25))
    band_points = np.array([[a, b, .23, .77] for a, b in product((0, .2, .5, .8, 1.), repeat=2)])
    band = partition.similarity.get_confusion_tensor(distance_mode="boundary", stimuli=band_points)
    np.testing.assert_allclose(band[25, 26], [[.04, .16], [.16, .64]])
    assert np.trace(band[25, 26]) == pytest.approx(.68)
    np.testing.assert_allclose(band.sum(axis=(2, 3)), 1)
    np.testing.assert_array_equal(band.transpose(1, 0, 3, 2), band)
    assert not band.flags.writeable
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("decay", [.2, .8])
def test_latest_graded_feedback_controls_search_and_range_with_subject_decay(decay):
    controller = dict(_config()["modules"]["hypo_transitions_mod"]["kwargs"]["search_controller"])
    controller.update(failure_decay=decay, baseline_probability=.2, error_gain=2.,
                      global_search=.3, global_search_failure_gain=.6)
    model = _model(search_controller=controller, exit_rate=0.)
    transition = _module(model)
    model.begin_trial(X)
    assert transition.transition_log[-1]["search_probability"] == 0.
    failure = 0.
    for feedback in (0., .5, 1., 0.):
        # Controller accepts the score trace; graded feedback is formally scored
        # only in C3. This direct hook does not call binary feedback learning.
        model.engine.posterior = model.engine.prior.copy()
        before = transition.current_event_probability
        transition.record_outcome((X, 1, feedback))
        assert transition.current_event_probability == before
        transition.process()
        failure = decay*failure + (1-decay)*(1-feedback)
        assert transition.failure_pressure == pytest.approx(failure)
        assert transition.current_event_probability == pytest.approx(1/(1+np.exp(-(np.log(.2/.8)+2*failure))))
        assert transition.current_g == pytest.approx(.3+.7*.6*failure)
        assert 1 <= len(transition.active) <= transition.capacity


def test_structure_and_functional_kernels_have_correct_masks_and_fallback():
    transition = _module(_model())
    remaining = np.array([0, 16])
    # 16 is a paired-sum rule with no legal single-edit neighbour.
    inactive = np.setdiff1d(np.arange(29), [0, 1, 16])
    functional, structural, fallback = transition._proposal_kernels(remaining, inactive)
    np.testing.assert_allclose(functional.sum(axis=1), 1.)
    np.testing.assert_allclose(structural.sum(axis=1), 1.)
    assert fallback.tolist() == [False, True]
    np.testing.assert_array_equal(structural[1], functional[1])
    assert np.all(structural[0, np.isin(inactive, [2, 3])] > 0)
    assert np.all(structural[0, np.isin(inactive, [16, 17, 18])] == 0)
    assert 1 not in inactive  # a just-exited rule cannot re-enter this event


def test_candidates_can_shrink_without_search_or_automatic_refill():
    model = _model(exit_rate=1.)
    _install(model, [0, 1, 2], [.8, .15, .05])
    transition = _module(model)
    transition.current_event_probability = 0.
    transition.trial_rng = _ProposalRng(3, exit=True)
    transition.process()
    assert transition.active.tolist() == [0]
    assert not transition.transition_log[-1]["search_event"]
    np.testing.assert_array_equal(model.engine.prior[:4], [1, 0, 0, 0])


@pytest.mark.parametrize("parent,newcomer,weight,source,fallback", [
    (0, 1, 1., "structural", False),
    (0, 1, 0., "functional", False),
    (16, 19, 1., "functional", True),
])
def test_actual_local_route_and_structural_fallback_never_become_global(parent, newcomer, weight, source, fallback):
    model = _model(local_structure_weight=weight, exit_rate=0.)
    _install(model, [parent], [1.], executed=parent)
    transition = _module(model)
    transition.current_event_probability = 1.
    transition.current_g = 0.
    transition.trial_rng = _ProposalRng(newcomer)
    transition.process()
    event = transition.transition_log[-1]
    assert event["proposal_source"] == source and event["structural_fallback"] == fallback
    assert event["proposal_parent"] == parent
    assert event["proposed_hypothesis"] == newcomer and event["admission_event"]


def test_search_can_be_rejected_without_changing_survivor_support():
    model = _model(max_capacity=2, exit_rate=0., transfer_scale=1e-5)
    _install(model, [0, 1], [.8, .2])
    transition = _module(model)
    transition.current_event_probability = transition.current_g = 1.
    transition.trial_rng = _ProposalRng(2)
    transition.process()
    event = transition.transition_log[-1]
    assert event["search_event"] and not event["admission_event"]
    assert event["proposed_hypothesis"] == 2
    assert event["swap_probability"] == 0.
    np.testing.assert_array_equal(model.engine.prior[:3], [.8, .2, 0.])


def _accepted_example(condition=1):
    model = _model(condition, max_capacity=2, exit_rate=0., transfer_scale=.1,
                   execution_inertia=1000.)
    _install(model, [0, 1], [.8, .2])
    transition = _module(model)
    distance = np.full((model.engine.set_size, model.engine.set_size), .5)
    np.fill_diagonal(distance, 0.)
    distance[:3, :3] = [[0, .2, .1], [.2, 0, .3], [.1, .3, 0]]
    transition._distance_matrix = distance
    transition.current_event_probability = transition.current_g = 1.
    transition.trial_rng = _ProposalRng(2)
    return model


def test_capacity_transport_matches_hand_calculation_and_resets_newcomer_precision():
    model = _accepted_example()
    transition = _module(model)
    beta = model.engine.get_module(ModuleRole.BETA)
    beta.beta[[0, 1, 2]] = [8., 9., 19.]
    transition.process()
    assert transition.active.tolist() == [0, 2]
    assert transition.transition_log[-1]["capacity_dropped"] == 1
    # Retained source 0 splits at logits (0,-1). Removed source 1 also splits
    # in the same ratio because its distances are (.2,.3), giving exact .731/.269.
    np.testing.assert_allclose(model.engine.prior[[0, 2]], [1/(1+np.exp(-1)), 1/(1+np.exp(1))])
    assert model.engine.prior[1] == 0.
    assert beta.beta[0] == 8. and beta.beta[2] == beta.beta_init
    assert beta.beta[1] == 0.
    assert np.isfinite(transition.execution_probabilities).all()
    assert transition.executed_hypothesis == 0


def test_newcomer_support_depends_on_relationship_not_a_fixed_budget():
    transition = _module(_model())
    distance = np.full((29, 29), .5)
    distance[[0, 1], 2] = [.1, .3]
    distance[[0, 1], 3] = [.5, .7]
    transition._distance_matrix = distance
    near = transition._temporary_weights(np.array([0, 1]), np.array([.8, .2]), 2)
    far = transition._temporary_weights(np.array([0, 1]), np.array([.8, .2]), 3)
    assert near[-1] == pytest.approx(.2246383117315095)
    assert far[-1] == pytest.approx(.005536490978308014)
    assert near.sum() == pytest.approx(1.) and far.sum() == pytest.approx(1.)


def test_pairing_survivor_conditionals_and_old_marginal_initialize_newcomer():
    model = _accepted_example(3)
    memory = model.engine.get_module(ModuleRole.MEMORY)
    memory.joint[:] = 0.
    memory.joint[0] = .8*np.array([.7, .2, .1])
    memory.joint[1] = .2*np.array([.1, .2, .7])
    old_marginal = memory.pairing_marginal()
    _module(model).process()
    np.testing.assert_allclose(memory.joint.sum(axis=1), model.engine.prior)
    np.testing.assert_allclose(memory.conditional_pairing()[0], [.7, .2, .1])
    np.testing.assert_allclose(memory.conditional_pairing()[2], old_marginal)
    assert memory.joint[1].sum() == 0.


def test_state_restoration_preserves_next_transition_and_execution_draw():
    model = _model(exit_rate=.3)
    model.begin_trial(X)
    model.complete_trial(1, 0.)
    saved = deepcopy(model.engine.state_dict())
    restored = _model(exit_rate=.3)
    restored.engine.load_state_dict(saved)
    # Compare the engine lifecycle directly; StateModel pending-trial counters
    # are managed by the particle filter rather than stored in engine state.
    _module(model).process()
    _module(restored).process()
    np.testing.assert_array_equal(model.engine.prior, restored.engine.prior)
    assert _module(model).active.tolist() == _module(restored).active.tolist()
    assert _module(model).executed_hypothesis == _module(restored).executed_hypothesis
    assert _module(model).failure_pressure == _module(restored).failure_pressure


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_choice_and_prefeedback_oral_share_execution_and_do_not_update_learning(condition):
    model = _model(condition)
    model.begin_trial(X)
    decision = _choose(model)
    snapshot = deepcopy(model.engine.state_dict())
    oral = model.predict_oral(1)
    executed = _module(model).executed_hypothesis
    assert np.flatnonzero(decision.rule_snapshot.rule_weights).tolist() == [executed]
    assert np.flatnonzero(oral.rule_weights_given_choice).tolist() == [executed]
    assert oral.probabilities.sum() == pytest.approx(1.)
    np.testing.assert_array_equal(model.engine.prior, snapshot["core"]["prior"])
    assert _module(model).executed_hypothesis == snapshot["modules"]["hypo_transitions_mod"]["executed_hypothesis"]
    model.complete_trial(1, .5 if condition == 3 else 0.)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_particle_filter_is_causal_normalized_and_deterministic(condition):
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    config = _config(condition)
    config["modules"]["hypo_transitions_mod"]["kwargs"]["search_controller"]["baseline_probability"] = .7
    config["modules"]["hypo_transitions_mod"]["kwargs"]["exit_rate"] = .3
    config["inference"]["resample_threshold_fraction"] = 1.
    arguments = dict(engine_config=config, subject_id=122, condition=condition,
                     stimulus=np.array([X, 1-X, X, 1-X]), choices=[1, 2, 1, 2],
                     particle_count=4, choice_readout_power=1., filter_seed=89)
    feedback = np.array([0., .5 if condition == 3 else 0., 1., 0.])
    first = run_state_model_particle_filter(feedback=feedback, **arguments)
    second = run_state_model_particle_filter(feedback=feedback, **arguments)
    np.testing.assert_array_equal(first.marginal_probabilities, second.marginal_probabilities)
    np.testing.assert_allclose(first.marginal_probabilities.sum(axis=1), 1.)
    assert np.isfinite(first.marginal_probabilities).all()
    assert np.all((first.marginal_active_probability.sum(axis=1) >= 1-1e-12)
                  & (first.marginal_active_probability.sum(axis=1) <= 3+1e-12))
    changed = feedback.copy()
    changed[-1] = 1.
    alternate = run_state_model_particle_filter(feedback=changed, **arguments)
    np.testing.assert_array_equal(first.marginal_probabilities, alternate.marginal_probabilities)
    if condition == 3:
        np.testing.assert_allclose(first.state_probabilities["pairing_prior"].sum(axis=1), 1.)


def test_fixed_parameter_provenance_captures_full_without_changing_b0_seed_inputs():
    full = infer_fixed_hyperparams_from_engine_config(_config())
    prefix = "engine.modules.hypo_transitions_mod.kwargs."
    assert full[prefix+"max_capacity"] == 3
    assert full[prefix+"search_controller"]["failure_decay"] == .6
    b0 = yaml.safe_load((ROOT/"configs/exp123/model_struct/model_0923_cond1_B0.yaml").read_text())
    baseline = infer_fixed_hyperparams_from_engine_config(b0)
    assert prefix+"search_controller" not in baseline
    assert prefix+"max_capacity" not in baseline
    provenance = build_model_provenance(_config(), repeat_aggregation="mean_probability")
    assert provenance["resolved"]["workspace_capacity"] == 3
    assert provenance["resolved"]["workspace_capacity_semantics"] == "upper_bound_variable_occupancy"
    assert provenance["resolved"]["initialization"]["active_count"] == 1
    assert provenance["resolved"]["hypothesis_similarity"]["tau_proposal"] == .1


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_name_and_saved_legacy_alias_have_identical_seeded_paths(condition):
    config = _config(condition)
    assert config["provenance"]["variant"] == "full"
    assert config["provenance"]["specification_version"] == "0923-full-v1"
    legacy = deepcopy(config)
    legacy["provenance"].update(variant="M1", specification_version="0923-M1-v1")
    assert infer_fixed_hyperparams_from_engine_config(config) == infer_fixed_hyperparams_from_engine_config(legacy)
    models = [StateModel(item, context=ModelContext(condition=condition)) for item in (config, legacy)]
    for stimulus, feedback in [(X, 0.), (1-X, .5 if condition == 3 else 1.), (X, 1.)]:
        decisions, reports = [], []
        for model in models:
            model.begin_trial(stimulus)
            decisions.append(_choose(model).observed_probabilities)
            reports.append(model.predict_oral(1).probabilities)
            model.complete_trial(1, feedback)
        np.testing.assert_array_equal(*decisions)
        np.testing.assert_array_equal(*reports)
        np.testing.assert_array_equal(models[0].engine.posterior, models[1].engine.posterior)
        np.testing.assert_array_equal(_module(models[0]).active, _module(models[1]).active)
        assert _module(models[0]).executed_hypothesis == _module(models[1]).executed_hypothesis


def test_snapshot_rejects_a_different_subject_parameterization():
    snapshot = _module(_model(transfer_scale=.1)).state_dict()
    changed = _module(_model(transfer_scale=.3))
    with pytest.raises(ValueError, match="configured mechanism"):
        changed.load_state_dict(snapshot)


@pytest.mark.parametrize("change", ["wrong_variant", "wrong_sampling", "fixed_budget", "wrong_capacity", "fixed_decay"])
def test_invalid_or_hidden_mechanisms_are_rejected(change):
    config = _config()
    transition = config["modules"]["hypo_transitions_mod"]["kwargs"]
    if change == "wrong_variant":
        config["provenance"]["variant"] = "B0"
    elif change == "wrong_sampling":
        transition["relation_samples"] = 100
    elif change == "fixed_budget":
        transition["newcomer_budget"] = .1
    elif change == "wrong_capacity":
        transition["max_capacity"] = 1
    else:
        transition["search_controller"]["failure_decay"] = 1.
    with pytest.raises(ValueError):
        StateModel(config, context=ModelContext(condition=1))
