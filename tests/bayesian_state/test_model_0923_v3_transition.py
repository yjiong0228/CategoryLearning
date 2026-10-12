"""Full-v3 transition and continuous choice invariants, independent of fitting."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule,
)
from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
from src.Bayesian_state.model.modules.readout.choice import (
    predict_choice_from_model, read_choice_probabilities_from_model,
    resolve_continuous_choice_weights, resolve_executed_hypothesis,
)


@pytest.fixture(scope="module")
def partition():
    return ContinuousPartition(4, 2)


def _make(partition, **overrides):
    count = len(partition.hypothesis_space)
    modules = {}
    engine = SimpleNamespace(
        partition=partition, set_size=count, distance_mode="boundary",
        prior=np.full(count, 1 / count), beta=np.full(count, 3.),
        get_module=lambda role, **kwargs: modules.get(role),
    )
    options = dict(
        mechanism_version="0923-full-v3", max_capacity=3, exit_rate=0.,
        local_structure_weight=.5, transfer_scale=.1, execution_inertia=0.,
        edit_costs={"feature": 1., "relation": 1.5, "reference": 1., "branch": 2.},
        search_controller=dict(baseline_probability=.2, error_gain=1., failure_decay=.6,
                               global_search=.3, global_search_failure_gain=.5),
        relation_samples=128, relation_seed=0, module_seed=923,
    )
    options.update(overrides)
    transition = MechanisticRuleSearchHypothesisTransitionModule(engine, **options)
    modules[ModuleRole.HYPOTHESIS_TRANSITION] = transition
    return transition, modules


def _install(transition, active, mass, dominant=0):
    transition.active = np.asarray(active, dtype=int)
    transition.old_active = transition.active.copy()
    transition.executed_hypothesis = dominant
    transition.trial_index = 1
    transition._apply_mask()
    prior = np.zeros(transition.total_hypo)
    prior[active] = mass
    transition.engine.prior = prior.copy()
    transition.engine.posterior = prior.copy()
    transition.execution_probabilities = prior.copy()


class ScriptedRng:
    def __init__(self, exits, newcomer=3, parent=0):
        self.exits = np.asarray(exits, dtype=float)
        self.newcomer, self.parent = newcomer, parent
        self.calls = []

    def random(self, size=None):
        return self.exits.copy() if size is not None else 0.

    def choice(self, values, p=None):
        self.calls.append((deepcopy(values), None if p is None else np.array(p)))
        if isinstance(values, int):
            return self.parent
        if p is not None:
            assert self.newcomer in values
            return self.newcomer
        return values[-1]


@pytest.mark.parametrize("mass,draws,expected", [
    ([.2, .8], [0., 1.], [1]),
    ([.2, .8], [0., 0.], [1]),
    ([.5, .5], [0., 0.], [1]),
    ([.8, .2, 0.], [0., 0., 1.], [0, 2]),
])
def test_exit_treats_dominant_equally_and_recovers_maximum_support(partition, mass, draws, expected):
    transition, _ = _make(partition, exit_rate=1.)
    _install(transition, list(range(len(mass))), mass)
    transition.current_event_probability = 0.
    transition.trial_rng = ScriptedRng(draws)
    transition.process()
    assert transition.active.tolist() == expected
    assert transition.engine.prior.sum() == pytest.approx(1.)
    expected_mass = np.asarray(mass)[expected]
    np.testing.assert_allclose(transition.engine.prior[expected], expected_mass / expected_mass.sum())


def test_capacity_releases_weakest_old_rule_before_single_transfer(partition):
    transition, _ = _make(partition)
    _install(transition, [0, 1, 2], [.4, .1, .5], dominant=1)
    distances = np.full((transition.total_hypo, transition.total_hypo), .5)
    distances[[0, 2], 3] = [.1, .3]
    transition._distance_matrix = distances
    transition.current_event_probability = transition.current_g = 1.
    transition.trial_rng = ScriptedRng([1., 1., 1.])
    transition.process()
    assert transition.active.tolist() == [0, 2, 3]
    event = transition.transition_log[-1]
    assert event["capacity_dropped"] == 1  # The previous dominant may leave.
    assert event["admission_event"] and event["swap_probability"] == 1.
    keep = np.array([4 / 9, 5 / 9])
    affinity = np.exp([-1., -3.])
    expected = np.r_[keep / (1 + affinity), np.sum(keep * affinity / (1 + affinity))]
    np.testing.assert_allclose(transition.engine.prior[[0, 2, 3]], expected)
    assert transition.engine.prior[1] == 0.


def test_distant_new_rule_is_admitted_and_capacity_ties_are_sampled(partition):
    transition, _ = _make(partition, max_capacity=2, transfer_scale=1e-5)
    _install(transition, [0, 1], [.5, .5], dominant=1)
    transition.current_event_probability = transition.current_g = 1.
    rng = transition.trial_rng = ScriptedRng([1., 1.], newcomer=2)
    transition.process()
    assert transition.active.tolist() == [0, 2]
    np.testing.assert_array_equal(rng.calls[-1][0], [0, 1])
    assert transition.transition_log[-1]["admission_event"]
    assert np.isfinite(transition.engine.prior).all()
    assert transition.engine.prior.sum() == pytest.approx(1.)


def test_local_parent_uses_retained_belief_and_pool_excludes_just_exited(partition):
    transition, _ = _make(partition, exit_rate=1.)
    _install(transition, [0, 1, 2], [.2, .3, .5], dominant=2)
    transition.current_event_probability, transition.current_g = 1., 0.
    rng = transition.trial_rng = ScriptedRng([1., 1., 0.], parent=1)
    transition.process()
    np.testing.assert_allclose(rng.calls[0][1], [.4, .6])
    assert transition.transition_log[-1]["proposal_parent"] == 1
    assert set(rng.calls[1][0]).isdisjoint({0, 1, 2})
    assert transition.transition_log[-1]["autonomous_exits"] == [2]


def test_new_pairing_inherits_all_old_rules_before_capacity_release(partition):
    transition, modules = _make(partition, max_capacity=2)
    _install(transition, [0, 1], [.8, .2])
    memory = HierarchicalPairingMemoryModule(transition.engine)
    modules[ModuleRole.MEMORY] = memory
    memory.joint[:] = 0.
    memory.joint[0], memory.joint[1] = [.56, .16, .08], [.02, .04, .14]
    transition.current_event_probability = transition.current_g = 1.
    transition.trial_rng = ScriptedRng([1., 1.], newcomer=2)
    transition.process()
    np.testing.assert_allclose(memory.conditional_pairing()[0], [.7, .2, .1])
    np.testing.assert_allclose(memory.conditional_pairing()[2], [.58, .2, .22])
    np.testing.assert_allclose(memory.joint.sum(axis=1), transition.engine.prior)


def test_continuous_choice_uses_all_weights_and_readout_is_pure(partition):
    transition, _ = _make(partition, execution_inertia=np.log(3.))
    _install(transition, [0, 1], [.4, .6])
    transition.current_event_probability = 0.
    before_rng = deepcopy(transition.execution_rng.bit_generator.state)
    transition.process()
    assert transition.execution_rng.bit_generator.state == before_rng
    expected_weights = np.array([2 / 3, 1 / 3])
    np.testing.assert_allclose(transition.execution_probabilities[:2], expected_weights)
    assert transition.dominant_hypothesis == 0
    assert resolve_executed_hypothesis(transition.engine) is None
    np.testing.assert_array_equal(resolve_continuous_choice_weights(transition.engine), transition.execution_probabilities)
    model = SimpleNamespace(engine=transition.engine, partition_model=partition, n_cats=2,
                            hypotheses_set=list(range(transition.total_hypo)), _oral_readout_parameters={})
    stimulus = np.array([.9, .1, .3, .7])
    observed, snapshot = read_choice_probabilities_from_model(model, stimulus, return_snapshot=True)
    np.testing.assert_allclose(snapshot.rule_weights, expected_weights)
    np.testing.assert_allclose(observed, expected_weights @ snapshot.category_probabilities)
    saved = deepcopy(transition.state_dict())
    choice = predict_choice_from_model(
        model, stimulus, trial_idx=1, choices=[1], feedback=[0.],
        choice_readout_config={"method": "expectation", "power": 1.},
        output_noise_config={"enabled": False}, rng=np.random.default_rng(10),
        sticky_state={}, post_error_lapse_state=0.,
    )
    np.testing.assert_allclose(choice.rule_snapshot.rule_weights[:2], expected_weights)
    np.testing.assert_allclose(choice.observed_probabilities, observed)
    np.testing.assert_array_equal(transition.engine.prior[:2], [.4, .6])
    assert transition.state_dict()["execution_rng"] == saved["execution_rng"]
    assert transition.transition_log[-1]["effective_participation"] == pytest.approx(1.8)


def test_dominant_ties_prefer_previous_then_use_saved_random_stream(partition):
    transition, _ = _make(partition)
    _install(transition, [0, 1, 2], [.4, .4, .2], dominant=1)
    transition.current_event_probability = 0.
    saved_rng = deepcopy(transition.execution_rng.bit_generator.state)
    transition.process()
    assert transition.dominant_hypothesis == 1
    assert transition.execution_rng.bit_generator.state == saved_rng
    transition.executed_hypothesis = 2
    saved = deepcopy(transition.state_dict())
    # The stored dominant is valid only when it maximizes the current weights.
    restored, _ = _make(partition)
    with pytest.raises(ValueError, match="dominant"):
        restored.load_state_dict(saved)
    transition.execution_probabilities[:] = 0.
    transition.execution_probabilities[2] = 1.
    restored.load_state_dict(transition.state_dict())
    restored.engine.posterior = transition.engine.posterior.copy()
    transition.process()
    restored.process()
    assert transition.dominant_hypothesis == restored.dominant_hypothesis
    assert transition.dominant_hypothesis in (0, 1)


def test_versions_reject_cross_loading_and_legacy_snapshot_schema_is_unchanged(partition):
    v3, _ = _make(partition)
    legacy, _ = _make(partition, mechanism_version="legacy")
    assert "mechanism_version" not in legacy.state_dict()
    assert v3.state_dict()["mechanism_version"] == "0923-full-v3"
    for source, target in [(v3, legacy), (legacy, v3)]:
        with pytest.raises(ValueError, match="mechanism version"):
            target.load_state_dict(source.state_dict())
    clone, _ = _make(partition, mechanism_version="legacy")
    clone.load_state_dict(legacy.state_dict())
    assert clone.configuration_signature == legacy.configuration_signature


def test_reseeded_v3_paths_restore_deterministically(partition):
    first, _ = _make(partition, exit_rate=.3)
    _install(first, [0, 1, 2], [.5, .3, .2])
    second, _ = _make(partition, exit_rate=.3)
    second.load_state_dict(first.state_dict())
    second.engine.posterior = first.engine.posterior.copy()
    for transition in (first, second):
        transition.reseed_future(701)
        transition.process()
    np.testing.assert_array_equal(first.active, second.active)
    np.testing.assert_array_equal(first.engine.prior, second.engine.prior)
    np.testing.assert_array_equal(first.execution_probabilities, second.execution_probabilities)
    assert first.dominant_hypothesis == second.dominant_hypothesis
