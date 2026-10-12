"""Independent checks of 0923 precision calibration and factorized forgetting."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.observation_model.likelihood import ObservationLikelihood
from src.Bayesian_state.hypothesis_space.observation_model.pairing import pairing_feedback_kernel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.beta import BetaModule
from src.Bayesian_state.model.modules.memory import DualMemoryModule
from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule


STIMULUS = np.array([.4, .1, .3, .7])


def _emission(distances, beta):
    logits = -np.asarray(distances) * np.asarray(beta)[..., None]
    values = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


class _DistancePartition:
    """Keep rule geometry fixed while independently forming softmax emissions."""

    VALID_DISTANCE_MODES = ("boundary",)
    n_dims = 4

    def __init__(self, distances):
        self.distances = np.asarray(distances, dtype=float)
        self.n_cats = self.distances.shape[1]
        self.boundary_geometry = self

    def category_distances(self, hypo, stimuli):
        return np.tile(self.distances[hypo], (len(stimuli), 1))

    def get_category_probabilities(self, hypo, data, beta, distance_mode):
        assert distance_mode == "boundary"
        return _emission(self.distances[hypo], beta)[:, None]

    def calc_likelihood(self, **kwargs):
        raise AssertionError("the gradient must use absolute category probabilities")


def _gradient_model(distances, *, beta=5., rate=.5, graded=False):
    partition = _DistancePartition(distances)
    engine = SimpleNamespace(
        set_size=len(distances), partition=partition, distance_mode="boundary",
        prior=np.full(len(distances), 1. / len(distances)),
        hypotheses_mask=np.ones(len(distances)),
    )
    engine.observation_likelihood = ObservationLikelihood(
        partition, distance_mode="boundary",
        feedback_likelihood_mode="hierarchical_pairing" if graded else "category_feedback",
    )
    memory = HierarchicalPairingMemoryModule(engine, forgetting_mode="factorized") if graded else None
    engine.get_module = lambda role: memory if role == ModuleRole.MEMORY else None
    module = BetaModule(
        engine, beta_update_mode="feedback_gradient", learning_rate=rate,
        beta_init=beta, beta_min=.1, beta_max=25.,
    )
    return engine, module, memory


def _event_probability(distances, beta, choice, feedback, pairing_weights=None):
    probabilities = _emission(distances, beta)
    if pairing_weights is not None:
        kernel = pairing_feedback_kernel(probabilities, choice, feedback)
        return (kernel * pairing_weights).sum(axis=1)
    if feedback == 1.:
        return probabilities[:, choice - 1]
    return probabilities[:, np.arange(probabilities.shape[1]) != choice - 1].sum(axis=1)


def test_correct_feedback_can_reduce_precision_and_improve_its_probability():
    distances = [[0., .1, .4, np.sqrt(.17)]]
    engine, beta, _ = _gradient_model(distances, rate=1.)
    initial = _event_probability(distances, beta.beta, 2, 1.)
    engine.observation = (STIMULUS, 2, 1.)

    beta.process()

    np.testing.assert_allclose(beta.beta, [4.989483856880223], atol=1e-14)
    updated = _event_probability(distances, beta.beta, 2, 1.)
    np.testing.assert_allclose(updated, [0.3245358363987926], atol=1e-14)
    assert updated[0] > initial[0] > .25
    np.testing.assert_array_equal(beta.beta_log, [[5.]])


@pytest.mark.parametrize("graded,feedback", [(False, 0.), (False, 1.), (True, 0.), (True, .5), (True, 1.)])
@pytest.mark.parametrize("choice", [1, 2, 3, 4])
def test_precision_gradient_matches_independent_finite_difference(graded, feedback, choice):
    distances = np.array([[0., .1, .4, np.sqrt(.17)], [.8, 0., .3, .9]])
    engine, beta, _ = _gradient_model(distances, graded=graded)
    betas = np.array([5., 2.])
    weights = np.array([[.65, .25, .1], [.1, .6, .3]]) if graded else None
    gradient = engine.observation_likelihood.feedback_precision_gradient(
        (STIMULUS, choice, feedback), [0, 1], betas, pairing_weights=weights,
    )
    epsilon = 1e-5
    expected = (
        np.log(_event_probability(distances, betas + epsilon, choice, feedback, weights))
        - np.log(_event_probability(distances, betas - epsilon, choice, feedback, weights))
    ) / (2. * epsilon)
    np.testing.assert_allclose(gradient, expected, rtol=1e-7, atol=2e-10)


@pytest.mark.parametrize("graded,n_categories", [(False, 2), (False, 4), (True, 4)])
def test_calibrated_feedback_has_zero_expected_unclipped_update(graded, n_categories):
    distances = np.array([[0., .3, .7, .4]])[:, :n_categories]
    engine, beta, _ = _gradient_model(distances, graded=graded)
    weights = np.array([[.7, .2, .1]]) if graded else None
    expected_gradient = 0.
    total_probability = 0.
    for feedback in ((0., .5, 1.) if graded else (0., 1.)):
        gradient = engine.observation_likelihood.feedback_precision_gradient(
            (STIMULUS, 1, feedback), [0], beta.beta, pairing_weights=weights,
        )[0]
        probability = _event_probability(distances, beta.beta, 1, feedback, weights)[0]
        expected_gradient += probability * gradient
        total_probability += probability
    assert total_probability == pytest.approx(1.)
    assert expected_gradient == pytest.approx(0., abs=1e-15)


@pytest.mark.parametrize("initial_beta", [.1, 5., 25.])
@pytest.mark.parametrize("learning_rate", [0., .3, 1.])
@pytest.mark.parametrize("feedback", [0., 1.])
def test_projected_gradient_preserves_bounds_and_feedback_likelihood(initial_beta, learning_rate, feedback):
    distances = np.array([[0., .1, .4, np.sqrt(.17)], [.6, 0., .2, .7]])
    _, beta, _ = _gradient_model(distances, beta=initial_beta, rate=learning_rate)
    initial = _event_probability(distances, beta.beta, 1, feedback)
    beta.update_beta(STIMULUS, 1, feedback)
    updated = _event_probability(distances, beta.beta, 1, feedback)
    assert np.all(updated >= initial - 1e-15)
    assert np.all((beta.beta >= .1) & (beta.beta <= 25.))


def test_gradient_uses_all_active_rules_and_fixed_initial_precision():
    engine, beta, _ = _gradient_model([[0., .4], [.4, 0.], [.1, 0.]], rate=1.)
    engine.hypotheses_mask[-1] = 0.
    beta.update_beta(STIMULUS, 1, 1.)
    assert beta.beta[0] > 5.
    assert beta.beta[1] < 5.
    assert beta.beta[2] == 0.
    beta.initialize_beta_for_hypotheses(np.array([0, 2]), np.array([.999, 0., .001]))
    np.testing.assert_array_equal(beta.beta[[0, 2]], [5., 5.])


def test_confident_error_event_retains_tiny_positive_probability():
    engine, beta, _ = _gradient_model([[0., 2., 2., 2.]], beta=25., rate=1.)
    beta.update_beta(STIMULUS, 1, 0.)
    assert beta.beta[0] == pytest.approx(23.)


@pytest.mark.parametrize("kwargs", [
    {}, {"learning_rate": np.nan}, {"learning_rate": np.inf},
    {"learning_rate": -.01}, {"learning_rate": 1.01},
    {"learning_rate": .5, "increase_rate": .04},
    {"learning_rate": .5, "decrease_rate": .15},
    {"learning_rate": .5, "correct_additive": 1.},
    {"learning_rate": .5, "update_scope": "executed_hypothesis"},
    {"learning_rate": .5, "probabilistic_feedback_lapse": .1},
    {"learning_rate": .5, "use_prior_scaling": True},
])
def test_gradient_configuration_requires_one_explicit_valid_rate(kwargs):
    with pytest.raises(ValueError):
        BetaModule(SimpleNamespace(set_size=2), beta_update_mode="feedback_gradient", **kwargs)


def _memory(joint, *, gamma=.8, mode="factorized", active=None):
    joint = np.asarray(joint, dtype=float)
    prior = joint.sum(axis=1)
    engine = SimpleNamespace(prior=prior, set_size=len(joint),
                             hypotheses_mask=(prior > 0.) if active is None else np.asarray(active))
    module = HierarchicalPairingMemoryModule(engine, gamma=gamma, forgetting_mode=mode)
    module.joint = joint.copy()
    return module


def test_factorized_forgetting_preserves_equal_rule_support_with_different_pairing_certainty():
    joint = np.array([[.4, .05, .05], [1/6, 1/6, 1/6]])
    new = _memory(joint)
    historical = _memory(joint, mode="joint")
    for module in (new, historical):
        module.prepare_feedback(np.full((2, 3), .6))
        module.process()
    np.testing.assert_allclose(new.engine.posterior, [.5, .5], atol=1e-15)
    np.testing.assert_allclose(historical.engine.posterior, [.4807769583158058, .5192230416841942])
    np.testing.assert_allclose(new.feedback_pairing_weights, historical.feedback_pairing_weights)


@pytest.mark.parametrize("gamma", [0., .6, 1.])
def test_factorized_update_matches_separate_rule_and_conditional_calculations(gamma):
    joint = np.array([[.24, .12, .04], [.06, .18, .36], [0., 0., 0.], [0., 0., 0.]])
    module = _memory(joint, gamma=gamma, active=[1, 1, 1, 0])
    conditional = np.array([[.6, .3, .1], [.1, .3, .6], [1/3]*3])
    prior = np.array([.4, .6, 0.])
    kernel = np.array([[.1, .2, .7], [.5, .3, .2], [.3, .4, .3], [.2, .2, .2]])
    if gamma == 0.:
        discounted_rules = np.ones(3) / 3.
        discounted_pairing = np.ones((3, 3)) / 3.
    else:
        discounted_rules = prior**gamma / (prior**gamma).sum()
        discounted_pairing = conditional**gamma
        discounted_pairing /= discounted_pairing.sum(axis=1, keepdims=True)
    evidence = (discounted_pairing * kernel[:3]).sum(axis=1)
    expected_rules = discounted_rules * evidence
    expected_rules /= expected_rules.sum()
    expected_pairing = discounted_pairing * kernel[:3] / evidence[:, None]
    module.prepare_feedback(kernel)
    np.testing.assert_allclose(module.feedback_pairing_weights[:3], discounted_pairing)
    np.testing.assert_allclose(module.feedback_evidence[:3], evidence)
    module.process()
    np.testing.assert_allclose(module.joint[:3], expected_rules[:, None] * expected_pairing)
    np.testing.assert_array_equal(module.joint[-1], 0.)
    np.testing.assert_allclose(module.engine.posterior[:3], expected_rules)


@pytest.mark.parametrize("gamma", [0., 1.])
def test_factorized_forgetting_matches_joint_update_at_retention_endpoints(gamma):
    joint = np.array([[.45, 0., .05], [.15, .3, .05], [0., 0., 0.]])
    kernel = np.array([[.1, .7, .2], [.3, .4, .3], [.2, .2, .6]])
    outputs = []
    for mode in ("joint", "factorized"):
        module = _memory(joint, gamma=gamma, mode=mode, active=[1, 1, 1])
        module.prepare_feedback(kernel)
        module.process()
        outputs.append(module.joint)
    np.testing.assert_allclose(*outputs, atol=1e-15)


def test_precision_consumes_cached_discounted_pairing_after_memory_updates():
    distances = np.array([[0., .1, .4, np.sqrt(.17)], [.3, 0., .2, .8]])
    engine, beta, memory = _gradient_model(distances, graded=True, rate=1.)
    memory.joint = np.array([[.4, .05, .05], [.05, .15, .3]])
    probabilities = _emission(distances, beta.beta)
    memory.prepare_feedback(pairing_feedback_kernel(probabilities, 1, .5))
    cached = memory.feedback_pairing_weights.copy()
    expected = engine.observation_likelihood.feedback_precision_gradient(
        (STIMULUS, 1, .5), [0, 1], beta.beta, pairing_weights=cached,
    )
    memory.process()
    assert not np.allclose(memory.conditional_pairing(), cached)
    beta.update_beta(STIMULUS, 1, .5)
    np.testing.assert_allclose(beta.beta, 5. + expected)
    np.testing.assert_array_equal(memory.feedback_pairing_weights, cached)


def test_pending_factorized_snapshot_preserves_cache_and_has_independent_storage():
    joint = np.array([[.4, .05, .05], [.05, .15, .3]])
    original = _memory(joint)
    original.prepare_feedback(np.array([[.1, .4, .2], [.6, .2, .4]]))
    snapshot = original.state_dict()
    restored = _memory(joint)
    restored.load_state_dict(snapshot)
    original.process()
    restored.process()
    np.testing.assert_array_equal(restored.joint, original.joint)
    np.testing.assert_array_equal(restored.feedback_pairing_weights, original.feedback_pairing_weights)
    restored.feedback_pairing_weights[0] = 1/3
    np.testing.assert_array_equal(snapshot["feedback_pairing_weights"], original.feedback_pairing_weights)
    original.reset()
    assert original.feedback_pairing_weights is None
    assert original.feedback_evidence is None
    assert original._pending_joint is None
    np.testing.assert_allclose(original.conditional_pairing(), 1/3)


def test_snapshot_modes_reject_cross_version_states_and_accept_legacy_snapshots():
    joint = np.full((2, 3), 1/6)
    new_memory = _memory(joint)
    old_memory = _memory(joint, mode="joint")
    old_snapshot = old_memory.state_dict()
    old_snapshot.pop("forgetting_mode")
    old_snapshot.pop("feedback_pairing_weights")
    old_memory.load_state_dict(old_snapshot)
    with pytest.raises(ValueError, match="forgetting_mode"):
        new_memory.load_state_dict(old_snapshot)
    with pytest.raises(ValueError, match="forgetting_mode"):
        old_memory.load_state_dict(new_memory.state_dict())

    engine, new_beta, _ = _gradient_model([[0., .3], [.2, 0.]])
    old_beta = BetaModule(SimpleNamespace(set_size=2))
    old_beta.load_state_dict({"beta": np.array([3., 4.])})
    with pytest.raises(ValueError, match="beta_update_mode"):
        new_beta.load_state_dict({"beta": np.array([3., 4.])})
    with pytest.raises(ValueError, match="beta_update_mode"):
        old_beta.load_state_dict(new_beta.state_dict())
    snapshot = new_beta.state_dict()
    new_beta.beta[:] = 8.
    new_beta.load_state_dict(snapshot)
    np.testing.assert_array_equal(new_beta.beta, [5., 5.])
    assert engine.beta is new_beta.beta
    new_beta.reset()
    np.testing.assert_array_equal(new_beta.beta, [5., 5.])


@pytest.mark.parametrize("bad_weights", [
    np.full((2, 3), np.nan), np.ones((2, 3)), np.ones((2, 2)) / 2.,
])
def test_pairing_cache_snapshot_validation_is_atomic(bad_weights):
    memory = _memory(np.full((2, 3), 1/6))
    snapshot = memory.state_dict()
    snapshot["feedback_pairing_weights"] = bad_weights
    before = memory.joint.copy()
    with pytest.raises(ValueError, match="pairing weights"):
        memory.load_state_dict(snapshot)
    np.testing.assert_array_equal(memory.joint, before)


def _binary_memory(prior, evidence, *, gamma=.8, mode="exact_power", active=None):
    prior = np.asarray(prior, dtype=float)
    engine = SimpleNamespace(
        set_size=len(prior), prior=prior, likelihood=np.asarray(evidence, dtype=float),
        hypotheses_mask=np.ones(len(prior)) if active is None else np.asarray(active),
    )
    return DualMemoryModule(engine, gamma=gamma, w0=0., feedback_gain=1., forgetting_mode=mode)


def test_exact_rule_memory_preserves_tiny_entrant_mass_before_forgetting():
    prior = np.array([1., np.exp(-100.)])
    exact = _binary_memory(prior, [.6, .6], gamma=.1)
    historical = _binary_memory(prior, [.6, .6], gamma=.1, mode="dual_trace")
    for module in (exact, historical):
        module.prepare_for_process()
        module.process()
    expected = np.exp(-10.) / (1. + np.exp(-10.))
    assert exact.engine.posterior[1] == pytest.approx(expected, rel=1e-13)
    assert historical.engine.posterior[1] == pytest.approx(.0306534300317155, rel=1e-12)


@pytest.mark.parametrize("gamma", [0., .4, 1.])
def test_exact_rule_memory_retains_small_evidence_and_handles_zero_prior(gamma):
    prior = np.array([.8, .2, 0., 0.])
    evidence = np.array([1e-22, .8, .6, 1.])
    module = _binary_memory(prior, evidence, gamma=gamma, active=[1, 1, 1, 0])
    powered = np.ones(3) if gamma == 0. else prior[:3]**gamma
    expected = powered * evidence[:3]
    expected /= expected.sum()
    module.prepare_for_process()
    module.process()
    np.testing.assert_allclose(module.engine.posterior[:3], expected, rtol=1e-13, atol=0.)
    assert module.engine.posterior[-1] == 0.
    assert (module.engine.posterior[2] > 0.) == (gamma == 0.)
    np.testing.assert_allclose(np.exp(module.state["fade"]), module.engine.posterior, rtol=1e-13, atol=0.)


def test_exact_rule_memory_follows_the_transition_prior_on_each_trial():
    module = _binary_memory([.9, .1, 0.], [.3, .8, .2], active=[1, 1, 0])
    module.process()
    module.engine.prior = np.array([.7, 0., .3])
    module.engine.hypotheses_mask = np.array([1, 0, 1])
    module.prepare_for_process()
    module.process()
    expected = np.array([.7**.8 * .3, 0., .3**.8 * .2])
    expected /= expected.sum()
    np.testing.assert_allclose(module.engine.posterior, expected)


@pytest.mark.parametrize("feedback,choice", [(0., 1), (1., 2)])
def test_exact_binary_likelihood_keeps_tiny_probability_and_matches_precision_gradient(feedback, choice):
    distances = np.array([[0., 2., 2., 2.], [.8, 0., .3, .4]])
    engine, beta, _ = _gradient_model(distances, beta=25.)
    observation = (STIMULUS, choice, feedback)
    evaluator = engine.observation_likelihood
    evidence = evaluator.binary_feedback_likelihood(observation, [0, 1], beta.beta)
    expected = _event_probability(distances, beta.beta, choice, feedback)
    np.testing.assert_allclose(evidence, expected, rtol=1e-14, atol=0.)
    assert 0. < evidence[0] < 1e-20
    epsilon = 1e-5
    difference = (
        np.log(evaluator.binary_feedback_likelihood(observation, [0, 1], beta.beta + epsilon))
        - np.log(evaluator.binary_feedback_likelihood(observation, [0, 1], beta.beta - epsilon))
    ) / (2. * epsilon)
    gradient = evaluator.feedback_precision_gradient(observation, [0, 1], beta.beta)
    np.testing.assert_allclose(gradient, difference, rtol=1e-7, atol=3e-10)
    memory = _binary_memory([.5, .5], evidence, gamma=1.)
    memory.process()
    np.testing.assert_allclose(memory.engine.posterior, evidence / evidence.sum(), rtol=1e-13, atol=0.)


def test_exact_binary_memory_snapshot_preserves_state_and_rejects_legacy_mode():
    module = _binary_memory([.9, .1], [.2, .8])
    module.process()
    snapshot = module.state_dict()
    restored = _binary_memory([.9, .1], [.2, .8])
    restored.load_state_dict(snapshot)
    np.testing.assert_array_equal(restored.state["fade"], module.state["fade"])
    restored.state["fade"][0] = 0.
    assert snapshot["state"]["fade"][0] != 0.
    legacy = _binary_memory([.9, .1], [.2, .8], mode="dual_trace")
    legacy_snapshot = legacy.state_dict()
    legacy_snapshot.pop("forgetting_mode")
    legacy.load_state_dict(legacy_snapshot)
    with pytest.raises(ValueError, match="forgetting_mode"):
        module.load_state_dict(legacy_snapshot)
    with pytest.raises(ValueError, match="forgetting_mode"):
        legacy.load_state_dict(snapshot)


@pytest.mark.parametrize("kwargs", [{"w0": .1}, {"feedback_gain": .5}, {"forgetting_mode": "invalid"}])
def test_exact_binary_memory_rejects_incompatible_configuration(kwargs):
    engine = SimpleNamespace(set_size=2, prior=np.array([.5, .5]), hypotheses_mask=np.ones(2))
    configuration = {"w0": 0., "feedback_gain": 1., "forgetting_mode": "exact_power", **kwargs}
    with pytest.raises(ValueError):
        DualMemoryModule(engine, **configuration)
