"""Check the 2026-11-01 design algebra, separate from a Cond3 implementation.

The joint arrays below enumerate the proposed hidden labels and pairings. They
are small independent references, not a second learning-model implementation.
Only the fixed-precision smoke check exercises the current Model 1031 itself.
"""
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space.observation_model.pairing import (
    PAIRING_SIBLINGS,
    pairing_feedback_kernel,
)
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.readout.choice import (
    resolve_choice_readout_config,
    resolve_output_noise_config,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("feedback", [0., .5, 1.])
def test_conditional_label_update_matches_one_joint_bayes_update(feedback):
    rng = np.random.default_rng(20261101)
    pi = rng.dirichlet(np.ones(3))
    pairing = rng.dirichlet(np.ones(3), size=3)
    labels = rng.dirichlet(np.ones(4), size=(3, 3))
    # Rows 0/1 are templates. Row 2 is a rule, independent of its dummy label.
    gains = np.array([.2, .85, 0.])
    emission = np.broadcast_to(
        (1 - gains[:, None, None, None]) / 4
        + gains[:, None, None, None] * np.eye(4)[None, None, :, :],
        (3, 3, 4, 4),
    ).copy()
    emission[2] = rng.dirichlet(np.ones(4))
    event = pairing_feedback_kernel(np.eye(4), 2, feedback).T
    label_likelihood = np.einsum("igzc,gc->igz", emission, event)
    likelihood = np.einsum("igz,igz->ig", labels, label_likelihood)

    prior = pi[:, None, None] * pairing[:, :, None] * labels
    enumerated = prior * label_likelihood
    enumerated /= enumerated.sum()

    joint_content_pairing = pi[:, None] * pairing * likelihood
    joint_content_pairing /= joint_content_pairing.sum()
    conditional_labels = labels * label_likelihood / likelihood[:, :, None]
    factorized = joint_content_pairing[:, :, None] * conditional_labels
    np.testing.assert_allclose(factorized, enumerated, atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(conditional_labels.sum(axis=-1), 1.)


def test_cond3_feedback_events_are_exhaustive_with_conditional_predictions():
    rng = np.random.default_rng(20261101)
    predictions = rng.dirichlet(np.ones(4), size=3)
    for choice in range(1, 5):
        kernels = [pairing_feedback_kernel(predictions, choice, f)
                   for f in (0., .5, 1.)]
        assert all(np.isfinite(k).all() and (k >= 0).all() for k in kernels)
        np.testing.assert_allclose(sum(kernels), 1., atol=1e-14)


def test_half_feedback_stores_label_pairing_dependence():
    # A first half-correct response 1 leaves all three pairings equally likely.
    # Its remembered label is the sibling of 1 conditional on each pairing.
    pairing = np.full(3, 1 / 3)
    labels = np.eye(4)[PAIRING_SIBLINGS[:, 0]]
    marginal_labels = pairing @ labels
    np.testing.assert_allclose(marginal_labels, [0., 1 / 3, 1 / 3, 1 / 3])
    correlated = np.diag(pairing_feedback_kernel(labels, 1, .5))
    independent = pairing_feedback_kernel(marginal_labels[None, :], 1, .5)[0]
    # At perfect retrieval, repeating response 1 predicts another half feedback.
    assert pairing @ correlated == pytest.approx(1.)
    assert pairing @ independent == pytest.approx(1 / 3)


def test_uniform_pairing_half_and_wrong_have_same_relative_rule_evidence():
    rng = np.random.default_rng(20261101)
    p = rng.dirichlet(np.ones(4), size=7)
    half = pairing_feedback_kernel(p, 1, .5).mean(axis=1)
    wrong = pairing_feedback_kernel(p, 1, 0.).mean(axis=1)
    np.testing.assert_allclose(half, (1 - p[:, 0]) / 3)
    np.testing.assert_allclose(wrong, 2 * half)
    np.testing.assert_allclose(half / half.sum(), wrong / wrong.sum())


def test_all_response_recodings_preserve_joint_label_pairing_update():
    rng = np.random.default_rng(20261101)
    joint = rng.dirichlet(np.ones(12)).reshape(3, 4)
    for permutation in permutations(range(4)):
        permutation = np.asarray(permutation)
        pairing_map = []
        for siblings in PAIRING_SIBLINGS:
            recoded = np.empty(4, int)
            recoded[permutation] = permutation[siblings]
            matches = np.flatnonzero(np.all(PAIRING_SIBLINGS == recoded, axis=1))
            assert len(matches) == 1
            pairing_map.append(int(matches[0]))
        destination = np.ix_(pairing_map, permutation)
        recoded_joint = np.empty_like(joint)
        recoded_joint[destination] = joint
        for choice in range(1, 5):
            for feedback in (0., .5, 1.):
                kernel = pairing_feedback_kernel(np.eye(4), choice, feedback).T
                posterior = joint * kernel
                posterior /= posterior.sum()
                recoded_kernel = pairing_feedback_kernel(
                    np.eye(4), int(permutation[choice - 1]) + 1, feedback).T
                recoded_posterior = recoded_joint * recoded_kernel
                recoded_posterior /= recoded_posterior.sum()
                np.testing.assert_allclose(recoded_posterior[destination], posterior,
                                           atol=1e-14, rtol=1e-13)


def test_separate_retention_preserves_equal_content_belief():
    pi = np.array([.5, .5])
    pairing = np.array([[1., 0., 0.], [1 / 3, 1 / 3, 1 / 3]])
    retained_pi = pi ** .5
    retained_pi /= retained_pi.sum()
    retained_pairing = pairing ** .5
    retained_pairing /= retained_pairing.sum(axis=1, keepdims=True)
    factorized = retained_pi[:, None] * retained_pairing
    np.testing.assert_allclose(factorized.sum(axis=1), pi)
    flattened = (pi[:, None] * pairing) ** .5
    flattened /= flattened.sum()
    assert flattened[1].sum() > flattened[0].sum()


def test_global_content_selection_dilutes_a_perfect_local_memory():
    # Binary example: one exact A memory and one distant, uninformative memory.
    probabilities = np.array([[1., 0.], [.5, .5]])
    mixture = np.array([.5, .5]) @ probabilities
    assert mixture[0] == pytest.approx(.75)


@pytest.mark.parametrize("condition", [1, 2])
def test_existing_zero_inertia_fixed_precision_boundary(condition):
    cfg = yaml.safe_load((ROOT / "configs/exp123/model_struct" /
                          f"model_1031_cond{condition}_experience_v1.yaml").read_text())
    cfg["modules"]["perception_mod"]["kwargs"] = dict(
        features=4, mean=[0.] * 4, std=[0.] * 4, module_seed=7)
    cfg["modules"]["hypo_transitions_mod"]["kwargs"].update(
        execution_inertia=0., baseline_probability=.8)
    cfg["modules"]["beta_mod"]["kwargs"].update(beta_init=3., learning_rate=0.)
    traces = []
    for _ in range(2):
        model = StateModel(cfg, context=ModelContext(condition=condition))
        rng = np.random.default_rng(20261101)
        predictions = []
        for _trial in range(12):
            x = rng.uniform(size=4)
            model.begin_trial(x)
            transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            active = transition.active
            np.testing.assert_allclose(model.engine.beta[active], 3., atol=1e-14)
            np.testing.assert_allclose(transition.execution_selection_probabilities,
                                       model.engine.prior, atol=1e-14)
            prediction = model.predict_choice(
                choices=[], feedback=[],
                choice_readout_config=resolve_choice_readout_config(None, model.engine_config),
                output_noise_config=resolve_output_noise_config(None, model.engine_config),
                rng=rng, sticky_state={}, post_error_lapse_state=0.)
            p = prediction.observed_probabilities
            assert np.isfinite(p).all() and (p >= 0.).all()
            np.testing.assert_allclose(p.sum(), 1.)
            predictions.append(p.copy())
            choice = int(rng.choice(model.n_cats, p=p)) + 1
            category = (1 + int(x[0] > .5) if condition == 1
                        else 1 + 2 * int(x[0] > .5) + int(x[1] > .5))
            model.complete_trial(choice, float(choice == category))
        traces.append(np.asarray(predictions))
    np.testing.assert_array_equal(*traces)
