import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.development import (
    categorical_log_score, normalized_exponential, oral_split, review_indices, token_truth,
    subset_log_probability, anchored_weights, condition_component_distribution)
from src.Bayesian_state.workflows.analysis.model_1024_oral_readout import (
    calibrated_frequency, mixture_readout, observed_atoms, shuffle_state)


def test_contiguous_holdout_and_actual_last_block_buffer():
    split = oral_split(70)
    assert (split[:4] == 'buffer').all()
    assert (split[4:28] == 'calibration').all()
    assert (split[36:60] == 'assessment').all()
    assert (split[64:] == 'buffer').all()
    assert len(review_indices(704, 128)) == 128
    np.testing.assert_array_equal(review_indices(5, 128), np.arange(5))


def test_atomic_score_is_normalized_and_report_length_has_equal_weight():
    p = normalized_exponential(np.array([[0., 1.], [0., 1.]]), 2.)
    s = categorical_log_score(p, [[1], [0, 1]])
    assert s[0] == pytest.approx(np.log(p[0, 1]))
    assert s[1] == pytest.approx(np.log(p[1]).mean())
    with pytest.raises(AssertionError):
        categorical_log_score(p * 2, [[1], [0]])
    np.testing.assert_allclose(normalized_exponential(np.ones((3, 4)), 0), .25)


def test_physical_truth_is_separate_from_geometric_entailment():
    tokens = ['facet:[[1.0,-1.0,0.0,0.0,0.0]]', 'facet:[[-1.0,0.0,0.0,0.0,-0.5]]', 'body:unknown']
    truth = token_truth(tokens, np.array([.8, .9, .1, .1]))
    np.testing.assert_array_equal(truth[:2], [1, 1])
    assert np.isnan(truth[2])


def test_readout_averages_normalized_components_and_keeps_low_mass_states():
    mass = np.array([[.9, .1]])
    scores = np.array([[1., 0., -1.], [-1., 0., 1.]])
    background = np.array([[.2, .5, .3]])
    actual = mixture_readout(mass, scores, background, 2.)
    raw = background * np.exp(2*scores)
    expected = mass @ (raw/raw.sum(1, keepdims=True))
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(mixture_readout(mass, scores, background, 0), background)


def test_assessment_words_do_not_change_calibration_frequency():
    words = [('a',), ('b', 'other'), ('a', 'b')]
    y = observed_atoms(words, ('a', 'b', 'OTHER'))
    mask = np.array([True, False, True])
    first = calibrated_frequency(y, mask, 1.)
    y[1] = [0, 0, 1]
    np.testing.assert_array_equal(first, calibrated_frequency(y, mask, 1.))
    assert np.all(first > 0) and first.sum() == pytest.approx(1.)


def test_state_shuffle_preserves_each_choice_block_and_is_reproducible():
    state = np.arange(140)[:, None, None]
    choices = np.arange(140) % 4 + 1
    a = shuffle_state(state, choices, 12)
    np.testing.assert_array_equal(a, shuffle_state(state, choices, 12))
    for block in range(3):
        for choice in range(1, 5):
            use = (np.arange(140)//64 == block) & (choices == choice)
            np.testing.assert_array_equal(np.sort(a[use, 0, 0]), state[use, 0, 0])


def test_subset_normalizer_equals_exhaustive_combinations():
    from itertools import combinations
    weights = np.array([[.1, .2, .3, .4], [4., 1., 2., 3.]])
    for size in [1, 2, 3, 4]:
        subsets = list(combinations(range(4), size))
        probabilities = np.stack([np.exp(subset_log_probability(weights, s)) for s in subsets], axis=1)
        expected = np.stack([np.prod(weights[:, s], axis=1) for s in subsets], axis=1)
        expected /= expected.sum(1, keepdims=True)
        np.testing.assert_allclose(probabilities, expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-14, rtol=0)
    with pytest.raises(ValueError):
        subset_log_probability(weights, [1, 1])


def test_oral_reweighting_retains_coherent_paths_and_identity_channel():
    from src.Bayesian_state.inference.backends.particle_filter import _trace_ancestral_indices
    from src.Bayesian_state.evaluation.particle_smoothing import ancestral_weights
    parents = np.array([[0, 0, 2], [2, 1, 1], [0, 1, 2]])
    paths = _trace_ancestral_indices(parents)
    terminal = np.array([.2, .3, .5])
    np.testing.assert_array_equal(anchored_weights(terminal, np.ones(3))[0], terminal)
    likelihood_at_anchor = np.array([.9, .2, .1])
    updated, evidence = anchored_weights(terminal, likelihood_at_anchor[paths[:, 0]])
    expected = terminal*likelihood_at_anchor[paths[:, 0]]
    assert evidence == pytest.approx(expected.sum())
    expected /= expected.sum()
    np.testing.assert_allclose(updated, expected)
    ancestry = ancestral_weights(paths, updated)['weights']
    for t in range(3):
        manual = np.zeros(3)
        for path, weight in zip(paths, expected):
            manual[path[t]] += weight
        np.testing.assert_allclose(ancestry[t], manual)


def test_rank_catalogue_covers_every_label_permutation_without_changing_base():
    from src.Bayesian_state.evaluation.oral.rank_catalogue import rank_hypotheses
    from src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit import catalogue
    base = catalogue(4)
    before = base.signature
    candidates = rank_hypotheses()
    assert len(candidates) == 48
    x = np.random.default_rng(11).uniform(size=(30, 4))
    for rule in candidates:
        actual = np.column_stack([np.all(x @ c.components[0].A.T <= c.components[0].b, axis=1)
                                  for c in rule.categories])
        assert (actual.sum(1) == 1).all()
        extreme = x.argmax(1) if rule.parameters['kind'] == 'longest' else x.argmin(1)
        for t in range(len(x)):
            assert rule.parameters['feature_for_category'][int(actual[t].argmax())] == extreme[t]
    assert catalogue(4).signature == before and len(catalogue(4)) == 152


def test_marginalized_execution_update_matches_enumerated_joint_posterior():
    particles = np.array([.3, .7])
    component = np.array([[[.8], [.2]], [[.1], [.9]]])
    likelihood = np.array([[[.1], [.9]], [[.4], [.2]]])
    conditional, evidence = condition_component_distribution(component, likelihood)
    posterior, _ = anchored_weights(particles, evidence)
    enumerated = particles[:, None, None]*component*likelihood
    enumerated /= enumerated.sum()
    np.testing.assert_allclose(posterior[:, None, None]*conditional, enumerated)
    # Particle weighting alone loses the oral information about which rule
    # generated the report within a zero-inertia mixture.
    assert not np.allclose(posterior[:, None, None]*component, enumerated)
