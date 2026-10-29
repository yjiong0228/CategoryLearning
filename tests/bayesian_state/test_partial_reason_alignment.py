import numpy as np

from src.Bayesian_state.evaluation.oral.partial_reason_alignment import (
    feature_coverage, predicate_coverage, project, stimulus_choice_reference, unresolved_reason_role)
from src.Bayesian_state.model.modules.readout.report_space import component_tokens


def test_extra_reasons_and_unspoken_basis_have_different_denominators():
    basis = np.array([[1, 1, 0, 0]], bool)
    np.testing.assert_allclose(feature_coverage(np.ones(4, bool), basis), [[.5, 1, 0]])
    np.testing.assert_allclose(feature_coverage(np.array([1, 0, 0, 0], bool), basis), [[1, .5, 0]])
    np.testing.assert_allclose(feature_coverage(basis[0], basis), [[1, 1, 1]])


def test_same_feature_with_opposite_direction_is_a_content_conflict():
    A = np.array([[1., 0, 0, 0]])
    short = tuple(component_tokens(A, np.array([.5])))
    long = tuple(component_tokens(-A, np.array([-.5])))
    np.testing.assert_allclose(predicate_coverage(short, A, np.array([.5])), [1, 0, 0, 1])
    np.testing.assert_allclose(predicate_coverage(long, A, np.array([.5])), [0, 1, 0, 0])


def test_extra_unconstrained_statement_remains_unexplained():
    a = np.eye(4)[[0, 1]]
    words = tuple(component_tokens(a, np.array([.5, .5])))
    np.testing.assert_allclose(predicate_coverage(words, a[:1], np.array([.5])), [.5, 0, .5, 1])


def test_probability_projection_does_not_pick_the_most_matching_rule():
    mass = np.array([[.2], [.8]])
    values = np.array([[[1., 0]], [[0., 1]]])
    np.testing.assert_allclose(project(mass, values), [.2, .8])


def test_stimulus_reference_weights_rules_before_components_and_flags_empty():
    A = np.array([[1., 0]])
    entries = [(0, 0, A, np.array([.5])), (0, 1, A, np.array([.5])),
               (1, 0, A, np.array([.5]))]
    mass, valid = stimulus_choice_reference(np.array([[.1, .2], [.9, .2]]), entries, 2, 2)
    np.testing.assert_allclose(mass[0], [[.25, .25], [.5, 0]])
    np.testing.assert_array_equal(valid, [True, False])
    assert np.isnan(mass[1]).all()


def test_negation_and_guessing_do_not_become_positive_reasons():
    assert unresolved_reason_role('不是因为头长，主要看腿')
    assert unresolved_reason_role('随机猜的')
    assert not unresolved_reason_role('腿短，头长')
