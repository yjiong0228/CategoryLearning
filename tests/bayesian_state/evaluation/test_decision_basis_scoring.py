"""Check complete-content scores, repeat pooling and frozen-prefix boundaries."""
import numpy as np
import pandas as pd
import pytest
import json

from src.Bayesian_state.evaluation.oral.decision_basis_scoring import (
    label_masses, pool_joint, pooled_mean_and_mcse, prefix_baseline, support_rule_masks,
)
from src.Bayesian_state.workflows.analysis.model_0923_decision_basis_alignment import (
    make_labels, run, digest,
)


def test_pool_choice_joint_before_normalizing_each_repeat():
    probabilities = np.array([[.9], [.1]])
    joint = np.array([[[.9, 0.]], [[0., .1]]])
    np.testing.assert_allclose(pool_joint(joint, probabilities), [[.9, .1]])
    assert not np.allclose(pool_joint(joint, probabilities), [[.5, .5]])


def test_labels_preserve_three_way_mass_and_missing_reports():
    mass = np.array([[[.2, .3], [.5, 0.]], [[.2, .3], [.5, 0.]]])
    labels = np.array([[[[0, 0], [1, 1]], [[2, 0], [-1, -1]]],
                       [[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]]])
    scores = label_masses(mass, labels)
    np.testing.assert_allclose(scores[0], [[.2, .3, .5], [.7, .3, 0.]])
    assert np.isnan(scores[1]).all()
    lower, upper = support_rule_masks(labels, 1)
    np.testing.assert_array_equal(lower, [[False, True], [False, False]])
    np.testing.assert_array_equal(upper, [[True, True], [False, False]])


def test_scored_report_cannot_silently_drop_a_component():
    with pytest.raises(AssertionError):
        label_masses(np.array([[[.5, .5]]]), np.array([[[[0, 0], [-1, -1]]]]))


def test_frozen_prefix_is_unchanged_by_later_states_and_reports():
    basis = np.array([[[1., 0.]], [[0., 1.]], [[.5, .5]], [[.8, .2]]])
    choices = np.array([1, 2, 1, 2])
    original = prefix_baseline(basis, choices, 2)
    changed = basis.copy(); changed[2:] = [[[.1, .9]], [[.2, .8]]]
    np.testing.assert_array_equal(original, prefix_baseline(changed, choices, 2))
    assert np.isnan(original[:2]).all()
    np.testing.assert_array_equal(original[2:], basis[:2])


def test_mcse_uses_whole_filter_replicates_and_empty_windows_are_missing():
    joint = np.array([[[.1], [.3]], [[.3], [.1]]])
    denominator = np.full((2, 2), .5)
    point, se, estimates = pooled_mean_and_mcse(joint, denominator, [True, True])
    np.testing.assert_allclose(point, [.4])
    np.testing.assert_allclose(se, [0.])
    np.testing.assert_allclose(estimates, [[.4], [.4]])
    point, se, estimates = pooled_mean_and_mcse(joint, denominator, [False, False])
    assert np.isnan(point).all() and np.isnan(se).all() and np.isnan(estimates).all()


def test_report_table_preserves_missing_and_complete_component_labels():
    trials = pd.DataFrame({'query_id': ['one', None, 'one']})
    matches = pd.DataFrame(dict(query_id=['one', 'one'], rule=[0, 1], component=[0, 0],
                                structural=['undetermined', 'conflict'], functional=['support', 'conflict']))
    labels = make_labels(trials, matches, 2, 1)
    np.testing.assert_array_equal(labels[0, :, 0], [[2, 0], [1, 1]])
    np.testing.assert_array_equal(labels[0], labels[2])
    assert (labels[1] == -1).all()


def test_incomplete_seed_groups_cannot_produce_a_scientific_summary(tmp_path):
    source, audit, output = [tmp_path / name for name in ('source', 'audit', 'output')]
    source.mkdir(); audit.mkdir(); (source / 'states').mkdir()
    (source / 'protocol.json').write_text(json.dumps(dict(smoke=False,
        settings=dict(subjects=[207], arms=['BASE'], seeds_per_arm=8))))
    hashes = {}
    for name in ('trials.csv', 'component_matches.csv'):
        (audit / name).write_text('unused\n')
        hashes[name] = digest(audit / name)
    (audit / 'manifest.json').write_text(json.dumps(dict(status='complete',input_sha256={},artifacts=hashes)))
    with pytest.raises(ValueError, match='Incomplete or duplicate seed group'):
        run(source, audit, output)
    assert not output.exists()
