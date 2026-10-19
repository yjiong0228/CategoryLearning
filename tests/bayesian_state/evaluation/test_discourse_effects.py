"""Text group position must not erase opposing evidence or state uncertainty."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.discourse_effects import group_effect_profiles


def masks():
    return ((np.arange(4)[:, None] >> np.arange(2)) & 1).astype(bool)


def test_opposing_and_supporting_features_keep_their_directions():
    q = np.array([[.7, .3, .9, .5]])
    groups = dict(before=np.array([False, True]), after=np.array([True, False]),
                  whole=np.ones(2, dtype=bool))
    rows = group_effect_profiles(q, masks(), groups, np.array([1.]))
    assert rows['before']['feature_means']['actual_effect'][0] == pytest.approx(-.2)
    assert rows['before']['means']['minimum_positive_effect'] == 0
    assert rows['before']['means']['minimum_absolute_effect'] == pytest.approx(.2)
    assert rows['after']['means']['minimum_positive_effect'] == pytest.approx(.4)
    assert rows['whole']['means']['joint_reflection_effect'] == pytest.approx(.2)
    assert rows['whole']['means']['minimum_positive_effect'] == 0
    assert rows['whole']['means']['minimum_absolute_effect'] == pytest.approx(.2)


def test_clause_names_do_not_assign_causal_direction():
    q = np.array([[.7, .3, .9, .5]])
    group = np.array([False, True])
    values = group_effect_profiles(q, masks(), {'before': group, 'after': group}, np.array([1.]))
    assert values['before']['means'] == values['after']['means']


def test_opposite_effects_across_states_do_not_cancel_absolute_relevance():
    q = np.array([[.9, .1, .9, .1], [.1, .9, .1, .9]])
    row = group_effect_profiles(q, masks(), {'topic': np.array([True, False])}, np.array([.5, .5]))['topic']
    assert row['feature_means']['actual_effect'][0] == pytest.approx(0)
    assert row['means']['minimum_positive_effect'] == pytest.approx(.4)
    assert row['means']['minimum_absolute_effect'] == pytest.approx(.8)


def test_uncertain_single_features_do_not_become_joint_use():
    q = np.array([[.9, .1, .9, .1], [.9, .9, .1, .1]])
    row = group_effect_profiles(q, masks(), {'whole': np.ones(2, dtype=bool)}, np.array([.5, .5]))['whole']
    np.testing.assert_allclose(row['feature_means']['positive_context_effect'], [.4, .4])
    assert row['means']['minimum_positive_effect'] == 0
    assert row['means']['minimum_absolute_effect'] == 0


def test_omitted_redundant_reason_can_vary_in_common_contexts():
    q = np.array([[.9, .9, .9, .1]])
    row = group_effect_profiles(q, masks(), {'topic': np.array([True, False])}, np.array([1.]))['topic']
    assert row['feature_means']['actual_effect'][0] == 0
    assert row['means']['minimum_positive_effect'] == pytest.approx(.8)


def test_unrelated_named_feature_cannot_inherit_group_effect():
    q = np.array([[.9, .1, .9, .1]])
    row = group_effect_profiles(q, masks(), {'whole': np.ones(2, dtype=bool)}, np.array([1.]))['whole']
    assert row['means']['joint_reflection_effect'] == pytest.approx(.8)
    assert row['means']['minimum_absolute_effect'] == 0


def test_probe_order_and_inputs_are_preserved():
    q = np.array([[.7, .3, .9, .5]]); m = masks(); original = q.copy()
    groups = {'whole': np.ones(2, dtype=bool)}; w = np.array([1.])
    a = group_effect_profiles(q, m, groups, w)['whole']
    order = [3, 1, 0, 2]
    b = group_effect_profiles(q[:, order], m[order], groups, w)['whole']
    for key in a['particle']:
        np.testing.assert_array_equal(a['particle'][key], b['particle'][key])
    np.testing.assert_array_equal(q, original)
    np.testing.assert_array_equal(m, masks())
    np.testing.assert_array_equal(w, [1.])


@pytest.mark.parametrize('bad', ['empty_group', 'integer_group', 'wrong_group_shape',
                               'unnamed', 'no_groups', 'weights_sum', 'negative_weight',
                               'weight_shape', 'nan_weight', 'mask_dimension'])
def test_invalid_groups_or_weights_are_rejected(bad):
    q = np.array([[.7, .3, .9, .5]]); m = masks()
    groups = {'whole': np.ones(2, dtype=bool)}; w = np.array([1.])
    if bad == 'empty_group': groups['whole'][:] = False
    if bad == 'integer_group': groups['whole'] = np.array([1, 1])
    if bad == 'wrong_group_shape': groups['whole'] = np.ones(3, dtype=bool)
    if bad == 'unnamed': groups = {'': np.ones(2, dtype=bool)}
    if bad == 'no_groups': groups = {}
    if bad == 'weights_sum': w = np.array([.5])
    if bad == 'negative_weight': w = np.array([-1.])
    if bad == 'weight_shape': w = np.array([.5, .5])
    if bad == 'nan_weight': w = np.array([np.nan])
    if bad == 'mask_dimension': m = np.array([True, False])
    with pytest.raises(ValueError): group_effect_profiles(q, m, groups, w)
