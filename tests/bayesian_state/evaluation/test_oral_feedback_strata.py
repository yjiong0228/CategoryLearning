"""Protect pooling, whole-filter uncertainty and report denominators by feedback."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_feedback_strata import stratified_estimates


def example():
    probability = np.array([[.1, .3, .2, .6], [.9, .7, .8, .4]])
    support = np.array([[.9, .2, .8, .3], [.1, .6, .2, .7]])
    conditional = np.stack((support, np.zeros_like(support), 1 - support), axis=-1)
    conditional = np.repeat(conditional[:, :, None, :], 2, axis=2)
    joint = conditional * probability[:, :, None, None]
    prior = np.broadcast_to(np.array([.4, .1, .5]), joint.shape).copy()
    return dict(joint=joint, probability=probability, prior=prior,
                complete=np.array([True, True, False, True]), feedback=np.array([0., 1., .5, 1.]), cut=2)


def test_pool_joint_before_conditioning_and_keep_filter_as_uncertainty_unit():
    result = stratified_estimates(**example())['training', 'incorrect']
    # One report: weighted support is (.1*.9 + .9*.1)/(.1+.9), not .5.
    np.testing.assert_allclose(result['point'][0, :, 0], .18)
    np.testing.assert_allclose(result['point'][1, :, 0], .4)
    np.testing.assert_allclose(result['point'][2, :, 0], -.22)
    np.testing.assert_allclose(result['deleted'][:, 0, 0, 0], [.1, .9])
    np.testing.assert_allclose(result['mcse'][0, :, 0], .4)
    np.testing.assert_allclose(result['mcse'][2, :, 0], .4)
    assert result['reports'] == result['trials'] == 1


def test_feedback_and_temporal_partitions_reconstruct_full_report_means():
    values = stratified_estimates(**example())
    for window in ('whole', 'training', 'test'):
        parts = [values[window, key] for key in ('incorrect', 'partial', 'correct')]
        full = values[window, 'all']
        assert sum(p['reports'] for p in parts) == full['reports']
        assert sum(p['trials'] for p in parts) == full['trials']
        for name in ('point', 'deleted'):
            reconstructed = sum(p[name] * p['reports'] for p in parts if p['reports']) / full['reports']
            np.testing.assert_allclose(reconstructed, full[name], atol=1e-12, rtol=0)
    for name in ('point', 'deleted'):
        reconstructed = (2 * values['training', 'all'][name] + values['test', 'all'][name]) / 3
        np.testing.assert_allclose(reconstructed, values['whole', 'all'][name], atol=1e-12, rtol=0)
    assert values['test', 'partial']['trials'] == 1
    assert values['test', 'partial']['reports'] == 0
    assert np.isnan(values['test', 'partial']['point']).all()
    assert np.isnan(values['test', 'incorrect']['mcse']).all()


def test_unscored_report_nan_is_preserved_but_complete_report_requires_labels():
    inputs = example()
    inputs['joint'][:, 2] = np.nan
    inputs['prior'][:, 2] = np.nan
    result = stratified_estimates(**inputs)
    assert result['whole', 'all']['reports'] == 3
    inputs['complete'][2] = True
    with pytest.raises(ValueError, match='Complete reports'):
        stratified_estimates(**inputs)


def test_complete_partial_feedback_report_remains_its_own_stratum():
    inputs = example()
    inputs['complete'][2] = True
    result = stratified_estimates(**inputs)
    partial = result['test', 'partial']
    assert partial['reports'] == partial['trials'] == 1
    np.testing.assert_allclose(partial['point'][0, :, 0], .32)
    assert result['test', 'correct']['reports'] == 1
    assert result['test', 'incorrect']['reports'] == 0
    assert result['whole', 'all']['reports'] == 4


@pytest.mark.parametrize('change', ['feedback', 'probability', 'boundary', 'mass'])
def test_invalid_feedback_probability_boundary_or_mass_is_rejected(change):
    inputs = example()
    if change == 'feedback':
        inputs['feedback'][0] = .25
    elif change == 'probability':
        inputs['probability'][0, 0] = 0
    elif change == 'boundary':
        inputs['cut'] = 4
    else:
        inputs['joint'][0, 0, 0, 0] += .1
    with pytest.raises((ValueError, AssertionError)):
        stratified_estimates(**inputs)
