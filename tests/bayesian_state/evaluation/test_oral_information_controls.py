"""The weighting controls retain joint pooling and prefix uncertainty."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.information_controls import (
    information_controls, jackknife_error,
)


def fixture():
    # Two seeds, two prefix trials and two later trials, two category regions.
    q = np.array([[.2, .8, .3, .7], [.8, .4, .9, .2]])
    basis = np.array([[[.1, .9], [.2, .8], [.8, .2], [.6, .4]],
                      [[.8, .2], [.5, .5], [.4, .6], [.3, .7]]])[..., None]
    prior = np.full_like(basis, .5)
    labels = np.tile(np.array([[[0, 0]], [[1, 1]]]), (4, 1, 1, 1))
    return basis*q[..., None, None], q, prior, labels, np.ones(4, int), 2, {'test': np.array([0, 0, 1, 1], bool)}


def test_joint_pooling_and_reestimated_prefix_in_jackknife():
    args = fixture(); out = information_controls(*args); row = out['windows']['test']
    pooled = args[0].sum(axis=0)/args[1].sum(axis=0)[:, None, None]
    np.testing.assert_allclose(row['point'][0, :, 0], pooled[2:, 0, 0].mean())
    np.testing.assert_allclose(row['point'][2, :, 0], pooled[:2, 0, 0].mean())
    per_seed = args[0]/args[1][..., None, None]
    expected = per_seed[:, 2:, 0, 0].mean(axis=1)-per_seed[:, :2, 0, 0].mean(axis=1)
    np.testing.assert_allclose(row['jackknife'][:, 4, 0, 0], expected[::-1])
    np.testing.assert_allclose(row['mcse'][4, 0, 0], jackknife_error(expected))
    np.testing.assert_allclose(out['trials'][3, :, 0, 0], pooled[:, 0, 0]-.5)
    assert out['trials'][3, 1, 0, 0] < 0


def test_static_control_excludes_unseen_choice_and_records_denominator():
    args = list(fixture()); args[4] = np.array([1, 1, 1, 2])
    row = information_controls(*args)['windows']['test']
    assert (row['requested_reports'], row['reports'], row['excluded_reports']) == (2, 1, 1)


def test_later_states_cannot_change_the_prefix_control():
    args = list(fixture()); first = information_controls(*args)
    args[0] = args[0].copy(); args[0][:, 2:] = args[0][:, 2:, ::-1]
    second = information_controls(*args)
    np.testing.assert_array_equal(first['trials'][2], second['trials'][2])


@pytest.mark.parametrize('error', ['single_seed', 'bad_mass', 'negative_prior', 'early_mask', 'missing_labels'])
def test_invalid_inputs_fail(error):
    args = list(fixture())
    if error == 'single_seed': args[:3] = [x[:1] for x in args[:3]]
    elif error == 'bad_mass': args[0] = args[0]*2
    elif error == 'negative_prior': args[2][0, 0, 0, 0] = -.1
    elif error == 'early_mask': args[6]['test'][0] = True
    elif error == 'missing_labels': args[3][2] = -1
    with pytest.raises((ValueError, AssertionError)): information_controls(*args)
