import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.particle_region_basis import particle_region_basis


def example():
    # The two particles associate different regions with response 1. Averaging
    # geometry and association separately would lose this dependence.
    q = np.array([[[[.7, .1, .1, .1], [.1, .7, .1, .1]],
                   [[.1, .1, .7, .1], [0., 0., 0., 0.]]]])
    a = np.full((1, 2, 2, 4, 4), .1)
    a[0, 0, 0, :, 1] = .7
    a[0, 0, 0, 0] = [.7, .1, .1, .1]
    a[0, 0, 1, :, 2] = .7
    a[0, 1, 0, :, 3] = .7
    a[0, 1, 0, 2] = [.7, .1, .1, .1]
    a[0, 1, 1] = 0.
    w = np.array([[[.6, .4], [1., 0.]]])
    prior = np.array([[.3, .7]])
    particle_p = np.einsum('tpk,tpkg,tpkgc->tpc', w, q, a)
    p = np.einsum('tp,tpc->tc', prior, particle_p)
    state = dict(rule_ids=np.array([[[0, 2], [0, -1]]]), total_rules=np.array(3),
        prior_weights=prior, choices=np.array([1]), geometry=q, associations=a, choice_weights=w,
        particle_prediction=particle_p, prediction=p, choice_probability=p[:, 0])
    rho = np.zeros((1, 2, 2, 4, 2)); rho[..., 0] = .25; rho[..., 1] = .75
    return state, rho


def test_particle_joint_mass_matches_direct_enumeration_with_sparse_padding():
    state, rho = example()
    result = particle_region_basis(state, rho)
    expected_prior = np.zeros((1, 3, 4, 2)); expected_joint = expected_prior.copy()
    for p in range(2):
        for k in range(2):
            rule = state['rule_ids'][0, p, k]
            if rule < 0:
                continue
            for g in range(4):
                for m in range(2):
                    mass = state['prior_weights'][0, p]*state['choice_weights'][0, p, k]*state['geometry'][0, p, k, g]*rho[0, p, k, g, m]
                    expected_prior[0, rule, g, m] += mass
                    expected_joint[0, rule, g, m] += mass*state['associations'][0, p, k, g, 0]
    np.testing.assert_allclose(result['prechoice_component_mass'], expected_prior, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result['choice_component_joint_mass'], expected_joint, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result['choice_component_joint_mass'].sum(), state['prediction'][0, 0], atol=1e-15, rtol=0)
    # The response-conditioned relative weight of the two rules differs from
    # the prior weight: local choice support is part of this reason projection.
    before = result['prechoice_component_mass'].sum((2, 3))[0]
    after = result['choice_component_joint_mass'].sum((2, 3))[0]/result['choice_probability'][0]
    assert not np.allclose(before, after)


@pytest.mark.parametrize('change', ['duplicate', 'padding', 'weights'])
def test_invalid_particle_alignment_is_rejected(change):
    state, rho = example()
    if change == 'duplicate':
        state['rule_ids'][0, 0, 1] = 0
    elif change == 'padding':
        state['associations'][0, 1, 1] = .25
    else:
        state['prior_weights'][0] = [1., 1.]
    with pytest.raises((ValueError, AssertionError)):
        particle_region_basis(state, rho)
