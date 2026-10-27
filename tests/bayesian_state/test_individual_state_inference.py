"""Joint-state enumeration checks for the individual-state observer."""
import numpy as np

from src.Bayesian_state.evaluation.individual_state_inference import IndividualStateCollector, summarize
from src.Bayesian_state.evaluation.oral.decision_basis import DecisionBasisCollector
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.inference.backends.particle_filter import _trace_ancestral_indices


def test_capture_matches_shared_oral_basis_and_choice_conditioned_execution():
    space = build_continuous_hypothesis_space(4, 2)
    h = len(space)
    active = np.zeros((2, h)); active[0, [0, 3]] = 1; active[1, 3] = 1
    belief = active.copy(); belief[0, [0, 3]] = [.25, .75]
    ell = np.zeros((2, h, 2)); ell[0, 0] = [.9, .1]; ell[:, 3] = [.1, .9]
    pred = np.einsum('ph,phc->pc', belief, ell)
    snapshot = dict(weights=np.array([.5, .5]), active=active, belief=belief,
        beta=active*5, choice_weights=belief, rule_choice_probabilities=ell,
        prediction=pred, perceived_stimulus=np.array([[.1, .3, .7, .9], [.2, .4, .6, .8]]),
        search=np.zeros(2), admission_event=np.zeros(2))
    c = IndividualStateCollector(space, np.array([1, 1]), 2)
    old = DecisionBasisCollector(space, np.array([1, 1]))
    for t in range(2):
        c(t, 'predictive', snapshot); old(t, 'predictive', snapshot)
        c(t, 'filtered', dict(weights=np.array([.75, .25])))
    raw = c.arrays(); previous = old.arrays()
    out = summarize(raw, np.array([[1, 1], [0, 0]]), _trace_ancestral_indices, [0])
    np.testing.assert_allclose(out['filtered_execution'][:, 0], .5625)
    np.testing.assert_allclose(out['filtered_source'], [[.25, .1875, .5625]]*2)
    np.testing.assert_allclose(out['filtered_component'],
        previous['choice_component_joint_mass']/previous['choice_probability'][:, None, None])
    # Both terminal particles descend from the particle without rule 0.
    np.testing.assert_allclose(out['full_target_present'], [0., .75])
    np.testing.assert_allclose(out['full_target_execution'], [0., .5625])
    np.testing.assert_allclose(out['full_component'].sum(axis=(1, 2)), 1.)
    np.testing.assert_allclose(out['filtered_active_precision'], out['filtered_active']*5.)


def test_shared_component_allocation_keeps_disconnected_category_ties():
    from types import SimpleNamespace
    from src.Bayesian_state.evaluation.oral.decision_basis import nearest_component_weights
    space = [SimpleNamespace(categories=[SimpleNamespace(components=[0, 1])])]
    geometry = SimpleNamespace(tolerance=1e-9,
        distances_to_polytope=lambda x, c: np.abs(x[:, 0]-c))
    x = np.array([[0.], [.5], [1.], [.5]])
    np.testing.assert_array_equal(nearest_component_weights(space, geometry, 0, 0, x),
        [[1., 0.], [.5, .5], [0., 1.], [.5, .5]])
