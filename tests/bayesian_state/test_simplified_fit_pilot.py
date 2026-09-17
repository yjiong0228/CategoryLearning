"""Scientific selection/scoring contracts of the isolated low-budget pilot."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.optimization.model_0826 import CAPACITY_PATH, EXECUTION_PATH, WORKSPACE_PROFILE_KEY
from src.Bayesian_state.workflows.analysis.pilot_model_0826_simplified_fit import (
    STATE_KEYS, compare_arrays, mixture_nll, paired_difference, point_id,
    score_one, select_diverse, validation_seeds,
)
from src.Bayesian_state.simulation.data import SubjectTrialDataLoader


def point(m, chi, beta):
    return {WORKSPACE_PROFILE_KEY: {CAPACITY_PATH: m, EXECUTION_PATH: bool(chi)},
            'engine.modules.beta_mod.kwargs.beta_init': beta}


def test_shortlist_keeps_competitive_distinct_structures_without_duplicates():
    points = [point(3, 0, 5), point(3, 0, 10), point(1, 0, 5), point(5, 1, 5)]
    rows = [{'hyperparams': p, 'aggregated_error': float(i)} for i,p in enumerate(points)]
    kept = select_diverse(rows, top_k=2, per_workspace=True)
    assert {point_id(r['hyperparams']) for r in kept} == {point_id(p) for p in points}
    assert len(kept) == 4
    assert len(select_diverse(rows, top_k=2, per_workspace=False)) == 2


def test_probability_mixture_and_mask_are_preserved():
    p = np.array([[[.1,.9],[.2,.8]], [[.9,.1],[.8,.2]]])
    result = mixture_nll(p, np.array([0,1]), np.array([False,True]))
    assert result == pytest.approx(-np.log(.5))
    assert result != pytest.approx((-np.log(.8)-np.log(.2))/2)


def test_seed_families_are_disjoint_deterministic_and_not_candidate_dependent():
    a = validation_seeds(5,129,'screen',8)
    b = validation_seeds(5,129,'confirmation',16)
    assert a == validation_seeds(5,129,'screen',8)
    assert len(set(a+b)) == 24
    assert not set(a).intersection(validation_seeds(5,229,'screen',8))


def arrays():
    p = np.tile(np.array([[[.3,.7],[.4,.6]]]), (4,1,1))
    result = {'probabilities':p, 'observed':np.array([0,1]),
              'mask':np.array([False,True]), 'seeds':np.arange(4)}
    for key in STATE_KEYS:
        result[key] = p.copy() if key in STATE_KEYS[:2] else p[:,:,0].copy()
    return result


def test_paired_numerical_difference_and_state_comparison_have_correct_direction():
    right = arrays()
    left = deepcopy(right)
    left['probabilities'][:,:,0] = .8
    left['probabilities'][:,:,1] = .2
    diff = paired_difference(left,right,7,100)
    assert diff['mean_nll_difference'] == pytest.approx(np.log(3))
    np.testing.assert_allclose(diff['paired_numerical_interval95'],np.log(3))
    assert compare_arrays(left,right)['mean_rule_total_variation'] == 0
    left['marginal_prior'][:] = [1.,0.]
    assert compare_arrays(left,right)['mean_rule_total_variation'] == pytest.approx(.65)
    left['seeds'][0] = 100
    with pytest.raises(AssertionError):
        paired_difference(left,right,7,100)


def test_validation_really_applies_candidate_parameters_to_the_shared_engine():
    engine = yaml.safe_load(Path('configs/exp123/model_struct/pmh_model_cond1_0826.yaml').read_text())
    loader = SubjectTrialDataLoader(engine)
    arrays = loader._extract_arrays(loader._get_subject_frame(129, 1.), 32)
    context = {'subject':129, 'condition':1, 'engine':engine, 'arrays':arrays,
               'processed_dir':Path('data/exp123/processed')}
    for capacity,chi in ((1,0),(5,1)):
        result = score_one(context,point(capacity,chi,5),2,8726)
        np.testing.assert_allclose(result['marginal_active_probability'].sum(axis=1),capacity)
    assert engine['modules']['hypo_transitions_mod']['kwargs']['capacity'] == 3
