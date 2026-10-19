"""Boundary and comparison checks for fixed-H precision state replay."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_fixed_h_precision_states as states
from src.Bayesian_state.workflows.analysis.model_0923_fixed_h_precision import INIT, RATE


def fixture():
    reference = {INIT: .1, RATE: .1, 'h_memory': .8}
    point = {'id': 'selected', 'hyperparams': {INIT: .5, RATE: .3, 'h_memory': .8}}
    family = {'precision': 'log', 'upper': 25., 'rate_scale': 1., 'anchor': [.5, .3]}
    protocol = {'cells': {'SELECT_LOG25': {'policy': 'SELECT', 'family': 'LOG25',
                                         'reference': {'hyperparams': reference}}},
                'settings': {'families': {'LOG25': family}, 'evaluation': {'particles': 512}},
                'seeds': {'evaluation': [31, 41]}}
    job = {'phase': 'evaluate', 'model': 'SEARCH', 'smoke': False,
           'fixed_h_precision_cell': 'SELECT_LOG25', 'execution_policy': 'SELECT',
           'candidate': point, 'arm_settings': {k:v for k,v in family.items() if k != 'anchor'},
           'seeds': [31], 'particles': 512}
    return job, deepcopy(point), protocol


def test_frozen_precision_point_preserves_h():
    job, nominee, protocol = fixture()
    states.check_point('SELECT_LOG25', job, nominee, protocol)


@pytest.mark.parametrize('field,value', [('phase','training'), ('particles',128),
    ('seeds',[31,41]), ('seeds',[90]), ('execution_policy','MIX'), ('smoke',True)])
def test_changed_evaluation_identity_rejected(field, value):
    job, nominee, protocol = fixture()
    job[field] = value
    with pytest.raises(ValueError, match='frozen evaluation'):
        states.check_point('SELECT_LOG25', job, nominee, protocol)


def test_changed_h_rejected_even_if_nominee_matches():
    job, nominee, protocol = fixture()
    job['candidate']['hyperparams']['h_memory'] = .2
    nominee['hyperparams']['h_memory'] = .2
    with pytest.raises(ValueError, match='fixed H'):
        states.check_point('SELECT_LOG25', job, nominee, protocol)


def test_wrong_precision_update_rejected():
    job, nominee, protocol = fixture()
    job['arm_settings']['precision'] = 'fast'
    with pytest.raises(ValueError):
        states.check_point('SELECT_LOG25', job, nominee, protocol)


def estimates():
    result = {}
    for policy in ['MIX', 'SELECT']:
        for family, value in [('ADD10', 0.), ('LOG25', 1.)]:
            result[107, policy+'_'+family] = {('later', 'functional', 'complete'):
                {'seeds': np.array([31,41,51]), 'point': np.array([.2,.3,.5])+value*np.array([.1,-.1,0.]),
                 'values': np.array([[.1,.2,.7],[.2,.3,.5],[.3,.4,.3]])+
                           value*np.array([[.1,-.1,0.],[.2,-.2,0.],[.3,-.3,0.]])}}
    return result


def test_precision_contrasts_pair_complete_filters_within_policy():
    table = states.paired_contrasts(estimates())
    assert set(table.policy) == {'MIX', 'SELECT'}
    np.testing.assert_allclose(table.support_difference, .1)
    np.testing.assert_allclose(table.conflict_difference, -.1)
    np.testing.assert_allclose(table.support_difference_mcse, np.sqrt(2*np.var([.1,.2,.3])))
    np.testing.assert_allclose(table.undetermined_difference_mcse, 0.)


def test_seed_mismatch_rejected():
    result = estimates()
    result[107, 'MIX_LOG25']['later', 'functional', 'complete']['seeds'] = np.array([31,41,61])
    with pytest.raises(AssertionError):
        states.paired_contrasts(result)


def test_smoke_states_cannot_enter_scientific_oral(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'smoke': True}))
    (tmp_path/'completion.json').write_text(json.dumps({'status':'complete', 'smoke':True}))
    with pytest.raises(ValueError, match='complete exact'):
        states.align(tmp_path, Path('unused_audit'), tmp_path/'unused_output')


def test_worker_budget_checked_before_source_read():
    with pytest.raises(ValueError, match='workers'):
        states.run(Path('unused_source'), Path('unused_output'), workers=129)
