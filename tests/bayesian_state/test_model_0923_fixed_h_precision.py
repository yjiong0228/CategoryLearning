"""Selection boundaries and fixed-coordinate invariants for the precision probe."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_fixed_h_precision import (
    INIT, RATE, choose_finalists, precision_points, seed_bank,
)


def test_grid_preserves_h_memory_and_input():
    reference = {INIT: .1, RATE: .1, 'memory': .99, 'capacity': 8, 'inertia': 2.}
    before = deepcopy(reference)
    points = precision_points(reference, [.1, .5], dict(upper=25., rates=[.01,.1], anchor=[.1,.1]))
    assert len(points) == len({p['id'] for p in points}) == 4
    assert reference == before
    for p in points:
        assert {k:v for k,v in p['hyperparams'].items() if k not in (INIT,RATE)} == {
            'memory': .99, 'capacity': 8, 'inertia': 2.}
    points[0]['hyperparams']['capacity'] = 1
    assert reference == before


def test_finalists_ignore_later_scores_and_preserve_anchor():
    rows = [dict(id='a',training_nll=.3,later_nll=.01),
            dict(id='b',training_nll=.2,later_nll=4.),
            dict(id='c',training_nll=.1,later_nll=8.)]
    assert choose_finalists(rows,'a',2) == ['c','b','a']
    for r in rows: r['later_nll'] = -r['later_nll']
    assert choose_finalists(rows,'a',2) == ['c','b','a']
    assert choose_finalists(rows,'c',2) == ['c','b']


def test_ties_are_deterministic():
    rows = [dict(id=x,training_nll=1.) for x in ['z','b','a']]
    assert choose_finalists(rows,'z',1) == ['a','z']
    assert choose_finalists(rows[::-1],'z',1) == ['a','z']


@pytest.mark.parametrize('rows,anchor', [
    ([dict(id='a',training_nll=np.nan)],'a'),
    ([dict(id='a',training_nll=.1)],'b'),
    ([dict(id='a',training_nll=.1),dict(id='a',training_nll=.2)],'a'),
])
def test_invalid_selection_is_rejected(rows,anchor):
    with pytest.raises(ValueError): choose_finalists(rows,anchor,1)


def test_stage_seeds_are_reproducible_and_disjoint():
    settings = dict(protocol='precision_test',seed=17,subject=107)
    stages = ['screen','confirmation','evaluation','autonomous']
    banks = [seed_bank(settings,s,64) for s in stages]
    assert banks == [seed_bank(settings,s,64) for s in stages]
    assert len(set(sum(banks,[]))) == 256
    assert seed_bank(dict(settings,subject=208),'screen',64) != banks[0]


@pytest.mark.parametrize('initials,family', [
    ([.1,.1],dict(upper=25.,rates=[.1],anchor=[.1,.1])),
    ([.1],dict(upper=25.,rates=[.1],anchor=[.5,.1])),
    ([0.],dict(upper=25.,rates=[.1],anchor=[0.,.1])),
])
def test_invalid_grid_is_rejected(initials,family):
    with pytest.raises(ValueError): precision_points({INIT:.1,RATE:.1},initials,family)
