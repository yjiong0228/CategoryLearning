"""A local parameter comparison preserves the matched behavior experiment."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_0923_capacity_memory_scores import (
    CAPACITY, MEMORY, choice_contrasts, grid_bank, reference_seeds,
)
from src.Bayesian_state.workflows.analysis.model_0923_recalled_capacity_memory_probe import points


def anchor():
    return candidate({MEMORY:0., CAPACITY:4, 'precision':3.}, 'fixture')


def grid_jobs():
    original = anchor()
    return [dict(candidate=row, anchor=row['id']==original['id'])
            for row in points(original['hyperparams'], [0.,.8,.99], [2,4])]


def test_complete_grid_reuses_one_point_across_simulation_chunks():
    jobs = grid_jobs()
    bank = grid_bank(jobs+deepcopy(jobs), anchor(), [0.,.8,.99], [2,4])
    assert len(bank) == 6
    assert all(x['hyperparams']['precision'] == 3. for x in bank)


@pytest.mark.parametrize('change', ['unrelated_coordinate', 'false_anchor', 'missing_cell'])
def test_probe_cannot_silently_change_other_parameters_or_omit_a_cell(change):
    jobs = grid_jobs()
    if change == 'unrelated_coordinate':
        h = dict(jobs[0]['candidate']['hyperparams'], precision=4.)
        jobs[0]['candidate'] = candidate(h, 'wrong')
    elif change == 'false_anchor':
        jobs[0]['anchor'] = not jobs[0]['anchor']
    else:
        jobs.pop()
    with pytest.raises(ValueError, match='grid'):
        grid_bank(jobs, anchor(), [0.,.8,.99], [2,4])


def references():
    return [dict(smoke=False, phase='evaluate', model='SEARCH', candidate=anchor(),
        particles=512, seeds=[seed], folder='prepared', runtime='archive',
        mechanism={'kind':'E'}, arm_settings={'precision':'fast'}, source_sha256={'core':'sha'})
        for seed in [12,8,42]]


def test_reference_seed_order_is_preserved():
    assert reference_seeds(references(),anchor()) == [12,8,42]


@pytest.mark.parametrize('key,value', [('particles',256), ('folder','different'),
    ('runtime','different'), ('mechanism',{'kind':'J'}), ('arm_settings',{'precision':'slow'}),
    ('source_sha256',{'core':'changed'}), ('smoke',True), ('seeds',[12])])
def test_reference_cannot_mix_numerical_budgets_or_cognitive_settings(key,value):
    jobs = references()
    jobs[1][key] = value
    with pytest.raises(ValueError):
        reference_seeds(jobs,anchor())


def test_scope_scores_preserve_order_and_joint_simulation_error():
    base = np.array([[.2,.3,.7,.8,.8],[.3,.4,.6,.8,.7],[.25,.4,.6,.75,.9]])
    p = base.copy()
    p[:,:2] *= .5
    p[:,2:] *= .9
    scores = {row['phase']:row for row in choice_contrasts(p,base,2)}
    assert scores['prefix']['delta_choice_nll'] == pytest.approx(np.log(2))
    assert scores['later']['delta_choice_nll'] == pytest.approx(-np.log(.9))
    assert scores['full']['choice_nll'] == pytest.approx(
        (2*scores['prefix']['choice_nll']+3*scores['later']['choice_nll'])/5)
    # Whole-filter differences are constant here despite noisy individual scores.
    assert all(row['delta_choice_nll_mc_se'] < 1e-14 for row in scores.values())
    same = choice_contrasts(base,base,2)
    assert all(row['delta_choice_nll'] == row['delta_choice_nll_mc_se'] == 0 for row in same)


@pytest.mark.parametrize('change', ['shape','nan','negative','boundary'])
def test_scope_probability_and_time_boundaries_are_checked(change):
    base = np.full((3,5),.5)
    p,cut = base.copy(),2
    if change=='shape':p=p[:,:4]
    elif change=='nan':p[0,0]=np.nan
    elif change=='negative':p[0,0]=-.1
    else:cut=5
    with pytest.raises(ValueError):
        choice_contrasts(p,base,cut)
