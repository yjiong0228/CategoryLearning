"""Distinguish the declared deduplication policy from extra search allocation."""
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.model_0923_refit_budget import budget_contrasts, expected_exchange


def point(name, score):
    return dict(id=name, hyperparams={'parameter':name}, mean_nll=score)


def banks():
    initial = {a:[point('anchor', 1.)] for a in ('POOL','SEARCH')}
    screen = {'POOL':[point('a', .2), point('b', .3), point('c', .4), point('d', .5)],
              'SEARCH':[point('a', .2), point('b', .3), point('d', .4), point('c', .5)]}
    return initial,screen,{'POOL':[],'SEARCH':[]}


def test_overlapping_leading_points_reduce_actual_exchange_work():
    assert set(expected_exchange(*banks(),3)) == {'anchor','a','b','c','d'}


def test_separate_leading_points_reach_the_same_seven_point_limit():
    initial,screen,refined = banks()
    screen['SEARCH'] = [point(x,.1+i/10) for i,x in enumerate(['e','f','g','h'])]
    assert len(expected_exchange(initial,screen,refined,3)) == 7


def test_refined_score_supersedes_earlier_score_before_union():
    initial,screen,refined = banks()
    assert set(expected_exchange(initial,screen,refined,1)) == {'anchor','a'}
    refined['POOL'] = [point('a', .9)]
    assert set(expected_exchange(initial,screen,refined,1)) == {'anchor','b','a'}


def test_parameter_identity_cannot_change_between_arms():
    initial,screen,refined = banks()
    screen['SEARCH'][0]['hyperparams'] = {'parameter':'changed'}
    with pytest.raises(ValueError, match='identity changes'):
        expected_exchange(initial,screen,refined,3)


def test_reference_point_must_be_shared():
    initial,screen,refined = banks()
    initial['SEARCH'] = [point('different_anchor', 1.)]
    with pytest.raises(ValueError, match='reference point'):
        expected_exchange(initial,screen,refined,3)


def frames():
    old = pd.DataFrame([dict(subject=208,model=arm,stage=stage,filter_runs=runs,state_updates=runs*128)
        for arm in ('POOL','SEARCH') for stage,runs in [('screen',80),('refine',36),('exchange',10)]])
    new = old.copy()
    new.loc[new.stage.eq('exchange'),['filter_runs','state_updates']] = [14,14*128]
    return old,new,old[old.stage.eq('exchange')].copy(),new[new.stage.eq('exchange')].copy()


def test_actual_exchange_work_is_exported_as_unequal():
    table = budget_contrasts(*frames())
    assert table[table.stage.ne('exchange')].actual_compute_equal.all()
    assert not table[table.stage.eq('exchange')].actual_compute_equal.any()
    assert table.delta_filter_runs.sum() == 8


def test_unexplained_extra_exchange_work_is_rejected():
    old,new,old_audit,new_audit = frames()
    new.loc[new.stage.eq('exchange'),'filter_runs'] += 1
    with pytest.raises(AssertionError):
        budget_contrasts(old,new,old_audit,new_audit)


@pytest.mark.parametrize('stage',['screen','refine'])
def test_deduplication_does_not_allow_other_stage_budget_changes(stage):
    old,new,old_audit,new_audit = frames()
    new.loc[new.stage.eq(stage),'state_updates'] += 128
    with pytest.raises(AssertionError):
        budget_contrasts(old,new,old_audit,new_audit)


def test_missing_stage_is_rejected():
    old,new,old_audit,new_audit = frames()
    with pytest.raises(AssertionError):
        budget_contrasts(old,new.iloc[1:],old_audit,new_audit)
