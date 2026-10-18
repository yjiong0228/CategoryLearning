"""Preserve a common fit policy while fixing the demonstrably inactive channel."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_0923_online_evidence_refit as fit


def settings():
    a=yaml.safe_load(Path('configs/shared/model0923_online_evidence_refit_s208.yaml').read_text())
    b=yaml.safe_load(Path('configs/shared/model0923_reentry_precision_refit_s208.yaml').read_text())
    return a,b


def test_only_inactive_channel_and_execution_budget_are_changed():
    a,b=settings();fit.validate_settings(a,b)


@pytest.mark.parametrize('change',['seed','particles','support','subjects','fixed','mechanism'])
def test_reject_changed_scientific_policy(change):
    a,b=settings();a=deepcopy(a)
    if change=='seed':a['seed']+=1
    elif change=='particles':a['budgets']['confirmation']['particles']=256
    elif change=='support':a['evidence_decay_values']=[.8]
    elif change=='subjects':a['subjects']=[207]
    elif change=='fixed':a['fixed_coordinates']['unrelated']=1
    elif change=='mechanism':a['belief_intervention']='other'
    with pytest.raises(ValueError):fit.validate_settings(a,b)


def test_projection_does_not_change_other_coordinates_or_original_records():
    point={fit.TRANSFER:1.,'capacity':4}
    frozen={'nominees':[{'candidate':{'id':'original','hyperparams':point}}]}
    projected=fit.projected_warm_starts(frozen,{fit.TRANSFER:.01})
    assert projected['nominees'][0]['candidate']['hyperparams']=={fit.TRANSFER:.01,'capacity':4}
    assert frozen['nominees'][0]['candidate']['hyperparams'][fit.TRANSFER]==1.
    space={fit.TRANSFER:[.01,.1,1.],'capacity':[2,4,6]}
    assert fit.fixed_space(space,{fit.TRANSFER:.01})=={fit.TRANSFER:[.01],'capacity':[2,4,6]}
    assert len(space[fit.TRANSFER])==3
    with pytest.raises(ValueError):fit.fixed_space(space,{fit.TRANSFER:.5})
