"""Protect the frozen subject and common filter bank in a pairing-only probe."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_pairing_retention_scores import checked_reference


def reference():
    base=dict(model='SEARCH',phase='evaluate',particles=128,seeds=[11,12,13,14],
        candidate={'hyperparams':{'engine.modules.memory_mod.kwargs.gamma':0.}},
        folder='prepared',runtime='archived',mechanism={'kind':'E'},retain_precision=True)
    plans=[dict(variant=label,job=dict(deepcopy(base),online_belief=flag))
           for label,flag in [('baseline',False),('online',True)]]
    return dict(smoke=False,plans=plans,score_settings={'subject':306}),dict(status='complete',smoke=False)


def test_valid_reference_preserves_both_original_variants():
    p,d=reference()
    assert checked_reference(p,d,306,128,4) is p['plans']


@pytest.mark.parametrize('key,value', [('particles',64),('seeds',[11,11,13,14]),
    ('online_belief',True),('retain_precision',False),('model','POOL'),('folder','another_subject')])
def test_changed_comparison_identity_is_rejected(key,value):
    p,d=reference();p['plans'][0]['job'][key]=value
    with pytest.raises(ValueError):checked_reference(p,d,306,128,4)


def test_changed_rule_retention_is_rejected():
    p,d=reference();p['plans'][0]['job']['candidate']['hyperparams']['engine.modules.memory_mod.kwargs.gamma']=.8
    with pytest.raises(ValueError):checked_reference(p,d,306,128,4)


@pytest.mark.parametrize('subject,smoke,status',[(308,False,'complete'),(306,True,'complete'),(306,False,'running')])
def test_wrong_participant_or_incomplete_screen_is_rejected(subject,smoke,status):
    p,d=reference();d.update(smoke=smoke,status=status)
    with pytest.raises(ValueError):checked_reference(p,d,subject,128,4)
