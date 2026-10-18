"""State replay must preserve the executed belief mechanism and filter identity."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_online_evidence_states import checked_job


def inputs(enabled=True):
    planned = dict(phase='evaluate', model='SEARCH', online_belief=enabled, retain_precision=True,
                   candidate={'id':'fixed'}, particles=128, seeds=[47])
    record = {key:deepcopy(planned[key]) for key in ('candidate','model','particles','seeds')}
    side = dict(enabled=enabled, evidence_matches_retained_state=True, exact_transition_timing=True)
    return planned, deepcopy(planned), record, side


@pytest.mark.parametrize('enabled',[False,True])
def test_accepts_original_completed_variant(enabled):
    checked_job(*inputs(enabled))


@pytest.mark.parametrize('change',['variant','parameters','particles','seed','extra','execution','timing','retention'])
def test_rejects_execution_or_filter_identity_changes(change):
    planned,actual,record,side=inputs()
    if change=='variant':actual['online_belief']=False
    elif change=='parameters':record['candidate']={'id':'changed'}
    elif change=='particles':record['particles']=512
    elif change=='seed':record['seeds']=[48]
    elif change=='extra':actual['unexplained']=True
    elif change=='execution':side['enabled']=False
    elif change=='timing':side['exact_transition_timing']=False
    elif change=='retention':actual['retain_precision']=False
    with pytest.raises(ValueError):checked_job(planned,actual,record,side)
