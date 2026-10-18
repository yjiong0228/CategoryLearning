"""Reject accidental participant or parameter changes in the fixed-point screen."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_online_evidence_development_scores import checked_banks


def banks():
    candidate = {'id':'fixed', 'hyperparams':{'capacity':6}}
    job = dict(candidate=candidate, model='SEARCH', retain_precision=True, phase='autonomous',
               smoke=False, folder='prepared',runtime='archived',mechanism={'kind':'E'},arm_settings={})
    return [dict(subject=107,schedule=s,model='SEARCH',candidate=deepcopy(candidate),
                 parameter_provenance='earlier_POOL_point',chunks=[deepcopy(job)]) for s in ('off','online')]


def test_preserves_original_parameter_provenance():
    original=banks()
    result=checked_banks(original,[107])
    assert result[107]==original[1]
    assert result[107]['parameter_provenance']=='earlier_POOL_point'


@pytest.mark.parametrize('subjects',[[],[107,107],[110]])
def test_rejects_undeclared_or_repeated_subjects(subjects):
    with pytest.raises(ValueError):
        checked_banks(banks(),subjects)


@pytest.mark.parametrize('field,value',[
    ('candidate',{'id':'changed','hyperparams':{'capacity':4}}),
    ('folder','different_trials'),('runtime','changed_engine'),
    ('retain_precision',False),('smoke',True),('phase','training')])
def test_rejects_mismatched_actual_development_job(field,value):
    original=banks()
    original[1]['chunks'][0][field]=value
    with pytest.raises(ValueError):
        checked_banks(original,[107])


def test_rejects_relabelled_parameter_origin():
    original=banks();original[1]['parameter_provenance']='new_SEARCH_fit'
    with pytest.raises(ValueError):
        checked_banks(original,[107])
