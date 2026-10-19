"""Protect matched search policies and explicit comparison labels."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from src.Bayesian_state.evaluation.model_0923_cohort_execution_comparison import (
    check_settings, presentation_columns,
)


def settings():
    old = dict(subjects=[207], belief_mechanism='SEPARATE', scope='common_policy_development_belief_refit',
               parallel_budget=48, seed=13, budgets={'screen': {'particles': 128}}, data='original_data')
    new = dict(deepcopy(old), scope='development_cohort_dominant_execution_refit',
               execution_policy='SELECT', execution_reference_fit='/tmp/original_cohort_fit')
    return old, new


def test_same_policy_accepts_only_execution_and_worker_changes():
    old, new = settings()
    new['parallel_budget'] = 32
    check_settings(old, new, 207, Path('/tmp/original_cohort_fit'))


@pytest.mark.parametrize('key,value', [('subjects',[306]), ('seed',14), ('data','different_data'),
                                      ('budgets',{'screen': {'particles': 256}}), ('belief_mechanism','ONLINE')])
def test_changed_scientific_policy_rejected(key,value):
    old, new = settings();new[key] = value
    with pytest.raises(ValueError):
        check_settings(old,new,207,Path('/tmp/original_cohort_fit'))


def test_wrong_reference_is_rejected():
    old, new = settings()
    with pytest.raises(ValueError):
        check_settings(old,new,207,Path('/tmp/unrelated_cohort_fit'))


def test_labels_change_without_changing_rows_or_source():
    source = pd.DataFrame({'online_nll':[.2,.4], 'separate_nll':[.1,.3], 'state_trial':[1,2]})
    before=source.copy()
    result=presentation_columns(source)
    assert list(result)==['mix_nll','select_nll','state_trial']
    pd.testing.assert_frame_equal(result.set_axis(source.columns,axis=1),source)
    pd.testing.assert_frame_equal(source,before)
    with pytest.raises(ValueError):
        presentation_columns(pd.DataFrame({'online_nll':[.2],'mix_nll':[.1]}))
