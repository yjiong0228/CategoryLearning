"""Task-provenance exclusions preserve the fitted cohort and every kept model."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_adaptive_task import select_task_nominees


def test_task_subset_keeps_all_models_and_does_not_change_original_fit():
    frozen = {'nominees': [dict(subject=s, model=m) for s in [106, 303] for m in ['B', 'R1']]}
    before = deepcopy(frozen)
    selected, reasons = select_task_nominees(frozen, {'excluded_subjects': {303: 'Documented task mismatch'}})
    assert selected == before['nominees'][:2]
    assert frozen == before
    assert reasons == [dict(subject=303, scope='adaptive_task_only', reason='Documented task mismatch')]
    assert select_task_nominees(frozen, {}) == (before['nominees'], [])


@pytest.mark.parametrize('excluded', [{999: 'unknown'}, {106: ''}, {'106': 'string ID'},
                                      {True: 'boolean'}, [106], {106: 'all removed'}])
def test_invalid_or_empty_task_cohort_is_rejected(excluded):
    with pytest.raises(ValueError):
        select_task_nominees({'nominees': [dict(subject=106, model='B')]}, {'excluded_subjects': excluded})
