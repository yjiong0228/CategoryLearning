import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_reason_groups import summarize_groups


def row(choice=2, text='尾巴最长。', scope='complete', **extra):
    return dict(reading='midpoint', model='MIX', arm='SEARCH', phase='training', text=text,
        choice=choice, scope=scope, correct=True, ordinal_trial=1,
        functional_support_any_component=1., functional_support=.2, prior_functional_support=.1,
        functional_conflict=.1, functional_undetermined=.7, functional_availability_any_component=.3,
        diagnostic_rule_presence=.3, diagnostic_rule_belief=.2, diagnostic_rule_response_share=.2) | extra


def test_same_words_with_different_choices_keep_different_catalogue_coverage():
    table = pd.DataFrame([row(), row(ordinal_trial=2), row(choice=1, functional_support_any_component=0.)])
    groups = summarize_groups(table).set_index('choice')
    assert groups.loc[2, 'reports'] == 2 and groups.loc[1, 'reports'] == 1
    assert groups.loc[2, 'catalogue_support_rules'] == 1 and groups.loc[1, 'catalogue_support_rules'] == 0


def test_unresolved_and_empty_reports_remain_in_groups():
    table = pd.DataFrame([row(text='脖子比头长很多。', scope='needs_review',
                             functional_support_any_component=np.nan, functional_support=np.nan),
                          row(text='', scope='missing', functional_support_any_component=np.nan, functional_support=np.nan)])
    groups = summarize_groups(table)
    assert groups.reports.sum() == 2 and set(groups.scope) == {'needs_review', 'missing'}
    assert groups.catalogue_support_rules.isna().all() and groups.functional_support.isna().all()


def test_inconsistent_catalogue_coverage_for_same_reason_choice_is_rejected():
    table = pd.DataFrame([row(), row(functional_support_any_component=0.)])
    with pytest.raises(ValueError, match='inconsistent catalogue coverage'):
        summarize_groups(table)
