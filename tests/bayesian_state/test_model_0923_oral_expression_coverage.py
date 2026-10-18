"""Distinguish separate clause availability from jointly represented content."""
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_expression_coverage import token_coverage


def test_separate_components_do_not_imply_a_joint_basis():
    components=[frozenset(['neck_short']),frozenset(['head_long'])]
    result=token_coverage(('neck_short','head_long'),components,components)
    assert result['all_individually_covered'] and result['combination_limited']
    assert not result['whole_structural_support']
    assert result['max_fraction_in_one_component']==.5


def test_actual_joint_basis_and_unreported_extra_predicate():
    components=[frozenset(['neck_short','head_long','tail_long'])]
    result=token_coverage(('neck_short','head_long'),components,components)
    assert result['whole_structural_support'] and not result['combination_limited']
    assert result['max_fraction_in_one_component']==1.


def test_wrong_choice_label_and_missing_catalogue_predicate_are_separate():
    selected=[frozenset(['neck_short'])]
    all_choices=selected+[frozenset(['neck_long'])]
    label=token_coverage(('neck_long',),selected,all_choices)
    assert label['any_uncovered_selected_choice'] and not label['any_uncovered_all_choices']
    absent=token_coverage(('neck_long','head_medium'),selected,all_choices)
    assert absent['any_uncovered_all_choices']


@pytest.mark.parametrize('tokens',[(),('neck_short','neck_short')])
def test_empty_or_duplicate_predicates_are_rejected(tokens):
    with pytest.raises(ValueError):token_coverage(tokens,[frozenset(['neck_short'])],[frozenset(['neck_short'])])
