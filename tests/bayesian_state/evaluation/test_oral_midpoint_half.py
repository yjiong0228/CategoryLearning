"""Use the user-confirmed range midpoint without changing legacy encodings."""
import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.oral.structure_0923 import (
    MIDPOINT_HALF_VERSION, encode_report, report_encoding_version,
)
from src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit import (
    encode_observations, numerical_constraints,
)
from src.Bayesian_state.evaluation.oral.constraint_alignment import evaluate_component

FEATURES = ('neck','head','leg','tail')


@pytest.mark.parametrize('text,ordinary', [
    ('脖子和头均短于一半','脖子和头短'), ('尾巴长于1/2','尾巴长'),
    ('腿大于二分之一','腿长'), ('脖子短于一半，头比腿长','脖子短，头比腿长'),
])
def test_midpoint_matches_existing_absolute_geometry_and_keeps_source(text,ordinary):
    names = tuple(reversed(FEATURES))
    coded = encode_report(text,names,half_reference='range_midpoint')
    expected = encode_report(ordinary,names)
    assert coded.status == expected.status == 'coded'
    assert coded.tokens == expected.tokens
    half = [r for r in coded.relations if r['kind']=='fraction_reference']
    assert half and all(r['reference']=='range_midpoint' and r['numerical_threshold']==.5 for r in half)
    assert all(r['item'] in text for r in half)
    assert 'half_reference_range_midpoint' in coded.flags


def test_confirmed_midpoint_has_support_conflict_and_correct_scale():
    coded = encode_report('脖子短于一半',FEATURES,half_reference='range_midpoint')
    for text, verdict in [('脖子短','support'),('脖子长','conflict'),('头短','undetermined')]:
        result = evaluate_component(coded.tokens,*numerical_constraints(encode_report(text,FEATURES).tokens))
        assert result.structural == result.functional == verdict
    A,b = numerical_constraints(coded.tokens)
    assert np.all(A@np.array([.49,0,0,0])<=b)
    assert not np.all(A@np.array([.51,0,0,0])<=b)
    # Physical range [0.25,1.25] has midpoint 0.75, hence normalized x=0.5.
    assert ((.25+1.25)/2-.25) == .5
    assert report_encoding_version(half_reference='range_midpoint') == MIDPOINT_HALF_VERSION


@pytest.mark.parametrize('text',['尾巴短于躯干的一半','尾巴不短于一半','尾巴大约一半'])
def test_distinct_referents_and_unsupported_syntax_remain_reviewable(text):
    assert encode_report(text,FEATURES,half_reference='range_midpoint').status=='needs_review'


def test_conflicting_midpoint_reports_remain_annotation_issues():
    frame=pd.DataFrame(dict(iSub=[308],condition=[3],choice=[1],text=['脖子长于一半，脖子短']))
    for i,name in enumerate(FEATURES,1):frame[f'feature{i}_name']=name
    result,queries=encode_observations(frame,half_reference='range_midpoint')
    assert result.scope.tolist()==['inconsistent_constraints'] and not queries


def test_reference_modes_are_explicit_and_mutually_exclusive():
    original=encode_report('脖子短于一半',FEATURES)
    assert original==encode_report('脖子短于一半',FEATURES,half_reference=None)
    assert original.status=='needs_review'
    with pytest.raises(ValueError,match='either'):
        encode_report('',FEATURES,symbolic_half_reference=True,half_reference='range_midpoint')
    with pytest.raises(ValueError,match='half_reference'):
        encode_observations(pd.DataFrame(),half_reference='maximum_half')
