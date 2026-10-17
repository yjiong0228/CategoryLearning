"""Preserve half-reference content without supplying its missing referent."""
import json

import pandas as pd
import pytest

from src.Bayesian_state.evaluation.oral.constraint_alignment import (
    InconsistentReportError, evaluate_component, validate_report_constraints,
)
from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit import (
    encode_observations, numerical_constraints,
)

FEATURES = ('neck', 'head', 'leg', 'tail')


def test_half_extension_is_opt_in_and_retains_feature_mapping_and_original_text():
    text = '脖子和头均短于一半。'
    original = encode_report(text, FEATURES)
    assert original.status == 'needs_review' and not original.tokens
    assert original == encode_report(text, FEATURES, symbolic_half_reference=False)
    coded = encode_report(text, tuple(reversed(FEATURES)), symbolic_half_reference=True)
    assert coded.status == 'coded'
    assert set(coded.tokens) == {'half_unknown:3:lt', 'half_unknown:2:lt'}
    assert all(relation['reference'] is None and relation['numerical_threshold'] is None
               and relation['fraction'] == [1, 2] for relation in coded.relations)
    assert 'half_reference_unknown' in coded.flags
    assert all('脖子和头均短于一半' in relation['item'] for relation in coded.relations)


@pytest.mark.parametrize('text', ['尾巴长于1/2', '尾巴大于二分之一', '尾巴长于一半'])
def test_fraction_spellings_preserve_unknown_support(text):
    coded = encode_report(text, FEATURES, symbolic_half_reference=True)
    assert coded.tokens == ('half_unknown:3:gt',)
    for rule in ('尾巴长', '尾巴短'):
        A, b = numerical_constraints(encode_report(rule, FEATURES).tokens)
        matched = evaluate_component(coded.tokens, A, b)
        assert matched.structural == matched.functional == 'undetermined'
        assert matched.symbolic_predicates == 1


def test_known_conflict_survives_unknown_half_reference():
    coded = encode_report('脖子长，尾巴短于一半', FEATURES, symbolic_half_reference=True)
    A, b = numerical_constraints(encode_report('脖子短', FEATURES).tokens)
    assert evaluate_component(coded.tokens, A, b).structural == 'conflict'


@pytest.mark.parametrize('text', [
    '尾巴短于躯干的一半', '尾巴不短于一半', '尾巴大约一半', '尾巴短于一半脖子长于一半',
])
def test_unsupported_referents_negation_and_incomplete_clauses_stay_reviewable(text):
    coded = encode_report(text, FEATURES, symbolic_half_reference=True)
    assert coded.status == 'needs_review' and not coded.tokens


def test_opposite_directions_share_unknown_referent_and_remain_annotation_issue():
    coded = encode_report('脖子长于一半，脖子短于一半', FEATURES, symbolic_half_reference=True)
    assert coded.status == 'needs_review'
    with pytest.raises(InconsistentReportError):
        validate_report_constraints(coded.tokens)
    # A body reference and an unspecified half reference need not share a threshold.
    validate_report_constraints(('body:0:lt', 'half_unknown:0:gt'))


@pytest.mark.parametrize('token', ['half_unknown:-1:lt', 'half_unknown:4:gt',
                                  'half_unknown:1:le', 'half_unknown:1:lt:extra'])
def test_invalid_symbolic_tokens_are_rejected(token):
    with pytest.raises(ValueError, match='Invalid symbolic'):
        validate_report_constraints((token,))


def test_audit_keeps_full_coverage_separate_from_numerical_interpretability():
    frame = pd.DataFrame(dict(iSub=[308], condition=[3], choice=[1], text=['脖子短于一半']))
    for index, name in enumerate(FEATURES, 1):
        frame[f'feature{index}_name'] = name
    old, old_queries = encode_observations(frame)
    coded, queries = encode_observations(frame, symbolic_half_reference=True)
    assert old.scope.tolist() == ['needs_review'] and not old_queries
    assert coded.scope.tolist() == ['complete'] and len(queries) == 1
    assert json.loads(coded['flags'].iloc[0]) == ['half_reference_unknown']
    with pytest.raises(ValueError, match='boolean'):
        encode_observations(frame, symbolic_half_reference='false')
