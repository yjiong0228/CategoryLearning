"""Semantic examples, joint-state weighting and raw trial-identity protection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.evaluation.oral.constraint_alignment import (
    InconsistentReportError, choice_conditioned_weights, evaluate_component,
    summarize_matches, validate_report_constraints,
)
from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit import (
    encode_observations, load_reports, numerical_constraints, run, run_examples,
)

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ('neck', 'head', 'leg', 'tail')


def tokens(text):
    result = encode_report(text, FEATURES)
    assert result.status in {'coded', 'missing'}
    return result.tokens


def match(report, rule):
    return evaluate_component(tokens(report), *numerical_constraints(tokens(rule)))


def test_declared_examples_have_their_prespecified_semantics():
    config = yaml.safe_load((ROOT / 'configs/shared/model0923_oral_constraint_audit.yaml').read_text())
    examples = run_examples(config)
    assert len(examples) == 14 and examples.passed.all()


def test_unspoken_predicate_is_not_a_contradiction():
    assert match('脖子短', '脖子短，头长').structural == 'support'
    assert match('脖子短，尾巴长', '脖子短').structural == 'undetermined'


def test_geometric_entailment_does_not_assert_explicit_comparison():
    result = match('头比腿长', '头长，腿短')
    assert result.functional == 'support'
    assert result.structural == 'undetermined'
    assert result.explicit_predicates == 0


def test_union_of_different_states_cannot_invent_a_joint_basis():
    report = '脖子短，尾巴长'
    values = [match(report, '脖子短'), match(report, '尾巴长')]
    result = summarize_matches(np.array([.5, .5]), values)
    assert result['structural_support'] == 0
    assert result['functional_support'] == 0
    assert result['functional_undetermined'] == 1
    assert match(report, report).structural == 'support'


def test_all_components_need_independently_fixed_weights():
    values = [match('脖子短', '脖子短'), match('脖子短', '脖子长')]
    result = summarize_matches(np.array([.2, .8]), values)
    assert result['functional_support'] == pytest.approx(.2)
    assert result['functional_conflict'] == pytest.approx(.8)
    for domain in ('structural', 'functional'):
        assert sum(v for k, v in result.items() if k.startswith(domain)) == pytest.approx(1)


def test_condition_joint_mass_before_normalization():
    # Particle 1 is unlikely to produce the observed choice. Equal averaging of
    # its normalized rule posterior would falsely give its rule half the mass.
    weights = choice_conditioned_weights(np.array([.5, .5]), np.eye(2),
                                         np.array([[.9, .1], [.9, .1]]))
    np.testing.assert_allclose(weights, [[.9, 0], [0, .1]])
    assert weights.sum() == pytest.approx(1)


def test_duplicates_order_and_positive_constraint_scaling_do_not_change_alignment():
    oral = tokens('脖子短，尾巴长')
    A, b = numerical_constraints(tokens('脖子短，尾巴长，头长'))
    first = evaluate_component(oral, A, b)
    second = evaluate_component(tuple(reversed(oral)) + oral, 7 * A, 7 * b)
    assert first == second


def test_feature_remapping_preserves_meaning():
    reverse = tuple(reversed(FEATURES))
    oral = encode_report('脖子短，头长', reverse)
    rule = encode_report('脖子短，头长，尾巴长', reverse)
    assert evaluate_component(oral.tokens, *numerical_constraints(rule.tokens)).structural == 'support'
    assert oral.tokens != tokens('脖子短，头长')


def test_body_reference_stays_unknown_but_known_conflict_is_retained():
    unknown = match('脖子比躯干短', '脖子短')
    assert unknown.symbolic_predicates == 1 and unknown.functional == 'undetermined'
    known = match('脖子长，尾巴比躯干长', '脖子短')
    assert known.functional == 'conflict'


def test_missing_report_and_inconsistent_report_are_not_state_failures():
    missing = match('', '脖子短')
    assert not missing.available
    with pytest.raises(ValueError, match='Missing report'):
        summarize_matches(np.array([1.]), [missing])
    with pytest.raises(ValueError, match='Report has no positive-volume'):
        evaluate_component(tokens('脖子短') + tokens('脖子长'),
                           *numerical_constraints(tokens('头长')))
    with pytest.raises(InconsistentReportError, match='opposing symbolic'):
        validate_report_constraints(['body:0:lt', 'body:0:gt'])


@pytest.mark.parametrize('weights', [[.2, .2], [1.1, -.1], [np.nan, 1.]])
def test_invalid_state_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        summarize_matches(np.array(weights), [match('脖子短', '脖子短')] * 2)


def test_impossible_choice_and_degenerate_rule_are_rejected():
    with pytest.raises(ValueError, match='impossible choice'):
        choice_conditioned_weights(np.array([1.]), np.array([[1.]]), np.array([[0.]]))
    with pytest.raises(ValueError, match='positive volume'):
        evaluate_component(tokens('头长'), *numerical_constraints(tokens('脖子短') + tokens('脖子长')))


def fixture_data(tmp_path, texts):
    frame = pd.DataFrame(dict(iSub=[107] * len(texts), condition=[1] * len(texts),
                              iSession=[1] * len(texts), iTrial=np.arange(1, len(texts) + 1),
                              choice=[1] * len(texts), text=texts))
    for i, name in enumerate(FEATURES, 1):
        frame[f'feature{i}_name'] = name
    frame.to_csv(tmp_path / 'processed.csv', index=False)
    raw = tmp_path / 'raw'; raw.mkdir()
    frame[['iSession', 'iTrial', 'text']].iloc[::-1].to_csv(raw / 'Task2_107_rec.csv', index=False)
    config = dict(processed='processed.csv', raw_directory='raw', subjects=[107],
                  sample_role='development', window=2, workers=1, tolerance=1e-9,
                  feature_order=list(FEATURES), examples=[dict(name='direct', report='脖子短',
                  rule='脖子短', expected=['support', 'support'])])
    return config, frame


def test_raw_join_preserves_order_and_scope_without_using_feedback(tmp_path):
    config, _ = fixture_data(tmp_path, ['脖子短', '腿中等，头长', None])
    frame, hashes = load_reports(config, tmp_path)
    result, queries = encode_observations(frame)
    assert result.iTrial.tolist() == [1, 2, 3]
    assert result.scope.tolist() == ['complete', 'parsed_fragment', 'missing']
    assert len(queries) == 2 and len(hashes) == 2
    assert not {'category', 'feedback'} & set(result.columns)


def test_contradictory_fragments_are_retained_as_annotation_issues(tmp_path):
    config, _ = fixture_data(tmp_path, ['脖子短，脖子长，腿中等', '脖子比躯干短，脖子比躯干长', '头长'])
    frame, _ = load_reports(config, tmp_path)
    result, queries = encode_observations(frame)
    assert result.scope.tolist() == ['inconsistent_constraints', 'inconsistent_constraints', 'complete']
    assert result.query_id.iloc[:2].eq('').all()
    assert result.constraint_issue.iloc[:2].str.len().gt(0).all()
    assert len(queries) == 1


@pytest.mark.parametrize('corruption', ['duplicate', 'text', 'missing'])
def test_raw_identity_or_text_corruption_fails(tmp_path, corruption):
    config, frame = fixture_data(tmp_path, ['脖子短', '头长'])
    raw = frame[['iSession', 'iTrial', 'text']].copy()
    if corruption == 'duplicate':
        raw = pd.concat([raw, raw.iloc[:1]])
    elif corruption == 'missing':
        raw = raw.iloc[:1]
    else:
        raw.loc[0, 'text'] = '脖子长'
    raw.to_csv(tmp_path / 'raw/Task2_107_rec.csv', index=False)
    with pytest.raises(ValueError):
        load_reports(config, tmp_path)


@pytest.mark.parametrize('texts', [['脖子短', '腿中等，头长', None], [None, None]])
def test_small_workflow_and_existing_output_guard(tmp_path, texts):
    config, _ = fixture_data(tmp_path, texts)
    config['processed'] = str(tmp_path / config['processed'])
    config['raw_directory'] = str(tmp_path / config['raw_directory'])
    # load_reports records repository-relative input provenance. For this
    # standalone fixture, substitute a reader with its own temporary root.
    import src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit as workflow
    from unittest.mock import patch
    original = workflow.load_reports
    def read_local(cfg):
        local = dict(cfg, processed='processed.csv', raw_directory='raw')
        frame, hashes = original(local, tmp_path)
        return frame, {str(tmp_path / key): value for key, value in hashes.items()}
    source = tmp_path / 'config.yaml'; source.write_text(yaml.safe_dump(config, allow_unicode=True))
    output = tmp_path / 'audit'
    with patch.object(workflow, 'load_reports', read_local):
        run(source, output, 1)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['trials'] == len(texts)
    assert manifest['result_kind'] == 'catalogue_representability_not_fitted_alignment'
    assert (output / 'index.html').exists()
    with pytest.raises(FileExistsError):
        run(source, output)
