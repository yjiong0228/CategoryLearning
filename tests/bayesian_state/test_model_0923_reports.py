"""Report semantics, probability conservation and immutable audit inputs."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.model.oral_observation import predict_choice_oral
from src.Bayesian_state.model.oral_report_space import (
    CatalogueReportSpace, ReportKernelParameters, body_token, facet_token, report_code,
)
from src.Bayesian_state.workflows.analysis.audit_model_0923_reports import audit_reports
from src.Bayesian_state.workflows.analysis.prepare_model_0923 import _write_csv

FEATURES = ('neck', 'head', 'leg', 'tail')
ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ReportKernelParameters(0.8, 0.02, 0.5, 0.75, 0.10)


@pytest.mark.parametrize('text,expected', [
    ('躯干长于脖子和腿。', ('body:0:lt', 'body:2:lt')),
    ('脖子和腿比躯干短。', ('body:0:lt', 'body:2:lt')),
    ('躯干短于脖子、长于尾巴。', ('body:0:gt', 'body:3:lt')),
    ('躯干短于脖子，长于尾巴。', ('body:0:gt', 'body:3:lt')),
    ('躯干短于脖子，且躯干长于尾巴。', ('body:0:gt', 'body:3:lt')),
    ('脖子比躯干长。', ('body:0:gt',)),
])
def test_body_direction_without_hard_threshold(text, expected):
    result = encode_report(text, FEATURES)
    assert result.status == 'coded'
    assert result.tokens == expected
    assert 'body_reference_uncertain' in result.flags
    for relation in result.relations:
        assert relation['physical_reference'] == 0.75
        assert relation['psychological_reference'] is None


def test_feature_order_and_equivalent_comparison_wording():
    first = encode_report('躯干长于脖子。', FEATURES)
    second = encode_report('躯干长于脖子。', ('leg', 'head', 'tail', 'neck'))
    assert first.tokens == ('body:0:lt',)
    assert second.tokens == ('body:3:lt',)
    assert encode_report('脖子比头短。', FEATURES).tokens == encode_report('头长于脖子。', FEATURES).tokens
    assert encode_report('脖子长，头短。', FEATURES).tokens == encode_report('头短，脖子长。', FEATURES).tokens


def test_weighted_sum_retains_four_legs_and_requires_physical_transform():
    result = encode_report('头、脖子、尾巴之和比四条腿之和长。', FEATURES)
    assert result.status == 'needs_review'
    assert result.tokens == ()
    assert result.relations[0]['right'] == {'腿': 4}
    assert result.relations[0]['left'] == {'头': 1, '脖子': 1, '尾巴': 1}
    assert 'physical_scale_required' in result.flags


@pytest.mark.parametrize('text', [
    '头加脖子长。', '三个部位很长。', '脖子长，但我不确定。', '去判断于脖子和尾巴。',
    '躯干短于脖子，躯干长于脖子。', '脖子比脖子长。', '脖子长，脖子短。', '脖子比头长，腿和尾巴不一样。',
])
def test_partial_unsupported_or_contradictory_text_is_not_confidently_scored(text):
    assert encode_report(text, FEATURES).status == 'needs_review'


def test_missing_is_not_empty_and_whole_parse_preserves_intensity_flag():
    assert encode_report('', FEATURES).status == 'missing'
    result = encode_report('腿很短，尾巴比较长。', FEATURES)
    assert result.status == 'coded'
    assert 'intensity_coarsened' in result.flags
    assert result.tokens == encode_report('腿短，尾巴长。', FEATURES).tokens


def test_near_equality_remains_a_two_sided_predicate():
    result = encode_report('脖子和头差不多长。', FEATURES)
    assert result.status == 'coded'
    assert len(result.tokens) == 1
    assert len(json.loads(result.tokens[0][6:])) == 2


def test_legacy_direction_diagnostic_is_separate_from_uncertain_reference():
    assert encode_report('躯干长于脖子和腿。', FEATURES).legacy_body_direction_disagreements == 2
    assert encode_report('脖子比躯干长。', FEATURES).legacy_body_direction_disagreements == 0


@pytest.mark.parametrize('n_categories', [2, 4])
@pytest.mark.parametrize('mention_probability', [0.0, 0.8, 1.0])
def test_kernel_is_a_distribution_in_report_space(n_categories, mention_probability):
    space = CatalogueReportSpace.from_catalogue(build_continuous_hypothesis_space(4, n_categories))
    matrix = space.matrix(replace(PARAMETERS, mention_probability=mention_probability))
    assert matrix.shape == (29 if n_categories == 2 else 116, n_categories, len(space.codes))
    assert np.isfinite(matrix).all() and np.all(matrix >= 0)
    np.testing.assert_allclose(matrix.sum(-1), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(matrix[..., 1], PARAMETERS.other_probability)
    if mention_probability == 0:
        np.testing.assert_allclose(matrix[..., 0], 1 - PARAMETERS.other_probability)
    np.testing.assert_array_equal(matrix, space.matrix(replace(PARAMETERS, mention_probability=mention_probability)))


def test_known_single_predicate_probabilities_and_body_direction():
    space = CatalogueReportSpace.from_catalogue(build_continuous_hypothesis_space(4, 2))
    matrix = space.matrix(PARAMETERS)
    # H0, category 1 is x0 < .5. Body wording softens reference position,
    # never the direction. Formula is derived independently of kernel code.
    p_body = 0.5 * np.exp(-0.5 * ((0.5 - 0.75) / 0.1)**2)
    body_lt = space.codes.index(report_code(('body:0:lt',)))
    body_gt = space.codes.index(report_code(('body:0:gt',)))
    simple = space.codes.index(report_code(encode_report('脖子短。', FEATURES).tokens))
    assert matrix[0, 0, 0] == pytest.approx(0.98 * 0.2)
    assert matrix[0, 0, body_lt] == pytest.approx(0.98 * 0.8 * p_body)
    assert matrix[0, 0, simple] == pytest.approx(0.98 * 0.8 * (1 - p_body))
    assert matrix[0, 0, body_gt] == 0
    wider = space.matrix(replace(PARAMETERS, body_reference_width=0.2))
    assert wider[0, 0, body_lt] > matrix[0, 0, body_lt]
    assert wider[0, 0, body_gt] == 0


def test_union_components_do_not_receive_extra_total_mass():
    space = CatalogueReportSpace.from_catalogue(build_continuous_hypothesis_space(4, 2))
    matrix = space.matrix(replace(PARAMETERS, mention_probability=1, other_probability=0,
                                body_naming_probability=0))
    # H19 is pairwise similarity: the far category has two components.
    nonzero = matrix[19, 1][matrix[19, 1] > 0]
    np.testing.assert_array_equal(nonzero, [0.5, 0.5])


def test_partial_report_has_multiple_sources_and_label_is_preserved():
    space = CatalogueReportSpace.from_catalogue(build_continuous_hypothesis_space(4, 4))
    token = report_code(encode_report('脖子短。', FEATURES).tokens)
    sources = space.source_index()[(token, 0)]
    assert len(sources['omission']) > 1
    opposite = report_code(encode_report('脖子长。', FEATURES).tokens)
    # Exact category support is checked separately, not relabeled to fit words.
    assert token != opposite


def test_kernel_integrates_with_choice_then_report_joint_readout():
    space = CatalogueReportSpace.from_catalogue(build_continuous_hypothesis_space(4, 2))
    kernel = space.matrix(PARAMETERS)
    choices = np.column_stack((np.linspace(.1, .9, 29), np.linspace(.9, .1, 29)))
    weights = np.arange(1, 30, dtype=float)
    weights /= weights.sum()
    result = predict_choice_oral(weights, choices, kernel)
    expected = np.einsum('h,hc,hcr->cr', weights, choices, kernel)
    np.testing.assert_allclose(result.joint_probabilities, expected)
    np.testing.assert_allclose(result.joint_probabilities.sum(1), weights @ choices)
    np.testing.assert_allclose(result.report_given_choice(0).sum(), 1)


@pytest.mark.parametrize('field,value', [('body_reference_width', 0), ('body_reference_width', float('nan')),
                                         ('mention_probability', 1.1), ('other_probability', -1),
                                         ('body_reference_center', 1.5)])
def test_invalid_measurement_parameters_rejected(field, value):
    with pytest.raises(ValueError):
        replace(PARAMETERS, **{field: value})


def _audit_inputs(tmp_path):
    inventory = tmp_path / 'inventory'
    inventory.mkdir()
    trials_path = tmp_path / 'trials.csv'
    rows = []
    for sid, features in [(314, FEATURES), (307, ('leg', 'head', 'tail', 'neck'))]:
        for trial, (choice, text) in enumerate([(1, '躯干长于脖子。'), (2, '躯干长于脖子。'), (2, '')], 1):
            rows.append({'iSub': sid, 'condition': 3, 'iSession': 1, 'iBlock': 1, 'iTrial': trial,
                         'trial': trial, 'choice': choice, 'text': text, 'oral_valid': text != '',
                         'scored': trial != 1, **{f'feature{i}_name': feature for i, feature in enumerate(features, 1)}})
    _write_csv(trials_path, rows)
    _write_csv(inventory / 'subjects.csv', [{'subject': sid, 'condition': 3, 'n_trials': 3} for sid in (314, 307)])
    _write_csv(inventory / 'report_review.csv', [{'status': 'unreviewed'}])
    (inventory / 'manifest.json').write_text(json.dumps({'n_trials': 6, 'sources': [
        {'path': str(trials_path), 'sha256': sha256(trials_path.read_bytes()).hexdigest()}]}))
    return dict(inventory_dir=inventory, trials_path=trials_path,
                config_path=ROOT / 'configs/exp123/oral_report_0923.yaml', output_dir=tmp_path / 'out')


def test_audit_preserves_feature_specific_identity_and_refuses_overwrite(tmp_path):
    import pandas as pd
    kwargs = _audit_inputs(tmp_path)
    original = kwargs['trials_path'].read_bytes()
    result = audit_reports(**kwargs)
    assert result['n_trials'] == 6
    assert result['n_feature_specific_reports'] == 6
    assert result['legacy_direction_disagreement_reports'] == 4
    assert result['coverage_counts']['missing'] == 2
    assert kwargs['trials_path'].read_bytes() == original
    rows = pd.read_csv(kwargs['output_dir'] / 'trials.csv')
    assert rows.choice.tolist() == [1, 2, 2, 1, 2, 2]
    assert rows.original_scored.tolist() == [False, True, True, False, True, True]
    with pytest.raises(FileExistsError):
        audit_reports(**kwargs)


def test_audit_rejects_changed_source_before_writing(tmp_path):
    kwargs = _audit_inputs(tmp_path)
    with kwargs['trials_path'].open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='hash'):
        audit_reports(**kwargs)
    assert not kwargs['output_dir'].exists()
