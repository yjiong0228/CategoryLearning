"""Text provenance, incomplete syntax and conjunction safeguards."""
from dataclasses import asdict
import json

import pandas as pd
import pytest

from src.Bayesian_state.evaluation.oral.discourse_encoding import encode_discourse
from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report

FEATURES = ('neck', 'tail', 'head', 'leg')


def test_concession_keeps_offsets_degrees_and_unassigned_roles():
    text = ' 虽然腿比较长，但是脖子很短。'
    r = encode_discourse(text, FEATURES)
    assert r.original.status == 'needs_review'
    assert r.content_complete and r.role_status == 'unassigned'
    assert r.content.tokens == encode_report('腿长，脖子短', FEATURES).tokens
    assert [m.text for m in r.contrast_markers] == ['但是']
    assert r.segments[0].concession.text == '虽然'
    for s in r.segments:
        for p in (s.span, s.content, *s.qualifiers):
            assert text[p.start:p.end] == p.text
    assert [q.text for s in r.segments for q in s.qualifiers] == ['比较', '很']
    assert 'intensity_coarsened' in r.content.flags


@pytest.mark.parametrize('text', ['头长，腿短。', '头长于一半，腿短于一半',
                                  '躯干短于脖子、长于尾巴', '', '头相对长一些'])
def test_unchanged_reports_reproduce_original_encoding(text):
    r = encode_discourse(text, FEATURES)
    expected = encode_report(text, FEATURES, half_reference='range_midpoint')
    assert asdict(r.original) == asdict(expected) == asdict(r.content)
    assert r.content_text == text and not r.contrast_markers


@pytest.mark.parametrize('text', ['头长但是头短', '头比腿长但腿比头长',
                                  '头比躯干长但是头比躯干短'])
def test_individually_valid_clauses_cannot_hide_contradictions(text):
    r = encode_discourse(text, FEATURES)
    assert all(s.encoding.status == 'coded' for s in r.segments)
    assert not r.content_complete and r.content.status == 'needs_review'
    assert any(f.startswith('contradictory') for f in r.content.flags)


@pytest.mark.parametrize('text', ['腿长但是脖子很短，头相对长一些。',
                                  '头和尾巴很长，脖子也挺长，但相对短一些，腿很短。',
                                  '但是头长', '头长但是'])
def test_relative_ellipsis_and_empty_contrast_segments_remain_incomplete(text):
    r = encode_discourse(text, FEATURES)
    assert not r.content_complete


@pytest.mark.parametrize('text', ['不但头长，腿也长', '不但是头长，腿也长', '头长不过腿'])
def test_correlative_and_ambiguous_connectives_are_preserved(text):
    r = encode_discourse(text, FEATURES)
    assert not r.contrast_markers and r.unhandled_connectives
    assert r.content_text == text and not r.content_complete


def test_multiple_markers_keep_order_and_selected_feature_mapping():
    text = '头长可是腿短然而脖子长'
    r = encode_discourse(text, FEATURES)
    assert r.content_complete
    assert [m.text for m in r.contrast_markers] == ['可是', '然而']
    assert r.content.tokens == encode_report('头长，腿短，脖子长', FEATURES).tokens
    reordered = encode_discourse(text, ('head', 'neck', 'leg', 'tail'))
    assert r.content.tokens != reordered.content.tokens


def test_nontext_input_rejected():
    with pytest.raises(TypeError): encode_discourse(None, FEATURES)


def test_relative_reading_spans_clauses_and_keeps_explicit_half_reference():
    from src.Bayesian_state.workflows.analysis.model_0923_oral_discourse_encoding import encode_text
    r = encode_text(('腿长但是脖子短', FEATURES))
    assert r['relative_reason'] == 'eligible'
    assert r['relative_scope'] == 'whole_content_conjunction'
    assert r['relative_tokens'] == encode_report('腿比脖子长', FEATURES).tokens
    explicit = encode_text(('腿长于一半但是脖子短于一半', FEATURES))
    assert explicit['content_complete']
    assert explicit['relative_reason'] == 'explicit_reference_or_ordering'
    assert not explicit['relative_tokens']


def test_workflow_reads_text_columns_preserves_identity_and_refuses_overwrite(tmp_path):
    from src.Bayesian_state.workflows.analysis import model_0923_oral_discourse_encoding as workflow
    audit = tmp_path/'audit'; audit.mkdir()
    rows = []
    for trial, text in enumerate(['头长', '腿长但是脖子短', ''], 1):
        old = encode_report(text, FEATURES, half_reference='range_midpoint')
        rows.append(dict(iSub=107, iSession=1, iTrial=trial, ordinal_trial=trial,
            text=text, status=old.status, scope='complete' if old.status == 'coded' else 'parsed_fragment',
            tokens=json.dumps(old.tokens), flags=json.dumps(old.flags), unparsed=json.dumps(old.unparsed),
            **dict(zip(workflow.FEATURES, FEATURES)), choice='must_not_read', model_score='must_not_read'))
    pd.DataFrame(rows).to_csv(audit/'trials.csv', index=False)
    (audit/'manifest.json').write_text(json.dumps(dict(status='complete',
        config=dict(sample_role='development', subjects=[107]),
        artifacts={'trials.csv': workflow.digest(audit/'trials.csv')})))
    output = tmp_path/'output'
    workflow.run(audit, output, 1)
    manifest = json.loads((output/'manifest.json').read_text())
    records = [json.loads(line) for line in (output/'reports.jsonl').read_text().splitlines()]
    assert manifest['status'] == 'complete' and not manifest['responses_read']
    assert 'choice' not in manifest['input_columns'] and 'model_score' not in manifest['input_columns']
    assert [r['iTrial'] for r in records] == [1, 2, 3]
    assert [r['content_complete'] for r in records] == [True, True, False]
    assert not any('must_not_read' in json.dumps(r) for r in records)
    with pytest.raises(FileExistsError): workflow.run(audit, output, 1)
