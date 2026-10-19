import json

import pytest

from src.Bayesian_state.evaluation.model_0923_h_validation import digest
from src.Bayesian_state.workflows.analysis.model_0923_cohort_execution_oral_followthrough import verify_report


def completed_report(tmp_path):
    artifact = tmp_path / 'scores.csv'
    artifact.write_text('support\n0.5\n')
    record = dict(status='complete', artifacts={'scores.csv': digest(artifact)})
    (tmp_path / 'manifest.json').write_text(json.dumps(record))
    return record


def test_accepts_actual_manifest_without_unwritten_completion_file(tmp_path):
    record = completed_report(tmp_path)
    assert not (tmp_path / 'completion.json').exists()
    assert verify_report(tmp_path) == record


def test_rejects_changed_completed_scores(tmp_path):
    completed_report(tmp_path)
    (tmp_path / 'scores.csv').write_text('support\n0.8\n')
    with pytest.raises(ValueError, match='artifact changed'):
        verify_report(tmp_path)


def test_rejects_incomplete_report_with_marker_name(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(status='running', artifacts={})))
    with pytest.raises(ValueError, match='complete oral manifest'):
        verify_report(tmp_path)
