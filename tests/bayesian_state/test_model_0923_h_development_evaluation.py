from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_development_evaluation import reuse_record
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import digest


def fixture(tmp_path):
    reference = tmp_path/'reference'; output = tmp_path/'output'
    file = reference/'evaluate/subject_101/B/original.npz'; file.parent.mkdir(parents=True)
    np.savez(file, probabilities=np.ones((2,3,2)) / 2)
    candidate = dict(id='original',hyperparams={'parameter':1.})
    row = dict(subject=101,model='B',candidate=candidate,predictions_sha256=digest(file))
    return reference, output, file, dict(subject=101,model='B',candidate=candidate), row


def test_unchanged_parameters_reuse_verified_arrays_with_provenance(tmp_path):
    reference, output, source, nominee, old = fixture(tmp_path)
    row = reuse_record(nominee, old, reference, output)
    target = output/'evaluate/subject_101/B/original.npz'
    assert target.is_symlink() and target.resolve() == source.resolve()
    assert row['reused'] and row['reused_record'] == old


def test_changed_parameter_point_cannot_reuse_saved_predictions(tmp_path):
    reference, output, _, nominee, old = fixture(tmp_path)
    nominee = dict(nominee, candidate=dict(id='changed',hyperparams={'parameter':2.}))
    with pytest.raises(ValueError, match='Changed parameters'):
        reuse_record(nominee, old, reference, output)


def test_changed_reference_is_rejected(tmp_path):
    reference, output, source, nominee, old = fixture(tmp_path)
    source.write_bytes(b'corrupted reference')
    with pytest.raises(ValueError, match='predictions changed'):
        reuse_record(nominee, old, reference, output)
