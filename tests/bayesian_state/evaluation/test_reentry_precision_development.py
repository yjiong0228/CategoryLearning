"""Require scientific, frozen references across both saved SEARCH layouts."""
import json
from types import SimpleNamespace

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_recalled_evidence_study import digest, read
from src.Bayesian_state.workflows.analysis.model_0923_reentry_precision_development import source_provenance


def write(path, value):
    path.write_text(json.dumps(value))


def test_old_evaluation_without_smoke_field_requires_frozen_scientific_fit(tmp_path):
    fit=tmp_path/'fit';fit.mkdir();frozen=fit/'nominees.json'
    write(frozen,dict(status='frozen_before_test',smoke=False))
    write(tmp_path/'completion.json',dict(status='complete'))
    write(tmp_path/'protocol.json',dict(fit=str(fit),nomination_sha256=digest(frozen)))
    v=SimpleNamespace(read=read,digest=digest)
    role,inputs=source_provenance(tmp_path,v)
    assert role=='joint_SEARCH_refit' and str(frozen) in inputs
    write(frozen,dict(status='frozen_before_test',smoke=True))
    with pytest.raises(ValueError,match='frozen scientific'):source_provenance(tmp_path,v)


def test_fixed_point_study_requires_explicit_scientific_status(tmp_path):
    write(tmp_path/'completion.json',dict(status='complete',smoke=False))
    write(tmp_path/'protocol.json',dict(smoke=False))
    v=SimpleNamespace(read=read,digest=digest)
    assert source_provenance(tmp_path,v)[0]=='fixed_prior_POOL_parameters_applied_to_SEARCH'
    write(tmp_path/'protocol.json',{})
    with pytest.raises(ValueError,match='scientific fixed'):source_provenance(tmp_path,v)


def test_unfinished_scientific_reference_is_rejected(tmp_path):
    write(tmp_path/'completion.json',dict(status='running',smoke=False))
    write(tmp_path/'protocol.json',dict(smoke=False))
    with pytest.raises(ValueError,match='Incomplete'):
        source_provenance(tmp_path,SimpleNamespace(read=read,digest=digest))
