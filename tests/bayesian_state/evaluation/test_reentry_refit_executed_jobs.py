"""The autonomous launcher adds two verified audit fields to evaluation plans."""
from copy import deepcopy
import json

import pytest

from src.Bayesian_state.evaluation import model_0923_reentry_refit as comparison


def fixture(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(comparison.refit, 'ROOT', tmp_path)
    folder=tmp_path/'prepared';folder.mkdir()
    (folder/'prepared.json').write_text(json.dumps({'subject':208}))
    audit=tmp_path/'diagnosis/audit/subject_208/audit.json';audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({'subject':208,'target':42}))
    target=tmp_path/'predictions'
    sidecar=target.with_suffix('.precision.json')
    sidecar.write_text(json.dumps({'retained_precision':True,'structural_intervention':comparison.refit.INTERVENTION}))
    planned=dict(phase=phase,folder=str(folder),target=str(target),seeds=[7],particles=512)
    references={'original_fit':'sha'}
    job=dict(planned,retain_precision=True,structural_intervention=comparison.refit.INTERVENTION,
             retention_reference_inputs=references)
    if phase=='autonomous':job.update(audit=str(audit),audit_sha256=comparison.digest(audit))
    return planned,job,dict(audit='diagnosis'),references,audit,sidecar


@pytest.mark.parametrize('phase',['autonomous','evaluate'])
def test_actual_launcher_fields_are_verified_and_bound(tmp_path,monkeypatch,phase):
    planned,job,settings,references,audit,sidecar=fixture(tmp_path,monkeypatch,phase)
    inputs=comparison.executed_retention_inputs(planned,job,settings,references)
    assert inputs[str(sidecar)]==comparison.digest(sidecar)
    assert (str(audit) in inputs)==(phase=='autonomous')


@pytest.mark.parametrize('change',['audit_path','audit_hash','audit_contents','particles','seed','extra','disabled'])
def test_execution_verifier_rejects_unexplained_changes(tmp_path,monkeypatch,change):
    planned,job,settings,references,audit,sidecar=fixture(tmp_path,monkeypatch,'autonomous')
    job=deepcopy(job)
    if change=='audit_path':job['audit']='wrong'
    elif change=='audit_hash':job['audit_sha256']='wrong'
    elif change=='audit_contents':audit.write_text(json.dumps({'subject':208,'target':43}))
    elif change=='particles':job['particles']=256
    elif change=='seed':job['seeds']=[8]
    elif change=='extra':job['unexplained']=True
    elif change=='disabled':sidecar.write_text(json.dumps({'retained_precision':False,'structural_intervention':comparison.refit.INTERVENTION}))
    with pytest.raises(ValueError):
        comparison.executed_retention_inputs(planned,job,settings,references)
