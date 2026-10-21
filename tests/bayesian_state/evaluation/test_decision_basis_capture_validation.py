"""Original-filter captures retain a checkable identity distinct from replay."""
import json

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_decision_basis_alignment import digest, validate_basis_record


def fixture(tmp_path):
    prediction = np.array([[.8, .2], [.3, .7]])
    choices, seeds = np.array([1, 2]), [9]
    score, captured, job = tmp_path / 'score.npz', tmp_path / 'state.npz', tmp_path / 'job.json'
    np.savez(score, probabilities=prediction[None], choices=choices, seeds=seeds)
    np.savez(captured, prediction=prediction, choices=choices, seeds=seeds)
    point = dict(id='example', hyperparams=dict(x=1))
    job.write_text(json.dumps(dict(target=str(score.with_suffix('')), particles=2, seeds=seeds, candidate=point)))
    record = dict(array_sha256=digest(captured), exact_behavior_replay=False, exact_original_filter_capture=True,
        additional_filters=0, timing='after_choice_before_current_feedback', score_file=str(score),
        score_sha256=digest(score), original_job=str(job), original_job_sha256=digest(job),
        particles=2, seeds=seeds, candidate=point)
    return captured, record


def test_capture_accepted_without_claiming_replay(tmp_path):
    target, record = fixture(tmp_path)
    validate_basis_record(target, record)
    assert record['exact_behavior_replay'] is False


@pytest.mark.parametrize('field,value', [('timing', 'after_feedback'), ('additional_filters', 1), ('particles', 3)])
def test_capture_identity_changes_rejected(tmp_path, field, value):
    target, record = fixture(tmp_path)
    record[field] = value
    with pytest.raises(ValueError):
        validate_basis_record(target, record)


def test_even_rehashed_states_must_reproduce_bound_prediction(tmp_path):
    target, record = fixture(tmp_path)
    np.savez(target, prediction=[[.2, .8], [.3, .7]], choices=[1, 2], seeds=[9])
    record['array_sha256'] = digest(target)
    with pytest.raises(AssertionError):
        validate_basis_record(target, record)
