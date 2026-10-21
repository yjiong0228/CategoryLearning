"""Known-score example and immutable-artifact checks for the new comparison."""
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.model_1021_simplification import summarize
from src.Bayesian_state.evaluation.model_0923_h_validation import digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    fit, evaluation = tmp_path / 'fit', tmp_path / 'evaluation'
    settings = dict(subjects=[101], arms=dict(REFERENCE={}, COMPACT={}), autonomous_paths=2,
                    budgets=dict(evaluate=dict(seeds=2)), features=dict(block_width=8))
    nominees, conditional, autonomous = [], [], []
    for arm, value in [('REFERENCE', .8), ('COMPACT', .7)]:
        point = dict(id=arm, hyperparams=dict(example=1))
        probabilities = np.tile([value, 1-value], (2, 16, 1))
        categories = np.ones(16, dtype=int)
        observed = np.full((2, 16), value)
        for stage in ('screen', 'refine', 'selection', 'confirmation'):
            record = dict(subject=101, model=arm, numeric_pools=[dict(num_threads=1)], particles=2,
                          trials=8, seeds=[11, 12], cpu_seconds=.1)
            write(fit / stage / 'subject_101' / arm / 'chunks' / 'one.json', record)
        confirmed = fit / 'confirmation' / 'subject_101' / arm / (arm + '.npz')
        np.savez(confirmed, observed_probability=observed[:, :8], seeds=[11, 12])
        write(confirmed.with_suffix('.json'), dict(predictions_sha256=digest(confirmed)))
        finalist = dict(point, score_file=str(confirmed.with_suffix('.json')))
        nominees.append(dict(subject=101, model=arm, candidate=point, finalists=[finalist], reference_id=arm))
        choices = np.ones((2, 16), dtype=int)
        if arm == 'COMPACT':
            choices[1] = 2
        for role in ('conditional', 'autonomous'):
            file = evaluation / role / 'evaluate' / 'subject_101' / arm / (arm + '.npz')
            file.parent.mkdir(parents=True)
            if role == 'conditional':
                np.savez(file, categories=categories, choices=categories, seeds=[101, 102],
                    probabilities=probabilities, observed_probability=observed, correct_probability=observed)
            else:
                np.savez(file, categories=categories, choices=choices, seeds=[201, 202],
                    probabilities=np.eye(2)[choices-1], present=np.ones((2, 16)), contribution=np.ones((2, 16)),
                    search=np.zeros((2, 16)), target_capacity_dropped=np.zeros((2, 16)), target_exited=np.zeros((2, 16)))
            row = dict(subject=101, model=arm, condition=1, candidate=point, training_trials=8,
                       predictions_sha256=digest(file))
            (conditional if role == 'conditional' else autonomous).append(row)
    write(fit / 'nominees.json', dict(smoke=False, status='frozen_before_test', settings=settings, nominees=nominees))
    write(fit / 'protocol.json', dict(settings=settings))
    write(evaluation / 'protocol.json', dict(nomination_sha256=digest(fit / 'nominees.json')))
    write(evaluation / 'completion.json', dict(status='complete'))
    for role, rows in [('conditional', conditional), ('autonomous', autonomous)]:
        write(evaluation / role / 'evaluation.json', dict(rows=rows))
    return fit, evaluation


def test_known_behavior_and_autonomous_differences(tmp_path):
    fit, evaluation = fixture(tmp_path)
    output = tmp_path / 'summary'
    summarize(fit, evaluation, output)
    contrasts = pd.read_csv(output / 'contrasts.csv')
    np.testing.assert_allclose(contrasts.delta_choice_nll, np.log(.8 / .7), atol=1e-12)
    np.testing.assert_allclose(contrasts.delta_energy, .25, atol=1e-12)
    np.testing.assert_allclose(contrasts.delta_choice_nll_mc_se, 0., atol=1e-12)
    done = json.loads((output / 'completion.json').read_text())
    assert done['within_behavior_working_tolerances'] is False
    assert done['development_sample'] is True and done['oral_and_recovery'] == 'pending separate analysis'


def test_changed_prediction_is_rejected(tmp_path):
    fit, evaluation = fixture(tmp_path)
    file = next((evaluation / 'conditional').glob('evaluate/*/*/*.npz'))
    file.write_bytes(b'changed scientific output')
    with pytest.raises(ValueError, match='changed'):
        summarize(fit, evaluation, tmp_path / 'summary')
    assert not (tmp_path / 'summary').exists()
