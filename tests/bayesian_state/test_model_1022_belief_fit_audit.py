"""Audit participant identities and seed banks independently of saved scores."""
import json

import numpy as np
import pytest

from src.Bayesian_state.evaluation import model_1022_belief_fit_audit as audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fitted(tmp_path):
    study = tmp_path / 'fit'
    data = tmp_path / 'data' / 'subject_108'
    data.mkdir(parents=True)
    np.savez_compressed(data / 'training.npz', choices=[1, 2, 1, 2])
    put(data / 'prepared.json', dict(subject=108, condition=1, training_trials=4,
        array_sha256={'training.npz': audit.digest(data / 'training.npz')}))
    settings = dict(subjects=[108], data=str(data.parent), protocol='audit_test', seed=3,
                    budgets={'screen': dict(particles=8, seeds=2)})
    policy = dict(variants=['SUPPORT', 'SLOT'], initial_candidates=1)
    put(study / 'protocol.json', dict(smoke=False, settings=settings, policy=policy, source_sha256={}))
    point = dict(id='point', hyperparams={'x': 1})
    seeds = [audit.stable_seed(dict(protocol='audit_test', base=3, stage='screen', subject=108, repeat=i))
             for i in range(2)]
    groups = []
    for arm in policy['variants']:
        target = study / arm / 'point.npz'
        target.parent.mkdir(parents=True)
        p = np.array([[.6, .7, .6, .7], [.5, .8, .5, .8]])
        np.savez_compressed(target, observed_probability=p, seeds=seeds)
        score = dict(subject=108, condition=1, particles=8, trials=4, candidate=point,
                     mean_nll=audit.nll(p), predictions_sha256=audit.digest(target))
        put(target.with_suffix('.json'), score)
        groups.append(dict(subject=108, model=arm, rows=[dict(point, mean_nll=score['mean_nll'],
                                                          score_file=str(target.with_suffix('.json')))]))
    put(study / 'starting_banks.json', dict(inputs_sha256={},
        banks=[dict(subject=108, arm=a, candidates=[point]) for a in policy['variants']]))
    put(study / 'screen_completion.json', dict(status='complete', points=2, groups=groups))
    return study


def test_complete_stage_scores_are_recomputed(fitted, tmp_path):
    output = tmp_path / 'audit'
    audit.run(fitted, output, ['screen'])
    done = audit.read(output / 'completion.json')
    assert done['candidate_evaluations'] == 2
    assert done['stages'] == ['screen'] and not done['nominations_changed']


@pytest.mark.parametrize('change', ['subject', 'seed'])
def test_identity_error_is_not_hidden_by_recomputed_hashes(fitted, tmp_path, change):
    target = fitted / 'SLOT' / 'point.npz'
    score_file = target.with_suffix('.json')
    score = audit.read(score_file)
    if change == 'subject':
        score['subject'] = 999
    else:
        with np.load(target) as z:
            p, seeds = z['observed_probability'], z['seeds']
        seeds[0] += 1
        np.savez_compressed(target, observed_probability=p, seeds=seeds)
        score['predictions_sha256'] = audit.digest(target)
    put(score_file, score)
    with pytest.raises((ValueError, AssertionError)):
        audit.run(fitted, tmp_path / 'invalid', ['screen'])
