"""Preserve full path ensembles and reject identity drift in training diagnosis."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_1022_belief_training_paths as audit


@pytest.fixture
def generated(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    categories, choices = np.array([1, 2, 1, 2]), np.array([1, 2, 2, 2])
    np.savez_compressed(data / 'training.npz', categories=categories, choices=choices)
    (data / 'prepared.json').write_text(json.dumps(dict(subject=108, condition=1, training_trials=4,
        array_sha256={'training.npz': audit.digest(data / 'training.npz')})))
    policy = dict(block_width=2, temporal_weight=.5)
    point = dict(id='point', hyperparams={'x': 1}, mean_nll=.6)
    paths = np.array([[1, 1, 1, 2], [2, 2, 2, 2], [1, 1, 2, 2],
                      [1, 2, 1, 2], [1, 2, 1, 1], [2, 1, 2, 1]])
    features = audit.metrics.behavior_features(paths, categories, n_categories=2, **policy)
    observed = audit.metrics.behavior_features(choices[None, :], categories, n_categories=2, **policy)[0]
    targets = []
    for i in (0, 3):
        target = tmp_path / f'chunk_{i}'
        np.savez_compressed(target.with_suffix('.npz'), categories=categories, choices=paths[i:i+3],
            features=features[i:i+3], observed_features=observed, seeds=np.arange(i, i+3))
        score = dict(subject=108, condition=1, candidate=point, data_boundary='training_only',
                     trials=4, paths=3, predictions_sha256=audit.digest(target.with_suffix('.npz')))
        target.with_suffix('.json').write_text(json.dumps(score))
        targets.append(str(target))
    case = dict(subject=108, arm='SLOT', candidate=point, folder=str(data), original_nominee=True,
                smoke=False, targets=targets, seeds=list(range(6)))
    return case, policy, audit.metrics.energy_u(features, observed)


@pytest.mark.parametrize('smoke', [False, True])
def test_scores_pool_all_paths_without_opening_future_data(generated, monkeypatch, smoke):
    case, policy, expected = generated
    case['smoke'] = smoke
    original = np.load

    def guarded(path, *args, **kwargs):
        assert 'sealed_test' not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', guarded)
    result = audit.combine(case, policy, {})
    assert result['training_trials'] == 4
    assert result['paths'] == 6
    assert result['training_energy'] == expected['value']
    assert result['training_energy_mcse'] == expected['mc_se']


@pytest.mark.parametrize('field', ['seeds', 'categories', 'features'])
def test_rehashed_generation_still_requires_original_identity(generated, field):
    case, policy, _ = generated
    target = Path(case['targets'][0])
    with np.load(target.with_suffix('.npz')) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays[field].flat[0] += 1
    np.savez_compressed(target.with_suffix('.npz'), **arrays)
    score = audit.read(target.with_suffix('.json'))
    score['predictions_sha256'] = audit.digest(target.with_suffix('.npz'))
    target.with_suffix('.json').write_text(json.dumps(score))
    with pytest.raises((ValueError, AssertionError)):
        audit.combine(case, policy, {})


def test_prefix_replay_uses_seed_identity_and_trial_order(tmp_path):
    reference, prefix = tmp_path / 'full.npz', tmp_path / 'prefix.npz'
    choices = np.array([[1, 2, 1], [2, 1, 2]])
    np.savez_compressed(reference, seeds=[11, 12], choices=choices, categories=[1, 2, 1])
    np.savez_compressed(prefix, seeds=[12, 11], choices=choices[::-1, :2], categories=[1, 2])
    audit.assert_prefix(reference, prefix, audit.digest(reference))
    np.savez_compressed(prefix, seeds=[12, 11], choices=choices[:, :2], categories=[1, 2])
    with pytest.raises(AssertionError):
        audit.assert_prefix(reference, prefix, audit.digest(reference))
