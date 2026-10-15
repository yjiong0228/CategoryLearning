"""Independent scoring arithmetic, equal budgets and training-only generation."""
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.Bayesian_state.metrics.autonomous_fit import behavior_features, energy_u
from src.Bayesian_state.workflows.analysis import model_0923_h_objective_comparison as objective


def test_features_preserve_time_and_category_specific_errors():
    labels = np.array([1, 1, 2, 2])
    choices = np.array([[1, 2, 2, 1], [2, 2, 1, 1]])
    actual = behavior_features(choices, labels, n_categories=2, block_width=2)
    expected = np.array([[.5*.5, .5*.5, .5/np.sqrt(8), .5/np.sqrt(8), .5/np.sqrt(8), .5/np.sqrt(8)],
                         [0, 0, 0, 1/np.sqrt(8), 1/np.sqrt(8), 0]])
    np.testing.assert_allclose(actual, expected)


def test_missing_category_mask_uses_schedule_only():
    result = behavior_features(np.array([[1, 2, 1]]), np.array([1, 1, 1]), n_categories=2, block_width=2)
    assert result.shape == (1, 4)
    assert np.isfinite(result).all()


@pytest.mark.parametrize('choices', [np.array([[0, 1]]), np.array([[3, 1]]), np.array([[1.5, 1]]), np.array([[np.nan, 1]])])
def test_invalid_response_labels_rejected(choices):
    with pytest.raises(ValueError):
        behavior_features(choices, np.array([1, 2]), n_categories=2)


def test_energy_and_jackknife_match_explicit_pair_enumeration():
    samples = np.array([[0., 0.], [1., 0.], [0., 2.], [2., 1.]])
    target = np.array([.4, .6])
    def direct(x):
        distances = [np.sqrt(np.square(a-b).sum()) for a, b in combinations(x, 2)]
        return np.mean([np.sqrt(np.square(a-target).sum()) for a in x])-.5*np.mean(distances)
    actual = energy_u(samples, target)
    deleted = np.array([direct(np.delete(samples, i, axis=0)) for i in range(4)])
    assert actual['value'] == pytest.approx(direct(samples))
    assert actual['mc_se'] == pytest.approx(np.sqrt(3/4*np.square(deleted-deleted.mean()).sum()))


def test_energy_avoids_finite_ensemble_self_pair_penalty():
    assert energy_u(np.array([[0.], [0.], [1.]]), np.array([0.]))['value'] == pytest.approx(0)
    assert energy_u(np.full((4, 1), 3.), np.array([1.]))['value'] == pytest.approx(2.)


@pytest.mark.parametrize('samples', [np.zeros((2, 2)), np.array([[0.], [np.nan], [1.]])])
def test_invalid_energy_samples_rejected(samples):
    with pytest.raises(ValueError):
        energy_u(samples, np.zeros(samples.shape[1]))


def test_protocol_requires_same_mechanism_and_equal_state_updates():
    settings = yaml.safe_load(Path('configs/shared/model0923_h_objective_comparison.yaml').read_text())
    objective.validate_settings(settings)
    settings['budgets']['screen']['paths'] += 1
    with pytest.raises(ValueError, match='budgets'):
        objective.validate_settings(settings)


def test_shared_finalist_union_does_not_retain_other_objective_scores():
    rows = {}
    for arm, ids in [('NLL', ['a', 'b']), ('GEN', ['b', 'c'])]:
        rows[(1, arm)] = [dict(id=i, hyperparams={'p': i}, sources=['test'], origin={}, objective_score=0.) for i in ids]
    result = objective.shared_union(rows, [1], 2)
    assert result[(1, 'NLL')] == result[(1, 'GEN')]
    assert len(result[(1, 'GEN')]) == 3
    assert all('objective_score' not in r for r in result[(1, 'GEN')])


def test_generation_fit_does_not_open_future_or_transmit_observed_responses(tmp_path, monkeypatch):
    v = objective.validation(); folder = tmp_path/'subject'; folder.mkdir()
    np.savez(folder/'training.npz', stimulus=np.zeros((4, 4)), categories=[1, 2, 1, 2], choices=[2, 1, 2, 1])
    (folder/'sealed_test.npz').write_bytes(b'unreadable future sentinel')
    v.write(folder/'engine.json', dict(modules=dict(hypo_transitions_mod=dict(kwargs={}))))
    v.write(folder/'prepared.json', dict(subject=101, condition=1,
        engine_sha256=v.digest(folder/'engine.json'), array_sha256={'training.npz': v.digest(folder/'training.npz')},
        paths={'processed_dir': str(tmp_path)}))
    calls = []
    def generate(**kwargs):
        assert 'choices' not in kwargs and 'feedback' not in kwargs
        calls.append(len(kwargs['stimulus']))
        return SimpleNamespace(trajectory=SimpleNamespace(choices=np.array([1, 2, 1, 2])))
    monkeypatch.setattr('src.Bayesian_state.simulation.autonomous.run_autonomous_category_learning', generate)
    job = dict(folder=str(folder), mechanism={'kind': 'B'}, candidate={'hyperparams': {}},
        smoke=False, features={'block_width': 2, 'temporal_weight': .5}, seeds=[1, 2, 3, 4],
        model='GEN', case_id=101, target=str(tmp_path/'score'))
    objective.generation_score(job, v)
    assert calls == [4]*4
    record = v.read(tmp_path/'score.json')
    assert record['data_boundary'] == 'training_only'
    assert record['objective_score'] > 0
    assert record['state_updates'] == 16


def test_recovery_fixture_changes_behavior_and_preserves_physical_inputs(tmp_path):
    from src.Bayesian_state.workflows.analysis.model_0923_h_objective_recovery import fixture
    v = objective.validation(); original = tmp_path/'original'; original.mkdir()
    v.write(original/'engine.json', {'example': 1})
    v.write(original/'prepared.json', {'subject': 101, 'condition': 1, 'paths': {'example': 'preserved'},
                                     'training_trials': 99, 'test_trials': 99})
    arrays = dict(stimulus=np.arange(16).reshape(4,4), categories=np.array([1,2,1,2]),
                  choices=np.array([2,2,1,1]), feedback=np.array([0,1,1,0]))
    meta = fixture(original, tmp_path/'synthetic', arrays)
    assert meta['subject'] == 101 and meta['training_trials'] == 4 and meta['test_trials'] == 0
    assert 'hyperparams' not in meta and 'synthetic_truth' not in meta
    with np.load(tmp_path/'synthetic/training.npz') as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(saved[key], value)
    with np.load(tmp_path/'synthetic/sealed_test.npz') as saved:
        assert all(len(saved[key]) == 0 for key in saved.files)


def test_choice_aggregation_accepts_a_named_objective_baseline():
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_h_validation import aggregate_choices
    frame = pd.DataFrame([dict(subject=1,model='NLL',trials=10,choice_nll=.3),
                          dict(subject=1,model='GEN',trials=10,choice_nll=.5)])
    contrasts = pd.DataFrame([dict(subject=1,model='GEN',choice_nll_mc_se=.01)])
    result = aggregate_choices(frame,contrasts,['NLL','GEN'],'NLL').set_index('model')
    assert result.loc['NLL','delta_total_negative_log_score']==0
    assert result.loc['GEN','delta_total_negative_log_score']==pytest.approx(2.)
    assert result.loc['GEN','delta_total_mc_se']==pytest.approx(.1)


def test_whole_feature_path_jackknife_matches_training_score_uncertainty():
    from src.Bayesian_state.evaluation.model_0923_h_objective_comparison import feature_jackknife
    from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se
    samples = np.array([[.1,.2], [.4,.6], [.2,.8], [.9,.3], [.7,.5]])
    target = np.array([.3,.4])
    assert jackknife_se(feature_jackknife(samples,target)) == pytest.approx(energy_u(samples,target)['mc_se'])
