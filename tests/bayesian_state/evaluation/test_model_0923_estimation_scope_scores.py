"""Scope scores preserve common random units and deterministic point identity."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_estimation_scope import (
    available_comparisons, digest, matched_choice_scores, paired_difference, summarize, validate_generation,
)
from src.Bayesian_state.workflows.analysis.model_0923_estimation_scope import score_scopes


def point():
    return dict(metrics={name: 1. for name in ('choice_nll', 'energy', 'crps')},
                jackknife={name: np.array([1., 2., 3.]) for name in ('choice_nll', 'energy', 'crps')},
                filter_seeds=np.array([1, 2, 3]), path_seeds=np.array([4, 5, 6]))


def test_two_scope_labels_for_one_point_have_zero_paired_difference():
    row = point()
    assert all(value == 0 for value in paired_difference(row, deepcopy(row)).values())


def test_paired_uncertainty_uses_differences_not_independent_errors():
    reference, target = point(), point()
    for name in target['metrics']:
        target['metrics'][name] += .5
        target['jackknife'][name] += .5
    result = paired_difference(target, reference)
    assert result['delta_energy'] == .5
    assert result['delta_energy_mc_se'] == 0
    target['path_seeds'] = target['path_seeds'][::-1]
    with pytest.raises(AssertionError):paired_difference(target, reference)


def test_generation_checks_task_feedback_and_absent_rule_contribution():
    arrays = dict(seeds=np.array([4, 5]), categories=np.array([1, 2]), probabilities=np.full((2, 2, 4), .25),
                  choices=np.array([[1, 2], [2, 1]]), feedback=np.array([[1., 1.], [0., 0.]]),
                  correct_probability=np.full((2, 2), .25), present=np.zeros((2, 2)),
                  contribution=np.zeros((2, 2)), beta=np.zeros((2, 2)), active_count=np.ones((2, 2)),
                  search=np.zeros((2, 2)), target_exited=np.zeros((2, 2)), target_capacity_dropped=np.zeros((2, 2)))
    record = dict(condition=2, candidate=dict(hyperparams={'engine.modules.hypo_transitions_mod.kwargs.max_capacity': 2}))
    validate_generation(arrays, record)
    arrays['contribution'][0, 0] = .2
    with pytest.raises(ValueError, match='presence'):validate_generation(arrays, record)
    arrays['contribution'][0, 0] = 0
    arrays['feedback'][0, 0] = 0
    with pytest.raises(AssertionError):validate_generation(arrays, record)


@pytest.fixture
def shared_bank(tmp_path):
    confirmed = {}
    observed = .2 + np.arange(3)[:, None]*.1 + np.arange(4)[None, :]*.02
    for model, factor in [('POOL', 1.), ('SEARCH', 1.1)]:
        file = tmp_path / (model + '.json'); array = file.with_suffix('.npz')
        probability = factor*observed
        np.savez(array, probabilities=np.stack([probability, 1-probability], axis=-1),
                 observed_probability=probability, seeds=[3, 5, 7], choices=[1, 1, 1, 1], categories=[1, 2, 1, 2])
        candidate = dict(id='same_point', hyperparams={'gamma': .9})
        record = dict(subject=308, model=model, candidate=candidate, particles=512,
                      seeds=[3, 5, 7], training_trials=2, predictions_sha256=digest(array))
        file.write_text(json.dumps(record))
        confirmed[308, model, 'same_point'] = dict(candidate, particles=512, score_file=str(file),
                                                 **score_scopes(probability, 2))
    return confirmed


def test_matched_mechanisms_use_shared_parameters_and_paired_filters(shared_bank):
    scores, contrasts, hashes = matched_choice_scores(shared_bank)
    assert len(scores) == 6 and len(contrasts) == 3 and len(hashes) == 4
    for row in contrasts:
        assert row['delta_choice_nll'] == pytest.approx(-np.log(1.1))
        assert row['delta_choice_nll_mc_se'] == pytest.approx(0., abs=1e-14)


@pytest.mark.parametrize('change', ['parameters', 'seed_order', 'missing_point'])
def test_matched_mechanisms_reject_misaligned_comparisons(shared_bank, change):
    from pathlib import Path
    point = shared_bank[308, 'SEARCH', 'same_point']
    file = Path(point['score_file']); record = json.loads(file.read_text())
    if change == 'parameters':
        record['candidate']['hyperparams']['gamma'] = .8
        point['hyperparams']['gamma'] = .8
    elif change == 'seed_order':
        array = file.with_suffix('.npz')
        with np.load(array) as saved: data = {key: saved[key] for key in saved.files}
        data['seeds'] = data['seeds'][::-1]; record['seeds'] = data['seeds'].tolist()
        np.savez(array, **data); record['predictions_sha256'] = digest(array)
    else:
        shared_bank.pop((308, 'SEARCH', 'same_point'))
    file.write_text(json.dumps(record))
    with pytest.raises((ValueError, AssertionError)):matched_choice_scores(shared_bank)


def test_confirmed_scope_contrasts_remain_complete_and_ordered():
    bindings = [dict(subject=308,model=model,objective=scope) for model in ('POOL','SEARCH')
                for scope in ('original','prefix','full')]
    rows = available_comparisons(bindings,308)
    assert len(rows) == 9
    assert rows[:3] == [('POOL','prefix','POOL','original'),('POOL','full','POOL','original'),
                       ('POOL','full','POOL','prefix')]
    assert rows[-1] == ('SEARCH','full','POOL','full')


def test_full_only_probe_keeps_all_four_available_contrasts():
    bindings = [dict(subject=308,model=model,objective=scope) for model in ('POOL','SEARCH')
                for scope in ('original','full')]
    assert available_comparisons(bindings,308) == [
        ('POOL','full','POOL','original'),('SEARCH','full','SEARCH','original'),
        ('SEARCH','original','POOL','original'),('SEARCH','full','POOL','full')]


@pytest.mark.parametrize('stage', [True,False])
def test_provisional_and_confirmed_outputs_require_their_explicit_mode(tmp_path, stage):
    generation = tmp_path/'generation';generation.mkdir()
    (generation/'generation.json').write_text(json.dumps(dict(status='complete')))
    (generation/'protocol.json').write_text(json.dumps(dict(selection_phase='confirmation' if stage else 'selection')))
    with pytest.raises(ValueError, match='selection phase'):
        summarize(tmp_path/'scope',generation,tmp_path/'summary',selection_stage=stage)
