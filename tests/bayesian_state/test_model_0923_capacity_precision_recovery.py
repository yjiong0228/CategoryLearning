from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_capacity_precision_recovery import (
    CAPACITY, crossed_cells, selected_estimates,
)
from src.Bayesian_state.workflows.analysis.model_0923_capacity_memory_recovery import disjoint_seeds
from src.Bayesian_state.evaluation.model_0923_h_validation import digest, nll


def points():
    return [dict(id=str(k), hyperparams={CAPACITY:k, 'learning_rate':.3, 'memory':0.}) for k in (4,2)]


def test_crossed_recovery_changes_only_capacity_and_storage():
    original = points()
    before = deepcopy(original)
    cells = crossed_cells(original)
    assert [c['label'] for c in cells] == ['K2_RESET','K2_RETAIN','K4_RESET','K4_RETAIN']
    assert [c['retain_precision'] for c in cells] == [False, True, False, True]
    assert all(c['candidate']['hyperparams']['learning_rate'] == .3 for c in cells)
    assert original == before
    cells[0]['candidate']['hyperparams']['memory'] = .8
    assert original == before


@pytest.mark.parametrize('change', ['capacity','nuisance','duplicate_id','missing'])
def test_recovery_rejects_nonmatched_generating_points(change):
    candidates = points()
    if change == 'capacity': candidates[0]['hyperparams'][CAPACITY] = 8
    elif change == 'nuisance': candidates[0]['hyperparams']['memory'] = .8
    elif change == 'duplicate_id': candidates[0]['id'] = candidates[1]['id']
    else: candidates.pop()
    with pytest.raises(ValueError):
        crossed_cells(candidates)


def test_sixteen_generating_seeds_are_disjoint_and_outcome_independent():
    bank = list(range(64))[::-1]
    chosen = [disjoint_seeds(bank, index, 4) for index in range(4)]
    assert chosen == [list(range(i*4, (i+1)*4)) for i in range(4)]
    assert len(set(seed for cell in chosen for seed in cell)) == 16


def score_table(tmp_path, probabilities, truth='K4_RETAIN'):
    rows = []
    for cell, p in zip(['K2_RESET','K2_RETAIN','K4_RESET','K4_RETAIN'], probabilities):
        file = tmp_path/(cell+'.npz')
        values = np.tile([p-.03,p+.03,p],(4,1))
        np.savez(file, seeds=np.array([10,20,30,40]), observed_probability=values)
        record = file.with_suffix('.json')
        record.write_text(json.dumps(dict(predictions_sha256=digest(file), mean_nll=nll(values))))
        rows.append(dict(case='case_1', true_cell=truth, true_capacity=int(truth[1]),
            true_variant=truth[3:], cell=cell, capacity=int(cell[1]), variant=cell[3:],
            score_file=str(record), mean_nll=nll(values)))
    return pd.DataFrame(rows)


def test_truth_and_two_factors_recover_when_its_sequence_probability_is_highest(tmp_path):
    scores = score_table(tmp_path, [.3,.4,.5,.8])
    result = selected_estimates(scores).iloc[0]
    assert result.selected_cell == 'K4_RETAIN'
    assert result.unique_joint_correct and result.truth_is_minimum
    assert result.capacity_correct and result.mechanism_correct
    assert result.truth_gap == 0 and result.truth_gap_mcse == 0
    assert result.runner_up_gap > 0


def test_tied_truth_is_retained_without_claiming_unique_recovery(tmp_path):
    scores = score_table(tmp_path, [.8,.4,.5,.8])
    result = selected_estimates(scores).iloc[0]
    assert result.selected_cell == 'K2_RESET'
    assert result.tied_minima == 2 and result.truth_is_minimum
    assert not result.unique_joint_correct and not result.selected_joint_correct
    assert result.truth_gap == 0 and result.runner_up_gap == 0


def test_capacity_can_recover_when_storage_mechanism_does_not(tmp_path):
    result = selected_estimates(score_table(tmp_path, [.3,.4,.8,.5])).iloc[0]
    assert result.capacity_correct and not result.mechanism_correct
    assert not result.truth_is_minimum
    assert result.truth_gap > 0


def test_missing_candidate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='every crossed candidate'):
        selected_estimates(score_table(tmp_path, [.3,.4,.5,.8]).iloc[:3])


def test_scoring_seed_mismatch_is_rejected(tmp_path):
    scores = score_table(tmp_path, [.3,.4,.5,.8])
    path = tmp_path/'K2_RESET.npz'
    with np.load(path) as old:
        values = old['observed_probability']
    np.savez(path, seeds=np.array([11,21,31,41]), observed_probability=values)
    record = path.with_suffix('.json')
    record.write_text(json.dumps(dict(predictions_sha256=digest(path), mean_nll=nll(values))))
    with pytest.raises(AssertionError):
        selected_estimates(scores)
