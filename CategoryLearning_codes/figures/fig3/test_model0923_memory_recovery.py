"""Numerical guards for the paired recovery-score figure."""
import json
from itertools import product

import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model0923_memory_recovery import (
    DECAY, MEMORY, digest, load_table, score_contrasts,
)
from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se


def test_pooled_probability_and_whole_seed_pairing():
    first = np.array([[.2, .8], [.4, .6]])
    second = np.array([[.4, .9], [.2, .7]])
    values, leave_one = score_contrasts([first, second])
    np.testing.assert_allclose(values, [-np.log([.3, .7]).mean(), -np.log([.3, .8]).mean()])
    expected = np.array([[-np.log(p[1]).mean(), -np.log(p[0]).mean()] for p in [first, second]])
    np.testing.assert_allclose(leave_one, expected)
    delta = expected[1] - expected[0]
    assert jackknife_se(leave_one[1]-leave_one[0]) == pytest.approx(abs(delta[1]-delta[0])/2)


def test_identical_candidates_have_zero_paired_error_despite_filter_variation():
    probability = np.array([[.2, .8], [.4, .6], [.3, .9]])
    scores, jack = score_contrasts([probability, probability.copy()])
    assert scores[0] == scores[1]
    assert jackknife_se(jack[0]) > 0
    assert jackknife_se(jack[1]-jack[0]) == 0


@pytest.mark.parametrize('arrays', [[], [np.ones(3)], [np.ones((1, 3))],
    [np.ones((2, 0))], [np.ones((2, 3)), np.ones((2, 4))],
    [np.array([[np.nan], [.5]])], [np.array([[1.01], [.5]])], [np.array([[-.1], [.5]])]])
def test_invalid_probability_banks_are_rejected(arrays):
    with pytest.raises(ValueError):
        score_contrasts(arrays)


def test_zero_probability_uses_existing_finite_nll_convention():
    scores, jack = score_contrasts([np.zeros((2, 3))])
    assert scores[0] == pytest.approx(-np.log(np.finfo(float).tiny))
    assert np.isfinite(jack).all()


@pytest.fixture
def recovery(tmp_path):
    """Analytic four-way tie with two paired seeds, isolated from research data."""
    case = 'S207_T00_R00'; folder = tmp_path / 'scores' / case; folder.mkdir(parents=True)
    protocol = dict(smoke=True, scope='two_memory_grid_recovery_with_all_other_parameters_known',
                    recovery=dict(memory_values=[0., .8], decay_values=[.8, 1.], seeds=2))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    truth = dict(label='S207_T00', subject=207, seeds=[9], true_memory=0., true_decay=.8,
                 hyperparams={MEMORY: 0., DECAY: .8, 'known_other_parameter': .2})
    (tmp_path/'truths.json').write_text(json.dumps(dict(rows=[truth])))
    p = np.array([[.2, .6, .8], [.4, .8, .6]])
    score = -np.log([.3, .7, .7]).mean()
    se = abs(-np.log(p[0]).mean()+np.log(p[1]).mean())/2
    rows = []
    for index, (gamma, delta) in enumerate(product([0., .8], [.8, 1.])):
        path = folder/f'p{index}.json'; arrays = path.with_suffix('.npz')
        np.savez(arrays, seeds=[1, 2], observed_probability=p)
        parameters = dict(truth['hyperparams'], **{MEMORY: gamma, DECAY: delta})
        record = dict(candidate=dict(id=f'p{index}', hyperparams=parameters), model='SEARCH',
                      subject=207, seeds=[1, 2], trials=3, mean_nll=score, predictions_sha256=digest(arrays))
        path.write_text(json.dumps(record))
        rows.append(dict(case=case, subject=207, repeat=0, true_memory=0., true_decay=.8,
                         memory=gamma, decay=delta, score_file=str(path), candidate_id=f'p{index}',
                         trials=3, mean_nll=score, mean_nll_mcse=se))
    pd.DataFrame(rows).to_csv(tmp_path/'scores.csv', index=False)
    pd.DataFrame([dict(case=case, memory=0., decay=.8)]).to_csv(tmp_path/'recovered_memory.csv', index=False)
    completion = dict(status='complete', smoke=True, nuisance_parameters_known=True, sequences=1,
                      grid_points=4, seed_jobs=8, protocol_sha256=digest(tmp_path/'protocol.json'))
    (tmp_path/'completion.json').write_text(json.dumps(completion))
    return tmp_path


def test_tied_selection_is_not_unique_recovery(recovery):
    table, meta = load_table(recovery, allow_smoke=True)
    assert meta['smoke']
    assert table.tied_minima.eq(4).all()
    assert table.truth_is_minimum.all() and not table.unique_correct.any()
    assert table.selected.sum() == table.is_truth.sum() == 1
    assert table.delta_to_truth.eq(0).all() and table.delta_to_truth_mcse.eq(0).all()


def test_smoke_requires_explicit_flag(recovery):
    with pytest.raises(ValueError, match='smoke requires'):
        load_table(recovery)


def test_missing_grid_point_is_rejected(recovery):
    table = pd.read_csv(recovery/'scores.csv'); table.iloc[:-1].to_csv(recovery/'scores.csv', index=False)
    with pytest.raises(ValueError, match='Incomplete recovery'):
        load_table(recovery, allow_smoke=True)


def test_unknown_nuisance_change_is_rejected(recovery):
    path = recovery/'scores'/'S207_T00_R00'/'p1.json'; record = json.loads(path.read_text())
    record['candidate']['hyperparams']['known_other_parameter'] = .9
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='parameter identities'):
        load_table(recovery, allow_smoke=True)


def test_unpaired_filter_seeds_are_rejected(recovery):
    path = recovery/'scores'/'S207_T00_R00'/'p1.json'; arrays = path.with_suffix('.npz')
    with np.load(arrays) as saved:
        p = saved['observed_probability']
    np.savez(arrays, seeds=[1, 3], observed_probability=p)
    record = json.loads(path.read_text()); record.update(seeds=[1, 3], predictions_sha256=digest(arrays))
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        load_table(recovery, allow_smoke=True)
