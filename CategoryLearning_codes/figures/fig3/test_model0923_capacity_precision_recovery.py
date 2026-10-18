"""Numerical and identity guards for the capacity/storage recovery figure."""
import json

import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model0923_capacity_precision_recovery import (
    CAPACITY, CELLS, digest, load_table,
)


@pytest.fixture
def recovery(tmp_path):
    """An analytic four-way tie keeps filter variability but no contrast error."""
    case = 'S208_T00_R00'
    folder = tmp_path/'scores'/case
    folder.mkdir(parents=True)
    (tmp_path/'jobs').mkdir()
    protocol = dict(smoke=True, settings=dict(protocol='model0923_capacity_precision_recovery_v1'),
                    jobs=4, filter_runs=8)
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    truth = dict(case=case, true_cell='K2_RESET', capacity=2, variant='RESET', trials=3,
                 candidate=dict(id='k2', hyperparams={CAPACITY:2, 'known_other_parameter':.2}))
    (tmp_path/'truths.json').write_text(json.dumps(dict(rows=[truth])))
    p = np.array([[.2,.6,.8], [.4,.8,.6]])
    score = -np.log([.3,.7,.7]).mean()
    se = abs(-np.log(p[0]).mean()+np.log(p[1]).mean())/2
    rows = []
    for cell in CELLS:
        capacity, variant = int(cell[1]), cell[3:]
        path = folder/(cell+'.json')
        np.savez(path.with_suffix('.npz'), seeds=[1,2], observed_probability=p)
        candidate = dict(id=f'k{capacity}', hyperparams={CAPACITY:capacity, 'known_other_parameter':.2})
        record = dict(candidate=candidate, model='SEARCH', subject=208, condition=2, particles=2,
            seeds=[1,2], trials=3, mean_nll=score, predictions_sha256=digest(path.with_suffix('.npz')))
        path.write_text(json.dumps(record))
        path.with_suffix('.precision.json').write_text(json.dumps(dict(retain_precision=variant == 'RETAIN')))
        job = dict(candidate=candidate, target=str(path.with_suffix('')), seeds=[1,2], retain_precision=variant == 'RETAIN')
        (tmp_path/'jobs'/('score_'+case+'_'+cell+'.json')).write_text(json.dumps(job))
        rows.append(dict(case=case, true_cell='K2_RESET', true_capacity=2, true_variant='RESET',
            cell=cell, capacity=capacity, variant=variant, candidate_id=candidate['id'],
            score_file=str(path), mean_nll=score, mean_nll_mcse=se, trials=3))
    pd.DataFrame(rows).to_csv(tmp_path/'scores.csv', index=False)
    pd.DataFrame([dict(case=case, true_cell='K2_RESET', selected_cell='K2_RESET',
        tied_minima=4, truth_is_minimum=True, unique_joint_correct=False)]).to_csv(tmp_path/'recovered.csv', index=False)
    done = dict(status='complete', smoke=True, nuisance_parameters_known=True, sequences=1, jobs=4,
        filter_runs=8, protocol_sha256=digest(tmp_path/'protocol.json'),
        artifacts={n:digest(tmp_path/n) for n in ('scores.csv','recovered.csv')})
    (tmp_path/'completion.json').write_text(json.dumps(done))
    return tmp_path


def edit(path, change):
    row = json.loads(path.read_text())
    change(row)
    path.write_text(json.dumps(row))


def test_all_tied_minima_are_shown_without_claiming_unique_recovery(recovery):
    table, meta = load_table(recovery, allow_smoke=True)
    assert meta['smoke'] and len(table) == 4
    assert table.is_minimum.all() and table.is_truth.sum() == 1
    assert not table.unique_correct.any() and table.tied_minima.eq(4).all()
    assert table.delta_to_truth.eq(0).all() and table.delta_to_truth_mcse.eq(0).all()
    assert table.mean_nll_mcse.gt(0).all()


def test_smoke_needs_explicit_opt_in(recovery):
    with pytest.raises(ValueError, match='smoke requires'):
        load_table(recovery)


def test_mutated_summary_is_rejected(recovery):
    with (recovery/'scores.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='table changed'):
        load_table(recovery, allow_smoke=True)


def test_missing_cell_is_rejected_even_with_updated_table_hash(recovery):
    path = recovery/'scores.csv'
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    edit(recovery/'completion.json', lambda d:d['artifacts'].update({'scores.csv':digest(path)}))
    with pytest.raises(ValueError, match='Incomplete recovery cases'):
        load_table(recovery, allow_smoke=True)


def test_unknown_parameter_change_is_rejected(recovery):
    for path in [recovery/'scores'/'S208_T00_R00'/'K2_RETAIN.json',
                 recovery/'jobs'/'score_S208_T00_R00_K2_RETAIN.json']:
        edit(path, lambda d:d['candidate']['hyperparams'].update(known_other_parameter=.9))
    with pytest.raises(ValueError, match='identity mismatch'):
        load_table(recovery, allow_smoke=True)


def test_storage_execution_must_match_candidate(recovery):
    path = recovery/'scores'/'S208_T00_R00'/'K2_RETAIN.precision.json'
    edit(path, lambda d:d.update(retain_precision=False))
    with pytest.raises(ValueError, match='identity mismatch'):
        load_table(recovery, allow_smoke=True)


def test_scoring_seeds_must_be_paired_across_candidates(recovery):
    path = recovery/'scores'/'S208_T00_R00'/'K2_RETAIN.json'
    arrays = path.with_suffix('.npz')
    with np.load(arrays) as z:
        p = z['observed_probability']
    np.savez(arrays, seeds=[1,3], observed_probability=p)
    edit(path, lambda d:d.update(seeds=[1,3], predictions_sha256=digest(arrays)))
    edit(recovery/'jobs'/'score_S208_T00_R00_K2_RETAIN.json', lambda d:d.update(seeds=[1,3]))
    with pytest.raises(AssertionError):
        load_table(recovery, allow_smoke=True)


def test_false_unique_recovery_label_is_rejected(recovery):
    path = recovery/'recovered.csv'
    data = pd.read_csv(path)
    data['unique_joint_correct'] = True
    data.to_csv(path, index=False)
    edit(recovery/'completion.json', lambda d:d['artifacts'].update({'recovered.csv':digest(path)}))
    with pytest.raises(ValueError, match='selection or ties'):
        load_table(recovery, allow_smoke=True)
