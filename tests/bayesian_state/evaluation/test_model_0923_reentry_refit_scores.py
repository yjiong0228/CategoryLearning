from copy import deepcopy
import json

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_reentry_refit import load_summary, match_task, window_estimate
from src.Bayesian_state.evaluation.model_0923_estimation_scope import paired_difference
from src.Bayesian_state.evaluation.model_0923_h_validation import digest


def data():
    c = dict(seeds=np.array([11, 22, 33]), choices=np.array([1, 2, 2, 1]),
             categories=np.array([1, 1, 2, 2]),
             observed_probability=np.array([[.8, .2, .7, .4], [.4, .3, .8, .5], [.6, .4, .6, .3]]))
    a = dict(seeds=np.array([44, 55, 66, 77]), categories=c['categories'].copy(),
             choices=np.array([[1, 2, 2, 1], [1, 1, 2, 1], [2, 1, 1, 2], [2, 2, 2, 2]]))
    row = dict(condition=2, training_trials=2, particles=512, target_rule=7)
    return c, a, row


def direct_scores(c, a):
    x = (a['choices'] == a['categories'][None, :]).reshape(len(a['seeds']), 2, 2).mean(axis=2)
    y = (c['choices'] == c['categories']).reshape(2, 2).mean(axis=1)
    energy = np.mean([np.linalg.norm(path-y) for path in x])
    energy -= .5*np.mean([np.linalg.norm(left-right) for left in x for right in x])
    return -np.log(c['observed_probability'].mean(axis=0)).mean(), energy/np.sqrt(2)


def test_window_scores_and_delete_complete_repeats():
    c, a, _ = data()
    result = window_estimate(c, a, 0, 4, 2)
    nll, energy = direct_scores(c, a)
    assert result['metrics']['choice_nll'] == pytest.approx(nll)
    assert result['metrics']['energy'] == pytest.approx(energy)
    assert result['metrics']['correct_loss']+result['metrics']['error_loss'] == pytest.approx(nll)
    for i in range(len(c['seeds'])):
        reduced = dict(c, observed_probability=np.delete(c['observed_probability'], i, axis=0))
        assert result['jackknife']['choice_nll'][i] == pytest.approx(direct_scores(reduced, a)[0])
    for i in range(len(a['seeds'])):
        reduced = {key: np.delete(value, i, axis=0) if key != 'categories' else value for key, value in a.items()}
        assert result['jackknife']['energy'][i] == pytest.approx(direct_scores(c, reduced)[1])


def test_pairing_is_by_complete_seed_and_identical_scores_have_zero_error():
    c, a, _ = data()
    result = window_estimate(c, a, 0, 4, 2)
    assert all(value == 0 for value in paired_difference(result, result).values())
    changed = deepcopy(result)
    changed['filter_seeds'] = changed['filter_seeds'][::-1]
    with pytest.raises(AssertionError):
        paired_difference(changed, result)


@pytest.mark.parametrize('field', ['condition', 'training_trials', 'particles', 'target_rule'])
def test_task_changes_are_rejected(field):
    original = data()
    changed = deepcopy(original)
    changed[2][field] += 1
    with pytest.raises(ValueError, match='task or evaluation budget'):
        match_task(original, changed)


@pytest.mark.parametrize('bank,field', [(0, 'seeds'), (0, 'choices'), (0, 'categories'),
                                       (1, 'seeds'), (1, 'categories')])
def test_task_and_random_seed_arrays_must_match(bank, field):
    original = data()
    changed = deepcopy(original)
    changed[bank][field][0] += 1
    with pytest.raises(AssertionError):
        match_task(original, changed)


def test_generated_behaviors_may_change_at_paired_seeds():
    original = data()
    changed = deepcopy(original)
    changed[1]['choices'][:] = 2
    match_task(original, changed)


def summary_fixture(tmp_path):
    fit, evaluation, summary = (tmp_path/name for name in ('fit', 'evaluation', 'summary'))
    for path in (fit, evaluation, summary):
        path.mkdir()
    (fit/'nominees.json').write_text(json.dumps(dict(smoke=False, status='frozen_before_test', source_sha256={})))
    sha = digest(fit/'nominees.json')
    (fit/'completion.json').write_text(json.dumps(dict(status='complete', smoke=False,
        test_choices_used=False, nomination_sha256=sha)))
    (evaluation/'protocol.json').write_text(json.dumps(dict(fit=str(fit), source_sha256={}, nomination_sha256=sha)))
    (evaluation/'completion.json').write_text(json.dumps(dict(status='complete', nomination_sha256=sha)))
    tables = {}
    for name in ('scores', 'budgets', 'parameters', 'paths', 'curves'):
        path = summary/(name+'.csv')
        path.write_text('subject,model\n')
        tables[path.name] = digest(path)
    (summary/'manifest.json').write_text(json.dumps(dict(status='complete', fit=str(fit),
        evaluation=str(evaluation), inputs_sha256={}, summary_sha256=tables)))
    return fit, evaluation, summary


def test_summary_checks_saved_table_hashes(tmp_path):
    _, _, summary = summary_fixture(tmp_path)
    load_summary(summary)
    (summary/'scores.csv').write_text('subject,model\n208,SEARCH\n')
    with pytest.raises(ValueError, match='input or table changed'):
        load_summary(summary)


@pytest.mark.parametrize('root,name,field,value', [
    ('fit', 'completion.json', 'test_choices_used', True),
    ('fit', 'completion.json', 'smoke', True),
    ('evaluation', 'completion.json', 'status', 'running'),
    ('evaluation', 'protocol.json', 'nomination_sha256', 'different'),
    ('evaluation', 'protocol.json', 'fit', '/different_fit'),
])
def test_incompatible_fit_or_evaluation_is_rejected(tmp_path, root, name, field, value):
    fit, evaluation, summary = summary_fixture(tmp_path)
    path = {'fit': fit, 'evaluation': evaluation}[root]/name
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='completed behavior-prefix fits'):
        load_summary(summary)
