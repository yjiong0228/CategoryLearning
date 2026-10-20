from copy import deepcopy
import json

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_finite_region_fit import (
    configured_engine, digest, finalist_ids, load_trials, parameter_points,
)


def grid():
    return dict(association_learning_rate=[.1, .5], beta_init=[5., 25.], delta=[.99, 1.], gamma=[.8, 1.])


def test_grid_identity_and_common_point_count():
    points = parameter_points(grid())
    assert len(points) == len({p['id'] for p in points}) == 16
    reordered = dict(reversed(list(grid().items())))
    assert parameter_points(reordered) == points
    integer = grid(); integer['beta_init'] = [5, 25]
    assert parameter_points(integer) == points


@pytest.mark.parametrize('key,value', [('beta_init', [0.]), ('delta', [-.1]), ('gamma', [1.1]),
                                     ('association_learning_rate', [np.nan]), ('beta_init', [5., 5.])])
def test_unsupported_coordinates_are_rejected(key, value):
    changed = grid(); changed[key] = value
    with pytest.raises(ValueError):
        parameter_points(changed)


def test_anchor_preserves_exact_budget_even_when_already_best():
    rows = [dict(id='a', nll=.2), dict(id='b', nll=.1), dict(id='c', nll=.3), dict(id='d', nll=.4)]
    assert finalist_ids(rows, 'b', 3) == ['b', 'a', 'c']
    assert finalist_ids(rows, 'd', 3) == ['d', 'b', 'a']
    # Later and oral scores cannot influence nomination.
    decorated = [dict(r, later_nll=-i, oral_support=i) for i, r in enumerate(rows)]
    assert finalist_ids(decorated, 'd', 3) == finalist_ids(rows, 'd', 3)
    with pytest.raises(ValueError):
        finalist_ids(rows + [rows[0]], 'd', 3)


def test_engine_changes_only_declared_parameters_and_metadata():
    template = dict(modules=dict(beta_mod=dict(kwargs=dict(beta_init=5., learning_rate=.1)),
        memory_mod=dict(kwargs=dict(gamma=.8)), hypo_transitions_mod=dict(kwargs=dict(
            intervention=dict(decay=.8, search='accumulate'), max_capacity=3))),
        oral_readout={'ignored': True}, provenance={'label_mapping': {'mode': 'old'}})
    before = deepcopy(template)
    p = parameter_points(grid())[-1]
    configured = configured_engine(template, p)
    assert template == before
    assert configured['modules']['beta_mod']['kwargs'] == {'beta_init': 25., 'learning_rate': .1}
    assert configured['modules']['hypo_transitions_mod']['kwargs'] == {
        'intervention': {'decay': 1., 'search': 'accumulate'}, 'max_capacity': 3}
    assert configured['oral_readout'] is None


def files(tmp_path):
    training, later = tmp_path/'training.npz', tmp_path/'later.npz'
    arrays = dict(stimulus=np.full((4, 4), .4), choices=np.array([1, 2, 3, 4]),
                  categories=np.array([1, 3, 3, 1]), feedback=np.array([1., 0., 1., 0.]))
    np.savez(training, **arrays); np.savez(later, **arrays)
    job = dict(training=str(training), training_trials=4, stage='screen', subject=207, arm='REHEARSE',
               point=parameter_points(grid())[0])
    return job, later


def test_training_opens_only_training_file_and_preserves_order(tmp_path, monkeypatch):
    job, later = files(tmp_path)
    original = np.load
    opened = []

    def read(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', read)
    data = load_trials(job)
    assert opened == [job['training']]
    np.testing.assert_array_equal(data['choices'], [1, 2, 3, 4])
    opened.clear()
    with pytest.raises(ValueError, match='evaluation inputs'):
        load_trials(dict(job, later=str(later)))
    assert str(later) not in opened


def test_later_choices_require_matching_frozen_point(tmp_path, monkeypatch):
    job, later = files(tmp_path)
    frozen = tmp_path/'nominees.json'
    frozen.write_text(json.dumps(dict(status='frozen_before_later_choices', nominees=[
        dict(subject=207, arm='FROZEN', point=job['point'])])))
    changed = dict(job, stage='evaluation', later=str(later), test_trials=4,
                   nominees=str(frozen), nomination_sha256=digest(frozen))
    original = np.load
    opened = []

    def read(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', read)
    with pytest.raises(ValueError, match='frozen nominee'):
        load_trials(changed)
    assert str(later) not in opened
    changed['arm'] = 'FROZEN'
    data = load_trials(changed)
    np.testing.assert_array_equal(data['choices'], [1, 2, 3, 4, 1, 2, 3, 4])


def test_binary_feedback_alignment_is_checked(tmp_path):
    job, _ = files(tmp_path)
    with np.load(job['training']) as z:
        arrays = {k: z[k] for k in z.files}
    arrays['feedback'][1] = 1.
    np.savez(job['training'], **arrays)
    with pytest.raises(ValueError, match='misaligned'):
        load_trials(job)
