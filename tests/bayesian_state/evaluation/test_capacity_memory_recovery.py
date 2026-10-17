"""Recovery data boundaries and outcome-independent generating-seed selection."""
import hashlib
import json

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_capacity_memory_recovery import (
    disjoint_seeds, training_prefix, verified_resource_dependency,
)


def test_generating_seed_ranks_are_deterministic_and_disjoint():
    bank = [70, 10, 60, 20, 50, 30, 40, 80]
    selected = [disjoint_seeds(bank, i, 2) for i in range(4)]
    assert selected == [[10, 20], [30, 40], [50, 60], [70, 80]]
    assert len(set(seed for pair in selected for seed in pair)) == 8


@pytest.mark.parametrize('bank,index,repeats', [([1, 1], 0, 1), ([1, 2], 1, 2),
    ([1, 2], -1, 1), ([1, 2], 0, 0), ([-1, 2], 0, 1), ([True, 2], 0, 1)])
def test_invalid_seed_selection_is_rejected(bank, index, repeats):
    with pytest.raises(ValueError):
        disjoint_seeds(bank, index, repeats)


def banks():
    physical = dict(stimulus=np.arange(20).reshape(5, 4)/20, categories=np.array([1, 2, 1, 2, 1]),
                    choices=np.full(5, 99), feedback=np.full(5, -1))
    generated = dict(seeds=np.array([7, 9]), categories=physical['categories'],
        choices=np.array([[2, 2, 1, 2, 1], [1, 1, 2, 1, 2]]),
        feedback=np.array([[0., 1., 1., 1., 1.], [1., 0., 0., 0., 0.]]))
    return physical, generated


def test_prefix_uses_synthetic_behavior_and_excludes_later_trials():
    physical, generated = banks(); prefix = training_prefix(physical, generated, 9, 2)
    np.testing.assert_array_equal(prefix['stimulus'], physical['stimulus'][:2])
    np.testing.assert_array_equal(prefix['categories'], [1, 2])
    np.testing.assert_array_equal(prefix['choices'], [1, 1])
    np.testing.assert_array_equal(prefix['feedback'], [1., 0.])
    generated['choices'][1, 2:] = 88
    np.testing.assert_array_equal(prefix['choices'], [1, 1])


def test_prefix_rejects_category_mismatch():
    physical, generated = banks(); generated['categories'] = np.array([2, 2, 1, 2, 1])
    with pytest.raises(AssertionError):
        training_prefix(physical, generated, 7, 2)


@pytest.mark.parametrize('seed,length', [(8, 2), (7, 0), (7, 6), (7, 1.5)])
def test_prefix_requires_unique_seed_and_valid_integer_length(seed, length):
    physical, generated = banks()
    with pytest.raises(ValueError):
        training_prefix(physical, generated, seed, length)


@pytest.mark.parametrize('problem', ['none', 'wrong_replay', 'changed_protocol'])
def test_cpu_dependency_uses_the_declared_completed_recovery(tmp_path, problem):
    replay = tmp_path/'replay'; replay.mkdir()
    (replay/'protocol.json').write_text(json.dumps(dict(smoke=False, role='fixed_parameter_replay_no_oral_input')))
    previous = tmp_path/'previous'; previous.mkdir(); protocol = previous/'protocol.json'
    protocol.write_text(json.dumps(dict(resource_predecessor=str(replay if problem != 'wrong_replay' else tmp_path/'other'))))
    done = previous/'completion.json'
    done.write_text(json.dumps(dict(status='complete', smoke=False,
                                   protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())))
    if problem == 'changed_protocol':
        protocol.write_text(protocol.read_text()+'\n')
    if problem == 'none':
        assert verified_resource_dependency(done, replay)['dependency_sha256'] == hashlib.sha256(done.read_bytes()).hexdigest()
    else:
        with pytest.raises(ValueError):
            verified_resource_dependency(done, replay)
