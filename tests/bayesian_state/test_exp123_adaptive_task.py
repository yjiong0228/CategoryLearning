"""Task source semantics and independent cognitive-generator parity."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.simulation.exp123_adaptive_task import (
    AdaptiveTask, allocate_species, block_performance, sample_block, stimulus_labels,
)


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_cube_geometry_counts_and_ambiguity(condition):
    block = sample_block(np.array([11, 33, 10, 10]), condition, np.random.default_rng(12))
    counts = np.bincount(block.categories)[1:]
    np.testing.assert_array_equal(counts, [44, 20] if condition == 1 else [11, 33, 10, 10])
    labels, ambiguous = stimulus_labels(block.stimulus, condition)
    np.testing.assert_array_equal(labels, block.categories)
    np.testing.assert_array_equal(ambiguous, block.ambiguous)


def test_matlab_nan_clip_and_empty_category_stop_semantics():
    accuracy, dprime = block_performance(np.repeat([1, 2], 3), np.ones(6, int), np.zeros(6), 2)
    np.testing.assert_array_equal(accuracy, [1, 0])
    np.testing.assert_array_equal(dprime, [-6, -6])
    accuracy, dprime = block_performance(np.ones(3, int), np.ones(3, int), np.zeros(3), 2)
    assert np.isnan(accuracy[1])
    assert not np.all(accuracy >= .9)
    assert np.isfinite(dprime).all()


def test_ambiguity_and_strict_criterion_use_exact_choices():
    cats = np.tile(np.arange(1, 5), 2)
    choices = cats.copy(); choices[0] = 2
    accuracy, _ = block_performance(cats, choices, np.zeros(8), 4)
    assert accuracy[0] == .5
    ambiguity = np.zeros(8); ambiguity[0] = 1
    accuracy, _ = block_performance(cats, choices, ambiguity, 4)
    np.testing.assert_array_equal(accuracy, np.ones(4))


@pytest.mark.parametrize('dprime,expected', [([0, 0, 0, 0], [16, 16, 16, 16]),
    ([6, -6, 6, 6], [5, 50, 5, 4]), ([-6, 6, -6, 6], [29, 3, 29, 3])])
def test_original_minimum_remainder_rule(dprime, expected):
    np.testing.assert_array_equal(allocate_species(np.array(dprime), 2), expected)


def test_stopping_is_at_first_passing_block_and_horizon_is_separate():
    first = sample_block(np.full(4, 16), 2, np.random.default_rng(1))
    task = AdaptiveTask(first, condition=2, seed=2, max_blocks=10)
    assert task.complete_block(first.categories) is None
    assert task.history[-1]['passed'] and len(task.history) == 1
    with pytest.raises(RuntimeError):
        task.complete_block(first.categories)
    task = AdaptiveTask(first, condition=2, seed=2, max_blocks=1)
    assert task.complete_block(np.ones(64, int)) is None
    assert not task.history[-1]['passed']


def test_session_first_block_and_previous_session_dprime_blend():
    first = sample_block(np.full(4, 16), 2, np.random.default_rng(1))
    task = AdaptiveTask(first, condition=2, seed=2, max_blocks=8)
    for _ in range(5):
        task.complete_block(np.ones(64, int))
    np.testing.assert_array_equal(task.current.stimulus, first.stimulus)
    task.complete_block(np.full(64, 2))
    # Correct choices except one category avoid both stopping and constant d'.
    response = task.current.categories.copy(); response[response == 1] = 2
    task.complete_block(response)
    np.testing.assert_allclose(task.history[-1]['used_dprime'],
        (task.history[4]['dprime'] + task.history[6]['dprime']) / 2)


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_adaptive_lifecycle_exactly_replays_through_existing_generator(condition, tmp_path):
    from src.Bayesian_state.simulation.adaptive_autonomous import run_adaptive_category_learning
    from src.Bayesian_state.simulation.autonomous import run_autonomous_category_learning
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    config['partition']['kwargs'].update(similarity_n_samples=128, similarity_cache_dir=str(tmp_path))
    config['oral_readout'] = None
    first = sample_block(np.full(4, 16), condition, np.random.default_rng(1))
    task = AdaptiveTask(first, condition=condition, seed=2, max_blocks=2)
    result = run_adaptive_category_learning(engine_config=config, subject_id=101, task=task, trajectory_seed=42)
    a = result['arrays']
    replay = run_autonomous_category_learning(engine_config=config, subject_id=101, condition=condition,
        stimulus=a['stimulus'], categories=a['categories'], trajectory_seed=42).trajectory
    np.testing.assert_array_equal(a['choices'], replay.choices)
    np.testing.assert_array_equal(a['feedback'], replay.feedback)
    np.testing.assert_array_equal(a['probabilities'], replay.observed_probabilities)
