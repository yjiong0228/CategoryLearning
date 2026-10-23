"""Exact replay rejects changed arrays even when provenance is regenerated."""
import hashlib

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1022_belief_capture import assert_replay


def test_capture_replay_requires_identical_predictions_and_seeds(tmp_path):
    original, replay = tmp_path / 'original.npz', tmp_path / 'replay.npz'
    arrays = dict(probabilities=np.array([[[.7, .3], [.4, .6]]]), choices=np.array([1, 2]), seeds=np.array([41]))
    np.savez_compressed(original, **arrays)
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    np.savez_compressed(replay, **arrays)
    assert_replay(original, replay, sha)
    np.savez_compressed(replay, **dict(arrays, seeds=np.array([42])))
    with pytest.raises(AssertionError, match='seeds'):
        assert_replay(original, replay, sha)
    changed = arrays['probabilities'].copy()
    changed[0, 0, :] += [1e-12, -1e-12]
    np.savez_compressed(replay, **dict(arrays, probabilities=changed))
    with pytest.raises(AssertionError, match='probabilities'):
        assert_replay(original, replay, sha)
    np.savez_compressed(original, **dict(arrays, choices=np.array([2, 1])))
    with pytest.raises(ValueError, match='changed'):
        assert_replay(original, replay, sha)
