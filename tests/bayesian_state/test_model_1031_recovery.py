"""End-to-end simulation truth and pre-feedback state-output validation."""
from pathlib import Path

import numpy as np
import yaml

from src.Bayesian_state.workflows.analysis.model_1031_finalization import (
    DEFAULT, engine_config, identity, save_arrays, write_new,
)
from src.Bayesian_state.workflows.analysis.model_1031_recovery import _custom_work, weighted_intervals


def test_weighted_intervals_respect_particle_probability_not_particle_count():
    values = np.array([[0., .5], [1., .7], [2., .9]])
    weights = np.array([.98, .01, .01])
    np.testing.assert_array_equal(weighted_intervals(values, weights), [[0., 0., 0.], [.5, .5, .5]])


def test_cond3_truth_and_state_recovery_share_the_same_pre_feedback_target(tmp_path):
    policy = yaml.safe_load(Path("configs/shared/model1031_finalization_20261101_v2.yaml").read_text())
    cfg = engine_config(3, dict(DEFAULT, capacity=2), policy["variants"]["compact"])
    cfg["modules"]["perception_mod"]["kwargs"] = dict(mean=[0.] * 4, std=[0.] * 4)
    key = identity(cfg)
    write_new(tmp_path / "configs" / f"{key}.json", cfg)
    x = np.random.default_rng(912).uniform(.1, .9, (6, 4))
    save_arrays(tmp_path / "inputs/s0.npz", stimulus=x, categories=np.array([1, 3, 2, 4, 1, 2]))
    generation = dict(output=str(tmp_path), kind="recovery_generate", subject=0, condition=3,
                      config=key, seed=81)
    generation["id"] = identity(generation)
    record = _custom_work(generation)
    assert record["trials"] == 6
    state = dict(output=str(tmp_path), kind="recovery_state", subject=0, condition=3, config=key,
                 generation_id=generation["id"], particles=4, seed=93)
    state["id"] = identity(state)
    record = _custom_work(state)
    assert record["trials"] == 6
    with np.load(tmp_path / "jobs" / f"{state['id']}.npz") as result:
        np.testing.assert_allclose(result["execution"].sum(axis=1), 1.)
        np.testing.assert_allclose(result["pairing"][0], 1 / 3)
        np.testing.assert_allclose(result["execution"][0], [1., 0., 0., 0.])
        assert result["group_intervals"].shape == (6, 4, 3)
    assert 0. <= record["marginal90_coverage"] <= 1.
