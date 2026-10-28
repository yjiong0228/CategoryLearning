"""Probabilistic task semantics, causal timing, and shared 1024 execution."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.inference.dispatcher import run_inference_backend
from src.Bayesian_state.metrics.probabilistic_task import probabilistic_task_metrics
from src.Bayesian_state.model import ModelContext
from src.Bayesian_state.model.model_1024 import model_1024_runtime
from src.Bayesian_state.model.modules.beta import BetaModule
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)
from src.Bayesian_state.model.modules.precision_interventions import precision_intervention
from src.Bayesian_state.simulation.probabilistic import (
    BinaryProbabilisticEnvironment, run_autonomous_probabilistic_learning,
)
from src.Bayesian_state.workflows.runs.run_model_1024_exp4 import load_config, run, _verify_replay
from tests.bayesian_state.test_model_1023_belief_execution import installed

CONFIG = Path("configs/exp4/model1024_smoke.yaml")


@pytest.fixture
def engine(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, "RUNTIME_CACHE_DIR", tmp_path / "similarity")
    _, config, _ = load_config(CONFIG)
    config["modules"]["hypo_transitions_mod"]["kwargs"]["relation_samples"] = 128
    config["provenance"]["hypothesis_similarity"]["sample_count"] = 128
    config["modules"]["perception_mod"]["kwargs"] = dict(
        features=4, mean=[0.] * 4, std=[0.] * 4)
    config["inference"]["particle_count"] = 4
    return config


def test_choice_contingent_feedback_and_rng_isolation():
    q = np.tile([.8, .2], (2000, 1))
    first, second = [BinaryProbabilisticEnvironment(q, seed=11) for _ in range(2)]
    a, b = [], []
    for t in range(len(q)):
        a.append(first(t, 1))
        b.append(second(t, 2))
    np.testing.assert_array_equal(first.uniform_draws, second.uniform_draws)
    np.testing.assert_array_equal(a, np.asarray(first.uniform_draws) < .8)
    np.testing.assert_array_equal(b, np.asarray(second.uniform_draws) < .2)
    assert .76 < np.mean(a) < .84 and .16 < np.mean(b) < .24
    with pytest.raises(ValueError, match="once per trial"):
        first(0, 1)
    edge = BinaryProbabilisticEnvironment(np.array([[0., 1.], [1., 0.]]), seed=12)
    assert [edge(0, 1), edge(1, 1)] == [0., 1.]


@pytest.mark.parametrize("q", [[], [[.4, .4]], [[np.nan, 1.]], [[1.1, -.1]], [[1.]]])
def test_invalid_task_probabilities_rejected(q):
    with pytest.raises(ValueError, match="normalized binary"):
        BinaryProbabilisticEnvironment(np.asarray(q), seed=1)


def test_quality_is_separate_from_realized_reward_and_ties_are_symmetric():
    metrics = probabilistic_task_metrics(choices=np.array([1, 2, 1]), feedback=np.array([0., 1., 0.]),
                                        target_probabilities=np.array([[.8, .2], [.5, .5], [.3, .7]]))
    assert metrics["optimal_choice_rate"] == 2 / 3
    assert metrics["mean_reward"] == 1 / 3
    assert metrics["expected_reward"] == pytest.approx(1.6 / 3)
    assert metrics["expected_regret"] == pytest.approx(.4 / 3)


def _generate(engine, q, seed=41):
    return run_autonomous_probabilistic_learning(
        engine_config=engine, subject_id=401, stimulus=np.tile([.7, .2, .4, .8], (len(q), 1)),
        target_probabilities=q, trajectory_seed=seed)


def test_runtime_matches_existing_adapters_and_restores_methods(engine):
    q = np.tile([.8, .2], (16, 1))
    original = Transition.__init__, BetaModule._update_beta_feedback_gradient
    with installed("STICKY"), precision_intervention(BetaModule, mode="fast", upper=25.):
        reference = _generate(engine, q)
    with model_1024_runtime():
        actual = _generate(engine, q)
        with pytest.raises(RuntimeError, match="nested"):
            with model_1024_runtime():
                pass
    assert original == (Transition.__init__, BetaModule._update_beta_feedback_gradient)
    for field in ("choices", "feedback", "prior", "posterior", "beta", "observed_probabilities"):
        np.testing.assert_array_equal(getattr(reference.trajectory, field), getattr(actual.trajectory, field))


def test_autonomous_seed_prefix_and_observation_replay(engine):
    q = np.tile([.8, .2], (16, 1))
    class SyntheticCase:
        def model_context(self):
            return ModelContext(condition=1, subject_id=401)
    with model_1024_runtime():
        a, b = _generate(engine, q), _generate(engine, q)
        prefix = _generate(engine, q[:7])
        _verify_replay(engine, SyntheticCase(), a)
    for field in ("choices", "feedback", "prior", "posterior", "beta", "observed_probabilities"):
        np.testing.assert_array_equal(getattr(a.trajectory, field), getattr(b.trajectory, field))
        np.testing.assert_array_equal(getattr(a.trajectory, field)[:7], getattr(prefix.trajectory, field))


def test_task_probabilities_and_current_feedback_cannot_change_current_choice(engine):
    q = np.tile([.8, .2], (12, 1))
    changed = q.copy()
    changed[5:] = [0., 1.]
    with model_1024_runtime():
        a, b = _generate(engine, q), _generate(engine, changed)
        # q_t is used only after choice_t. A changed future environment leaves
        # the entire pre-feedback prefix unchanged, including choice at t=5.
        for field in ("choices", "prior", "beta", "observed_probabilities"):
            np.testing.assert_array_equal(getattr(a.trajectory, field)[:6], getattr(b.trajectory, field)[:6])
        outcomes = a.trajectory.feedback.copy()
        outcomes[5] = 1. - outcomes[5]
        kwargs = dict(engine_config=engine, condition=1, subject_id=401,
                      stimulus=a.trajectory.stimulus, choices=a.trajectory.choices, inference_seed=19)
        first = run_inference_backend(**kwargs, feedback=a.trajectory.feedback)
        second = run_inference_backend(**kwargs, feedback=outcomes)
    np.testing.assert_array_equal(first.marginal_probabilities[:6], second.marginal_probabilities[:6])
    assert not np.array_equal(first.marginal_probabilities[6:], second.marginal_probabilities[6:])


def test_real_exp4_entry_point_and_output_protection(tmp_path):
    settings = yaml.safe_load(CONFIG.read_text())
    settings.update(input_config=str((CONFIG.parent / settings["input_config"]).resolve()),
                    base_engine_config=str((CONFIG.parent / settings["base_engine_config"]).resolve()),
                    max_trials=6, autonomous_paths=1)
    overrides = settings["engine_overrides"]
    overrides["inference"]["particle_count"] = 4
    overrides["modules"]["hypo_transitions_mod"]["kwargs"]["relation_samples"] = 128
    overrides["provenance"]["hypothesis_similarity"] = dict(sample_count=128)
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(settings))
    output = tmp_path / "run"
    result = run(config, output)
    assert result["status"] == "closed_loop_smoke_passed" and all(result["checks"].values())
    assert result["subject_metadata"]["feedback_probability_disclosed"] is True
    assert result["executed_trials"] == 6
    with np.load(output / "observed.npz", allow_pickle=False) as observed:
        assert observed["pre_choice_probabilities"].shape == (6, 2)
        assert "target_probabilities" not in observed
    with pytest.raises(FileExistsError):
        run(config, output)


def test_smoke_budget_and_readout_consistency(tmp_path):
    settings = yaml.safe_load(CONFIG.read_text())
    config = tmp_path / "too_large.yaml"
    settings["max_trials"] = 960
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="max_trials"):
        load_config(config)
