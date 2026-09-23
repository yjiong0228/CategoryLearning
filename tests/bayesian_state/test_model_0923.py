"""Scientific invariants of the 0923 baseline and pre-feedback report kernel."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model.config import ModelConfig
from src.Bayesian_state.model.modules.hypothesis_transition.contracts import HypothesisSelection
from src.Bayesian_state.model.modules.hypothesis_transition.unified_rule_search import (
    UnifiedRuleSearchHypothesisTransitionModule,
)
from src.Bayesian_state.model.oral_observation import predict_choice_oral

ROOT = Path(__file__).resolve().parents[2]


class _Partition:
    def get_similarity_matrix(self, **kwargs):
        positions = np.arange(8, dtype=float)
        return np.exp(-abs(positions[:, None] - positions[None, :]))


class _Engine:
    def __init__(self):
        self.set_size = 8
        self.prior = np.full(8, 1 / 8)
        self.posterior = None
        self.hypotheses_mask = None
        self.partition = _Partition()
        self.distance_mode = "boundary"

    def get_module(self, role, *, required=False):
        if required:
            raise ValueError(role)
        return None


def _module(*, gain=1.0, capacity=3, **kwargs):
    engine = _Engine()
    module = UnifiedRuleSearchHypothesisTransitionModule(
        engine, capacity=capacity, tau_local=0.1, module_seed=20260923,
        search_controller={"baseline_probability": 0.2, "error_gain": gain,
                           "global_search": 0.3, "failure_decay": 0.6},
        **kwargs,
    )
    module.reseed_future(20260923)
    return engine, module


def _config(condition):
    path = ROOT / f"configs/exp123/model_struct/model_0923_cond{condition}_B0.yaml"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("gain", [0.0, 1.0, 3.0])
def test_trace_uses_only_completed_feedback_and_retains_partial_success(gain):
    engine, module = _module(gain=gain)
    module.process()
    assert module.transition_log[-1]["swap_probability"] == 0.0
    failure = 0.0
    for feedback in [0.0, 0.5, 1.0, 0.0, 1.0]:
        before = module.current_event_probability
        engine.posterior = engine.prior.copy()
        module.record_outcome((np.zeros(4), 1, feedback))
        assert module.current_event_probability == before
        module.process()
        failure = 0.6 * failure + 0.4 * (1.0 - feedback)
        expected = 1 / (1 + np.exp(-(np.log(0.2 / 0.8) + gain * failure)))
        assert module.failure_pressure == pytest.approx(failure)
        assert module.current_event_probability == pytest.approx(expected)
        assert module.current_g == 0.3
        assert module.transition_log[-1]["swap_probability"] == pytest.approx(expected)
        assert len(module.last_transition_result.selection.newcomers) <= 1
        assert module.transition_log[-1]["transition_method"] == "single_candidate_bernoulli"
        np.testing.assert_allclose(engine.prior.sum(), 1.0)


@pytest.mark.parametrize("capacity", [2, 3, 4])
def test_revision_event_probability_is_independent_of_workspace_capacity(capacity):
    _, module = _module(capacity=capacity)
    module.current_event_probability = 0.4
    count = np.array([module._draw_replacement_count(capacity, 0.01) for _ in range(4000)])
    assert set(count) == {0, 1}
    assert abs(count.mean() - 0.4) < 0.025


def test_single_revision_preserves_survivor_mass_even_for_a_weak_dropped_rule():
    _, module = _module()
    old = np.array([0.89, 0.10, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    selection = HypothesisSelection.from_active_sets([0, 1, 2], [0, 1, 3])
    prior, _ = module._mass_preserving_similarity_transport_prior(old, selection)
    np.testing.assert_allclose(prior, [0.89, 0.10, 0, 0.01, 0, 0, 0, 0], atol=1e-15)
    assert module.persistent_execution_enabled is False


def test_resampled_state_retains_controller_trace():
    engine, module = _module()
    module.process()
    engine.posterior = engine.prior.copy()
    module.record_outcome((np.zeros(4), 1, 0.5))
    module.process()
    saved = deepcopy(module.state_dict())
    _, restored = _module()
    restored.load_state_dict(saved)
    assert restored.failure_pressure == module.failure_pressure
    assert restored.current_event_probability == module.current_event_probability
    assert restored.previous_feedback == 0.5
    assert restored._draw_replacement_count(3, 0.1) == module._draw_replacement_count(3, 0.1)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_condition_profiles_are_valid_and_do_not_mutate_input(condition):
    config = _config(condition)
    before = deepcopy(config)
    ModelConfig.from_mapping(config)
    assert config == before


@pytest.mark.parametrize("change", ["dynamic_precision", "persistent_readout", "wrong_feedback", "unknown_variant"])
def test_baseline_rejects_hidden_extra_mechanisms(change):
    config = _config(3)
    if change == "dynamic_precision":
        config["modules"]["beta_mod"]["kwargs"]["increase_rate"] = 0.04
    elif change == "persistent_readout":
        config["choice_readout"]["kwargs"]["method"] = "sticky"
    elif change == "wrong_feedback":
        config["likelihood"]["feedback_likelihood_mode"] = "category_feedback"
    else:
        config["provenance"]["variant"] = "B0_plus_unregistered"
    with pytest.raises(ValueError):
        ModelConfig.from_mapping(config)


def test_transition_rejects_old_execution_settings():
    with pytest.raises(ValueError, match="unsupported transition keys"):
        _module(persistent_execution={"enabled": True})


def test_choice_and_report_share_the_same_latent_rule():
    weights = np.array([0.6, 0.4])
    choices = np.array([[0.9, 0.1], [0.2, 0.8]])
    mapping = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=float)
    prediction = predict_choice_oral(weights, choices, mapping)
    # Joint probability, not a product of independently averaged marginals.
    np.testing.assert_allclose(prediction.joint_probabilities, [[0.54, 0.08], [0.06, 0.32]])
    np.testing.assert_allclose(prediction.choice_probabilities, [0.62, 0.38])
    np.testing.assert_allclose(prediction.report_given_choice(0), [27 / 31, 4 / 31])
    assert prediction.joint_probabilities.sum() == pytest.approx(1.0)


def test_report_noise_does_not_change_choice_prediction():
    weights = np.array([0.6, 0.4])
    choices = np.array([[0.9, 0.1], [0.2, 0.8]])
    mapping = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=float)
    prediction = predict_choice_oral(weights, choices, mapping, reliability=0.0)
    np.testing.assert_allclose(prediction.joint_probabilities, [[0.31, 0.31], [0.19, 0.19]])
    np.testing.assert_allclose(prediction.choice_probabilities, [0.62, 0.38])


def test_report_kernel_rejects_unnormalized_compatibility_scores():
    with pytest.raises(ValueError, match="rows must sum to 1"):
        predict_choice_oral(np.ones(2), np.full((2, 2), 0.5), np.ones((2, 2, 3)))


def test_report_conditioning_rejects_an_impossible_choice():
    prediction = predict_choice_oral(
        np.ones(2), np.array([[1, 0], [1, 0]]), np.ones((2, 2, 1))
    )
    with pytest.raises(ValueError, match="impossible choice"):
        prediction.report_given_choice(1)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_particle_filter_is_deterministic_and_current_feedback_cannot_change_prediction(condition):
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter

    config = _config(condition)
    config["modules"]["perception_mod"]["kwargs"] = {
        "features": 4, "mean": [0.0] * 4, "std": [0.0] * 4,
    }
    stimulus = np.array([[0.2, 0.3, 0.7, 0.4], [0.8, 0.6, 0.2, 0.7],
                         [0.4, 0.8, 0.3, 0.6], [0.9, 0.4, 0.7, 0.2]])
    feedback = np.array([0.0, 0.5 if condition == 3 else 0.0, 1.0, 0.0])
    arguments = dict(engine_config=config, subject_id=122, condition=condition,
                     stimulus=stimulus, choices=[1, 2, 1, 2], particle_count=4,
                     choice_readout_power=1.0, filter_seed=20260923)
    first = run_state_model_particle_filter(feedback=feedback, **arguments)
    second = run_state_model_particle_filter(feedback=feedback, **arguments)
    np.testing.assert_array_equal(first.marginal_probabilities, second.marginal_probabilities)
    np.testing.assert_allclose(first.marginal_probabilities.sum(axis=1), 1.0)
    assert np.all(np.isfinite(first.marginal_probabilities))
    assert np.all(first.predictive_replacement_fraction <= 1 / 3 + 1e-12)
    changed_feedback = feedback.copy()
    changed_feedback[-1] = 1.0
    changed = run_state_model_particle_filter(feedback=changed_feedback, **arguments)
    np.testing.assert_array_equal(first.marginal_probabilities, changed.marginal_probabilities)


def test_old_condition3_profile_still_rejects_graded_search():
    from src.Bayesian_state.model import ModelContext, StateModel

    config = yaml.safe_load((ROOT / "configs/exp123/model_struct/pmh_model_cond3_0826.yaml").read_text())
    config["modules"]["perception_mod"]["kwargs"] = {
        "features": 4, "mean": [0.0] * 4, "std": [0.0] * 4,
    }
    config["modules"]["hypo_transitions_mod"]["kwargs"]["feedback_interpretation"] = "graded"
    with pytest.raises(ValueError, match="full_success workspace search"):
        StateModel(config, context=ModelContext(condition=3, subject_id=314))


def test_baseline_rejects_wrong_condition_context():
    from src.Bayesian_state.model import ModelContext, StateModel

    with pytest.raises(ValueError, match="declared condition"):
        StateModel(_config(2), context=ModelContext(condition=1, subject_id=122))


def test_module_override_cannot_silently_enable_dynamic_precision():
    from src.Bayesian_state.model import ModelContext, StateModel

    config = _config(1)
    beta = deepcopy(config["modules"]["beta_mod"]["kwargs"])
    beta["increase_rate"] = 0.1
    with pytest.raises(ValueError, match="fixed rule precision"):
        StateModel(config, context=ModelContext(condition=1, subject_id=122),
                   module_overrides={"beta_mod": {"kwargs": beta}})
