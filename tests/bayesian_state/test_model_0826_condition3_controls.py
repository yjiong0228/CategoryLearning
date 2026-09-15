"""Condition 3 confidence and search controls preserve feedback timing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.beta import BetaModule
from src.Bayesian_state.model.modules.hypothesis_transition.feedback_reactive import (
    FeedbackReactiveHypothesisTransitionModule,
)
from src.Bayesian_state.model.modules.hypothesis_transition.nested_feedback_accumulator import (
    NestedFeedbackAccumulatorHypothesisTransitionModule,
)


class _ControlPartition:
    n_cats = 4

    def get_similarity_matrix(self, *, kind, distance_mode, **kwargs):
        del kwargs
        assert kind == "assignment_agreement"
        assert distance_mode == "boundary"
        positions = np.arange(6, dtype=float)
        return np.exp(-np.abs(positions[:, None] - positions[None, :]))


class _ControlEngine:
    """Small workspace with a supplied, already cached memory evidence vector."""

    def __init__(self, evidence=None):
        self.set_size = 6
        self.prior = np.full(6, 1.0 / 6.0)
        self.posterior = None
        self.hypotheses_mask = np.asarray([1, 1, 1, 0, 0, 0], dtype=float)
        self.distance_mode = "boundary"
        self.partition = _ControlPartition()
        self.memory = SimpleNamespace(feedback_evidence=evidence)
        self.transition = None

    def get_module(self, role, *, required=False):
        module = {
            ModuleRole.MEMORY: self.memory,
            ModuleRole.HYPOTHESIS_TRANSITION: self.transition,
        }.get(role)
        if required and module is None:
            raise ValueError(f"missing module: {role}")
        return module


def _beta(evidence, **kwargs):
    engine = _ControlEngine(np.asarray(evidence, dtype=float))
    module = BetaModule(
        engine,
        **{
            "beta_init": 10.0,
            "beta_min": 0.1,
            "beta_max": 30.0,
            "increase_rate": 0.2,
            "decrease_rate": 0.3,
            "beta_update_mode": "hierarchical_feedback",
            "use_prior_scaling": False,
            **kwargs,
        },
    )
    return engine, module


@pytest.mark.parametrize("feedback, chance", [(1.0, 0.25), (0.5, 0.25), (0.0, 0.5)])
def test_hierarchical_beta_is_neutral_for_each_feedback_chance(feedback, chance):
    _, module = _beta([chance] * 6)

    module.update_beta(np.asarray([0.5]), 1, feedback)

    np.testing.assert_array_equal(module.beta, [10.0, 10.0, 10.0, 0.0, 0.0, 0.0])


def test_hierarchical_beta_uses_absolute_cached_evidence_and_logs_predictive_beta():
    engine, module = _beta([0.5, 0.125, 0.25, 0.0, 0.0, 0.0])
    cached = engine.memory.feedback_evidence.copy()
    engine.observation = (np.asarray([0.5]), 1, 0.5)
    # Post-feedback pair weights favor a different partner. Beta must consume
    # the cached pre-feedback evidence even after memory has updated its state.
    engine.memory.pairing_weights = np.tile([0.0, 0.0, 1.0], (6, 1))

    module.process()

    np.testing.assert_allclose(module.beta, [34.0 / 3.0, 9.0, 10.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(module.beta_log[0], [10.0] * 6)
    np.testing.assert_array_equal(engine.memory.feedback_evidence, cached)
    assert engine.beta is module.beta


def test_uniform_pairings_make_partial_and_zero_feedback_beta_equivalent():
    # Uniform pairing with choice probabilities .7, .1 and .25 gives these
    # absolute partial/zero likelihoods, respectively.
    _, partial = _beta([0.1, 0.3, 0.25, 0.0, 0.0, 0.0])
    _, wrong_family = _beta([0.2, 0.6, 0.5, 0.0, 0.0, 0.0])

    partial.update_beta(np.asarray([0.5]), 1, 0.5)
    wrong_family.update_beta(np.asarray([0.5]), 1, 0.0)

    np.testing.assert_allclose(partial.beta[:3], [61.0 / 7.0, 114.0 / 11.0, 10.0])
    np.testing.assert_allclose(wrong_family.beta, partial.beta)


def test_hierarchical_beta_preserves_scope_and_bounds():
    engine, module = _beta(
        [0.5, 0.0, 0.25, 0.0, 0.0, 0.0],
        update_scope="executed_hypothesis",
        decrease_rate=1.0,
    )
    engine.transition = SimpleNamespace(
        persistent_execution_enabled=True,
        executed_hypothesis=1,
    )

    module.update_beta(np.asarray([0.5]), 1, 0.5)

    np.testing.assert_array_equal(module.beta, [10.0, 0.1, 10.0, 0.0, 0.0, 0.0])


def test_hierarchical_beta_rejects_nonzero_feedback_lapse():
    with pytest.raises(ValueError, match="hierarchical_feedback.*lapse"):
        _beta([0.25] * 6, probabilistic_feedback_lapse=0.1)


@pytest.mark.parametrize("feedback", [-1.0, 0.2, 2.0, np.nan])
def test_hierarchical_beta_rejects_non_task_feedback(feedback):
    _, module = _beta([0.25] * 6)
    with pytest.raises(ValueError, match="feedback"):
        module.update_beta(np.asarray([0.5]), 1, feedback)


@pytest.mark.parametrize("evidence", [None, [0.25] * 3, [np.nan] * 6, [1.1] * 6])
def test_hierarchical_beta_requires_valid_cached_evidence(evidence):
    engine, module = _beta([0.25] * 6)
    engine.memory.feedback_evidence = evidence
    with pytest.raises((ValueError, RuntimeError), match="feedback_evidence"):
        module.update_beta(np.asarray([0.5]), 1, 0.5)


def _search(kind, interpretation="full_success", *, lagged=True):
    engine = _ControlEngine()
    controller = {
        "event_after_correct": 0.2,
        "event_after_error": 0.6,
        "initial_event_probability": 0.2,
        "global_search": 0.3,
    }
    kwargs = {
        "capacity": 3,
        "init_hypotheses": [0, 1, 2],
        "module_seed": 17,
    }
    if interpretation is not None:
        kwargs["feedback_interpretation"] = interpretation
    if kind == "reactive":
        module = FeedbackReactiveHypothesisTransitionModule(
            engine, feedback_reactive_controller=controller, **kwargs
        )
    else:
        controller.update(
            accumulator_decay=0.5,
            accumulator_logit_gain=2.0 if kind == "active" else 0.0,
            global_search_failure_gain=0.5 if kind == "active" else 0.0,
            event_history_excludes_latest_error=lagged,
        )
        module = NestedFeedbackAccumulatorHypothesisTransitionModule(
            engine, nested_feedback_accumulator_controller=controller, **kwargs
        )
    return engine, module


def _advance(engine, module, feedback):
    engine.posterior = engine.prior.copy()
    module.record_outcome((np.asarray([0.5]), 1, feedback))
    module.process()


@pytest.mark.parametrize("kind", ["reactive", "inactive", "active"])
def test_full_success_counts_partial_feedback_as_failure_only_in_controls(kind):
    engine, module = _search(kind)
    module.process()
    module.record_outcome((np.asarray([0.5]), 1, 0.5))
    assert module.current_event_probability == pytest.approx(0.2)
    assert module.previous_feedback == 0.5
    engine.posterior = engine.prior.copy()

    module.process()

    assert module.current_event_probability == pytest.approx(0.6)
    assert module.previous_feedback == 0.5
    assert module.transition_log[-1]["previous_feedback"] == 0.5
    assert module.state_dict()["previous_feedback"] == 0.5
    assert module.failure_pressure == pytest.approx(0.5 if kind == "active" else 1.0)
    assert module.current_g == pytest.approx(0.475 if kind == "active" else 0.3)


@pytest.mark.parametrize("lagged, expected_event", [(True, 0.6), (False, 0.803049686686)])
def test_full_success_preserves_lagged_event_and_current_range_history(lagged, expected_event):
    engine, module = _search("active", lagged=lagged)
    module.process()

    _advance(engine, module, 0.5)

    assert module.current_event_probability == pytest.approx(expected_event)
    assert module.event_history_failure == pytest.approx(0.0 if lagged else 0.5)
    assert module.current_g == pytest.approx(0.475)

    _advance(engine, module, 1.0)

    assert module.failure_pressure == pytest.approx(0.25)
    assert module.mastery_evidence == pytest.approx(0.75)
    assert module.current_g == pytest.approx(0.3875)


@pytest.mark.parametrize("kind", ["reactive", "inactive", "active"])
def test_default_search_retains_graded_partial_feedback(kind):
    engine, module = _search(kind, interpretation=None)
    module.process()

    _advance(engine, module, 0.5)

    assert module.current_event_probability == pytest.approx(0.4)
    assert module.failure_pressure == pytest.approx(0.25 if kind == "active" else 0.5)


def test_full_success_zero_gain_nested_search_is_exact_reactive_boundary():
    reactive_engine, reactive = _search("reactive")
    nested_engine, nested = _search("inactive")
    reactive.process()
    nested.process()

    for feedback in (0.5, 1.0, 0.0, 0.5):
        _advance(reactive_engine, reactive, feedback)
        _advance(nested_engine, nested, feedback)
        np.testing.assert_array_equal(reactive.active, nested.active)
        np.testing.assert_array_equal(reactive_engine.prior, nested_engine.prior)
        assert nested.current_event_probability == reactive.current_event_probability
        assert nested.failure_pressure == reactive.failure_pressure
        assert nested.failure_pressure == float(feedback != 1.0)


@pytest.mark.parametrize("kind", ["reactive", "inactive", "active"])
def test_search_rejects_unknown_feedback_interpretation(kind):
    with pytest.raises(ValueError, match="feedback_interpretation"):
        _search(kind, interpretation="partial_success")
