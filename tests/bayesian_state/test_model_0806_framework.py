from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.Bayesian_state.simulation.data import TrialArrays
from src.Bayesian_state.simulation.execution import evaluate_state_model_run
from src.Bayesian_state.inference.dispatcher import run_inference_backend
from src.Bayesian_state.inference.backends.particle_filter import (
    _trace_ancestral_indices,
    run_state_model_particle_filter,
)
from src.Bayesian_state.inference.results import (
    InferenceResult,
    TrajectoryInferenceResult,
)
from src.Bayesian_state.inference.posterior_predictive import (
    run_conditioned_condition1_rollouts,
)
from src.Bayesian_state.model import ModelContext, ModuleRole, StateModel
from src.Bayesian_state.model.modules.beta import BetaModule
from src.Bayesian_state.model.modules.hypothesis_transition.dynamic_adaptive_control import (
    DynamicAdaptiveControlHypothesisTransitionModule,
)
from src.Bayesian_state.model.modules.hypothesis_transition.contracts import (
    HypothesisSelection,
)
from src.Bayesian_state.model.modules.hypothesis_transition.dynamic_discrete_strategy import (
    DynamicDiscreteStrategyHypothesisTransitionModule,
)
from src.Bayesian_state.model.modules.hypothesis_transition.fixed_strategy import (
    FixedFeedbackSwapHypothesisTransitionModule,
    FixedWorkspaceHypothesisTransitionModule,
    FixedStrategyHypothesisTransitionModule,
)
from src.Bayesian_state.model.modules.hypothesis_transition.feedback_reactive import (
    FeedbackReactiveHypothesisTransitionModule,
)
from src.Bayesian_state.model.readout import (
    apply_rule_commitment_choice_confidence,
    apply_strategy_conditioned_choice_confidence,
    read_oral_report,
    read_reaction_time,
)
from src.Bayesian_state.simulation.autonomous import (
    run_autonomous_category_learning,
)


class _TinyPartition:
    """Small partition sufficient for engine/particle integration tests."""

    VALID_DISTANCE_MODES = ("prototype",)

    def __init__(self, **kwargs):
        del kwargs
        self.length = 6
        self.n_cats = 2
        positions = np.arange(self.length, dtype=float)
        self._similarity = np.exp(-np.abs(positions[:, None] - positions[None, :]))

    @property
    def similarity_matrix(self) -> np.ndarray:
        return self._similarity

    def _probability(self, hypo: int, stimulus: np.ndarray) -> np.ndarray:
        x = float(np.asarray(stimulus, dtype=float).reshape(-1)[0])
        boundary = (int(hypo) + 1.0) / (self.length + 1.0)
        category_one = x <= boundary if int(hypo) % 2 == 0 else x > boundary
        return np.asarray([0.85, 0.15] if category_one else [0.15, 0.85])

    def get_category_assignment(
        self,
        hypo,
        stimulus,
        distance_mode="prototype",
        beta=1.0,
    ):
        del distance_mode, beta
        return int(np.argmax(self._probability(int(hypo), np.asarray(stimulus))))

    def get_category_probabilities(self, hypo, data, beta, distance_mode, **kwargs):
        del beta, distance_mode, kwargs
        return self._probability(int(hypo), np.asarray(data[0][0]))[:, None]

    def calc_likelihood(
        self,
        hypos,
        data,
        beta,
        distance_mode,
        normalized,
        **kwargs,
    ):
        del beta, distance_mode, normalized, kwargs
        choice = int(data[1][0]) - 1
        feedback = float(data[2][0])
        compatible = choice if feedback >= 0.5 else 1 - choice
        return np.asarray(
            [[self._probability(int(hypo), data[0][0])[compatible] for hypo in hypos]],
            dtype=float,
        )


class _TinyEngine:
    def __init__(self):
        self.set_size = 6
        self.prior = np.full(6, 1.0 / 6.0)
        self.posterior = None
        self.likelihood = np.ones(6)
        self.partition = _TinyPartition()
        self.modules = {}
        self.hypotheses_mask = None

    def get_module(self, role, *, required=False):
        names = {
            ModuleRole.HYPOTHESIS_TRANSITION: "hypo_transitions_mod",
            ModuleRole.BETA: "beta_mod",
            ModuleRole.PERCEPTION: "perception_mod",
            ModuleRole.MEMORY: "memory_mod",
        }
        module = self.modules.get(names[role])
        if module is None and required:
            raise ValueError(role)
        return module


def _engine_config() -> dict:
    return {
        "partition": {"class": _TinyPartition, "kwargs": {}},
        "inference": {
            "backend": "particle_filter",
            "particle_count": 8,
            "resample_threshold_fraction": 0.95,
        },
        "likelihood": {"distance_mode": "prototype"},
        "modules": {
            "perception_mod": {
                "class": "src.Bayesian_state.model.modules.perception.PerceptionModule",
                "kwargs": {"features": 1, "mean": [0.0], "std": [0.0]},
            },
            "beta_mod": {
                "class": "src.Bayesian_state.model.modules.beta.BetaModule",
                "kwargs": {
                    "beta_init": 5.0,
                    "decrease_rate": 0.0,
                    "correct_additive": 0.0,
                    "beta_update_mode": "probabilistic_feedback",
                    "use_prior_scaling": False,
                },
            },
            "hypo_transitions_mod": {
                "class": (
                    "src.Bayesian_state.model.modules."
                    "hypothesis_transition.dynamic_adaptive_control."
                    "DynamicAdaptiveControlHypothesisTransitionModule"
                ),
                "kwargs": {
                    "capacity": 2,
                    "m": 0.35,
                    "m_phi": 0.4,
                    "m_beta_surprise": 0.6,
                    "surprise_center": 0.8,
                    "surprise_scale": 0.5,
                    "g": 0.35,
                },
            },
            "memory_mod": {
                "class": "src.Bayesian_state.model.modules.memory.DualMemoryModule",
                "kwargs": {"gamma": 0.7, "w0": 0.4},
            },
        },
        "choice_readout": {
            "kwargs": {"method": "sharpened_expectation", "power": 2.0}
        },
        "output_noise": {
            "kwargs": {
                "enabled": True,
                "base_lapse": 0.05,
                "post_error_lapse": 0.0,
                "low_accuracy_lapse": 0.0,
                "latent_volatility_lapse": 0.0,
                "lapse_target": "uniform",
            }
        },
        "agenda": [
            "perception_mod",
            "hypo_transitions_mod",
            "memory_mod",
            "beta_mod",
        ],
    }


def test_adaptive_rate_uses_previous_feedback_and_restores_state():
    engine = _TinyEngine()
    module = DynamicAdaptiveControlHypothesisTransitionModule(
        engine,
        capacity=2,
        init_hypotheses=[0, 1],
        m=0.2,
        m_phi=0.5,
        m_beta_surprise=0.8,
        surprise_center=0.4,
        surprise_scale=0.5,
        g=1.0,
        module_seed=7,
    )

    module.process()
    predictive_prior = module.predictive_prior.copy()
    engine.likelihood = np.asarray([0.25, 0.75, 0.0, 0.0, 0.0, 0.0])
    raw_posterior = predictive_prior * engine.likelihood
    engine.posterior = raw_posterior / raw_posterior.sum()
    expected_surprise = -math.log(float(np.sum(predictive_prior * engine.likelihood)))
    expected_logit = module.baseline_logit + 0.8 * (
        (expected_surprise - 0.4) / 0.5
    )

    module.process()

    assert np.isclose(module.feedback_surprise, expected_surprise)
    assert np.isclose(module.control_logit, expected_logit)
    assert np.isclose(module.current_m, 1.0 / (1.0 + math.exp(-expected_logit)))
    assert module.active.size == 2
    assert np.isclose(engine.prior.sum(), 1.0)
    assert np.count_nonzero(engine.hypotheses_mask) == 2

    saved = module.state_dict()
    saved_active = module.active.copy()
    module.process()
    module.load_state_dict(saved)
    assert np.array_equal(module.active, saved_active)
    assert module.trial_index == saved["trial_index"]


def _failure_accumulator_module(
    capacity: int,
    seed: int,
    prior_reset_max_strength: float = 0.0,
    execution_switch_scale: float | None = None,
    misconception_capture: dict | None = None,
    rule_commitment: dict | None = None,
):
    engine = _TinyEngine()
    module = DynamicAdaptiveControlHypothesisTransitionModule(
        engine,
        capacity=capacity,
        init_hypotheses=list(range(capacity)),
        continuous_controller={
            "mode": "failure_accumulator_v2",
            "state": {
                "failure_decay": 0.60,
                "mastery_decay": 0.90,
            },
            "exploration": {
                "event_min": 0.05,
                "event_max": 0.65,
                "failure_threshold": 0.55,
                "failure_gain": 10.0,
                "uncertainty_weight": 0.0,
                "mastery_weight": 1.0,
                "surprise_weight": 0.0,
                "rise_rate": 0.80,
                "recovery_rate": 0.20,
            },
            "range": {
                "global_min": 0.05,
                "global_max": 0.80,
                "failure_threshold": 0.75,
                "failure_gain": 12.0,
                "uncertainty_weight": 0.0,
                "mastery_weight": 0.0,
                "surprise_weight": 0.0,
                "rise_rate": 0.80,
                "recovery_rate": 0.20,
            },
            "prior_reset": {
                "max_strength": prior_reset_max_strength,
            },
            **(
                {
                    "execution": {
                        "enabled": True,
                        "switch_scale": execution_switch_scale,
                        **({"misconception_capture": misconception_capture} if misconception_capture else {}),
                        **({"rule_commitment": rule_commitment} if rule_commitment else {}),
                    }
                }
                if execution_switch_scale is not None
                else {}
            ),
        },
        module_seed=seed,
    )
    module.process()
    return engine, module


def _advance_failure_controller(engine, module, feedback: float) -> tuple[float, float]:
    module.record_outcome((np.asarray([0.2]), 1, feedback))
    engine.likelihood = np.ones(engine.set_size, dtype=float)
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    event = module.transition_log[-1]
    return float(event["swap_probability"]), float(event["predictive_g"])


def test_strategy_conditioned_choice_confidence_is_symmetric_and_bounded():
    base = np.asarray([0.70, 0.30])
    confident, details = apply_strategy_conditioned_choice_confidence(
        base,
        mastery_evidence=0.90,
        failure_pressure=0.10,
        gain=2.0,
    )
    reversed_confident, reversed_details = (
        apply_strategy_conditioned_choice_confidence(
            base[::-1],
            mastery_evidence=0.90,
            failure_pressure=0.10,
            gain=2.0,
        )
    )
    np.testing.assert_allclose(confident, reversed_confident[::-1])
    assert confident[0] > base[0]
    assert details["strategy_confidence_signal"] == pytest.approx(0.64)
    assert details["strategy_choice_precision"] == pytest.approx(2.28)
    assert details == reversed_details

    unchanged, disabled = apply_strategy_conditioned_choice_confidence(
        base,
        mastery_evidence=np.nan,
        failure_pressure=np.nan,
        gain=0.0,
    )
    np.testing.assert_allclose(unchanged, base)
    assert disabled["strategy_choice_precision"] == 1.0


def test_rule_commitment_confidence_is_gated_and_symmetric():
    base = np.asarray([0.70, 0.30])
    inactive, inactive_details = apply_rule_commitment_choice_confidence(
        base,
        committed=False,
        choice_compatibility=np.nan,
        gain=4.0,
    )
    np.testing.assert_allclose(inactive, base)
    assert inactive_details["rule_commitment_choice_precision"] == 1.0

    committed, details = apply_rule_commitment_choice_confidence(
        base,
        committed=True,
        choice_compatibility=0.80,
        gain=2.0,
    )
    reversed_committed, reversed_details = (
        apply_rule_commitment_choice_confidence(
            base[::-1],
            committed=True,
            choice_compatibility=0.80,
            gain=2.0,
        )
    )
    assert committed[0] > base[0]
    np.testing.assert_allclose(committed, reversed_committed[::-1])
    assert details["rule_commitment_confidence_signal"] == pytest.approx(0.36)
    assert details["rule_commitment_choice_precision"] == pytest.approx(1.72)
    assert details == reversed_details


def test_guided_rule_commitment_enters_from_full_space_and_recovers():
    engine, module = _failure_accumulator_module(
        capacity=3,
        seed=47,
        execution_switch_scale=0.0,
        rule_commitment={
            "enabled": True,
            "choice_decay": 0.50,
            "failure_threshold": 0.60,
            "min_evidence_trials": 4,
            "min_choice_compatibility": 0.75,
            "min_runner_up_margin": 0.05,
            "entry_probability": 1.0,
            "min_dwell_trials": 3,
            "disconfirmation_decay": 0.50,
            "recovery_threshold": 1.0,
            "reentry_cooldown_trials": 2,
        },
    )
    module.executed_hypothesis = 0
    module.dynamic_controls = False
    module.current_event_probability = 0.0
    module.current_m = 0.0
    module.failure_pressure = 0.90
    module.choice_compatibility_observations = 8
    module.choice_compatibility[:] = 0.50
    module.choice_compatibility[4] = 0.78
    module.choice_compatibility[5] = 0.90
    assert 5 not in module.active

    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    entered = module.transition_log[-1]
    assert entered["rule_commitment_eligible"] is True
    assert entered["rule_commitment_entry_event"] is True
    assert entered["rule_commitment_active"] is True
    assert entered["rule_commitment_candidate_hypothesis"] == 5
    assert entered["rule_commitment_forced_newcomer"] is True
    assert entered["replacement_count"] == 1
    assert entered["executed_hypothesis"] == 5
    assert 5 in entered["active_after"]
    assert entered["execution_switch_event"] is True

    saved = module.state_dict()
    module.rule_commitment_active = False
    module.load_state_dict(saved)
    assert module.rule_commitment_active is True
    assert module.rule_commitment_age == 1
    assert module.executed_hypothesis == 5

    module.rule_commitment_disconfirmation = 2.0
    for expected_age in (2, 3):
        engine.posterior = np.asarray(engine.prior, dtype=float).copy()
        module.process()
        held = module.transition_log[-1]
        assert held["rule_commitment_active"] is True
        assert held["rule_commitment_exit_event"] is False
        assert held["rule_commitment_age"] == expected_age
        assert held["executed_hypothesis"] == 5

    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    recovered = module.transition_log[-1]
    assert recovered["rule_commitment_active"] is False
    assert recovered["rule_commitment_exit_event"] is True
    assert recovered["rule_commitment_cooldown_remaining"] == 2


def test_guided_rule_commitment_protects_an_already_active_target():
    engine, module = _failure_accumulator_module(
        capacity=3,
        seed=51,
        execution_switch_scale=1.0,
        rule_commitment={
            "enabled": True,
            "failure_threshold": 0.0,
            "min_evidence_trials": 1,
            "min_choice_compatibility": 0.75,
            "min_runner_up_margin": 0.05,
            "entry_probability": 1.0,
        },
    )
    module.executed_hypothesis = 0
    module.dynamic_controls = False
    module.current_event_probability = 1.0
    module.current_m = 1.0
    module.failure_pressure = 1.0
    module.choice_compatibility_observations = 8
    module.choice_compatibility[:] = 0.50
    module.choice_compatibility[2] = 0.70
    module.choice_compatibility[1] = 0.90
    assert 1 in module.active

    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    event = module.transition_log[-1]
    assert event["rule_commitment_entry_event"] is True
    assert event["rule_commitment_forced_newcomer"] is False
    assert event["executed_hypothesis"] == 1
    assert 1 not in event["dropped_hypotheses"]
    assert event["replacement_count"] <= 1


def test_rule_commitment_gate_leaves_smooth_learning_state_unchanged():
    engine, module = _failure_accumulator_module(
        capacity=3,
        seed=53,
        execution_switch_scale=0.0,
        rule_commitment={
            "enabled": True,
            "failure_threshold": 0.60,
            "min_evidence_trials": 4,
            "min_choice_compatibility": 0.75,
            "min_runner_up_margin": 0.05,
            "entry_probability": 1.0,
        },
    )
    module.dynamic_controls = False
    module.current_event_probability = 0.0
    module.current_m = 0.0
    module.failure_pressure = 0.20
    module.choice_compatibility_observations = 8
    module.choice_compatibility[:] = 0.50
    module.choice_compatibility[5] = 0.95
    active_before = module.active.copy()
    executed_before = int(module.executed_hypothesis)

    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    event = module.transition_log[-1]
    assert event["rule_commitment_eligible"] is False
    assert event["rule_commitment_entry_event"] is False
    assert event["rule_commitment_active"] is False
    assert event["replacement_count"] == 0
    np.testing.assert_array_equal(module.active, active_before)
    assert int(module.executed_hypothesis) == executed_before


def test_rule_commitment_requires_prior_mastery_and_releases_when_support_collapses():
    engine, module = _failure_accumulator_module(
        capacity=3,
        seed=59,
        execution_switch_scale=0.0,
        rule_commitment={
            "enabled": True,
            "failure_threshold": 0.60,
            "min_evidence_trials": 4,
            "min_prior_mastery": 0.70,
            "min_choice_compatibility": 0.80,
            "min_runner_up_margin": 0.10,
            "entry_probability": 1.0,
            "min_dwell_trials": 2,
            "min_hold_choice_compatibility": 0.60,
            "recovery_threshold": 100.0,
        },
    )
    module.dynamic_controls = False
    module.current_event_probability = 0.0
    module.current_m = 0.0
    module.failure_pressure = 0.90
    module.choice_compatibility_observations = 8
    module.choice_compatibility[:] = 0.50
    module.choice_compatibility[5] = 0.90
    module.peak_mastery_evidence = 0.69

    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    assert module.transition_log[-1]["rule_commitment_eligible"] is False

    module.peak_mastery_evidence = 0.80
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    entered = module.transition_log[-1]
    assert entered["rule_commitment_entry_event"] is True
    assert entered["rule_commitment_active"] is True
    assert entered["peak_mastery_evidence"] == pytest.approx(0.80)

    committed = int(module.executed_hypothesis)
    module.choice_compatibility[committed] = 0.55
    for expected_active in (True, False):
        engine.posterior = np.asarray(engine.prior, dtype=float).copy()
        module.process()
        assert module.transition_log[-1]["rule_commitment_active"] is expected_active
    assert module.transition_log[-1]["rule_commitment_exit_event"] is True


def test_failure_accumulator_v2_escalates_and_is_capacity_neutral():
    engine2, module2 = _failure_accumulator_module(capacity=2, seed=21)
    engine3, module3 = _failure_accumulator_module(capacity=3, seed=22)

    trajectories = []
    for engine, module in ((engine2, module2), (engine3, module3)):
        after_correct = _advance_failure_controller(engine, module, 1.0)
        after_one_error = _advance_failure_controller(engine, module, 0.0)
        after_two_errors = _advance_failure_controller(engine, module, 0.0)
        after_three_errors = _advance_failure_controller(engine, module, 0.0)
        after_recovery = _advance_failure_controller(engine, module, 1.0)
        trajectories.append(
            (
                after_correct,
                after_one_error,
                after_two_errors,
                after_three_errors,
                after_recovery,
            )
        )

        assert after_two_errors[0] > after_one_error[0]
        assert after_three_errors[1] > after_two_errors[1]
        assert after_recovery[0] < after_three_errors[0]
        assert module.failure_pressure < 0.784
        assert module.mastery_evidence > 0.36

        saved = module.state_dict()
        _advance_failure_controller(engine, module, 0.0)
        module.load_state_dict(saved)
        assert module.failure_pressure == pytest.approx(saved["failure_pressure"])
        assert module.mastery_evidence == pytest.approx(saved["mastery_evidence"])
        assert module.outcome_pending == saved["outcome_pending"]

    event_probabilities2 = [item[0] for item in trajectories[0]]
    event_probabilities3 = [item[0] for item in trajectories[1]]
    np.testing.assert_allclose(event_probabilities2, event_probabilities3)
    assert module2.current_m != pytest.approx(module3.current_m)


def test_persistent_execution_protects_switches_and_restores_state():
    engine, module = _failure_accumulator_module(
        capacity=3,
        seed=31,
        execution_switch_scale=1.0,
    )
    initial_executed = int(module.executed_hypothesis)
    assert module.transition_log[0]["execution_dwell_trials"] == 1

    # Isolate the execution contract: a certain search event replaces both
    # non-executed slots, while the protected overt rule stays available until
    # the explicit switch is completed.
    module.dynamic_controls = False
    module.current_event_probability = 1.0
    module.current_m = 1.0
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    event = module.transition_log[-1]

    assert event["swap_probability"] == pytest.approx(1.0)
    assert event["execution_search_slot_count"] == 2
    assert event["replacement_count"] == 2
    assert event["execution_switch_probability"] == pytest.approx(1.0)
    assert event["execution_switch_event"] is True
    assert initial_executed not in event["dropped_hypotheses"]
    assert initial_executed in event["active_after"]
    assert int(event["executed_hypothesis"]) != initial_executed
    assert int(event["executed_hypothesis"]) in event["active_after"]
    assert event["execution_dwell_trials"] == 1

    saved = module.state_dict()
    saved_active = module.active.copy()
    saved_executed = int(module.executed_hypothesis)
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    module.load_state_dict(saved)
    assert np.array_equal(module.active, saved_active)
    assert int(module.executed_hypothesis) == saved_executed
    assert module.execution_dwell_trials == saved["execution_dwell_trials"]
    assert module.execution_switch_count == saved["execution_switch_count"]


def test_misconception_capture_uses_past_choices_and_enforces_minimum_dwell():
    engine, module = _failure_accumulator_module(
        capacity=3,
        seed=37,
        execution_switch_scale=1.0,
        misconception_capture={
            "enabled": True,
            "choice_decay": 0.0,
            "failure_threshold": 0.0,
            "min_evidence_trials": 1,
            "min_advantage": 0.05,
            "min_dwell_trials": 3,
        },
    )
    module.executed_hypothesis = 0
    assert np.all(module.choice_compatibility == 0.5)

    # The completed choice updates the rule-compatibility trace, but cannot
    # alter the already logged trial-0 transition.
    module.record_outcome((np.asarray([0.10]), 2, 0.0))
    expected = np.asarray(
        [
            float(
                engine.partition.get_category_assignment(
                    hypothesis,
                    np.asarray([0.10]),
                )
                == 1
            )
            for hypothesis in range(engine.set_size)
        ]
    )
    np.testing.assert_allclose(module.choice_compatibility, expected)
    assert module.transition_log[0]["choice_compatibility_observations"] == 0

    # Force a search event.  The overt rule is captured by the active
    # alternative that best explains the previous choice.
    module.dynamic_controls = False
    module.failure_pressure = 1.0
    module.current_event_probability = 1.0
    module.current_m = 1.0
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    captured = module.transition_log[-1]

    assert captured["misconception_capture_search_bias"] is True
    assert captured["misconception_capture_eligible"] is True
    assert captured["misconception_capture_switch_event"] is True
    assert captured["execution_switch_event"] is True
    assert captured["misconception_capture_hold_remaining"] == 2
    assert captured["executed_hypothesis"] != 0
    assert expected[int(captured["executed_hypothesis"])] == 1.0

    saved = module.state_dict()
    captured_hypothesis = int(module.executed_hypothesis)
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.process()
    held = module.transition_log[-1]
    assert held["execution_switch_probability"] == 0.0
    assert held["execution_switch_event"] is False
    assert int(held["executed_hypothesis"]) == captured_hypothesis
    assert held["misconception_capture_hold_remaining"] == 1

    module.choice_compatibility[:] = 0.5
    module.misconception_capture_hold_remaining = 0
    module.load_state_dict(saved)
    np.testing.assert_allclose(
        module.choice_compatibility,
        saved["choice_compatibility"],
    )
    assert module.misconception_capture_hold_remaining == 2


def test_misconception_capture_requires_absolute_choice_compatibility():
    _, module = _failure_accumulator_module(
        capacity=3,
        seed=41,
        execution_switch_scale=1.0,
        misconception_capture={
            "enabled": True,
            "failure_threshold": 0.0,
            "min_evidence_trials": 1,
            "min_advantage": 0.05,
            "min_choice_compatibility": 0.70,
            "min_dwell_trials": 3,
        },
    )
    module.executed_hypothesis = 0
    module.failure_pressure = 1.0
    module.choice_compatibility_observations = 8
    module.choice_compatibility[:3] = [0.50, 0.60, 0.65]

    target, current, best, advantage, eligible = (
        module._misconception_capture_target(np.asarray([1, 2]))
    )
    assert target == 2
    assert current == pytest.approx(0.50)
    assert best == pytest.approx(0.65)
    assert advantage == pytest.approx(0.15)
    assert eligible is False

    module.choice_compatibility[2] = 0.75
    _, _, best, advantage, eligible = module._misconception_capture_target(
        np.asarray([1, 2])
    )
    assert best == pytest.approx(0.75)
    assert advantage == pytest.approx(0.25)
    assert eligible is True


def test_misconception_threshold_variants_preserve_future_execution_rng():
    modules = []
    for threshold in (0.0, 0.70):
        engine, module = _failure_accumulator_module(
            capacity=3,
            seed=43,
            execution_switch_scale=1.0,
            misconception_capture={
                "enabled": True,
                "failure_threshold": 0.0,
                "min_evidence_trials": 1,
                "min_advantage": 0.05,
                "min_choice_compatibility": threshold,
                "min_dwell_trials": 3,
            },
        )
        module.executed_hypothesis = 0
        module.dynamic_controls = False
        module.failure_pressure = 1.0
        module.choice_compatibility_observations = 8
        module.choice_compatibility[:] = 0.65
        module.choice_compatibility[0] = 0.50
        module.current_event_probability = 1.0
        module.current_m = 1.0
        engine.posterior = np.asarray(engine.prior, dtype=float).copy()
        modules.append(module)

    for module in modules:
        module.process()

    permissive_event = modules[0].transition_log[-1]
    strict_event = modules[1].transition_log[-1]
    assert permissive_event["misconception_capture_switch_event"] is True
    assert strict_event["misconception_capture_switch_event"] is False
    assert (
        permissive_event["misconception_capture_target_hypothesis"]
        == strict_event["misconception_capture_target_hypothesis"]
    )
    assert modules[0].execution_rng.random() == pytest.approx(
        modules[1].execution_rng.random()
    )


def test_beta_executed_hypothesis_scope_updates_only_overt_rule():
    engine = _TinyEngine()
    engine.hypotheses_mask = np.asarray([1, 1, 0, 0, 0, 0], dtype=float)
    engine.modules["hypo_transitions_mod"] = SimpleNamespace(
        persistent_execution_enabled=True,
        executed_hypothesis=1,
    )
    beta = BetaModule(
        engine,
        beta_init=5.0,
        beta_min=0.1,
        beta_max=25.0,
        decrease_rate=0.2,
        correct_additive=1.0,
        beta_update_mode="probabilistic_feedback",
        update_scope="executed_hypothesis",
        use_prior_scaling=False,
    )
    before = beta.beta.copy()

    beta.update_beta(
        stimulus=np.asarray([0.1]),
        choice=1,
        feedback=1.0,
        active_mask=engine.hypotheses_mask,
    )

    assert beta.beta[0] == pytest.approx(before[0])
    assert beta.beta[1] != pytest.approx(before[1])
    np.testing.assert_allclose(beta.beta[2:], 0.0)

    with pytest.raises(ValueError, match="update_scope"):
        BetaModule(_TinyEngine(), update_scope="not_a_scope")


def test_beta_increase_rate_is_exact_legacy_reparameterization():
    legacy = BetaModule(
        _TinyEngine(),
        beta_init=5.0,
        beta_min=0.1,
        beta_max=25.0,
        decrease_rate=0.15,
        correct_additive=1.0,
        beta_update_mode="probabilistic_feedback",
        use_prior_scaling=False,
    )
    canonical = BetaModule(
        _TinyEngine(),
        beta_init=5.0,
        beta_min=0.1,
        beta_max=25.0,
        decrease_rate=0.15,
        increase_rate=0.04,
        beta_update_mode="probabilistic_feedback",
        use_prior_scaling=False,
    )
    active = np.asarray([1, 1, 1, 0, 0, 0], dtype=float)

    for stimulus, choice, feedback in (
        (np.asarray([0.10]), 1, 1.0),
        (np.asarray([0.80]), 1, 0.0),
        (np.asarray([0.35]), 2, 1.0),
        (np.asarray([0.60]), 2, 0.0),
    ):
        legacy.update_beta(stimulus, choice, feedback, active)
        canonical.update_beta(stimulus, choice, feedback, active)
        np.testing.assert_allclose(canonical.beta, legacy.beta, atol=0.0, rtol=0.0)

    assert canonical.increase_rate == pytest.approx(0.04)
    assert canonical.correct_additive == pytest.approx(1.0)
    assert canonical.increase_parameterization == "increase_rate"
    assert legacy.increase_parameterization == "correct_additive_legacy"


def test_beta_increase_parameterization_rejects_ambiguous_or_invalid_rates():
    with pytest.raises(ValueError, match="increase_rate or legacy"):
        BetaModule(
            _TinyEngine(),
            beta_max=25.0,
            increase_rate=0.04,
            correct_additive=1.0,
        )
    for invalid in (-0.01, 1.01, np.inf):
        with pytest.raises(ValueError, match="increase_rate"):
            BetaModule(_TinyEngine(), increase_rate=invalid)


def test_failure_accumulator_v2_global_reset_broadens_newcomer_prior_only():
    _, module = _failure_accumulator_module(
        capacity=2,
        seed=23,
        prior_reset_max_strength=0.35,
    )
    posterior = np.asarray([0.9, 0.1, 0.0, 0.0, 0.0, 0.0])
    module._pending_transition = {"posterior": posterior}
    module.current_prior_reset_strength = 0.35
    selection = HypothesisSelection.from_active_sets(
        [0, 1],
        [0, 2],
        replacement_pairs=((1, 2),),
    )

    reset_prior = module.assign_prior(None, selection)

    np.testing.assert_allclose(reset_prior, [0.76, 0.0, 0.24, 0.0, 0.0, 0.0])
    assert reset_prior[2] > posterior[1]
    assert module._pending_transition["prior_reset_strength"] == pytest.approx(0.35)
    assert module._pending_transition["prior_reset_mass_shift"] == pytest.approx(0.14)

    module._pending_transition = {"posterior": posterior}
    no_replacement = HypothesisSelection.from_active_sets([0, 1], [0, 1])
    unchanged_prior = module.assign_prior(None, no_replacement)
    np.testing.assert_allclose(unchanged_prior, posterior)
    assert module._pending_transition["prior_reset_strength"] == 0.0


def test_dynamic_search_range_uses_previous_feedback_and_restores_state():
    engine = _TinyEngine()
    module = DynamicAdaptiveControlHypothesisTransitionModule(
        engine,
        capacity=2,
        init_hypotheses=[0, 1],
        m=0.0,
        g=0.35,
        range_controller={
            "g_phi": 0.5,
            "g_beta_surprise": 0.6,
            "g_beta_uncertainty": 0.0,
            "g_surprise_center": 0.4,
            "g_surprise_scale": 0.5,
        },
        module_seed=9,
    )

    module.process()
    predictive_prior = module.predictive_prior.copy()
    engine.likelihood = np.asarray([0.25, 0.75, 0.0, 0.0, 0.0, 0.0])
    posterior = predictive_prior * engine.likelihood
    engine.posterior = posterior / posterior.sum()
    surprise = -math.log(float(np.sum(predictive_prior * engine.likelihood)))
    expected_logit = module.g_baseline_logit + 0.6 * ((surprise - 0.4) / 0.5)

    module.process()

    assert np.isclose(module.g_control_logit, expected_logit)
    assert np.isclose(module.current_g, 1.0 / (1.0 + math.exp(-expected_logit)))
    assert np.isclose(module.transition_log[-1]["predictive_g"], module.current_g)
    saved = module.state_dict()
    module.g_control_logit = -20.0
    module.current_g = 0.0
    module.load_state_dict(saved)
    assert np.isclose(module.g_control_logit, expected_logit)
    assert np.isclose(module.current_g, saved["current_g"])


def test_static_bounded_workspace_uses_common_transition_contract():
    engine = _TinyEngine()
    module = FixedWorkspaceHypothesisTransitionModule(
        engine,
        capacity=2,
        init_hypotheses=[0, 1],
        m=0.0,
        g=1.0,
        module_seed=4,
    )

    module.process()

    assert module.active.size == 2
    assert np.isclose(np.sum(engine.prior), 1.0)
    assert module.last_transition_result is not None
    assert module.last_transition_result.diagnostics["strategy_mode"] == "static"


def test_feedback_reactive_workspace_uses_only_previous_outcome_and_restores_state():
    engine = _TinyEngine()
    module = FeedbackReactiveHypothesisTransitionModule(
        engine,
        capacity=2,
        init_hypotheses=[0, 1],
        feedback_reactive_controller={
            "event_after_correct": 0.10,
            "event_after_error": 0.70,
            "initial_event_probability": 0.10,
            "global_search": 0.25,
        },
        module_seed=17,
    )

    module.process()
    assert module.transition_log[-1]["swap_probability"] == pytest.approx(0.0)

    module.record_outcome((np.asarray([0.2]), 1, 1.0))
    module.process()
    assert module.transition_log[-1]["previous_feedback"] == pytest.approx(1.0)
    assert module.transition_log[-1]["swap_probability"] == pytest.approx(0.10)
    assert module.transition_log[-1]["predictive_g"] == pytest.approx(0.25)

    module.record_outcome((np.asarray([0.4]), 1, 0.0))
    saved = module.state_dict()
    module.process()
    assert module.transition_log[-1]["previous_feedback"] == pytest.approx(0.0)
    assert module.transition_log[-1]["swap_probability"] == pytest.approx(0.70)

    module.load_state_dict(saved)
    assert module.controller_mode == "feedback_reactive_v1"
    assert module.outcome_pending is True
    module.process()
    assert module.transition_log[-1]["swap_probability"] == pytest.approx(0.70)


def test_feedback_reactive_workspace_requires_more_exploration_after_error():
    with pytest.raises(ValueError, match="event_after_error"):
        FeedbackReactiveHypothesisTransitionModule(
            _TinyEngine(),
            capacity=2,
            feedback_reactive_controller={
                "event_after_correct": 0.60,
                "event_after_error": 0.20,
                "global_search": 0.30,
            },
        )


def test_static_strategy_uses_common_selection_then_prior_contract():
    engine = _TinyEngine()
    engine.posterior = np.asarray([0.7, 0.3, 0.0, 0.0, 0.0, 0.0])
    engine.observation = (np.asarray([0.2]), 1, 1.0)
    module = FixedStrategyHypothesisTransitionModule(
        engine,
        module_seed=11,
        selection_strategy={
            "method": "strategy_chain",
            "init_num": 2,
            "init_hypotheses": [0, 1],
            "max_active_hypotheses": 2,
            "strategies": [
                {
                    "label": "retain",
                    "amount": "fixed",
                    "value": 1,
                    "method": "top_posterior",
                    "pool": "active",
                },
                {
                    "label": "explore",
                    "amount": "fixed",
                    "value": 1,
                    "method": "random",
                    "pool": "inactive",
                },
            ],
        },
        prior_assignment={
            "method": "conservative_carryover",
            "newcomer_mass": 0.1,
        },
    )

    module.process()

    result = module.last_transition_result
    assert result is not None
    assert result.diagnostics["strategy_mode"] == "static"
    assert result.selection.active_after.size == 2
    assert result.selection.newcomers.size == 1
    assert np.isclose(np.sum(result.prior_after), 1.0)
    assert np.all(result.prior_after[engine.hypotheses_mask == 0.0] == 0.0)


def test_dynamic_discrete_has_explicit_trial_level_strategy_state():
    engine = _TinyEngine()
    engine.posterior = np.asarray([0.65, 0.35, 0.0, 0.0, 0.0, 0.0])
    engine.observation = (np.asarray([0.2]), 1, 1.0)
    module = DynamicDiscreteStrategyHypothesisTransitionModule(
        engine,
        module_seed=12,
        init_num=2,
        init_hypotheses=[0, 1],
        max_active_hypotheses=2,
        state_controller={
            "method": "feedback_gated_softmax",
            "features": {
                "recent_accuracy_window": 2,
                "accuracy_delta_window": 2,
                "padding": "chance",
                "feedback_mode": "exact",
                "trial_progress_scale": 10,
            },
            "activation": {"temperature": 1.0},
            "states": [
                {
                    "id": "only_state",
                    "activation": {"bias": 0.0},
                    "strategies": [
                        {
                            "amount": "fixed",
                            "value": 1,
                            "method": "top_posterior",
                            "pool": "active",
                        },
                        {
                            "amount": "fixed",
                            "value": 1,
                            "method": "random",
                            "pool": "inactive",
                        },
                    ],
                    "post_to_prior": {
                        "method": "conservative_carryover",
                        "newcomer_mass": 0.1,
                    },
                }
            ],
        },
    )

    module.process()

    assert module.strategy_counts_log[-1]["selected_state"] == "only_state"
    assert module.strategy_counts_log[-1]["strategy_mode"] == "dynamic_discrete"
    assert module.last_transition_result is not None
    assert (
        module.last_transition_result.selection.diagnostics["selected_state"]
        == "only_state"
    )


def test_transition_mode_boundaries_fail_fast():
    with pytest.raises(ValueError, match="does not accept a state controller"):
        FixedStrategyHypothesisTransitionModule(
            _TinyEngine(),
            strategies=[
                {
                    "amount": "fixed",
                    "value": 1,
                    "method": "random",
                    "pool": "active",
                }
            ],
            state_controller={"states": []},
        )

    with pytest.raises(ValueError, match="requires at least one trial-varying control"):
        DynamicAdaptiveControlHypothesisTransitionModule(
            _TinyEngine(),
            capacity=2,
            m=0.2,
            g=0.35,
        )


def test_feedback_swap_is_a_static_reactive_strategy():
    engine = _TinyEngine()
    engine.observation = (np.asarray([0.2]), 1, 0.0)
    module = FixedFeedbackSwapHypothesisTransitionModule(
        engine,
        capacity=2,
        theta=1.0,
        init_hypotheses=[0, 1],
        module_seed=15,
    )

    module.process()
    assert not module.transition_log[-1]["swap_event"]
    module.record_outcome(engine.observation)

    engine.posterior = np.asarray([0.9, 0.1, 0.0, 0.0, 0.0, 0.0])
    engine.observation = (np.asarray([0.3]), 1, 1.0)
    module.process()
    module.record_outcome(engine.observation)

    event = module.transition_log[-1]
    assert event["strategy_mode"] == "static"
    assert event["swap_event"]
    assert event["dropped_hypothesis"] == 1
    assert event["new_hypothesis"] in {2, 3, 4, 5}
    assert np.isclose(engine.prior[event["new_hypothesis"]], 0.5)
    assert module.last_transition_result is not None


def test_model_structure_module_class_paths_are_importable():
    for config_path in sorted(Path("configs/model_struct").glob("*.yaml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for module_config in (payload.get("modules") or {}).values():
            class_path = module_config.get("class")
            if not isinstance(class_path, str):
                continue
            module_name, class_name = class_path.rsplit(".", 1)
            loaded_module = importlib.import_module(module_name)
            assert hasattr(loaded_module, class_name), (
                f"{config_path} references missing class {class_path}"
            )


def test_all_hypothesis_transition_config_paths_are_current_and_resolvable():
    old_fragments = (
        "hypo_transition.core",
        "hypo_transition.static_strategy.",
        "hypo_transition.profile_dynamic",
        "hypo_transition.continuous_dynamic",
        "hypo_transition.finite_workspace",
        "hypo_transition.strategy_chain",
        "modules.hypo_transitions.",
        "modules.static_hypo_transition.",
        "modules.profile_dynamic_hypo_transition.",
        "modules.continuous_dynamic_hypo_transition.",
        "modules/finite_workspace_transition",
        "modules/hypo_transition_strategies",
    )

    def visit(value, config_path: Path):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "class" and isinstance(child, str) and (
                    "modules.hypothesis_transition." in child
                ):
                    module_name, class_name = child.rsplit(".", 1)
                    loaded_module = importlib.import_module(module_name)
                    assert hasattr(loaded_module, class_name), (
                        f"{config_path} references missing H class {child}"
                    )
                if key == "path" and isinstance(child, str) and (
                    "candidates/hypothesis_transition" in child
                ):
                    resolved_path = (config_path.parent / child).resolve()
                    assert resolved_path.is_file(), (
                        f"{config_path} references missing H candidate {child}"
                    )
                visit(child, config_path)
        elif isinstance(value, list):
            for child in value:
                visit(child, config_path)

    for root in (Path("configs"), Path("configs_exp4"), Path("configs_exp5")):
        for config_path in sorted(root.rglob("*.yaml")):
            raw = config_path.read_text(encoding="utf-8")
            assert not any(fragment in raw for fragment in old_fragments), config_path
            if "state_controller" in raw:
                assert "DynamicDiscreteStrategyHypothesisTransitionModule" in raw, config_path
            visit(yaml.safe_load(raw) or {}, config_path)


@pytest.mark.parametrize(
    "config_name, expected_mode",
    [
        ("pmh_model_cond1.yaml", "static"),
        ("pmh_model_cond1_active_set.yaml", "static"),
        ("pmh_model_cond1_v13.yaml", "dynamic_discrete"),
        ("pmh_model_cond1_v14.yaml", "dynamic_discrete"),
        ("pmh_model_cond1_0806.yaml", "dynamic_continuous"),
    ],
)
def test_canonical_transition_configs_run_the_declared_mode(
    config_name: str,
    expected_mode: str,
):
    config_path = Path("configs/model_struct") / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    transition_config = payload["modules"]["hypo_transitions_mod"]
    module_name, class_name = transition_config["class"].rsplit(".", 1)
    transition_class = getattr(importlib.import_module(module_name), class_name)
    engine = _TinyEngine()
    engine.observation = (np.asarray([0.2]), 1, 1.0)
    transition = transition_class(engine, **transition_config.get("kwargs", {}))

    transition.process()

    assert transition.strategy_mode == expected_mode
    assert transition.last_transition_result is not None
    assert np.isclose(np.sum(engine.prior), 1.0)


def test_0806_hyper_candidates_bind_static_and_dynamic_classes_to_controls():
    config_path = Path("configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_payload = yaml.safe_load(
        Path("configs/model_struct/pmh_model_cond1_0806.yaml").read_text(
            encoding="utf-8"
        )
    )
    base_kwargs = model_payload["modules"]["hypo_transitions_mod"]["kwargs"]
    candidates = payload["hyperparam_space"]["__profile_candidate__"]["values"]
    class_key = "engine.modules.hypo_transitions_mod.class"
    rate_key = "engine.modules.hypo_transitions_mod.kwargs.rate_controller"
    range_key = "engine.modules.hypo_transitions_mod.kwargs.range_controller"

    assert len(candidates) == 28
    static_candidates = [
        item for item in candidates if "FixedWorkspace" in item[class_key]
    ]
    assert len(static_candidates) == 1
    static_candidate = static_candidates[0]
    assert static_candidate[rate_key]["m_beta_surprise"] == 0.0
    assert static_candidate[rate_key]["m_beta_uncertainty"] == 0.0
    assert static_candidate[range_key]["g_beta_surprise"] == 0.0
    assert static_candidate[range_key]["g_beta_uncertainty"] == 0.0

    for candidate in candidates:
        has_dynamic_control = any(
            float(candidate[rate_key][key]) > 0.0
            for key in ("m_beta_surprise", "m_beta_uncertainty")
        ) or any(
            float(candidate[range_key][key]) > 0.0
            for key in ("g_beta_surprise", "g_beta_uncertainty")
        )
        assert has_dynamic_control == (
            "DynamicAdaptiveControlHypothesisTransitionModule" in candidate[class_key]
        )

        module_name, class_name = candidate[class_key].rsplit(".", 1)
        transition_class = getattr(importlib.import_module(module_name), class_name)
        transition_kwargs = dict(base_kwargs)
        transition_kwargs["rate_controller"] = candidate[rate_key]
        transition_kwargs["range_controller"] = candidate[range_key]
        engine = _TinyEngine()
        engine.observation = (np.asarray([0.2]), 1, 1.0)
        transition = transition_class(engine, **transition_kwargs)
        transition.process()
        assert np.isclose(np.sum(engine.prior), 1.0)


def test_standard_runner_dispatches_model_0806_to_particle_backend():
    n_trials = 18
    stimulus = np.linspace(0.05, 0.95, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    choices = np.where(np.arange(n_trials) % 4 == 0, 3 - categories, categories)
    feedback = (choices == categories).astype(float)
    target_probs = np.eye(2, dtype=float)[categories - 1]
    arrays = TrialArrays(
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        categories=categories,
        target_probs=target_probs,
    )

    result = evaluate_state_model_run(
        subject_id=1,
        condition=1,
        arrays=arrays,
        params={},
        engine_config_template=_engine_config(),
        processed_data_dir=Path("."),
        window_size=4,
        keep_logs=True,
        prediction_mode="prior_t",
        selection_prediction_mode="prior_t",
        loss_metric="choice_brier",
        trajectory_seed=20260806,
    )

    metrics = result.metrics_by_mode["prior_t"]
    probabilities = np.asarray(metrics["pred_category_probs"])
    assert probabilities.shape == (n_trials, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert not bool(np.asarray(metrics["valid_trial_mask"])[0])
    assert np.isfinite(result.mean_error)
    assert result.state_log is not None
    assert result.posterior_log is None
    assert np.asarray(result.prior_log).shape == (n_trials, 6)
    assert np.asarray(result.state_log["transition_rate"]).shape == (n_trials,)
    assert np.asarray(result.state_log["replacement_fraction"]).shape == (n_trials,)
    assert np.asarray(result.state_log["newcomer_distance"]).shape == (n_trials,)
    assert result.transition_counts is not None
    assert len(result.transition_counts) == n_trials
    assert np.asarray(result.state_log["search_range"]).shape == (n_trials,)
    strategy = np.column_stack(
        [
            result.state_log["predictive_strategy_exploit"],
            result.state_log["predictive_strategy_local_explore"],
            result.state_log["predictive_strategy_global_explore"],
        ]
    )
    assert strategy.shape == (n_trials, 3)
    np.testing.assert_allclose(strategy.sum(axis=1), 1.0)
    np.testing.assert_allclose(strategy[0], [1.0, 0.0, 0.0])
    assert np.asarray(result.state_log["predictive_swap_probability"]).shape == (
        n_trials,
    )
    assert result.state_log["predictive_swap_probability"][0] == 0.0


def test_particle_filter_v2_uses_only_previous_feedback_for_controller_state():
    feedback = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    stimulus = np.linspace(0.1, 0.9, feedback.size)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    choices = np.where(feedback > 0.5, categories, 3 - categories)
    arrays = TrialArrays(
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        categories=categories,
        target_probs=np.eye(2, dtype=float)[categories - 1],
    )
    config = _engine_config()
    config["modules"]["hypo_transitions_mod"]["kwargs"] = {
        "capacity": 2,
        "continuous_controller": {
            "mode": "failure_accumulator_v2",
            "state": {
                "failure_decay": 0.60,
                "mastery_decay": 0.90,
            },
            "exploration": {
                "event_min": 0.05,
                "event_max": 0.65,
                "failure_threshold": 0.55,
                "failure_gain": 10.0,
                "rise_rate": 0.80,
                "recovery_rate": 0.20,
            },
            "range": {
                "global_min": 0.05,
                "global_max": 0.80,
                "failure_threshold": 0.75,
                "failure_gain": 12.0,
                "rise_rate": 0.80,
                "recovery_rate": 0.20,
            },
        },
    }
    config["choice_readout"]["kwargs"]["strategy_confidence_gain"] = 2.0

    result = evaluate_state_model_run(
        subject_id=1,
        condition=1,
        arrays=arrays,
        params={},
        engine_config_template=config,
        processed_data_dir=Path("."),
        window_size=2,
        keep_logs=True,
        prediction_mode="prior_t",
        selection_prediction_mode="prior_t",
        loss_metric="choice_brier",
        trajectory_seed=20260810,
    )

    state = result.state_log
    assert state is not None
    np.testing.assert_allclose(
        state["predictive_failure_pressure"],
        [0.0, 0.0, 0.4, 0.64, 0.784, 0.4704],
    )
    np.testing.assert_allclose(
        state["predictive_mastery_evidence"],
        [0.5, 0.55, 0.495, 0.4455, 0.40095, 0.460855],
    )
    expected_signal = np.square(
        np.maximum(
            np.asarray(state["predictive_mastery_evidence"])
            - np.asarray(state["predictive_failure_pressure"]),
            0.0,
        )
    )
    np.testing.assert_allclose(
        state["predictive_choice_confidence_signal"],
        expected_signal,
    )
    np.testing.assert_allclose(
        state["predictive_strategy_choice_precision"],
        1.0 + 2.0 * expected_signal,
    )
    exploration = 1.0 - np.asarray(state["predictive_strategy_exploit"])
    assert exploration[3] > exploration[2]
    assert exploration[4] > exploration[3]
    assert exploration[5] < exploration[4]
    assert state["predictive_search_range"][4] > state["predictive_search_range"][3]
    assert np.asarray(state["predictive_prior_reset_strength"]).shape == (
        feedback.size,
    )
    assert np.all(np.asarray(state["predictive_prior_reset_strength"]) == 0.0)


def test_particle_filter_rule_commitment_and_confidence_are_marginalized():
    n_trials = 6
    stimulus = np.full((n_trials, 1), 0.10, dtype=float)
    choices = np.full(n_trials, 2, dtype=int)
    feedback = np.zeros(n_trials, dtype=float)
    config = _engine_config()
    config["inference"]["choice_transmission_audit"] = True
    config["modules"]["beta_mod"]["kwargs"]["update_scope"] = (
        "executed_hypothesis"
    )
    config["modules"]["hypo_transitions_mod"]["kwargs"] = {
        "capacity": 2,
        "continuous_controller": {
            "mode": "failure_accumulator_v2",
            "state": {"failure_decay": 0.60, "mastery_decay": 0.90},
            "exploration": {
                "event_min": 0.05,
                "event_max": 0.65,
                "failure_threshold": 0.55,
                "failure_gain": 10.0,
                "rise_rate": 0.80,
                "recovery_rate": 0.20,
            },
            "range": {
                "global_min": 0.05,
                "global_max": 0.80,
                "failure_threshold": 0.75,
                "failure_gain": 12.0,
                "rise_rate": 0.80,
                "recovery_rate": 0.20,
            },
            "execution": {
                "enabled": True,
                "switch_scale": 0.0,
                "rule_commitment": {
                    "enabled": True,
                    "choice_decay": 0.0,
                    "failure_threshold": 0.0,
                    "min_evidence_trials": 1,
                    "min_choice_compatibility": 0.75,
                    "min_runner_up_margin": 0.0,
                    "entry_probability": 1.0,
                    "min_dwell_trials": 3,
                    "recovery_threshold": 100.0,
                    "reentry_cooldown_trials": 2,
                },
            },
        },
    }

    result = run_inference_backend(
        engine_config=config,
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        inference_seed=20260812,
        choice_readout_power=2.0,
        strategy_confidence_gain=0.0,
        rule_commitment_confidence_gain=2.0,
        output_lapse=0.02,
        processed_data_dir=Path("."),
    )
    commitment = np.asarray(
        result.latent_summaries["predictive_rule_commitment_probability"],
        dtype=float,
    )
    entry = np.asarray(
        result.latent_summaries[
            "predictive_rule_commitment_entry_event_probability"
        ],
        dtype=float,
    )
    precision = np.asarray(
        result.latent_summaries[
            "predictive_rule_commitment_choice_precision"
        ],
        dtype=float,
    )
    peak_mastery = np.asarray(
        result.latent_summaries["predictive_peak_mastery_evidence"],
        dtype=float,
    )
    assert commitment.shape == entry.shape == precision.shape == peak_mastery.shape == (
        n_trials,
    )
    assert np.all(np.diff(peak_mastery) >= -1e-12)
    assert commitment[0] == 0.0
    assert commitment[1] == pytest.approx(1.0)
    assert entry[1] == pytest.approx(1.0)
    assert precision[0] == pytest.approx(1.0)
    assert precision[1] > 1.0
    np.testing.assert_allclose(
        result.observation_probabilities["prior_t"].sum(axis=1),
        1.0,
    )


def test_particle_filter_persistent_execution_survives_resampling():
    n_trials = 12
    stimulus = np.linspace(0.1, 0.9, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    feedback = np.asarray([1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    choices = np.where(feedback > 0.5, categories, 3 - categories)
    config = _engine_config()
    config["inference"]["choice_transmission_audit"] = True
    config["modules"]["beta_mod"]["kwargs"].update(
        {
            "decrease_rate": 0.2,
            "correct_additive": 1.0,
            "update_scope": "executed_hypothesis",
        }
    )
    config["modules"]["hypo_transitions_mod"]["kwargs"] = {
        "capacity": 2,
        "continuous_controller": {
            "mode": "failure_accumulator_v2",
            "state": {
                "failure_decay": 0.60,
                "mastery_decay": 0.90,
            },
            "exploration": {
                "event_min": 0.50,
                "event_max": 0.90,
                "failure_threshold": 0.30,
                "failure_gain": 10.0,
                "rise_rate": 0.80,
                "recovery_rate": 0.20,
            },
            "range": {
                "global_min": 0.05,
                "global_max": 0.80,
                "failure_threshold": 0.50,
                "failure_gain": 12.0,
                "rise_rate": 0.80,
                "recovery_rate": 0.20,
            },
            "execution": {
                "enabled": True,
                "switch_scale": 1.0,
                "misconception_capture": {
                    "enabled": True,
                    "choice_decay": 0.50,
                    "failure_threshold": 0.0,
                    "min_evidence_trials": 1,
                    "min_advantage": 0.0,
                    "min_dwell_trials": 3,
                },
            },
        },
    }

    result = run_inference_backend(
        engine_config=config,
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        inference_seed=20260811,
        choice_readout_power=2.0,
        strategy_confidence_gain=0.0,
        output_lapse=0.02,
        processed_data_dir=Path("."),
    )
    counterfactual = run_state_model_particle_filter(
        engine_config=config,
        subject_id=1,
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        particle_count=8,
        choice_readout_power=2.0,
        strategy_confidence_gain=0.0,
        output_lapse=0.02,
        filter_seed=20260811,
        resample_threshold_fraction=0.95,
        choice_transmission_audit=True,
        choice_transmission_counterfactual_gain=2.0,
        processed_data_dir=Path("."),
    )
    np.testing.assert_allclose(
        counterfactual.observation_probabilities["prior_t"],
        result.observation_probabilities["prior_t"],
        rtol=0.0,
        atol=0.0,
    )

    executed = np.asarray(
        result.state_probabilities["executed_probability"], dtype=float
    )
    filtered_executed = np.asarray(
        result.state_probabilities["filtered_executed_probability"], dtype=float
    )
    assert executed.shape == filtered_executed.shape == (n_trials, 6)
    np.testing.assert_allclose(executed.sum(axis=1), 1.0)
    np.testing.assert_allclose(filtered_executed.sum(axis=1), 1.0)
    switch_probability = np.asarray(
        result.latent_summaries["predictive_execution_switch_probability"],
        dtype=float,
    )
    swap_probability = np.asarray(
        result.latent_summaries["predictive_swap_probability"], dtype=float
    )
    assert np.all(switch_probability <= swap_probability + 1e-12)
    assert np.any(switch_probability < swap_probability - 1e-12)
    executed_beta = np.asarray(
        result.latent_summaries["predictive_executed_beta"],
        dtype=float,
    )
    filtered_beta = np.asarray(
        result.latent_summaries["filtered_executed_beta"],
        dtype=float,
    )
    assert executed_beta.shape == filtered_beta.shape == (n_trials,)
    assert np.all(np.isfinite(executed_beta))
    assert executed_beta[0] == pytest.approx(5.0)
    assert np.any(np.abs(np.diff(executed_beta)) > 1e-9)
    capture_hold = np.asarray(
        result.latent_summaries[
            "predictive_misconception_capture_hold_probability"
        ],
        dtype=float,
    )
    capture_switch = np.asarray(
        result.latent_summaries[
            "predictive_misconception_capture_switch_event_probability"
        ],
        dtype=float,
    )
    assert capture_hold.shape == capture_switch.shape == (n_trials,)
    assert np.all((capture_hold >= 0.0) & (capture_hold <= 1.0))
    assert np.all((capture_switch >= 0.0) & (capture_switch <= 1.0))
    assert np.any(capture_hold > 0.0)

    persistent = np.asarray(
        result.observation_probabilities[
            "audit_persistent_execution_no_lapse"
        ],
        dtype=float,
    )
    assert persistent.shape == (n_trials, 2)
    np.testing.assert_allclose(persistent.sum(axis=1), 1.0)
    persistent_without_strategy = np.asarray(
        result.observation_probabilities[
            "audit_persistent_execution_no_strategy_no_lapse"
        ],
        dtype=float,
    )
    assert persistent_without_strategy.shape == (n_trials, 2)
    np.testing.assert_allclose(
        persistent_without_strategy.sum(axis=1),
        1.0,
    )
    np.testing.assert_allclose(persistent, persistent_without_strategy)
    counterfactual_strategy = np.asarray(
        counterfactual.observation_probabilities[
            "audit_persistent_execution_counterfactual_strategy_no_lapse"
        ],
        dtype=float,
    )
    assert counterfactual_strategy.shape == (n_trials, 2)
    np.testing.assert_allclose(counterfactual_strategy.sum(axis=1), 1.0)
    assert np.any(
        np.abs(counterfactual_strategy - persistent_without_strategy) > 1e-9
    )

    ancestral = result.artifacts["audit_ancestral_paths"]
    executed_paths = np.asarray(ancestral["executed_hypothesis"], dtype=int)
    switch_paths = np.asarray(ancestral["execution_switch_event"], dtype=float)
    swap_paths = np.asarray(ancestral["swap_event"], dtype=float)
    dwell_paths = np.asarray(ancestral["execution_dwell_trials"], dtype=float)
    assert executed_paths.shape == switch_paths.shape == swap_paths.shape
    assert dwell_paths.shape == executed_paths.shape
    assert np.any(switch_paths[:, 1:] > 0.5)
    assert np.all(switch_paths <= swap_paths)
    changed = executed_paths[:, 1:] != executed_paths[:, :-1]
    assert np.all(switch_paths[:, 1:][changed] > 0.5)
    assert np.all(dwell_paths[switch_paths > 0.5] == 1.0)


def test_dispatcher_runs_single_trajectory_backend():
    n_trials = 8
    stimulus = np.linspace(0.1, 0.9, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    feedback = np.ones(n_trials, dtype=float)
    config = _engine_config()
    config["inference"] = {"backend": "trajectory"}

    result = run_inference_backend(
        engine_config=config,
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        choices=categories,
        feedback=feedback,
        inference_seed=20260806,
        processed_data_dir=Path("."),
    )

    assert isinstance(result, InferenceResult)
    assert isinstance(result, TrajectoryInferenceResult)
    assert result.backend == "trajectory"
    assert result.state_probabilities["hypothesis_prior"] is result.prior_log
    assert np.asarray(result.prior_log).shape == (n_trials, 6)
    assert np.asarray(result.posterior_log).shape == (n_trials, 6)
    assert len(result.step_log) == n_trials


def test_standard_runner_trajectory_uses_shared_choice_readout():
    n_trials = 10
    stimulus = np.linspace(0.1, 0.9, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    feedback = np.ones(n_trials, dtype=float)
    config = _engine_config()
    config["inference"] = {"backend": "trajectory"}
    arrays = TrialArrays(
        stimulus=stimulus,
        choices=categories,
        feedback=feedback,
        categories=categories,
        target_probs=np.eye(2, dtype=float)[categories - 1],
    )

    result = evaluate_state_model_run(
        subject_id=1,
        condition=1,
        arrays=arrays,
        params={},
        engine_config_template=config,
        processed_data_dir=Path("."),
        window_size=4,
        keep_logs=False,
        prediction_mode="prior_t",
        selection_prediction_mode="prior_t",
        loss_metric="choice_brier",
        trajectory_seed=20260806,
    )

    metrics = result.metrics_by_mode["prior_t"]
    probabilities = np.asarray(metrics["pred_category_probs"], dtype=float)
    valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    assert np.allclose(probabilities[valid].sum(axis=1), 1.0)
    assert metrics["choice_readout_method"] == "sharpened_expectation"
    assert np.isfinite(result.mean_error)


def test_particle_backend_uses_common_inference_result_contract():
    n_trials = 6
    stimulus = np.linspace(0.1, 0.9, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    config = _engine_config()
    config["inference"]["choice_transmission_audit"] = True
    config["modules"]["hypo_transitions_mod"]["kwargs"]["range_controller"] = {
        "g_phi": 0.0,
        "g_beta_surprise": 0.5,
        "g_beta_uncertainty": 0.0,
        "g_surprise_center": 0.0,
        "g_surprise_scale": 1.0,
    }
    result = run_inference_backend(
        engine_config=config,
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        choices=categories,
        feedback=np.ones(n_trials, dtype=float),
        inference_seed=20260806,
        choice_readout_power=2.0,
        output_lapse=0.05,
        processed_data_dir=Path("."),
    )

    assert isinstance(result, InferenceResult)
    assert result.backend == "particle_filter"
    assert result.observation_probabilities["prior_t"] is result.marginal_probabilities
    search_range = np.asarray(result.latent_summaries["search_range"], dtype=float)
    assert search_range.shape == (n_trials,)
    assert np.isclose(search_range[0], 0.35)
    assert np.any(search_range[1:] > search_range[0])
    assert np.allclose(result.marginal_probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        result.latent_summaries["predictive_strategy_choice_precision"],
        1.0,
    )
    for key in (
        "audit_hypothesis_map",
        "audit_adaptive_sharpening",
        "audit_exploration_lapse",
        "audit_unsharpened_expectation",
        "audit_sharpened_no_lapse",
        "audit_strategy_confidence_no_lapse",
    ):
        probabilities = np.asarray(result.observation_probabilities[key], dtype=float)
        assert probabilities.shape == (n_trials, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    q10 = np.asarray(result.diagnostics["audit_particle_correct_q10"], dtype=float)
    q50 = np.asarray(result.diagnostics["audit_particle_correct_q50"], dtype=float)
    q90 = np.asarray(result.diagnostics["audit_particle_correct_q90"], dtype=float)
    assert q10.shape == q50.shape == q90.shape == (n_trials,)
    assert np.all(q10 <= q50)
    assert np.all(q50 <= q90)
    for key in (
        "audit_correct_predicting_available_probability",
        "audit_correct_predicting_prior_mass",
        "audit_best_active_correct_probability",
    ):
        values = np.asarray(result.diagnostics[key], dtype=float)
        assert values.shape == (n_trials,)
        assert np.all((values >= 0.0) & (values <= 1.0))
    ancestral = result.artifacts["audit_ancestral_paths"]
    n_particles = int(result.metadata["particle_count"])
    assert np.asarray(ancestral["particle_indices"]).shape == (
        n_particles,
        n_trials,
    )
    weights = np.asarray(ancestral["weights"], dtype=float)
    assert weights.shape == (n_particles,)
    assert np.isclose(np.sum(weights), 1.0)
    for key in (
        "correct_probability",
        "strategy_exploit",
        "strategy_local_explore",
        "strategy_global_explore",
        "swap_event",
    ):
        assert np.asarray(ancestral[key]).shape == (n_particles, n_trials)
    np.testing.assert_allclose(
        np.asarray(ancestral["strategy_exploit"])
        + np.asarray(ancestral["strategy_local_explore"])
        + np.asarray(ancestral["strategy_global_explore"]),
        1.0,
    )

    config["inference"].pop("choice_transmission_audit")
    baseline = run_inference_backend(
        engine_config=config,
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        choices=categories,
        feedback=np.ones(n_trials, dtype=float),
        inference_seed=20260806,
        choice_readout_power=2.0,
        output_lapse=0.05,
        processed_data_dir=Path("."),
    )
    np.testing.assert_allclose(
        result.marginal_probabilities,
        baseline.marginal_probabilities,
    )


def test_particle_filter_analysis_controls_separate_weighting_and_resampling():
    n_trials = 18
    stimulus = np.linspace(0.05, 0.95, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    choices = np.where(np.arange(n_trials) % 4 == 0, 3 - categories, categories)
    feedback = (choices == categories).astype(float)
    common = {
        "engine_config": _engine_config(),
        "subject_id": 1,
        "stimulus": stimulus,
        "choices": choices,
        "feedback": feedback,
        "particle_count": 8,
        "choice_readout_power": 2.0,
        "output_lapse": 0.05,
        "filter_seed": 20260814,
        "resample_threshold_fraction": 0.0,
        "processed_data_dir": Path("."),
    }
    unweighted = run_state_model_particle_filter(
        **common,
        condition_on_observed_choice=False,
    )
    weighted = run_state_model_particle_filter(
        **common,
        condition_on_observed_choice=True,
    )

    assert unweighted.condition_on_observed_choice is False
    np.testing.assert_allclose(unweighted.pre_choice_ess, 8.0)
    np.testing.assert_allclose(unweighted.post_choice_ess, 8.0)
    np.testing.assert_allclose(unweighted.final_weights, 1.0 / 8.0)
    assert not np.any(unweighted.resampled)

    assert weighted.condition_on_observed_choice is True
    assert not np.any(weighted.resampled)
    assert np.any(np.asarray(weighted.post_choice_ess) < 8.0)
    assert np.isclose(np.sum(weighted.final_weights), 1.0)

    with pytest.raises(ValueError, match="must lie in"):
        run_state_model_particle_filter(
            **{**common, "resample_threshold_fraction": -0.1}
        )


def test_particle_ancestral_indices_follow_resampling_parents():
    parents = np.asarray(
        [
            [0, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
            [0, 1, 2],
        ],
        dtype=int,
    )
    traced = _trace_ancestral_indices(parents)
    np.testing.assert_array_equal(
        traced,
        [
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [2, 2, 2, 2],
        ],
    )


def test_rt_and_oral_readouts_return_normalized_measurements():
    rt = read_reaction_time(
        [0.8, 0.2],
        trial_index=4,
        replacement_fraction=0.5,
        newcomer_distance=0.25,
        config={
            "intercept": 1.0,
            "choice_uncertainty": 0.2,
            "replacement_fraction": 0.4,
            "newcomer_distance": 0.3,
            "scale": 0.5,
            "degrees_of_freedom": 6.0,
        },
    )
    assert np.isfinite(rt.log_location)
    assert rt.scale == 0.5
    assert 0.0 <= rt.choice_uncertainty <= 1.0

    oral = read_oral_report(
        [0.75, 0.25],
        [[0.9, 0.1], [0.2, 0.8]],
        reliability=0.8,
    )
    assert np.all(oral.probabilities >= 0.0)
    assert np.isclose(np.sum(oral.probabilities), 1.0)


def _feedback_swap_engine_config() -> dict:
    config = _engine_config()
    config["inference"] = {"backend": "trajectory"}
    config["modules"]["perception_mod"]["kwargs"]["module_seed"] = 101
    config["modules"]["hypo_transitions_mod"] = {
        "class": (
            "src.Bayesian_state.model.modules.hypothesis_transition.fixed_strategy."
            "FixedFeedbackSwapHypothesisTransitionModule"
        ),
        "kwargs": {
            "capacity": 2,
            "theta": 1.0,
            "init_hypotheses": [0, 1],
            "module_seed": 202,
        },
    }
    return config


def test_state_model_fit_uses_the_single_public_trial_lifecycle():
    stimulus = np.asarray([[0.15], [0.35], [0.65], [0.85]], dtype=float)
    choices = np.asarray([1, 2, 2, 1], dtype=int)
    feedback = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float)
    trials = list(zip(stimulus, choices, feedback))

    phased_model = StateModel(
        _feedback_swap_engine_config(),
        context=ModelContext(condition=1, subject_id=1),
    )
    phased_posterior, phased_prior = phased_model.fit_step_by_step(trials)

    manual_model = StateModel(
        _feedback_swap_engine_config(),
        context=ModelContext(condition=1, subject_id=1),
    )
    manual_posterior = []
    manual_prior = []
    for trial in trials:
        manual_model.begin_trial(trial[0])
        posterior, prior, _ = manual_model.complete_trial(trial[1], trial[2])
        manual_posterior.append(posterior)
        manual_prior.append(prior)

    assert not hasattr(manual_model.engine, "infer_single")
    assert np.allclose(phased_prior, manual_prior)
    assert np.allclose(phased_posterior, manual_posterior)
    phased_events = phased_model.engine.hypo_transitions_mod.transition_log
    legacy_events = manual_model.engine.hypo_transitions_mod.transition_log
    assert [event["active_after"] for event in phased_events] == [
        event["active_after"] for event in legacy_events
    ]
    assert [event["current_feedback_recorded"] for event in phased_events] == list(
        feedback
    )


def test_autonomous_model_path_samples_before_task_feedback_and_is_reproducible():
    stimulus = np.linspace(0.1, 0.9, 8)[:, None]
    categories = np.where(stimulus[:, 0] <= 0.5, 1, 2)

    first = run_autonomous_category_learning(
        engine_config=_feedback_swap_engine_config(),
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        categories=categories,
        trajectory_seed=303,
    )
    second = run_autonomous_category_learning(
        engine_config=_feedback_swap_engine_config(),
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        categories=categories,
        trajectory_seed=303,
    )

    trajectory = first.trajectory
    assert np.array_equal(trajectory.choices, second.trajectory.choices)
    assert np.array_equal(
        trajectory.feedback,
        (trajectory.choices == categories).astype(float),
    )
    assert np.allclose(trajectory.observed_probabilities.sum(axis=1), 1.0)
    assert np.allclose(trajectory.cognitive_probabilities.sum(axis=1), 1.0)
    assert trajectory.prior.shape == trajectory.posterior.shape == (8, 6)
    assert trajectory.beta.shape == (8, 6)
    assert [event["current_feedback_recorded"] for event in trajectory.transition_log] == list(
        trajectory.feedback
    )
    assert trajectory.transition_log[0]["feedback_used"] is None
    assert trajectory.transition_log[1]["feedback_used"] == trajectory.feedback[0]


def test_conditioned_rollout_reuses_phased_state_model_lifecycle():
    stimulus = np.linspace(0.1, 0.9, 6)[:, None]
    categories = np.where(stimulus[:, 0] <= 0.5, 1, 2)
    prefix_choices = categories[:3].copy()
    prefix_feedback = np.ones(3, dtype=float)

    result = run_conditioned_condition1_rollouts(
        engine_config=_feedback_swap_engine_config(),
        subject_id=1,
        stimulus=stimulus,
        categories=categories,
        observed_prefix_choices=prefix_choices,
        observed_prefix_feedback=prefix_feedback,
        particle_count=2,
        rollout_count=2,
        rho=2.0,
        epsilon=0.05,
        filter_seed=404,
        rollout_seed=405,
        processed_data_dir=Path("."),
    )

    assert result.choices.shape == (2, 3)
    assert np.array_equal(
        result.feedback,
        (result.choices == categories[3:][None, :]).astype(np.int8),
    )
    assert np.allclose(result.probabilities.sum(axis=2), 1.0)
