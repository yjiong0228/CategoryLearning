"""Feedback-reactive search with nested accumulated-failure controls."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .feedback_reactive import FeedbackReactiveHypothesisTransitionModule


class NestedFeedbackAccumulatorHypothesisTransitionModule(
    FeedbackReactiveHypothesisTransitionModule
):
    """Nest constant, one-step reactive, and accumulated-failure search.

    Let ``e_(t-1) = 1 - feedback_(t-1)`` and

    ``F_t = decay * F_(t-1) + (1 - decay) * e_(t-1)``.

    The probability of at least one workspace replacement is

    ``logit(E_t) = logit(E_reactive,t) + accumulator_logit_gain * F_t``,

    and the conditional probability that a newcomer is proposed globally is

    ``g_t = g_0 + (1 - g_0) * global_search_failure_gain * F_t``.

    where ``E_reactive,t`` is ``event_after_correct`` or
    ``event_after_error`` according to the previous completed outcome.
    Both controls reuse the same failure state; no second range accumulator is
    introduced.  Setting both gains to zero delegates to the parent controller
    exactly, so the transition path is the one-step feedback-reactive model.
    If the two reactive event probabilities are also equal, the path is the
    constant-event model.
    """

    MODE = "nested_feedback_accumulator_v1"
    strategy_mode = "nested_feedback_accumulator"
    # H4/H5 use the same persistent-execution state machine as the continuous
    # controller.  The binary enabled flag is a subject-level structural
    # coordinate; switch_scale remains fixed by the model specification.
    allows_legacy_persistent_execution = True

    def __init__(self, engine, **kwargs):
        resolved = dict(kwargs)
        raw = resolved.pop("nested_feedback_accumulator_controller", None)
        if not isinstance(raw, Mapping):
            raise ValueError(
                "nested_feedback_accumulator_controller must be a mapping."
            )
        unknown = set(raw) - {
            "event_after_correct",
            "event_after_error",
            "initial_event_probability",
            "global_search",
            "accumulator_decay",
            "accumulator_logit_gain",
            "global_search_failure_gain",
            "initial_failure",
        }
        if unknown:
            raise ValueError(
                "nested_feedback_accumulator_controller has unsupported keys: "
                f"{sorted(unknown)}."
            )
        if "feedback_reactive_controller" in resolved:
            raise ValueError(
                "Configure nested_feedback_accumulator_controller only; do not "
                "also provide feedback_reactive_controller."
            )

        accumulator_decay = self._probability(
            raw.get("accumulator_decay", 0.60),
            "accumulator_decay",
        )
        if accumulator_decay >= 1.0:
            raise ValueError("accumulator_decay must be smaller than 1.")
        accumulator_gain = self._nonnegative(
            raw.get("accumulator_logit_gain", 0.0),
            "accumulator_logit_gain",
        )
        global_search_failure_gain = self._probability(
            raw.get("global_search_failure_gain", 0.0),
            "global_search_failure_gain",
        )
        initial_failure = self._probability(
            raw.get("initial_failure", 0.0),
            "initial_failure",
        )

        resolved["feedback_reactive_controller"] = {
            "event_after_correct": raw.get("event_after_correct"),
            "event_after_error": raw.get("event_after_error"),
            "initial_event_probability": raw.get(
                "initial_event_probability",
                raw.get("event_after_correct"),
            ),
            "global_search": raw.get("global_search"),
        }
        super().__init__(engine, **resolved)

        self.controller_mode = self.MODE
        self.accumulator_decay = float(accumulator_decay)
        self.accumulator_logit_gain = float(accumulator_gain)
        self.global_search_failure_gain = float(global_search_failure_gain)
        self.accumulator_initial_failure = float(initial_failure)
        self.event_accumulator_active = bool(self.accumulator_logit_gain > 0.0)
        self.global_range_accumulator_active = bool(
            self.global_search_failure_gain > 0.0
        )
        self.accumulator_active = bool(
            self.event_accumulator_active
            or self.global_range_accumulator_active
        )
        self.immediate_error_logit_gain = float(
            self._safe_logit(self.event_after_error)
            - self._safe_logit(self.event_after_correct)
        )
        if self.accumulator_active:
            self.failure_pressure = float(self.accumulator_initial_failure)
            # There is only one scientific history state.  This complementary
            # value is retained solely for the shared diagnostic contract.
            self.mastery_evidence = float(1.0 - self.failure_pressure)
            self.peak_mastery_evidence = float(self.mastery_evidence)

    def _transition_signals(self) -> Mapping[str, Any]:
        signals = dict(super()._transition_signals())
        signals.update(
            {
                "accumulator_logit_gain": float(
                    self.accumulator_logit_gain
                ),
                "global_search_failure_gain": float(
                    self.global_search_failure_gain
                ),
                "event_accumulator_active": bool(
                    self.event_accumulator_active
                ),
                "global_range_accumulator_active": bool(
                    self.global_range_accumulator_active
                ),
            }
        )
        return signals

    @staticmethod
    def _nonnegative(value: Any, name: str) -> float:
        parsed = float(value)
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return parsed

    def _update_transition_controls(self) -> tuple[float, float]:
        # This explicit boundary is intentional: it makes the simpler model a
        # literal member of the supermodel, including its diagnostic states and
        # random transition path under a shared seed.
        if not self.accumulator_active:
            return super()._update_transition_controls()

        self.feedback_surprise = float("nan")
        self.feedback_uncertainty = float("nan")
        if self.outcome_pending and np.isfinite(self.previous_feedback):
            feedback = float(np.clip(self.previous_feedback, 0.0, 1.0))
            error = float(1.0 - feedback)
            self.failure_pressure = float(
                self.accumulator_decay * self.failure_pressure
                + (1.0 - self.accumulator_decay) * error
            )
            self.mastery_evidence = float(1.0 - self.failure_pressure)
            self.peak_mastery_evidence = float(
                max(self.peak_mastery_evidence, self.mastery_evidence)
            )
            reactive_event = float(
                feedback * self.event_after_correct
                + error * self.event_after_error
            )
            self.exploration_target = float(
                self._expit(
                    self._safe_logit(reactive_event)
                    + self.accumulator_logit_gain * self.failure_pressure
                )
            )
            self.outcome_pending = False
        else:
            self.exploration_target = float(self.initial_event_probability)

        self.current_event_probability = float(self.exploration_target)
        self.current_m = self._event_probability_to_slot_rate(
            self.current_event_probability
        )
        self.control_logit = self._safe_logit(self.current_m)
        self.current_g = float(
            self.g
            + (1.0 - self.g)
            * self.global_search_failure_gain
            * self.failure_pressure
        )
        self.g_control_logit = self._safe_logit(self.current_g)
        self.global_target = float(self.current_g)
        self.current_prior_reset_strength = 0.0
        return float("nan"), float("nan")


__all__ = ["NestedFeedbackAccumulatorHypothesisTransitionModule"]
