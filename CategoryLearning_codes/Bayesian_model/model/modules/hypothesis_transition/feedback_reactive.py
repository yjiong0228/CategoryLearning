"""One-step feedback-reactive bounded-workspace control."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .execution import WorkspaceTransitionExecutionMixin
from .workspace import AdaptiveWorkspaceController


class FeedbackReactiveHypothesisTransitionModule(
    WorkspaceTransitionExecutionMixin,
    AdaptiveWorkspaceController,
):
    """Use the previous outcome to set the next replacement probability.

    The workspace selection rule, local/global newcomer proposal, and selected
    prior-assignment policy are identical to the other bounded-workspace H
    modes.  Only the probability of at least one replacement changes:

    ``E_t = feedback * E_correct + (1-feedback) * E_error``.

    Completed feedback from trial ``t-1`` is recorded through
    :meth:`record_outcome` and consumed before trial ``t``.  The current
    trial's choice and feedback therefore cannot affect its own transition.
    """

    MODE = "feedback_reactive_v1"
    strategy_mode = "feedback_reactive"
    dynamic_controls = True
    # H3 remains the execution-off architecture screen.  The nested H4/H5
    # subclass explicitly widens this capability without changing H3.
    allows_legacy_persistent_execution = False

    def __init__(self, engine, **kwargs):
        resolved = dict(kwargs)
        raw = resolved.pop("feedback_reactive_controller", None)
        if not isinstance(raw, Mapping):
            raise ValueError("feedback_reactive_controller must be a mapping.")
        unknown = set(raw) - {
            "event_after_correct",
            "event_after_error",
            "initial_event_probability",
            "global_search",
        }
        if unknown:
            raise ValueError(
                "feedback_reactive_controller has unsupported keys: "
                f"{sorted(unknown)}."
            )
        forbidden = {
            "continuous_controller",
            "rate_controller",
            "range_controller",
            "controller_mode",
            "failure_accumulator_controller",
            "m",
            "g",
        }.intersection(resolved)
        if forbidden:
            raise ValueError(
                "feedback-reactive control cannot be combined with other "
                f"workspace controllers: {sorted(forbidden)}."
            )

        event_correct = self._probability(
            raw.get("event_after_correct"), "event_after_correct"
        )
        event_error = self._probability(
            raw.get("event_after_error"), "event_after_error"
        )
        if event_error < event_correct:
            raise ValueError(
                "event_after_error must be at least event_after_correct."
            )
        initial_event = self._probability(
            raw.get("initial_event_probability", event_correct),
            "initial_event_probability",
        )
        global_search = self._probability(
            raw.get("global_search"), "global_search"
        )

        prior_spec = resolved.get("prior_assignment")
        if prior_spec is not None:
            if not isinstance(prior_spec, Mapping):
                raise ValueError("prior_assignment must be a mapping.")
            if (
                str(prior_spec.get("method", ""))
                not in self.VALID_PRIOR_ASSIGNMENTS
            ):
                raise ValueError(
                    "Feedback-reactive bounded-workspace control requires a "
                    "supported bounded-workspace prior_assignment method."
                )

        selection_spec = resolved.pop("selection_strategy", None)
        if selection_spec is not None:
            if not isinstance(selection_spec, Mapping):
                raise ValueError("selection_strategy must be a mapping.")
            if str(selection_spec.get("method", "")) != "bounded_workspace":
                raise ValueError(
                    "FeedbackReactiveHypothesisTransitionModule requires "
                    "selection_strategy.method='bounded_workspace'."
                )
            for key, value in selection_spec.items():
                if key != "method":
                    resolved[key] = value

        capacity = self._positive_integer(
            resolved.get(
                "capacity",
                resolved.get("max_active_hypotheses", resolved.get("init_num", 3)),
            ),
            "capacity",
        )
        resolved["m"] = self._event_to_slot_rate(initial_event, capacity)
        resolved["g"] = global_search
        resolved["controller_mode"] = self.LEGACY_CONTROLLER_MODE
        super().__init__(engine, **resolved)

        if (
            self.persistent_execution_enabled
            and not self.allows_legacy_persistent_execution
        ):
            raise ValueError(
                "feedback-reactive H is an execution-off architecture screen."
            )
        self.controller_mode = self.MODE
        self.uses_outcome_feedback_controller = True
        self.dynamic_rate = True
        self.dynamic_range = False
        self.event_after_correct = float(event_correct)
        self.event_after_error = float(event_error)
        self.initial_event_probability = float(initial_event)
        self.current_event_probability = float(initial_event)
        self.current_m = self._event_probability_to_slot_rate(initial_event)
        self.control_logit = self._safe_logit(self.current_m)
        self.current_g = float(global_search)
        self.g_control_logit = self._safe_logit(self.current_g)
        self.exploration_target = float(initial_event)
        self.global_target = float(global_search)
        self.current_prior_reset_strength = 0.0
        self._pending_transition: dict[str, Any] | None = None

    @staticmethod
    def _probability(value: Any, name: str) -> float:
        if value is None:
            raise ValueError(f"{name} is required.")
        parsed = float(value)
        if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{name} must be a finite probability in [0, 1].")
        return parsed

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer.")
        parsed = int(value)
        if float(value) != float(parsed) or parsed <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return parsed

    @staticmethod
    def _event_to_slot_rate(event_probability: float, capacity: int) -> float:
        return float(1.0 - (1.0 - float(event_probability)) ** (1.0 / int(capacity)))

    def _update_transition_controls(self) -> tuple[float, float]:
        self.feedback_surprise = float("nan")
        self.feedback_uncertainty = float("nan")
        if self.outcome_pending and np.isfinite(self.previous_feedback):
            feedback = float(np.clip(self.previous_feedback, 0.0, 1.0))
            self.exploration_target = float(
                feedback * self.event_after_correct
                + (1.0 - feedback) * self.event_after_error
            )
            # Retain one-step controller summaries for the common diagnostic
            # contract.  They do not enter choice readout in the H screen.
            self.failure_pressure = float(1.0 - feedback)
            self.mastery_evidence = float(feedback)
            self.peak_mastery_evidence = float(
                max(self.peak_mastery_evidence, self.mastery_evidence)
            )
            self.outcome_pending = False
        else:
            self.exploration_target = float(self.initial_event_probability)

        self.current_event_probability = float(self.exploration_target)
        self.current_m = self._event_probability_to_slot_rate(
            self.current_event_probability
        )
        self.control_logit = self._safe_logit(self.current_m)
        self.current_g = float(self.g)
        self.g_control_logit = self._safe_logit(self.current_g)
        self.global_target = float(self.current_g)
        self.current_prior_reset_strength = 0.0
        return float("nan"), float("nan")

    def record_outcome(
        self,
        observation: tuple[np.ndarray, int, float],
    ) -> None:
        if observation is None or len(observation) < 3:
            raise ValueError(
                "feedback_reactive_v1 requires "
                "observation=(stimulus, choice, feedback)."
            )
        self.previous_feedback = self._probability(observation[2], "feedback")
        self.outcome_pending = True


__all__ = ["FeedbackReactiveHypothesisTransitionModule"]
