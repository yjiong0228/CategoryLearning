"""Model 0923's unified controller, using the shared workspace lifecycle."""

from __future__ import annotations

from typing import Any, Mapping

from .nested_feedback_accumulator import (
    NestedFeedbackAccumulatorHypothesisTransitionModule,
)


class UnifiedRuleSearchHypothesisTransitionModule(
    NestedFeedbackAccumulatorHypothesisTransitionModule
):
    """One failure trace, one Bernoulli revision, no persistent execution.

    Local proposal geometry and low-support removal are inherited unchanged.
    A single newcomer inherits the removed candidate's belief mass. This is
    the executable B0 baseline, not a fitted or oral-validated model.
    """

    MODE = "unified_rule_search_0923_v1"
    strategy_mode = "unified_rule_search"
    replacement_count_method = "single_candidate_bernoulli"

    def __init__(self, engine: Any, **kwargs: Any) -> None:
        allowed = {"capacity", "tau_local", "epsilon", "search_controller", "module_seed"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"0923 B0 has unsupported transition keys: {sorted(unknown)}")
        resolved = dict(kwargs)
        controller = resolved.pop("search_controller", None)
        fields = {"baseline_probability", "error_gain", "global_search", "failure_decay"}
        if not isinstance(controller, Mapping) or set(controller) != fields:
            raise ValueError(f"0923 search_controller requires exactly {sorted(fields)}")
        if self._probability(controller["failure_decay"], "failure_decay") != 0.60:
            raise ValueError("0923 B0 fixes failure_decay=0.60 for this comparison")
        baseline = self._probability(controller["baseline_probability"], "baseline_probability")
        if not 0.0 < baseline < 1.0:
            raise ValueError("0923 baseline_probability must be strictly between 0 and 1")
        resolved["nested_feedback_accumulator_controller"] = {
            "event_after_correct": baseline,
            "event_after_error": baseline,
            "initial_event_probability": baseline,
            "global_search": controller["global_search"],
            "accumulator_decay": controller["failure_decay"],
            "accumulator_logit_gain": controller["error_gain"],
            "global_search_failure_gain": 0.0,
            "initial_failure": 0.0,
            "event_history_excludes_latest_error": False,
        }
        # A half-correct result contributes half an error to this control trace;
        # the separate condition-3 feedback likelihood still uses joint pairing.
        resolved["feedback_interpretation"] = "graded"
        resolved["prior_assignment"] = {
            "method": "mass_preserving_similarity_transport"
        }
        resolved["persistent_execution"] = {"enabled": False, "switch_scale": 0.0}
        super().__init__(engine, **resolved)
        # Keep the documented trace even at zero gain. The parent optimizes
        # away an unused accumulator; B0 retains it for comparable diagnostics.
        self.accumulator_active = True

    def _draw_replacement_count(self, slot_count: int, slot_rate: float) -> int:
        del slot_count, slot_rate
        return int(self.trial_rng.binomial(1, self.current_event_probability))


__all__ = ["UnifiedRuleSearchHypothesisTransitionModule"]
