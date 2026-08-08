"""Trial-varying discrete-state hypothesis-transition strategies.

The subject has one fitted state controller.  On each trial the controller
selects a discrete strategy state (for example conservative, stable,
aggressive, or stubborn); that state supplies the selection and
prior-assignment policies used by the common two-step transition lifecycle.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .process import (
    HypothesisSelection,
    TransitionContext,
    TwoStepHypothesisTransitionMixin,
)
from ._internal.strategy_policy import StrategyPolicyRuntime


class DynamicDiscreteHypothesisTransitionModule(
    TwoStepHypothesisTransitionMixin,
    StrategyPolicyRuntime,
):
    """Run a trial-varying discrete strategy-state controller."""

    strategy_mode = "dynamic_discrete"

    def __init__(self, engine, **kwargs):
        resolved = dict(kwargs)
        if "module_seed" not in resolved and "random_seed" in resolved:
            resolved["module_seed"] = resolved["random_seed"]
        if resolved.get("state_controller") is None:
            raise ValueError(
                "DynamicDiscreteHypothesisTransitionModule requires a non-empty "
                "state_controller."
            )
        if "selection_strategy" in resolved or "prior_assignment" in resolved:
            raise ValueError(
                "Dynamic-discrete mode obtains selection and prior-assignment "
                "policies from its states; subject-level static strategies "
                "cannot be configured alongside the state controller."
            )
        super().__init__(engine, **resolved)

    def _prepare_hypothesis_transition(self, **kwargs) -> None:
        del kwargs
        self._update_latent_volatility_state()

    def _transition_signals(self) -> Mapping[str, Any]:
        return {"latent_volatility": float(self.latent_volatility_state)}

    def select_hypotheses(
        self,
        context: TransitionContext,
        **kwargs,
    ) -> HypothesisSelection:
        del context
        self._transition(**kwargs)
        latest = self.strategy_counts_log[-1] if self.strategy_counts_log else {}
        return HypothesisSelection.from_active_sets(
            self.old_active,
            self.active,
            diagnostics={
                "strategy_mode": self.strategy_mode,
                "selected_state": latest.get("selected_state"),
                "state_probabilities": latest.get("state_probabilities", {}),
            },
        )

    def assign_prior(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        **kwargs,
    ) -> np.ndarray:
        del context, selection, kwargs
        self._posterior_to_prior_transition()
        return np.asarray(self.engine.prior, dtype=float)

    def _finish_hypothesis_transition(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        prior: np.ndarray,
        **kwargs,
    ) -> Mapping[str, Any]:
        del context, selection, prior, kwargs
        self._record_feedback_from_observation()
        self._record_previous_observation()
        latest = self.strategy_counts_log[-1] if self.strategy_counts_log else {}
        latest["strategy_mode"] = self.strategy_mode
        return latest


__all__ = ["DynamicDiscreteHypothesisTransitionModule"]
