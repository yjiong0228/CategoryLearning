"""Hypothesis-transition (H) module package.

Only explicit cognitive modes are public.  Shared mechanism files are internal
implementation components and are not alternative H modules.
"""

from .contracts import (
    HypothesisSelection,
    HypothesisTransitionResult,
    TransitionContext,
    TwoStepHypothesisTransitionMixin,
)
from .fixed_strategy import (
    FIXED_STRATEGY_SPACE,
    FixedFeedbackSwapHypothesisTransitionModule,
    FixedWorkspaceHypothesisTransitionModule,
    FixedHypothesisStrategySpace,
    FixedStrategyHypothesisTransitionModule,
)
from .dynamic_discrete_strategy import DynamicDiscreteStrategyHypothesisTransitionModule
from .dynamic_adaptive_control import DynamicAdaptiveControlHypothesisTransitionModule

__all__ = [
    "DynamicAdaptiveControlHypothesisTransitionModule",
    "DynamicDiscreteStrategyHypothesisTransitionModule",
    "HypothesisSelection",
    "HypothesisTransitionResult",
    "FIXED_STRATEGY_SPACE",
    "FixedFeedbackSwapHypothesisTransitionModule",
    "FixedWorkspaceHypothesisTransitionModule",
    "FixedHypothesisStrategySpace",
    "FixedStrategyHypothesisTransitionModule",
    "TransitionContext",
    "TwoStepHypothesisTransitionMixin",
]
