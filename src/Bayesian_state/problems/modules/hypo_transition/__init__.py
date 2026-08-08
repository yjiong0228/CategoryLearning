"""Hypothesis-transition (H) module package.

Only explicit cognitive modes are public.  Shared mechanism files are internal
implementation components and are not alternative H modules.
"""

from .process import (
    HypothesisSelection,
    HypothesisTransitionResult,
    TransitionContext,
    TwoStepHypothesisTransitionMixin,
)
from .static import (
    STATIC_STRATEGY_SPACE,
    StaticFeedbackSwapHypothesisTransitionModule,
    StaticWorkspaceHypothesisTransitionModule,
    StaticHypothesisStrategySpace,
    StaticHypothesisTransitionModule,
)
from .dynamic_discrete import DynamicDiscreteHypothesisTransitionModule
from .dynamic_continuous import DynamicContinuousHypothesisTransitionModule

__all__ = [
    "DynamicContinuousHypothesisTransitionModule",
    "DynamicDiscreteHypothesisTransitionModule",
    "HypothesisSelection",
    "HypothesisTransitionResult",
    "STATIC_STRATEGY_SPACE",
    "StaticFeedbackSwapHypothesisTransitionModule",
    "StaticWorkspaceHypothesisTransitionModule",
    "StaticHypothesisStrategySpace",
    "StaticHypothesisTransitionModule",
    "TransitionContext",
    "TwoStepHypothesisTransitionMixin",
]
