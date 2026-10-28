"""One-rule baselines using the existing fixed-workspace transition algorithm.

The 0923-simple-v1 config contract fixes capacity=1 and global proposals. Setting
m=0 retains the initial rule; m>0 replaces it with an inactive rule with constant
probability. Pairing evidence follows pairwise transport; precision is reset on
rule admission by the inherited implementation. See the simple-baseline protocol
in docs/model_architecture/model_0923_simple_baselines.md.
"""
from .fixed_strategy import FixedWorkspaceHypothesisTransitionModule


class SimpleRuleHypothesisTransitionModule(FixedWorkspaceHypothesisTransitionModule):
    """Expose the condition-3 contract for constant, feedback-independent search."""

    # The controller ignores feedback. This tag satisfies the shared assembly
    # contract; the likelihood and pairing memory retain all three feedback levels.
    feedback_interpretation = "full_success"
