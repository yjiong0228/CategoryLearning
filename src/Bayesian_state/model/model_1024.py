"""Compose the existing Model 1024 research mechanisms for a serial worker.

The runtime uses process-wide adapters. Enter once around construction and all
execution of the models; use separate processes for independent parallel work.
Configuration retains the 0923 full-v3 carrier validation and explicitly supplies
the compact-belief coordinates. No learning equations are reimplemented here.
"""
from contextlib import ExitStack, contextmanager

from .modules.beta import BetaModule
from .modules.hypothesis_transition.belief_error_entry import belief_error_entry
from .modules.hypothesis_transition.belief_execution import belief_execution
from .modules.hypothesis_transition.belief_outcome_retention import belief_outcome_retention
from .modules.hypothesis_transition.belief_slot_transfer import belief_slot_transfer
from .modules.hypothesis_transition.compact_belief import compact_belief
from .modules.hypothesis_transition.feedback_guided import retention_roundoff_guard
from .modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)
from .modules.precision_interventions import precision_intervention

_runtime_active = False


@contextmanager
def model_1024_runtime():
    """Install belief, outcome retention, error entry, ADD10 and STICKY execution.

    This composition is the binary-task Model 1024 research candidate. Existing
    adapters validate the mechanism's fixed coordinates at model construction.
    All patched methods are restored on normal exit or an exception.
    """
    global _runtime_active
    if _runtime_active:
        raise RuntimeError("Model 1024 runtime must not be nested or used concurrently")
    _runtime_active = True
    try:
        with ExitStack() as stack:
            audits = {}
            audits["roundoff"] = stack.enter_context(retention_roundoff_guard(Transition))
            stack.enter_context(precision_intervention(BetaModule, mode="fast", upper=25.))
            for adapter in (compact_belief, belief_slot_transfer, belief_outcome_retention,
                            belief_error_entry):
                audits[adapter.__name__] = stack.enter_context(adapter(Transition))
            audits["execution"] = stack.enter_context(belief_execution(Transition, mode="STICKY"))
            yield audits
    finally:
        _runtime_active = False
