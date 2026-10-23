"""Outcome-dependent admission preserves beliefs, inheritance and replay."""
from contextlib import ExitStack
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.compact_belief import compact_belief
from src.Bayesian_state.model.modules.hypothesis_transition.belief_slot_transfer import belief_slot_transfer
from src.Bayesian_state.model.modules.hypothesis_transition.belief_outcome_retention import belief_outcome_retention
from src.Bayesian_state.model.modules.hypothesis_transition.belief_error_entry import (
    belief_error_entry, newcomer_share)
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition)
from tests.bayesian_state.test_model_1021_compact_evidence import compact_config


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('outcome_retention', [False, True])
def test_admission_inheritance_and_replay(condition, outcome_retention, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(condition)
    with ExitStack() as stack:
        stack.enter_context(compact_belief(Transition))
        stack.enter_context(belief_slot_transfer(Transition))
        if outcome_retention:
            stack.enter_context(belief_outcome_retention(Transition))
        audit = stack.enter_context(belief_error_entry(Transition))
        model = StateModel(cfg, context=ModelContext(condition=condition))
        transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        memory = model.engine.get_module(ModuleRole.MEMORY)
        for trial in range(15):
            old = transition.active.copy()
            posterior = model.engine.posterior.copy() if trial else model.engine.prior.copy()
            beta = model.engine.get_module(ModuleRole.BETA).beta.copy()
            marginal = memory.pairing_marginal() if condition == 3 else None
            feedback = transition.previous_feedback
            transition.baseline_probability = .999
            decision = model.begin_trial([.8, .3, .2, .4])
            selection = transition.last_transition_result.selection
            if len(selection.dropped):
                mass = feedback * posterior[selection.dropped[0]] + (1 - feedback) / transition.capacity
                np.testing.assert_allclose(decision.prior[selection.newcomers], mass)
                retained = selection.survivors
                np.testing.assert_allclose(decision.prior[retained] / decision.prior[retained].sum(),
                                           posterior[retained] / posterior[retained].sum())
                assert decision.prior[selection.newcomers[0]] >= posterior[selection.dropped[0]] - 1e-12
            if len(selection.newcomers):
                np.testing.assert_allclose(decision.beta[selection.newcomers], np.average(beta[old], weights=posterior[old]))
                if condition == 3:
                    np.testing.assert_allclose(memory.conditional_pairing()[selection.newcomers[0]], marginal)
            assert not hasattr(transition, 'retention_evidence')
            np.testing.assert_allclose(decision.prior.sum(), 1., atol=1e-12)
            model.complete_trial(1, (0., .5, 1.)[trial % 3] if condition == 3 else float(trial % 2))
        assert audit['replacements'] > 0
        clone = StateModel(cfg, context=ModelContext(condition=condition))
        clone.engine.load_state_dict(deepcopy(model.engine.state_dict()))
        for m in (model, clone):
            m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION).baseline_probability = .999
        for feedback in (0., 1., 0.):
            a, b = [m.begin_trial([.2, .7, .8, .6]) for m in (model, clone)]
            np.testing.assert_array_equal(a.prior, b.prior)
            np.testing.assert_array_equal(a.beta, b.beta)
            np.testing.assert_array_equal(*[m.complete_trial(1, feedback)[0] for m in (model, clone)])


def test_zero_belief_can_receive_mass_after_error_and_invalid_values_fail():
    assert newcomer_share(0., 4, 0.) == .25
    assert newcomer_share(0., 4, .5) == .125
    assert newcomer_share(0., 4, 1.) == 0.
    for args in [(-.1, 4, 0), (.4, 4, 0), (.1, 0, 1), (.1, 4, .3), (float('nan'), 4, 0)]:
        with pytest.raises(ValueError):
            newcomer_share(*args)


def test_requires_belief_and_restores_methods():
    original = Transition.__init__, Transition._assign_v3_prior
    with pytest.raises(ValueError, match='normalized rule beliefs'):
        with belief_error_entry(Transition):
            StateModel(compact_config(1), context=ModelContext(condition=1))
    assert original == (Transition.__init__, Transition._assign_v3_prior)
