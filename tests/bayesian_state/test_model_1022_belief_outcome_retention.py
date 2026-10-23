"""Outcome-gated belief updates, preserved modules and particle continuation."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.compact_belief import compact_belief
from src.Bayesian_state.model.modules.hypothesis_transition.belief_slot_transfer import belief_slot_transfer
from src.Bayesian_state.model.modules.hypothesis_transition.belief_outcome_retention import (
    belief_outcome_retention, effective_retention)
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition)
from tests.bayesian_state.test_model_1021_compact_evidence import compact_config


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('retention', [0., .8, 1.])
def test_feedback_update_and_replay(condition, retention, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(condition)
    cfg['modules']['hypo_transitions_mod']['kwargs']['intervention']['decay'] = retention
    with compact_belief(Transition), belief_slot_transfer(Transition), belief_outcome_retention(Transition) as audit:
        model = StateModel(cfg, context=ModelContext(condition=condition))
        transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        memory = model.engine.get_module(ModuleRole.MEMORY)
        for trial in range(9):
            decision = model.begin_trial([.8, .3, .2, .4])
            prior = decision.prior.copy()
            active = transition.active.copy()
            feedback = (0., .5, 1.)[trial % 3] if condition == 3 else float(trial % 2)
            if condition == 3:
                old_pairing = memory.conditional_pairing().copy()
                old_gamma = memory.gamma
            model.complete_trial(1, feedback)
            exponent = retention + (1 - retention) * feedback
            expected = np.zeros_like(prior)
            expected[active] = (1. if exponent == 0 else prior[active] ** exponent) * model.engine.likelihood[active]
            expected /= expected.sum()
            np.testing.assert_allclose(model.engine.posterior, expected, atol=1e-12)
            assert transition.retention_decay == retention
            assert not hasattr(transition, 'retention_evidence')
            assert 'retention_evidence' not in transition.state_dict()
            if condition == 3:
                pairing = old_pairing[active] ** old_gamma
                pairing /= pairing.sum(1, keepdims=True)
                np.testing.assert_allclose(memory.feedback_pairing_weights[active], pairing)
                assert memory.gamma == old_gamma
        assert audit['updates'] == 9
        clone = StateModel(cfg, context=ModelContext(condition=condition))
        clone.engine.load_state_dict(deepcopy(model.engine.state_dict()))
        for feedback in (1., 0., 1.):
            a, b = [m.begin_trial([.2, .7, .8, .6]) for m in (model, clone)]
            np.testing.assert_array_equal(a.prior, b.prior)
            np.testing.assert_array_equal(a.beta, b.beta)
            posteriors = [m.complete_trial(1, feedback)[0] for m in (model, clone)]
            np.testing.assert_array_equal(*posteriors)


def test_validation_and_restoration():
    from src.Bayesian_state.model.modules.memory import DualMemoryModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    original = Transition.__init__, DualMemoryModule.process, HierarchicalPairingMemoryModule.prepare_feedback
    with pytest.raises(ValueError, match='normalized rule beliefs'):
        with belief_outcome_retention(Transition):
            StateModel(compact_config(1), context=ModelContext(condition=1))
    assert original == (Transition.__init__, DualMemoryModule.process, HierarchicalPairingMemoryModule.prepare_feedback)
    for retention, feedback in [(-.1, 1), (1.1, 0), (.8, .3), (float('nan'), 1), (.8, float('nan'))]:
        with pytest.raises(ValueError):
            effective_retention(retention, feedback)
