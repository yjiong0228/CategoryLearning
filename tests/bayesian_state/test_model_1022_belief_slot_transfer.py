"""Replacement conserves beliefs, pairing inheritance and particle replay."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.compact_belief import compact_belief
from src.Bayesian_state.model.modules.hypothesis_transition.belief_slot_transfer import belief_slot_transfer
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition)
from tests.bayesian_state.test_model_1021_compact_evidence import compact_config


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_replacement_keeps_survivor_probabilities_and_inherits_pairing(condition, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(condition)
    with compact_belief(Transition), belief_slot_transfer(Transition) as audit:
        model = StateModel(cfg, context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(12):
            posterior = model.engine.posterior.copy() if trial else np.zeros(h.total_hypo)
            beta = model.engine.get_module(ModuleRole.BETA).beta.copy()
            old = h.active.copy()
            memory = model.engine.get_module(ModuleRole.MEMORY)
            pairing = memory.pairing_marginal().copy() if condition == 3 else None
            h.baseline_probability = .999
            decision = model.begin_trial([.8, .3, .2, .4])
            selected = h.last_transition_result.selection
            if selected.dropped.size:
                np.testing.assert_allclose(decision.prior[selected.survivors], posterior[selected.survivors],
                                           atol=1e-14, rtol=0)
                np.testing.assert_allclose(decision.prior[selected.newcomers], posterior[selected.dropped])
                assert decision.prior[selected.newcomers[0]] <= 1 / len(h.active) + 1e-12
                np.testing.assert_allclose(decision.beta[selected.newcomers], np.average(beta[old], weights=posterior[old]))
            elif selected.newcomers.size:
                np.testing.assert_allclose(decision.prior[selected.newcomers], 1 / len(h.active))
            if condition == 3 and selected.newcomers.size:
                np.testing.assert_allclose(memory.conditional_pairing()[selected.newcomers[0]], pairing)
                np.testing.assert_allclose(memory.joint.sum(axis=1), decision.prior, atol=1e-12)
            assert not hasattr(h, 'retention_evidence')
            np.testing.assert_allclose(decision.prior.sum(), 1., atol=1e-12)
            model.complete_trial(1, .5 if condition == 3 and trial % 2 else float(trial % 2))
        assert audit['replacements'] > 0
        clone = StateModel(cfg, context=ModelContext(condition=condition))
        clone.engine.load_state_dict(deepcopy(model.engine.state_dict()))
        for m in (model, clone):
            m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION).baseline_probability = .999
        for _ in range(3):
            a, b = [m.begin_trial([.2, .7, .8, .6]) for m in (model, clone)]
            np.testing.assert_array_equal(a.prior, b.prior)
            np.testing.assert_array_equal(a.beta, b.beta)
            for m in (model, clone):
                m.complete_trial(1, 0.)


def test_requires_belief_framework_and_restores_methods():
    original = Transition.__init__, Transition._assign_v3_prior
    with pytest.raises(ValueError, match='normalized-belief'):
        with belief_slot_transfer(Transition):
            StateModel(compact_config(1), context=ModelContext(condition=1))
    assert original == (Transition.__init__, Transition._assign_v3_prior)
