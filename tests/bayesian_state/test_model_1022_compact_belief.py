"""Belief feedback timing, admission, module preservation and particle replay."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.compact_belief import compact_belief
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition)
from tests.bayesian_state.test_model_1021_compact_evidence import compact_config


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('decay', [0., .8, 1.])
def test_belief_learning_admission_precision_and_replay(condition, decay, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(condition)
    cfg['modules']['hypo_transitions_mod']['kwargs']['intervention']['decay'] = decay
    with compact_belief(Transition) as audit:
        model = StateModel(cfg, context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(8):
            previous = model.engine.posterior.copy() if trial else np.zeros(h.total_hypo)
            before = h.active.copy()
            beta = model.engine.get_module(ModuleRole.BETA).beta[before].copy()
            h.baseline_probability = 1e-12 if trial == 4 else .999
            decision = model.begin_trial([.8, .3, .2, .4])
            selected = h.last_transition_result.selection
            np.testing.assert_allclose(decision.prior, h.execution_probabilities)
            assert 'retention_evidence' not in h.state_dict()
            assert not hasattr(h, 'retention_evidence')
            assert 1 <= len(h.active) <= h.capacity
            if selected.newcomers.size:
                np.testing.assert_allclose(decision.prior[selected.newcomers], 1 / len(h.active))
                expected = np.average(beta, weights=previous[before])
                np.testing.assert_allclose(decision.beta[selected.newcomers], expected)
            if trial and len(selected.survivors) > 1:
                retained = selected.survivors
                np.testing.assert_allclose(decision.prior[retained] / decision.prior[retained].sum(),
                                           previous[retained] / previous[retained].sum())
            if selected.dropped.size:
                assert previous[selected.dropped[0]] == previous[before].min()
            if trial == 4:
                np.testing.assert_allclose(decision.prior, previous)
            prior = decision.prior.copy()
            feedback = .5 if condition == 3 and trial % 2 else float(trial % 2)
            model.complete_trial(1, feedback)
            expected = np.zeros_like(prior)
            active = h.active
            expected[active] = (1. if decay == 0 else prior[active] ** decay) * model.engine.likelihood[active]
            expected /= expected.sum()
            np.testing.assert_allclose(model.engine.posterior, expected, atol=1e-12)
        assert audit['replacements'] > 0 and audit['feedback_updates'] == 8
        clone = StateModel(cfg, context=ModelContext(condition=condition))
        clone.engine.load_state_dict(deepcopy(model.engine.state_dict()))
        other = clone.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        h.baseline_probability = other.baseline_probability = .999
        for _ in range(3):
            a, b = [m.begin_trial([.2, .7, .8, .6]) for m in (model, clone)]
            np.testing.assert_array_equal(a.prior, b.prior)
            np.testing.assert_array_equal(a.beta, b.beta)
            np.testing.assert_array_equal(h.active, other.active)
            for m in (model, clone):
                m.complete_trial(1, 0.)


def test_methods_restore_after_exception():
    from src.Bayesian_state.model.modules.memory import DualMemoryModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    original = Transition.__init__, DualMemoryModule.process, HierarchicalPairingMemoryModule.prepare_feedback
    with pytest.raises(RuntimeError):
        with compact_belief(Transition):
            raise RuntimeError('stop')
    assert original == (Transition.__init__, DualMemoryModule.process, HierarchicalPairingMemoryModule.prepare_feedback)
