"""Scientific contracts for compact online evidence and causal state replay."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.compact_evidence import compact_evidence, evidence_prior
from tests.bayesian_state.test_model_0923_continuous_recalled_belief import config


def compact_config(condition):
    cfg = config(condition)
    h = cfg['modules']['hypo_transitions_mod']['kwargs']
    h.update(exit_rate=0., execution_inertia=0., local_structure_weight=1., transfer_scale=.01)
    h['intervention'].update(capacity='evidence', search='accumulate')
    h['search_controller']['global_search_failure_gain'] = 0.
    h['search_controller']['baseline_probability'] = .999
    return cfg


def test_evidence_normalization_extreme_values():
    p = evidence_prior(np.ones(4), np.array([1, 2]), np.array([1e5, -1e5, -1e5, 0.]))
    np.testing.assert_array_equal(p, [0., .5, .5, 0.])
    with pytest.raises(ValueError):
        evidence_prior(np.ones(4), np.array([1, 1]), np.zeros(4))


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_causal_online_learning_neutral_entry_and_exact_replay(condition, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(condition)
    with compact_evidence(Transition) as audit:
        a = StateModel(cfg, context=ModelContext(condition=condition))
        h = a.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        assert not hasattr(h, 'admission_history') and not hasattr(h, 'reentry_precision_seen')
        for trial in range(8):
            before = h.active.copy()
            if trial:
                expected = .8 * h.retention_evidence[before] + h._previous_log_evidence()
            pending = a.begin_trial([.8, .3, .2, .4])
            s = h.last_transition_result.selection
            if trial:
                indices = [int(np.flatnonzero(before == rule).item()) for rule in s.survivors]
                np.testing.assert_allclose(h.retention_evidence[s.survivors], expected[indices], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(h.retention_evidence[s.newcomers], 0.)
            np.testing.assert_array_equal(h.retention_evidence[s.dropped], 0.)
            np.testing.assert_allclose(pending.prior.sum(), 1., atol=1e-12)
            np.testing.assert_allclose(h.execution_probabilities, pending.prior, atol=1e-12)
            assert s.diagnostics['autonomous_exits'] == [] and len(h.active) <= h.capacity
            if s.dropped.size:
                assert len(before) == h.capacity
                old = dict(zip(before, expected))
                assert old[int(s.dropped[0])] == min(old.values())
            a.complete_trial(1, .5 if condition == 3 and trial % 2 == 0 else float(trial % 2))
        assert audit['replacements'] > 0
        b = StateModel(cfg, context=ModelContext(condition=condition))
        b.engine.load_state_dict(deepcopy(a.engine.state_dict()))
        hb = b.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(3):
            pa, pb = [m.begin_trial([.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(pa.prior, pb.prior)
            np.testing.assert_array_equal(pa.beta, pb.beta)
            np.testing.assert_array_equal(h.active, hb.active)
            assert h.trial_rng.bit_generator.state == hb.trial_rng.bit_generator.state
            for m in (a, b):
                m.complete_trial(1, float(trial % 2))


def test_unused_m_posterior_cannot_change_selection(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(1)
    with compact_evidence(Transition):
        m = StateModel(cfg, context=ModelContext(condition=1))
        h = m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for _ in range(4):
            m.begin_trial([.8, .3, .2, .4]); m.complete_trial(1, 0.)
        h._prepare_hypothesis_transition()
        context = h._build_transition_context()
        state = deepcopy(h.state_dict())
        a = h.select_hypotheses(context)
        h.load_state_dict(state)
        modified = np.ones(h.total_hypo) / h.total_hypo
        b = h.select_hypotheses(replace(context, posterior=modified))
        np.testing.assert_array_equal(a.active_after, b.active_after)
        assert a.diagnostics['proposal_parent'] == b.diagnostics['proposal_parent']


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_failure_decay_has_zero_effect_when_gain_is_zero(condition, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(condition)
    control = cfg['modules']['hypo_transitions_mod']['kwargs']['search_controller']
    control.update(error_gain=0., baseline_probability=.35, failure_decay=0.)
    other = deepcopy(cfg)
    other['modules']['hypo_transitions_mod']['kwargs']['search_controller']['failure_decay'] = .99
    with compact_evidence(Transition):
        models = [StateModel(c, context=ModelContext(condition=condition)) for c in (cfg, other)]
        hs = [m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in models]
        different_failure_states = False
        for trial in range(12):
            first, second = [m.begin_trial([.8, .3, .2, .4]) for m in models]
            np.testing.assert_array_equal(first.prior, second.prior)
            np.testing.assert_array_equal(first.beta, second.beta)
            np.testing.assert_array_equal(hs[0].active, hs[1].active)
            assert hs[0].current_event_probability == hs[1].current_event_probability
            assert hs[0].current_g == hs[1].current_g
            different_failure_states |= hs[0].accumulated_failure != hs[1].accumulated_failure
            for model in models:
                model.complete_trial(1, float(trial % 2))
        assert different_failure_states


def test_context_restores_methods_after_failure():
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    original = Transition._select_v3
    with pytest.raises(RuntimeError):
        with compact_evidence(Transition):
            raise RuntimeError('interrupted task')
    assert Transition._select_v3 is original
