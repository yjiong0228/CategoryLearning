"""Online evidence timing, newcomer initialization and state cloning."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import episodic_admission
from src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief import online_evidence_belief
from tests.bayesian_state.test_model_0923_continuous_recalled_belief import config


def test_disabled_context_leaves_class_unchanged():
    class Empty:
        pass
    with online_evidence_belief(Empty) as events:
        assert not hasattr(Empty, '_assign_v3_prior') and events == []


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_online_evidence_is_consumed_once_and_newcomer_inherits_admission(condition, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = config(condition)
    with episodic_admission(Transition, mode='POOL'), online_evidence_belief(Transition, enabled=True) as events:
        model = StateModel(cfg, context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        saw_newcomer = False
        for trial in range(8):
            before = h.active.copy()
            previous = h.retention_evidence.copy()
            if trial:
                feedback = h.previous_feedback
                chance = (.5 if feedback == 0 else .25) if condition == 3 else (
                    1 / model.engine.partition.n_cats if feedback == 1 else 1 - 1 / model.engine.partition.n_cats)
                expected = .8 * previous[before] + np.log(np.maximum(
                    model.engine.likelihood[before], np.finfo(float).tiny)) - np.log(chance)
            h.baseline_probability = 1e-12 if trial == 2 else .999
            pending = model.begin_trial([.8, .3, .2, .4])
            selection = h.last_transition_result.selection
            if trial:
                old = selection.survivors
                indices = [int(np.flatnonzero(before == rule).item()) for rule in old]
                np.testing.assert_allclose(h.retention_evidence[old], expected[indices], atol=1e-12, rtol=0)
                if len(selection.newcomers):
                    saw_newcomer = True
                    assert h.retention_evidence[selection.newcomers[0]] == selection.diagnostics['candidate_initial_evidence']
                active = selection.active_after
                score = h.retention_evidence[active]
                weighted = h.base_prior[active] * np.exp(score - score.max())
                expected_prior = np.zeros(h.total_hypo)
                expected_prior[active] = weighted / weighted.sum()
                np.testing.assert_allclose(pending.prior, expected_prior, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(h.retention_evidence[selection.dropped], 0.)
            model.complete_trial(1, .5 if condition == 3 and trial % 2 == 0 else float(trial % 2))
        assert saw_newcomer and len(events) == 7
        assert any(event['newcomer_count'] == 0 for event in events)
        assert [event['trial'] for event in events] == list(range(1, 8))
        state = deepcopy(model.engine.state_dict())
        clone = StateModel(cfg, context=ModelContext(condition=condition))
        clone.engine.load_state_dict(state)
        other = clone.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(3):
            a, b = [m.begin_trial([.2, .7, .8, .6]) for m in (model, clone)]
            np.testing.assert_array_equal(a.prior, b.prior)
            np.testing.assert_array_equal(h.execution_probabilities, other.execution_probabilities)
            np.testing.assert_array_equal(h.retention_evidence, other.retention_evidence)
            assert h.trial_rng.bit_generator.state == other.trial_rng.bit_generator.state
            for m in (model, clone):
                m.complete_trial(1, float(trial % 2))
