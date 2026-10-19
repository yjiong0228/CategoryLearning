"""Same-history rule selection, unchanged belief learning and particle cloning."""
from contextlib import nullcontext
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.dominant_rule_execution import (
    POLICY, dominant_rule_execution,
)
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import episodic_admission
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)
from src.Bayesian_state.model.modules.hypothesis_transition.separate_belief_evidence import separate_belief_evidence
from tests.bayesian_state.test_model_0923_separate_belief_evidence import configuration


@pytest.fixture(autouse=True)
def isolated_similarity(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)


def history(condition, adapter):
    rows = []
    with episodic_admission(Transition, mode='POOL'), separate_belief_evidence(Transition, enabled=True), adapter:
        model = StateModel(configuration(condition, .3, .8), context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(12):
            h.baseline_probability = 1e-12 if trial == 2 else .999
            pending = model.begin_trial([.8, .3, .2, .4])
            rows.append(dict(prior=pending.prior.copy(), active=h.active.copy(), beta=pending.beta.copy(),
                evidence=h.belief_evidence.copy(), h_evidence=h.retention_evidence.copy(),
                weights=h.execution_probabilities.copy(), dominant=h.dominant_hypothesis))
            model.complete_trial(1, .5 if condition == 3 and trial % 2 == 0 else float(trial % 2))
    return rows


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_same_history_keeps_candidates_beliefs_precision_and_dominant_choice(condition):
    old = history(condition, nullcontext())
    new = history(condition, dominant_rule_execution(Transition, enabled=True))
    assert any(np.count_nonzero(x['weights']) > 1 for x in old)
    for baseline, selected in zip(old, new):
        for name in ('prior', 'active', 'beta', 'evidence', 'h_evidence', 'dominant'):
            np.testing.assert_array_equal(selected[name], baseline[name])
        expected = np.zeros_like(baseline['weights'])
        expected[baseline['dominant']] = 1.
        np.testing.assert_array_equal(selected['weights'], expected)


def test_disabled_is_exact_and_class_is_restored():
    before = Transition._finish_hypothesis_transition
    old = history(2, nullcontext())
    new = history(2, dominant_rule_execution(Transition))
    for a, b in zip(old, new):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert Transition._finish_hypothesis_transition is before
    with pytest.raises(ValueError, match='explicitly'):
        with dominant_rule_execution(Transition, enabled=1):
            pass
    with pytest.raises(RuntimeError, match='intentional'):
        with dominant_rule_execution(Transition, enabled=True):
            raise RuntimeError('intentional restoration check')
    assert Transition._finish_hypothesis_transition is before


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_snapshot_continuation_and_reject_mixed_weights(condition):
    cfg = configuration(condition, .3, .8)
    with episodic_admission(Transition, mode='POOL'), separate_belief_evidence(Transition, enabled=True), \
            dominant_rule_execution(Transition, enabled=True) as events:
        model = StateModel(cfg, context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(8):
            model.begin_trial([.8, .3, .2, .4])
            model.complete_trial(1, float(trial % 2))
        saved = deepcopy(model.engine.state_dict())
        clone = StateModel(cfg, context=ModelContext(condition=condition))
        clone.engine.load_state_dict(saved)
        other = clone.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        assert h.state_dict()['rule_execution_policy'] == POLICY
        assert not np.shares_memory(h.execution_probabilities, other.execution_probabilities)
        for trial in range(4):
            a, b = (m.begin_trial([.2, .7, .8, .6]) for m in (model, clone))
            np.testing.assert_array_equal(a.prior, b.prior)
            np.testing.assert_array_equal(a.beta, b.beta)
            np.testing.assert_array_equal(h.execution_probabilities, other.execution_probabilities)
            np.testing.assert_array_equal(h.belief_evidence, other.belief_evidence)
            for m in (model, clone):
                m.complete_trial(1, float(trial % 2))
        # Response selection occurs on the first trial as well; feedback
        # evidence updates start one trial later.
        assert Counter(e['trial'] for e in events) == {t: 1 if t < 8 else 2 for t in range(12)}
        state = deepcopy(h.state_dict())
        state['execution_probabilities'] = np.full(h.total_hypo, 1/h.total_hypo)
        with pytest.raises(ValueError, match='weights'):
            h.load_state_dict(state)
        state = deepcopy(h.state_dict()); state['rule_execution_policy'] = 'mixture'
        with pytest.raises(ValueError, match='policy'):
            h.load_state_dict(state)
