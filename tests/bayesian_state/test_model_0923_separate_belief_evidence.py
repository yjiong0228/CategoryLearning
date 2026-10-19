"""Nested-model equivalence, feedback timing and particle-state continuation."""
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import (
    Episode, episodic_admission, recalled_episodes, recalled_log_evidence)
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition)
from src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief import online_evidence_belief
from src.Bayesian_state.model.modules.hypothesis_transition.separate_belief_evidence import separate_belief_evidence
from tests.bayesian_state.test_model_0923_continuous_recalled_belief import config


@pytest.fixture(autouse=True)
def isolated_similarity(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)


def configuration(condition, gamma, delta):
    cfg = config(condition)
    cfg['modules']['memory_mod']['kwargs']['gamma'] = gamma
    cfg['modules']['hypo_transitions_mod']['kwargs']['intervention']['decay'] = delta
    return cfg


def trajectory(cfg, condition, adapter):
    rows = []
    with episodic_admission(Transition, mode='POOL'), adapter:
        model = StateModel(cfg, context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(12):
            h.baseline_probability = 1e-12 if trial == 2 else .999
            pending = model.begin_trial([.8, .3, .2, .4])
            rows.append((pending.prior.copy(), h.execution_probabilities.copy(), h.active.copy(),
                         pending.beta.copy(), h.retention_evidence.copy()))
            model.complete_trial(1, .5 if condition == 3 and trial % 2 == 0 else float(trial % 2))
    return rows


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('decay', [0., .8, 1.])
def test_equal_decays_exactly_reproduce_online(condition, decay):
    cfg = configuration(condition, decay, decay)
    old = trajectory(cfg, condition, online_evidence_belief(Transition, enabled=True))
    new = trajectory(cfg, condition, separate_belief_evidence(Transition, enabled=True))
    for first, second in zip(old, new):
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


def test_disabled_is_exact_and_methods_are_restored():
    cfg = configuration(2, .3, .8)
    method = Transition._prepare_hypothesis_transition
    old = trajectory(cfg, 2, nullcontext())
    new = trajectory(cfg, 2, separate_belief_evidence(Transition))
    for first, second in zip(old, new):
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
    assert Transition._prepare_hypothesis_transition is method
    with pytest.raises(ValueError, match='explicitly'):
        with separate_belief_evidence(Transition, enabled=1):
            pass


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('gamma', [0., .3, 1.])
def test_separate_history_timing_reentry_and_clone(condition, gamma):
    cfg = configuration(condition, gamma, .8)
    with episodic_admission(Transition, mode='POOL'), separate_belief_evidence(Transition, enabled=True) as events:
        model = StateModel(cfg, context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        entries, drops = 0, 0
        for trial in range(12):
            before = h.active.copy()
            old = h.belief_evidence.copy()
            h_old = h.retention_evidence.copy()
            if trial:
                feedback = h.previous_feedback
                categories = model.engine.partition.n_cats
                chance = (.5 if feedback == 0 else .25) if condition == 3 else (
                    1 / categories if feedback == 1 else 1 - 1 / categories)
                evidence = np.log(np.maximum(model.engine.likelihood[before], np.finfo(float).tiny)) - np.log(chance)
            h.baseline_probability = 1e-12 if trial == 2 else .999
            pending = model.begin_trial([.8, .3, .2, .4])
            if trial:
                selection = h.last_transition_result.selection
                keep = selection.survivors
                indices = [int(np.flatnonzero(before == rule).item()) for rule in keep]
                np.testing.assert_allclose(h.belief_evidence[keep], gamma * old[keep] + evidence[indices], atol=1e-12, rtol=0)
                np.testing.assert_allclose(h.retention_evidence[keep], .8 * h_old[keep] + evidence[indices], atol=1e-12, rtol=0)
                if selection.newcomers.size:
                    episodes, weights = recalled_episodes(h.admission_history, gamma)
                    values, _ = recalled_log_evidence(model.engine.observation_likelihood, episodes,
                        selection.newcomers, model.engine.get_module(ModuleRole.BETA).beta_init)
                    np.testing.assert_allclose(h.belief_evidence[selection.newcomers], weights @ values, atol=1e-12, rtol=0)
                    entries += 1
                drops += len(selection.dropped)
                inactive = np.setdiff1d(np.arange(h.total_hypo), h.active)
                np.testing.assert_array_equal(h.belief_evidence[inactive], 0.)
                scores = h.belief_evidence[h.active]
                weights = h.base_prior[h.active] * np.exp(scores - scores.max())
                np.testing.assert_allclose(pending.prior[h.active], weights / weights.sum(), atol=1e-12, rtol=0)
            consumed = h.belief_evidence.copy()
            model.complete_trial(1, .5 if condition == 3 and trial % 2 == 0 else float(trial % 2))
            np.testing.assert_array_equal(h.belief_evidence, consumed)
        assert entries and drops
        assert len(events) == 11 and any(e['newcomer_count'] == 0 for e in events)
        assert all(e['gamma'] == gamma and e['delta'] == .8 for e in events)
        state = deepcopy(model.engine.state_dict())
        clone = StateModel(cfg, context=ModelContext(condition=condition))
        clone.engine.load_state_dict(state)
        other = clone.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        assert not np.shares_memory(h.belief_evidence, other.belief_evidence)
        for trial in range(3):
            a, b = [m.begin_trial([.2, .7, .8, .6]) for m in (model, clone)]
            np.testing.assert_array_equal(a.prior, b.prior)
            np.testing.assert_array_equal(h.execution_probabilities, other.execution_probabilities)
            np.testing.assert_array_equal(h.belief_evidence, other.belief_evidence)
            np.testing.assert_array_equal(h.belief_exposure, other.belief_exposure)
            for m in (model, clone):
                m.complete_trial(1, float(trial % 2))
        for key, value in [('belief_gamma', .2), ('belief_evidence', np.full(h.total_hypo, np.nan)),
                           ('belief_exposure', np.full(h.total_hypo, -1.))]:
            broken = deepcopy(h.state_dict()); broken[key] = value
            with pytest.raises(ValueError, match='[Bb]elief'):
                h.load_state_dict(broken)


def test_newcomer_gamma_history_includes_episodes_beyond_delta_underflow():
    cfg = configuration(2, 1., .3)
    with episodic_admission(Transition, mode='POOL'), separate_belief_evidence(Transition, enabled=True):
        model = StateModel(cfg, context=ModelContext(condition=2))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 0.)
        head = None
        for _ in range(700):
            head = Episode((.8, .3, .2, .4), 1, 0., None, head)
        h.admission_history = head
        model.begin_trial([.8, .3, .2, .4])
        selection = h.last_transition_result.selection
        assert selection.newcomers.size == 1
        assert selection.diagnostics['recalled_episodes'] < 700
        assert selection.diagnostics['belief_newcomer_episodes'] == 700
        assert h.belief_exposure[selection.newcomers[0]] == 700
