"""Completed-feedback timing, cache cloning and continuous H belief assignment."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.continuous_recalled_belief import continuous_recalled_belief
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import (
    Episode, episodic_admission, recalled_episodes, recalled_log_evidence)
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal import (
    episodic_proposal, recalled_prior)


def config(condition):
    cfg = yaml.safe_load(Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    cfg['oral_readout'] = None
    cfg['modules']['perception_mod']['kwargs'] = dict(features=4, mean=[0.] * 4, std=[0.] * 4, module_seed=2)
    h = cfg['modules']['hypo_transitions_mod']['kwargs']
    h.update(intervention=dict(kind='E', decay=.8, capacity='evidence_mean', search='accumulate'),
             relation_samples=128, module_seed=3, max_capacity=3, exit_rate=0.)
    h['search_controller']['baseline_probability'] = .999
    cfg['provenance']['hypothesis_similarity']['sample_count'] = 128
    return cfg


def test_off_schedule_does_not_patch_methods():
    class Fake:
        pass
    with continuous_recalled_belief(Fake) as events:
        assert events == [] and not hasattr(Fake, '_assign_v3_prior')
    with pytest.raises(ValueError, match='schedule'):
        with continuous_recalled_belief(Fake, schedule='unknown'):
            pass


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('schedule', ['entry', 'every_trial'])
def test_real_engine_recall_uses_only_completed_feedback_and_clone_is_exact(condition, schedule, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = config(condition)
    with episodic_admission(Transition, mode='POOL'), continuous_recalled_belief(Transition, schedule=schedule) as events:
        a = StateModel(cfg, context=ModelContext(condition=condition))
        h = a.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(4):
            # Exercise a known non-admission step as well as the search branch.
            h.baseline_probability = 1e-12 if trial == 2 else .999
            pending = a.begin_trial([.8, .3, .2, .4])
            if h.belief_recall_cache is not None:
                assert h.belief_recall_cache.count <= trial
                episodes, weights = recalled_episodes(h.admission_history, .8)
                evidence, _ = recalled_log_evidence(a.engine.observation_likelihood, episodes,
                    np.arange(h.total_hypo), a.engine.get_module(ModuleRole.BETA).beta_init)
                expected = recalled_prior(h.base_prior, h.active, np.arange(h.total_hypo), weights @ evidence)
                if schedule == 'every_trial' or h.last_transition_result.selection.newcomers.size:
                    assert h.belief_recall_cache.count == trial
                    np.testing.assert_allclose(pending.prior, expected, atol=1e-11, rtol=1e-12)
            a.complete_trial(1, .5 if condition == 3 else 0.)
        state = deepcopy(a.engine.state_dict())
        b = StateModel(cfg, context=ModelContext(condition=condition))
        b.engine.load_state_dict(state)
        other = b.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        assert h.belief_recall_cache is other.belief_recall_cache
        for trial in range(4, 7):
            first, second = [m.begin_trial([.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(first.prior, second.prior)
            np.testing.assert_array_equal(h.execution_probabilities, other.execution_probabilities)
            assert h.trial_rng.bit_generator.state == other.trial_rng.bit_generator.state
            for model in (a, b):
                model.complete_trial(1, float(trial % 2))
        assert events
        if schedule == 'every_trial':
            assert any(event['newcomer_count'] == 0 for event in events)
            assert all(event['belief_recalled_episodes'] == event['trial'] for event in events)
            assert all(event['belief_new_episodes'] == 1 for event in events)
        else:
            assert all(event['newcomer_count'] == 1 for event in events)
        broken = deepcopy(h.state_dict())
        broken['admission_history'] = Episode((.1, .2, .3, .4), 1, 0., None)
        with pytest.raises(ValueError, match='history'):
            h.load_state_dict(broken)


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_entry_schedule_reproduces_existing_entry_only_prior(condition, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = config(condition)
    trajectories = []
    for legacy in (True, False):
        context = episodic_proposal(Transition, prior=True) if legacy else continuous_recalled_belief(Transition, schedule='entry')
        with episodic_admission(Transition, mode='POOL'), context:
            model = StateModel(cfg, context=ModelContext(condition=condition))
            h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            rows = []
            for trial in range(10):
                pending = model.begin_trial([.8, .3, .2, .4])
                rows.append((pending.prior.copy(), h.execution_probabilities.copy(), h.active.copy()))
                model.complete_trial(1, .5 if condition == 3 and trial % 2 else float(trial % 2))
            trajectories.append(rows)
    for first, second in zip(*trajectories):
        np.testing.assert_allclose(first[0], second[0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(first[1], second[1], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(first[2], second[2])
