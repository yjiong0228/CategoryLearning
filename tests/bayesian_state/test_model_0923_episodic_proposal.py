"""Recalled proposal support and one-time prior reconstruction invariants."""
import numpy as np
import pytest
from copy import deepcopy
from pathlib import Path
import yaml

from src.Bayesian_state.model.modules.hypothesis_transition import episodic_proposal as module
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import Episode
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal import (
    _RecalledDraw, advance_recall, episodic_proposal, recalled_prior, tilt_log_evidence,
)


def test_log_tilt_preserves_support_and_is_invariant_to_score_offset():
    q = np.array([.2, 0., .8])
    scores = np.array([-10000., 10000., -10002.])
    value = tilt_log_evidence(q, scores)
    assert value[1] == 0 and np.isfinite(value).all()
    np.testing.assert_allclose(value.sum(), 1.)
    np.testing.assert_allclose(value, tilt_log_evidence(q, scores + 20000.))
    np.testing.assert_allclose(value[0]/value[2], .25*np.exp(2))


def test_prior_uses_final_set_and_common_evidence_once():
    base = np.array([.1, .2, .3, .4])
    value = recalled_prior(base, [3, 1], [0, 1, 3], [100., np.log(2), 0.])
    np.testing.assert_allclose(value, [0., .5, 0., .5])
    with pytest.raises(ValueError, match='Every retained rule'):
        recalled_prior(base, [2, 3], [0, 1, 3], [0., 0., 0.])


def test_draw_delegates_route_rng_and_changes_only_catalogue_probability():
    class RNG:
        def __init__(self): self.received = []
        def random(self): return .72
        def choice(self, a, *args, **kwargs):
            self.received.append((a, kwargs['p'].copy()))
            return np.asarray(a)[0] if np.ndim(a) else 0
    rng = RNG(); adapter = _RecalledDraw(rng, np.array([4, 8]), np.log([1., 3.]))
    assert adapter.random() == .72
    adapter.choice(2, p=np.array([.4, .6]))
    adapter.choice(np.array([4, 8]), p=np.array([.5, .5]))
    np.testing.assert_array_equal(rng.received[0][1], [.4, .6])
    np.testing.assert_allclose(rng.received[1][1], [.25, .75])
    assert adapter.calls == 1


def test_unmodified_arm_and_nonadmission_prior_delegate_without_rng():
    class Module:
        def __init__(self):
            self.admission_history = None
            self.configuration_signature = 'original'
        def _draw_v3_proposal(self, *args): return 17
        def _assign_v3_prior(self, context, selection): return 'original_prior'
    originals = [Module.__init__, Module._draw_v3_proposal, Module._assign_v3_prior]
    with episodic_proposal(Module):
        assert originals == [Module.__init__, Module._draw_v3_proposal, Module._assign_v3_prior]
    class Selection:
        newcomers = np.array([], int)
    with episodic_proposal(Module, prior=True):
        module = Module()
        assert module.configuration_signature != 'original'
        assert module._assign_v3_prior(None, Selection()) == 'original_prior'
        assert module._draw_v3_proposal(None, None, None, {}) == 17
    assert originals == [Module.__init__, Module._draw_v3_proposal, Module._assign_v3_prior]


@pytest.mark.parametrize('decay', [0., .8, 1.])
def test_incremental_cache_matches_full_episode_recall_and_survives_clone(monkeypatch, decay):
    calls = []
    def evidence(likelihood, episodes, rules, beta):
        calls.append(len(episodes))
        return np.array([[e.stimulus[0] * (h + 1) for h in rules] for e in episodes]), .9
    monkeypatch.setattr(module, 'recalled_log_evidence', evidence)
    chain, head = [], None
    for value in [.2, .5, .9, .4]:
        head = Episode((value,), 1, 1., None, head)
        chain.append(head)
    options = dict(decay=decay, likelihood=None, rules=np.arange(3), beta=5.)
    first, _ = advance_recall(chain[1], None, **options)
    assert deepcopy(first) is first
    cached, _ = advance_recall(chain[-1], deepcopy(first), **options)
    full, _ = advance_recall(chain[-1], None, **options)
    assert calls == [2, 2, 4]
    np.testing.assert_allclose(cached.total, full.total, atol=1e-14, rtol=0)
    np.testing.assert_allclose(cached.exposure, full.exposure, atol=1e-14, rtol=0)
    assert cached.count == 4
    same, _ = advance_recall(chain[-1], cached, **options)
    assert same is cached and calls == [2, 2, 4]
    with pytest.raises(ValueError, match='ancestor'):
        advance_recall(chain[0], cached, **options)


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('proposal,prior', [(True, False), (False, True), (True, True)])
def test_real_engine_clone_preserves_new_mechanisms_and_full_recall(condition, proposal, prior, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import (
        episodic_admission, recalled_episodes, recalled_log_evidence)
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = yaml.safe_load(Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    cfg['oral_readout'] = None
    cfg['modules']['perception_mod']['kwargs'] = dict(features=4, mean=[0.]*4, std=[0.]*4, module_seed=2)
    h_cfg = cfg['modules']['hypo_transitions_mod']['kwargs']
    h_cfg.update(intervention=dict(kind='E', decay=.8, capacity='evidence_mean', search='accumulate'),
                 relation_samples=128, module_seed=3, max_capacity=2, exit_rate=0.)
    h_cfg['search_controller']['baseline_probability'] = .999
    cfg['provenance']['hypothesis_similarity']['sample_count'] = 128
    with episodic_admission(Transition, mode='POOL'), episodic_proposal(Transition, proposal=proposal, prior=prior):
        a = StateModel(cfg, context=ModelContext(condition=condition))
        for _ in range(3):
            a.begin_trial([.8, .3, .2, .4]); a.complete_trial(1, .5 if condition == 3 else 0.)
        h_a = a.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        b = StateModel(cfg, context=ModelContext(condition=condition))
        b.engine.load_state_dict(deepcopy(a.engine.state_dict()))
        h_b = b.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        if proposal:
            assert h_a.proposal_recall_cache is h_b.proposal_recall_cache
        for t in range(5):
            p_a, p_b = [m.begin_trial([.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(p_a.prior, p_b.prior)
            np.testing.assert_array_equal(h_a.execution_probabilities, h_b.execution_probabilities)
            np.testing.assert_array_equal(h_a.active, h_b.active)
            assert h_a.trial_rng.bit_generator.state == h_b.trial_rng.bit_generator.state
            if proposal:
                cache = h_a.proposal_recall_cache
                episodes, weights = recalled_episodes(cache.head, .8)
                evidence, _ = recalled_log_evidence(a.engine.observation_likelihood, episodes,
                    np.arange(h_a.total_hypo), a.engine.get_module(ModuleRole.BETA).beta_init)
                np.testing.assert_allclose(cache.total, weights @ evidence, atol=1e-11, rtol=1e-12)
            for m in (a, b):
                m.complete_trial(1, float(t % 2))
