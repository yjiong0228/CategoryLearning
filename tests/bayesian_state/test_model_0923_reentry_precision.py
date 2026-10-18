"""Rule-specific precision retention, snapshot isolation and exact continuation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision import reentry_precision


class TinyTransition:
    def __init__(self):
        self.continuous_choice_enabled = True
        self._STATE_FIELDS = ()
        self.total_hypo = 4
        self.configuration_signature = 'base'
        self.transition_log, self.transition_rate_log = [], []
        beta = SimpleNamespace(beta=np.array([3., 8., 0., 0.]), beta_min=.01,
                               beta_max=25., beta_update_mode='feedback_gradient')
        self.engine = SimpleNamespace(get_module=lambda role: beta, beta=beta.beta.copy())

    def _finish_hypothesis_transition(self, context, selection, prior, **kwargs):
        beta = self.engine.get_module(None)
        beta.beta[selection.newcomers] = 1.
        beta.beta[np.setdiff1d(np.arange(4), selection.active_after)] = 0.
        self.engine.beta = beta.beta.copy()
        self.transition_log.append({}); self.transition_rate_log.append({})
        return self.transition_log[-1]

    def load_state_dict(self, state):
        if state['signature'] != self.configuration_signature:
            raise ValueError('different mechanism')
        for key in self._STATE_FIELDS:
            setattr(self, key, deepcopy(state[key]))

    def state_dict(self):
        return dict(signature=self.configuration_signature,
                    **{key: deepcopy(getattr(self, key)) for key in self._STATE_FIELDS})


def step(module, before, after, trial):
    context = SimpleNamespace(active_before=np.array(before), trial_index=trial)
    selection = SimpleNamespace(newcomers=np.setdiff1d(after, before), active_after=np.array(after))
    return module._finish_hypothesis_transition(context, selection, np.ones(4)/4)


def test_last_active_value_returns_and_absent_rule_receives_no_learning():
    with reentry_precision(TinyTransition) as records:
        h = TinyTransition(); beta = h.engine.get_module(None)
        step(h, [0, 1], [1, 2], 0)
        np.testing.assert_array_equal(beta.beta, [0, 8, 1, 0])
        np.testing.assert_array_equal(h.reentry_precision_values, [3, 8, 0, 0])
        beta.beta[1], beta.beta[2] = 9, 4
        step(h, [1, 2], [1, 2], 1)
        assert h.reentry_precision_values[0] == 3
        event = step(h, [1, 2], [0, 2], 2)
        np.testing.assert_array_equal(beta.beta, [3, 0, 4, 0])
        np.testing.assert_array_equal(h.engine.beta, beta.beta)
        assert event['precision_restored_rules'] == [0]
        assert event['precision_restored_values'] == [3]
        step(h, [0, 2], [1, 2], 3)
        assert beta.beta[1] == 9
        assert [row['trial'] for row in records] == [0, 1, 2, 3]


def test_disabled_mode_changes_no_methods_state_or_rng_contract():
    methods = [getattr(TinyTransition, k) for k in ('__init__', '_finish_hypothesis_transition', 'load_state_dict')]
    with reentry_precision(TinyTransition, enabled=False) as records:
        h = TinyTransition(); step(h, [0, 1], [1, 2], 0)
        assert h._STATE_FIELDS == () and records == []
        assert methods == [getattr(TinyTransition, k) for k in ('__init__', '_finish_hypothesis_transition', 'load_state_dict')]
    with pytest.raises(RuntimeError):
        with reentry_precision(TinyTransition): raise RuntimeError('fixture failure')
    assert methods == [getattr(TinyTransition, k) for k in ('__init__', '_finish_hypothesis_transition', 'load_state_dict')]


@pytest.mark.parametrize('change', ['nan', 'negative', 'above_bound', 'unseen_nonzero', 'seen_dtype', 'missing', 'shape'])
def test_invalid_cache_is_rejected_before_import(change):
    with reentry_precision(TinyTransition):
        h = TinyTransition(); step(h, [0, 1], [1, 2], 0); state = h.state_dict()
        if change == 'nan': state['reentry_precision_values'][0] = np.nan
        elif change == 'negative': state['reentry_precision_values'][0] = -.1
        elif change == 'above_bound': state['reentry_precision_values'][0] = 26
        elif change == 'unseen_nonzero': state['reentry_precision_values'][3] = 1
        elif change == 'seen_dtype': state['reentry_precision_seen'] = state['reentry_precision_seen'].astype(int)
        elif change == 'missing': del state['reentry_precision_values']
        else: state['reentry_precision_values'] = np.zeros(5)
        before = h.reentry_precision_values.copy()
        with pytest.raises(ValueError, match='snapshot'): h.load_state_dict(state)
        np.testing.assert_array_equal(before, h.reentry_precision_values)


def test_snapshot_cache_copies_are_isolated():
    with reentry_precision(TinyTransition):
        a = TinyTransition(); step(a, [0, 1], [1, 2], 0)
        snapshot = a.state_dict(); b = TinyTransition(); b.load_state_dict(snapshot)
        snapshot['reentry_precision_values'][0] = 20
        b.reentry_precision_values[0] = 10
        assert a.reentry_precision_values[0] == 3


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_real_search_engine_snapshot_continues_exactly(condition, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import episodic_admission
    from src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal import episodic_proposal
    cfg = yaml.safe_load(Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    cfg['oral_readout'] = None
    cfg['modules']['perception_mod']['kwargs'] = dict(features=4, mean=[0.]*4, std=[0.]*4, module_seed=2)
    h_cfg = cfg['modules']['hypo_transitions_mod']['kwargs']
    h_cfg.update(intervention=dict(kind='E', decay=.8, capacity='evidence_mean', search='accumulate'),
                 relation_samples=128, module_seed=3, max_capacity=2, exit_rate=0.)
    h_cfg['search_controller']['baseline_probability'] = .999
    cfg['provenance']['hypothesis_similarity']['sample_count'] = 128
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    with episodic_admission(Transition, mode='POOL'), episodic_proposal(Transition, proposal=True), reentry_precision(Transition):
        a = StateModel(cfg, context=ModelContext(condition=condition))
        for _ in range(3):
            a.begin_trial([.8, .3, .2, .4]); a.complete_trial(1, .5 if condition == 3 else 0.)
        b = StateModel(cfg, context=ModelContext(condition=condition))
        b.engine.load_state_dict(deepcopy(a.engine.state_dict()))
        ha, hb = [m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in (a, b)]
        assert ha.reentry_precision_values is not hb.reentry_precision_values
        assert ha.reentry_precision_seen.any()
        for trial in range(5):
            pa, pb = [m.begin_trial([.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(pa.prior, pb.prior)
            np.testing.assert_array_equal(pa.beta, pb.beta)
            np.testing.assert_array_equal(ha.active, hb.active)
            np.testing.assert_array_equal(ha.execution_probabilities, hb.execution_probabilities)
            np.testing.assert_array_equal(ha.reentry_precision_seen, hb.reentry_precision_seen)
            np.testing.assert_array_equal(ha.reentry_precision_values, hb.reentry_precision_values)
            assert ha.trial_rng.bit_generator.state == hb.trial_rng.bit_generator.state
            for m in (a, b): m.complete_trial(1, float(trial % 2))
