"""Causal timing, admission competition and resumable intervention state."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.special import expit, logit

from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
from src.Bayesian_state.simulation.provenance import build_model_provenance


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    partition = ContinuousPartition(4, 2)
    def build(intervention, **overrides):
        n = len(partition.hypothesis_space)
        engine = SimpleNamespace(partition=partition, set_size=n, distance_mode='boundary',
            prior=np.full(n, 1/n), beta=np.ones(n), get_module=lambda *a, **kw: None)
        kwargs = dict(mechanism_version='0923-full-v3', max_capacity=2, exit_rate=.1,
            transfer_scale=.1, execution_inertia=0., relation_samples=128, relation_seed=0, module_seed=23,
            edit_costs=dict(feature=1., relation=1.5, reference=1., branch=2.),
            search_controller=dict(baseline_probability=.005, error_gain=1., failure_decay=.6,
                global_search=.3, global_search_failure_gain=.5), intervention=intervention)
        kwargs.update(overrides)
        return Transition(engine, **kwargs)
    return build


class Draws:
    def random(self, size=None):
        return np.ones(size) if size is not None else 0.
    def choice(self, values, p=None):
        return 2 if p is not None and not isinstance(values, int) else (0 if isinstance(values, int) else values[0])


def install(model, mass=(.8, .2)):
    model.active = np.array([0, 1])
    model.executed_hypothesis = 0
    model.trial_index = 1
    model.engine.posterior = np.zeros(model.total_hypo)
    model.engine.posterior[:2] = mass
    model.engine.prior = model.engine.posterior.copy()
    model.execution_probabilities = model.engine.posterior.copy()
    model.current_event_probability = model.current_g = 1.
    model._apply_mask()


def test_failure_accumulation_can_exceed_old_search_ceiling_without_invalid_range(make):
    model = make({'kind': 'S1'})
    for _ in range(12):
        model.record_outcome((None, 1, 0.))
        model._prepare_hypothesis_transition()
    assert model.accumulated_failure == 12
    assert model.current_event_probability > .9
    assert 0 <= model.current_g <= 1 and model.failure_pressure <= 1
    model.record_outcome((None, 1, 1.))
    model._prepare_hypothesis_transition()
    assert model.accumulated_failure == pytest.approx(7.2)


def test_search_control_multiplies_odds_exactly(make):
    base, control = make({'kind':'B'}), make({'kind':'S_control','factor':4.})
    for model in (base, control):
        model.record_outcome((None, 1, .5))
        model._prepare_hypothesis_transition()
    assert logit(control.current_event_probability)-logit(base.current_event_probability) == pytest.approx(np.log(4))
    assert base.current_g == control.current_g


def test_c1_rejects_distant_newcomer_and_preserves_old_beliefs(make):
    model = make({'kind':'C1'})
    install(model)
    model._distance_matrix = np.full((model.total_hypo, model.total_hypo), 1.)
    model.trial_rng = Draws()
    model.process()
    np.testing.assert_array_equal(model.active, [0, 1])
    np.testing.assert_allclose(model.engine.prior[:2], [.8, .2])
    event = model.transition_log[-1]
    assert event['search_event'] and not event['admission_event']
    assert event['newcomer_rejected'] and event['capacity_dropped'] is None
    assert event['swap_probability'] == 0


def test_c1_accepts_newcomer_and_removes_weakest_old_after_common_comparison(make):
    model = make({'kind':'C1'})
    install(model)
    model._distance_matrix = np.zeros((model.total_hypo, model.total_hypo))
    model.trial_rng = Draws()
    # The smallest temporary mass is old rule 1; draw ties by actual index.
    model.trial_rng.choice = lambda values, p=None: 2 if p is not None else values[0]
    model.process()
    assert model.active.tolist() == [0, 2]
    np.testing.assert_allclose(model.engine.prior[[0, 2]], [.5, .5])
    assert model.transition_log[-1]['swap_probability'] == pytest.approx(1)


def test_r1_uses_previous_completed_likelihood_and_sliding_window(make):
    model = make({'kind':'R1','window':4})
    install(model)
    model.engine.likelihood = np.ones(model.total_hypo)*.25
    for _ in range(5):
        model.record_outcome((None, 1, 1.))
        model._prepare_hypothesis_transition()
    np.testing.assert_allclose(model.recent_log_evidence[:2].sum(axis=1), 4*np.log(.5))
    assert model.recent_evidence_count[:2].tolist() == [4, 4]
    model.engine.likelihood[:2] = 1.
    model.record_outcome((None, 1, 1.))
    model._prepare_hypothesis_transition()
    np.testing.assert_allclose(model.recent_log_evidence[:2].sum(axis=1), 2*np.log(.5))


@pytest.mark.parametrize('intervention', [{'kind':'R1','window':0}, {'kind':'R1','window':True},
    {'kind':'S_control','factor':0}, {'kind':'R_control','factor':2}, {'kind':'S1','extra':1}])
def test_invalid_interventions_rejected(make, intervention):
    with pytest.raises(ValueError):
        make(intervention)


@pytest.mark.parametrize('condition', [1,2,3])
@pytest.mark.parametrize('intervention', [{'kind':'S1'}, {'kind':'R1','window':4}, {'kind':'C1'}])
def test_full_model_state_resume_includes_causal_intervention_state(condition, intervention, monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    path = Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml')
    cfg = yaml.safe_load(path.read_text())
    cfg['modules']['perception_mod']['kwargs'] = dict(features=4, mean=[0.]*4, std=[0.]*4, module_seed=2)
    cfg['modules']['hypo_transitions_mod']['kwargs'].update(intervention=intervention, relation_samples=128, module_seed=3)
    cfg['provenance']['hypothesis_similarity']['sample_count'] = 128
    a = StateModel(cfg, context=ModelContext(condition=condition))
    a.begin_trial([.8,.3,.2,.4]); a.complete_trial(1, .5 if condition==3 else 0.)
    snapshot = a.engine.state_dict()
    b = StateModel(cfg, context=ModelContext(condition=condition))
    b.engine.load_state_dict(deepcopy(snapshot))
    pa, pb = a.begin_trial([.2,.7,.8,.6]), b.begin_trial([.2,.7,.8,.6])
    np.testing.assert_array_equal(pa.prior, pb.prior)
    ta, tb = [m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in (a,b)]
    assert ta.state_dict().keys() == tb.state_dict().keys()
    for key in ta._STATE_FIELDS:
        np.testing.assert_equal(getattr(ta,key),getattr(tb,key))
    if intervention['kind']=='R1':
        assert ta.recent_evidence_count.max() <= 1
    assert build_model_provenance(cfg, repeat_aggregation='mean')['resolved']['intervention'] == intervention


def test_r1_preserves_positive_evidence_rule_and_clears_exited_history(make):
    model = make({'kind':'R1','window':4}, exit_rate=1.)
    install(model)
    model.recent_log_evidence[0, -1] = -2.
    model.recent_log_evidence[1, -1] = 1.
    model.recent_evidence_count[:2] = 1
    model.trial_rng = Draws()
    model.trial_rng.random = lambda size=None: np.zeros(size) if size is not None else 1.
    model.process()
    np.testing.assert_array_equal(model.active, [1])
    assert model.recent_evidence_count[0] == 0
    assert model.recent_log_evidence[0].sum() == 0
    assert model.recent_log_evidence[1].sum() == 1
    assert model.transition_log[-1]['autonomous_exits'] == [0]


def test_c1_admission_probability_includes_random_tie_rejection(make):
    # Scale 1 makes exp(-log(2)) exactly 0.5 in floating point.
    model = make({'kind':'C1'}, transfer_scale=1.)
    install(model, mass=(.5,.5))
    model._distance_matrix = np.full((model.total_hypo,model.total_hypo), model.transfer_scale*np.log(2))
    model.trial_rng = Draws()
    model.trial_rng.choice = lambda values,p=None: 2 if p is not None else values[-1]
    model.process()
    assert model.transition_log[-1]['newcomer_rejected']
    assert model.transition_log[-1]['swap_probability'] == pytest.approx(2/3)
    np.testing.assert_array_equal(model.active,[0,1])
