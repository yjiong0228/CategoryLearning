"""The report bridge conditions within states before particle marginalization."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.decision_basis import DecisionBasisCollector
from src.Bayesian_state.evaluation.oral.response_basis import ResponseBasisCollector
from src.Bayesian_state.model.modules.readout.category_power import category_power


class Space(list):
    n_cats=2
    n_dims=1
    signature='test_single_component_rules'


def test_joint_bridge_preserves_within_state_basis_and_updates_particle_weights():
    space=Space([SimpleNamespace(categories=[SimpleNamespace(components=[None]) for _ in range(2)]) for _ in range(2)])
    geometry=SimpleNamespace(space=space)
    w=np.array([[.8,.2],[.2,.8]])
    likelihood=np.tile([[.9,.1],[.2,.8]],(2,1,1));p=np.array([.4,.6])
    raw=np.einsum('ph,phc->pc',w,likelihood);q=category_power(raw,2.)
    snapshot=dict(weights=p,choice_weights=w,rule_choice_probabilities=likelihood,
        perceived_stimulus=np.array([[.2],[.8]]),active=np.ones((2,2)),prediction=q,
        belief=w,beta=np.ones((2,2))*4,search=np.array([0,1]),admission_event=np.array([0,1]))
    collector=ResponseBasisCollector(space,np.array([1]),geometry=geometry,exponent=2.)
    collector(0,'predictive',snapshot);arrays=collector.arrays()
    within=w*likelihood[:,:,0]/raw[:,0,None]
    expected=(p[:,None]*q[:,0,None]*within).sum(0)
    np.testing.assert_allclose(arrays['choice_component_joint_mass'][0,:,0],expected,atol=1e-14,rtol=0)
    np.testing.assert_allclose(arrays['conditional_particle_weights'][0],p*q[:,0]/(p@q[:,0]),atol=1e-14,rtol=0)
    np.testing.assert_allclose(arrays['prediction'][0],p@q,atol=1e-14,rtol=0)
    assert not np.allclose(p@q,category_power(p@raw,2.))
    np.testing.assert_array_equal(snapshot['rule_choice_probabilities'],likelihood)
    original=dict(snapshot,prediction=raw)
    a=DecisionBasisCollector(space,[1],geometry=geometry)
    b=ResponseBasisCollector(space,[1],geometry=geometry,exponent=1.)
    for c in (a,b):c(0,'predictive',original)
    for k,v in a.arrays().items():np.testing.assert_array_equal(v,b.arrays()[k])


@pytest.mark.parametrize('condition',[1,2,3])
def test_response_basis_observation_leaves_filter_and_terminal_states_unchanged(condition,monkeypatch,tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    from src.Bayesian_state.model.modules.hypothesis_transition.compact_evidence import compact_evidence
    from src.Bayesian_state.model.modules.hypothesis_transition.entry_precision import entry_precision
    from src.Bayesian_state.model.modules.readout.category_power import category_power_readout
    from src.Bayesian_state.simulation.filtered_continuation import capture_prefix
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.metrics.task import category_learning_feedback
    from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
    from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
    from tests.bayesian_state.test_model_1021_compact_evidence import compact_config
    monkeypatch.setattr(ContinuousSimilarity,'RUNTIME_CACHE_DIR',tmp_path)
    space=build_continuous_hypothesis_space(4,2) if condition==1 else build_relational_0923_space()
    choices=np.tile(np.arange(1,space.n_cats+1),8//space.n_cats)
    categories=np.roll(choices,1)
    feedback=np.array([category_learning_feedback(int(a),int(c),condition=condition) for a,c in zip(choices,categories)])
    args=dict(engine_config=compact_config(condition),subject_id=108,condition=condition,
        stimulus=np.tile([[.8,.3,.2,.4],[.2,.7,.8,.6]],(4,1)),choices=choices,feedback=feedback,
        particle_count=4,particle_blocks=1,choice_readout_power=1.,filter_seed=17)
    collector=ResponseBasisCollector(space,choices,exponent=2.)
    with compact_evidence(Transition),entry_precision(Transition),category_power_readout(2.):
        plain_models,plain=capture_prefix(**args)
        observed_models,observed=capture_prefix(**args,state_observer=collector,observe_decision_basis=True)
    np.testing.assert_array_equal(plain.observation_probabilities['prior_t'],observed.observation_probabilities['prior_t'])
    np.testing.assert_array_equal(plain.final_weights,observed.final_weights)
    saved=collector.arrays()
    np.testing.assert_allclose(saved['prediction'],observed.observation_probabilities['prior_t'],atol=1e-14,rtol=0)
    np.testing.assert_allclose(saved['choice_component_joint_mass'].sum((1,2)),saved['choice_probability'],atol=1e-12,rtol=0)
    for a,b in zip(plain_models,observed_models):
        np.testing.assert_array_equal(a.engine.prior,b.engine.prior)
        np.testing.assert_array_equal(a.engine.beta,b.engine.beta)
        ha,hb=[m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in (a,b)]
        np.testing.assert_array_equal(ha.active,hb.active)
        np.testing.assert_array_equal(ha.retention_evidence,hb.retention_evidence)
        assert ha.trial_rng.bit_generator.state==hb.trial_rng.bit_generator.state
