"""Category response power leaves cognitive feedback semantics intact."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model.modules.readout.category_power import category_power, category_power_readout


def test_power_is_stable_and_one_is_exact():
    p=np.array([[.4,.3,.2,.1],[.5,.5,0.,0.]])
    np.testing.assert_array_equal(category_power(p,1),p)
    np.testing.assert_allclose(category_power(p,2)[0],np.array([16,9,4,1])/30)
    np.testing.assert_array_equal(category_power(p,1e6),[[1,0,0,0],[.5,.5,0,0]])
    for value in (np.zeros(4),np.array([-.1,1.1]),np.array([np.nan,1.])):
        with pytest.raises(ValueError):category_power(value,2)
    for value in (0,-1,np.nan,np.inf):
        with pytest.raises(ValueError):
            with category_power_readout(value):pass


def test_single_rule_power_matches_precision_scaling_but_mixture_does_not():
    from src.Bayesian_state.hypothesis_space.geometry.boundary import BoundaryGeometry
    from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
    geometry=BoundaryGeometry(build_relational_0923_space())
    x=np.array([[.1,.8,.3,.2],[.9,.6,.1,.2]])
    p=geometry.category_probabilities(42,x,8.).T
    doubled=geometry.category_probabilities(42,x,16.).T
    np.testing.assert_allclose(category_power(p,2.),doubled,atol=1e-12,rtol=0)
    other=geometry.category_probabilities(0,x,5.).T
    other_doubled=geometry.category_probabilities(0,x,10.).T
    mixed=.6*p+.4*other
    assert not np.allclose(category_power(mixed,2.),.6*doubled+.4*other_doubled,atol=1e-6)


@pytest.mark.parametrize('condition',[1,2,3])
def test_filter_and_generation_read_same_power_without_changing_states(condition,monkeypatch,tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext,StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    from src.Bayesian_state.model.modules.hypothesis_transition.compact_evidence import compact_evidence
    from src.Bayesian_state.model.modules.hypothesis_transition.entry_precision import entry_precision
    from src.Bayesian_state.model.modules.readout import choice
    from src.Bayesian_state.inference.backends.particle_filter import _choice_probability
    from tests.bayesian_state.test_model_1021_compact_evidence import compact_config
    monkeypatch.setattr(ContinuousSimilarity,'RUNTIME_CACHE_DIR',tmp_path)
    with compact_evidence(Transition),entry_precision(Transition):
        model=StateModel(compact_config(condition),context=ModelContext(condition=condition))
        pending=model.begin_trial([.8,.3,.2,.4])
        settings=dict(choices=[],feedback=[],choice_readout_config=choice.resolve_choice_readout_config(None,model.engine_config),
            output_noise_config={'enabled':False},rng=np.random.default_rng(6),sticky_state={},post_error_lapse_state=0.)
        original=model.predict_choice(**settings)
        raw,snapshot=choice.read_choice_probabilities_from_model(model,pending.perceived_stimulus,return_snapshot=True)
        np.testing.assert_allclose(original.observed_probabilities,raw,atol=1e-12,rtol=0)
        prior,beta=model.engine.prior.copy(),model.engine.beta.copy()
        h=model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        rng_state=deepcopy(h.trial_rng.bit_generator.state)
        read,predict=choice.read_choice_probabilities_from_model,choice.predict_choice_from_model
        with category_power_readout(2.) as audit:
            filtered,retained=_choice_probability(model,pending.perceived_stimulus,1.,0.,return_snapshot=True)
            simulated=model.predict_choice(**settings)
            np.testing.assert_allclose(filtered,category_power(raw,2.),atol=1e-12,rtol=0)
            np.testing.assert_allclose(simulated.observed_probabilities,filtered,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(simulated.cognitive_probabilities,original.cognitive_probabilities)
            np.testing.assert_array_equal(retained.rule_weights,snapshot.rule_weights)
            np.testing.assert_array_equal(retained.category_probabilities,snapshot.category_probabilities)
            assert audit['filter_reads']==audit['trajectory_reads']==1
            np.testing.assert_array_equal(model.engine.prior,prior)
            np.testing.assert_array_equal(model.engine.beta,beta)
            assert h.trial_rng.bit_generator.state==rng_state
        assert choice.read_choice_probabilities_from_model is read
        assert choice.predict_choice_from_model is predict
        with pytest.raises(RuntimeError):
            with category_power_readout(2.):raise RuntimeError('deliberate exit')
        assert choice.read_choice_probabilities_from_model is read
        with category_power_readout(1.):
            assert choice.read_choice_probabilities_from_model is read
