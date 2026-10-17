"""Check observational state capture, weighting timing, and numerical invariance."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml

from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.evaluation.model_0923_postfit import ParticleStateCollector, weighted_quantiles, score_probabilities


def config(condition,tmp_path):
    path=Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml')
    engine=yaml.safe_load(path.read_text())
    engine['oral_readout']=None
    engine['modules']['perception_mod']['kwargs'] = {'features':4, 'mean':[0.]*4, 'std':[0.]*4}
    engine['partition']['kwargs']['similarity_cache_dir']=str(tmp_path)
    engine['modules']['hypo_transitions_mod']['kwargs']['relation_samples']=32
    engine['provenance']['hypothesis_similarity']['sample_count']=32
    return engine


@pytest.mark.parametrize('condition',[1,2,3])
def test_observer_preserves_filter_and_collects_current_choice_timing(tmp_path,condition):
    engine=config(condition,tmp_path)
    stimulus=np.asarray([[.1,.8,.3,.6],[.7,.2,.5,.8],[.4,.9,.8,.2],[.6,.4,.1,.9]])
    choices=[1,2,1,2]
    feedback=[1.,0.,1.,.5 if condition==3 else 0.]
    arguments=dict(engine_config=engine,subject_id=1,condition=condition,stimulus=stimulus,
                   choices=choices,feedback=feedback,particle_count=4,choice_readout_power=1.,filter_seed=71)
    original=run_state_model_particle_filter(**arguments)
    collector=ParticleStateCollector()
    captured=[]
    def observer(t,phase,snapshot):
        assert not snapshot['belief'].flags.writeable
        collector(t,phase,snapshot)
        captured.append((t,phase,snapshot))
        # Even making a detached array writable cannot alter the owning filter.
        snapshot['belief'].setflags(write=True)
        snapshot['belief'].fill(-99.)
    observed=run_state_model_particle_filter(**arguments,state_observer=observer,
                                             observe_decision_basis=True)
    for name in ('observation_probabilities','state_probabilities','latent_summaries','diagnostics'):
        first,second=getattr(original,name),getattr(observed,name)
        assert first.keys()==second.keys()
        for key in first:np.testing.assert_array_equal(first[key],second[key],err_msg=key)
    arrays=collector.arrays()
    np.testing.assert_allclose(arrays['predictive_choice_weights'],observed.state_probabilities['choice_weights'],atol=1e-15)
    np.testing.assert_allclose(arrays['filtered_choice_weights'],observed.state_probabilities['filtered_choice_weights'],atol=1e-15)
    for t in range(4):
        before,after=captured[2*t][2],captured[2*t+1][2]
        selected=before['prediction'][:,choices[t]-1]
        reconstructed=np.einsum('ph,phc->pc', before['choice_weights'],
                                 before['rule_choice_probabilities'])
        np.testing.assert_allclose(reconstructed,before['prediction'],atol=1e-14,rtol=0)
        np.testing.assert_array_equal(before['perceived_stimulus'],
                                       np.broadcast_to(stimulus[t],(4,4)))
        posterior=before['weights']*selected
        posterior/=posterior.sum()
        np.testing.assert_allclose(after['weights'],posterior,atol=1e-15)
        np.testing.assert_allclose(before['weights']@before['prediction'],observed.observation_probabilities['prior_t'][t],atol=1e-15)


def test_participation_distinguishes_inference_uncertainty_from_path_mixture():
    collector=ParticleStateCollector()
    one=np.eye(2)
    values={'weights':np.array([.5,.5]),'belief':one,'active':one,'beta':one*5,
            'choice_weights':one,'dominant':one}
    for name in ('search','global_search','failure','search_event','admission_event'):values[name]=np.zeros(2)
    for phase in ('predictive','filtered'):collector(0,phase,values)
    result=collector.arrays()
    np.testing.assert_array_equal(result['predictive_effective_participation'],[[1.,1.]])
    np.testing.assert_array_equal(result['predictive_choice_weights'],[[.5,.5]])


def test_weighted_quantiles_ignore_zero_mass_and_preserve_joint_seed_weights():
    values=np.array([[0.,1.,2.,10.]])
    weights=np.array([[0.,.1,.8,.1]])
    np.testing.assert_array_equal(weighted_quantiles(values,weights),[[1.,2.,2.]])
    with pytest.raises(ValueError):weighted_quantiles(values,np.zeros_like(weights))


def test_choice_scores_exclude_initialization_and_use_probability_averaging():
    probabilities=np.array([[.99,.01],[.25,.75],[.6,.4]])
    score=score_probabilities(probabilities,np.array([2,2,1]))
    assert score['mean_choice_nll']==pytest.approx(-np.log([.75,.6]).mean())


def test_particle_blocks_are_rejected_before_observer_dispatch(tmp_path):
    with pytest.raises(ValueError,match='particle_blocks=1'):
        run_state_model_particle_filter(engine_config=config(1,tmp_path),subject_id=1,
            stimulus=[[.2]*4],choices=[1],feedback=[1.],particle_count=4,particle_blocks=2,
            choice_readout_power=1.,state_observer=lambda *a:None)


def test_decision_basis_requires_observer_and_explicit_mixture(tmp_path):
    arguments=dict(engine_config=config(1,tmp_path),subject_id=1,
                   stimulus=[[.2]*4],choices=[1],feedback=[1.],particle_count=4,
                   choice_readout_power=1.,observe_decision_basis=True)
    with pytest.raises(ValueError,match='requires state_observer'):
        run_state_model_particle_filter(**arguments)
    with pytest.raises(ValueError,match='untransformed rule-mixture'):
        run_state_model_particle_filter(**arguments,state_observer=lambda *a:None,output_lapse=.1)
