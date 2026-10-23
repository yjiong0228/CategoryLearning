"""Continuation keeps future observations sealed and retains filter grouping."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1023_prefix_forecast import forecast_inputs, write, digest
from src.Bayesian_state.evaluation.model_1023_prefix_forecast import assemble, conditional_mean
from src.Bayesian_state.simulation.continuation_diagnostics import ContinuationObserver
from src.Bayesian_state.model.modules.base_module import ModuleRole


class TaskOnly(dict):
    def __getitem__(self, key):
        if key not in ('stimulus', 'categories'):
            raise AssertionError('Future observed behavior was accessed')
        return super().__getitem__(key)


def test_future_inputs_expose_only_task_and_do_not_alias_source():
    training = dict(stimulus=np.ones((16,4)), categories=np.ones(16,int),
                    choices=np.full(16,2), feedback=np.zeros(16))
    future = TaskOnly(stimulus=np.zeros((3,4)), categories=np.full(3,3),
                      choices=np.full(3,4), feedback=np.ones(3))
    prefix, task = forecast_inputs(training,future,False)
    assert set(task)=={'stimulus','categories'}
    assert set(prefix)=={'stimulus','choices','feedback'}
    task['categories'][0]=1
    assert future['categories'][0]==3
    prefix, task=forecast_inputs(training,TaskOnly(),True)
    assert len(prefix['choices'])==8 and task['stimulus'].shape==(8,4)


def test_assembly_retains_endpoint_cloud_and_branch_ancestry(tmp_path):
    jobs=[]
    for seed, probabilities in [(11,[.9,.1]),(22,[.2,.8])]:
        target=tmp_path/str(seed)
        np.savez_compressed(target.with_suffix('.npz'),categories=[1,2],filter_seed=seed,
            terminal_weights=probabilities,endpoint_present=[1,0],prefix_probabilities=np.ones((4,2))/2,
            branch_seeds=[seed+100,seed+200],ancestors=[0,1],choices=[[1,2],[2,1]])
        write(target.with_suffix('.json'),dict(status='complete',prefix_exact=True,filter_seed=seed,
            predictions_sha256=digest(target.with_suffix('.npz'))))
        jobs.append(dict(target=str(target),seeds=[seed]))
    arrays,_=assemble(jobs)
    np.testing.assert_array_equal(arrays['branch_filter_seed'],[11,11,22,22])
    np.testing.assert_array_equal(arrays['terminal_weights'],[[.9,.1],[.2,.8]])
    assert arrays['endpoint_present'].shape==(2,2)
    assert arrays['choices'].shape==(4,2)
    (tmp_path/'11.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        assemble(jobs)


def test_diagnostic_uses_joint_rule_weight_without_treating_it_as_accuracy():
    h=SimpleNamespace(active=[0,2],transition_log=[dict(search_event=True,proposed_hypothesis=2,capacity_dropped=None)])
    memory=SimpleNamespace(conditional_pairing=lambda:np.array([[.7,.2,.1],[1/3]*3,[.1,.3,.6]]),
        pairing_marginal=lambda:np.array([.4,.25,.35]))
    engine=SimpleNamespace(beta=np.array([4.,8.,12.]),prior=np.array([.75,0,.25]),
        get_module=lambda role:h if role==ModuleRole.HYPOTHESIS_TRANSITION else memory)
    probabilities=np.array([[.8,.2],[.1,.9]])
    weights=np.array([.75,.25])
    prediction=SimpleNamespace(rule_snapshot=SimpleNamespace(hypothesis_indices=(0,2),rule_weights=weights,
        category_probabilities=probabilities),observed_probabilities=weights@probabilities)
    model=SimpleNamespace(engine=engine,n_cats=2)
    row=ContinuationObserver(2)(model,prediction)
    assert row['contribution']==.25 and row['beta']==12
    np.testing.assert_array_equal(row['target_probabilities'],[.1,.9])
    row['target_probabilities'][0]=99
    assert probabilities[1,0]==.1
    missing=ContinuationObserver(1)(model,prediction)
    assert not missing['present'] and missing['beta']==0 and missing['contribution']==0
    np.testing.assert_array_equal(missing['target_pairing'],np.zeros(3))


def test_precision_mean_conditions_on_presence():
    values=np.array([[0.,20.],[10.,0.]])
    mask=np.array([[False,True],[True,False]])
    assert conditional_mean(values,mask)==15
    assert np.isnan(conditional_mean(values,np.zeros_like(mask)))


def test_fixed_state_probe_retains_ties_absence_and_extreme_power_stability():
    from src.Bayesian_state.evaluation.model_1023_prefix_readout_probe import readout_probabilities
    p=np.array([[.4,.3,.2,.1],[.5,.5,0.,0.],[0.,0.,0.,0.]])
    sharp,hard=readout_probabilities(p,2.)
    np.testing.assert_allclose(sharp[0],np.array([16,9,4,1])/30)
    np.testing.assert_array_equal(hard,[[1,0,0,0],[.5,.5,0,0],[0,0,0,0]])
    np.testing.assert_array_equal(sharp[2],np.zeros(4))
    very_sharp,_=readout_probabilities(p,1e6)
    np.testing.assert_array_equal(very_sharp,hard)
    with pytest.raises(ValueError):readout_probabilities(p,-1)


@pytest.mark.parametrize('condition', [2,3])
def test_shared_ancestor_branches_restore_transfer_state_and_feedback(condition,monkeypatch,tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext,StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    from src.Bayesian_state.model.modules.hypothesis_transition.compact_evidence import compact_evidence
    from src.Bayesian_state.model.modules.hypothesis_transition.entry_precision import entry_precision
    from src.Bayesian_state.simulation.filtered_continuation import generate_continuations
    from src.Bayesian_state.metrics.task import category_learning_feedback
    from tests.bayesian_state.test_model_1021_compact_evidence import compact_config
    monkeypatch.setattr(ContinuousSimilarity,'RUNTIME_CACHE_DIR',tmp_path)
    cfg=compact_config(condition)
    x=np.tile([.8,.3,.2,.4],(16,1));categories=np.tile(np.arange(1,5),4)
    choices=np.ones(8,int)
    feedback=np.array([category_learning_feedback(1,int(c),condition=condition) for c in categories[:8]])
    with compact_evidence(Transition),entry_precision(Transition):
        model=StateModel(cfg,context=ModelContext(condition=condition))
        for stimulus,choice,outcome in zip(x[:8],choices,feedback):
            model.begin_trial(stimulus);model.complete_trial(int(choice),float(outcome))
        target=int(model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION).active[0])
        kwargs=dict(models=[model],weights=np.ones(1),observed_choices=choices,observed_feedback=feedback,
            stimulus=x[8:],categories=categories[8:],condition=condition)
        seeds=[101,202,303]
        plain=generate_continuations(**kwargs,branch_seeds=seeds)
        captured=generate_continuations(**kwargs,branch_seeds=seeds,observer=ContinuationObserver(target))
        reverse=generate_continuations(**kwargs,branch_seeds=seeds[::-1],observer=ContinuationObserver(target))
        np.testing.assert_array_equal(captured['ancestors'],[0,0,0])
        for key,value in plain.items():
            np.testing.assert_array_equal(value,captured[key],err_msg=key)
        for key,value in captured.items():
            other=reverse[key] if key in ('categories','terminal_weights') else reverse[key][::-1]
            np.testing.assert_array_equal(value,other,err_msg=key)
        expected=np.array([[category_learning_feedback(int(a),int(c),condition=condition)
                            for a,c in zip(path,categories[8:])] for path in captured['choices']])
        np.testing.assert_array_equal(captured['feedback'],expected)
        assert model._completed_trial_count==8 and model._pending_trial is None
