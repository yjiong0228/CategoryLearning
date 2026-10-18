import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_association_autonomous import generate, block_scores
from src.Bayesian_state.workflows.analysis.model_0923_association_diagnostic import replay
from src.Bayesian_state.metrics.task import category_learning_feedback


@pytest.mark.parametrize('condition', [2,3])
@pytest.mark.parametrize('learn', [False,True])
def test_own_history_replays_exactly_without_task_truth_in_learner(condition,learn):
    q=np.random.default_rng(23).dirichlet(np.ones(4),size=(18,3)); labels=np.arange(18)%4+1
    arm=dict(learn_associations=learn,association_retention='stable')
    a=generate(q,labels,gamma=.8,arm=arm,condition=condition,seed=72)
    b=generate(q,labels,gamma=.8,arm=arm,condition=condition,seed=72)
    c=replay(q,a['choices'],a['feedback'],gamma=.8,arm=arm,condition=condition)
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(a['probabilities'],c['probabilities'])
    expected=[category_learning_feedback(y,c,condition=condition) for y,c in zip(a['choices'],labels)]
    np.testing.assert_array_equal(a['feedback'],expected)


def test_future_task_truth_cannot_change_current_prediction_or_choice():
    q=np.random.default_rng(43).dirichlet(np.ones(4),size=(10,3));labels=np.arange(10)%4+1
    arm=dict(learn_associations=True,association_retention='stable')
    a=generate(q,labels,gamma=.8,arm=arm,condition=3,seed=44)
    changed=labels.copy();changed[4:]=labels[4:]%4+1
    b=generate(q,changed,gamma=.8,arm=arm,condition=3,seed=44)
    np.testing.assert_array_equal(a['probabilities'][:5],b['probabilities'][:5])
    np.testing.assert_array_equal(a['choices'][:5],b['choices'][:5])
    assert not np.array_equal(a['probabilities'][5:],b['probabilities'][5:])


def test_u_energy_matches_independent_pair_enumeration():
    x=np.array([[0.,1.],[.5,.5],[1.,0.],[1.,1.]]);y=np.array([.75,.75])
    metrics,jack=block_scores(x,y)
    def value(z):
        target=sum(np.linalg.norm(a-y) for a in z)/len(z)
        pairs=sum(np.linalg.norm(a-b) for i,a in enumerate(z) for j,b in enumerate(z) if i!=j)
        return (target-pairs/(2*len(z)*(len(z)-1)))/np.sqrt(2)
    assert metrics['energy_u']==pytest.approx(value(x))
    np.testing.assert_allclose(jack['energy_u'],[value(np.delete(x,i,axis=0)) for i in range(len(x))])


def test_common_streams_preserve_equivalent_fixed_binary_arms():
    q=np.random.default_rng(12).dirichlet(np.ones(4),size=(18,2));labels=np.arange(18)%4+1
    outputs=[generate(q,labels,gamma=.3,arm=dict(learn_associations=False,association_retention=mode),
                      condition=2,seed=44) for mode in ['shared','stable']]
    for key in outputs[0]:np.testing.assert_array_equal(outputs[0][key],outputs[1][key])
