import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_region_association_autonomous import generate
from src.Bayesian_state.workflows.analysis.model_0923_region_association_diagnostic import replay


@pytest.mark.parametrize('rate',[None,.5])
def test_generated_history_replays_exactly_and_feedback_uses_own_choice(rate):
    q=np.tile(np.array([[[.1,.2,.3,.4],[.4,.3,.2,.1]]]),(12,1,1));labels=np.tile([1,2,3,4],3)
    a=generate(q,labels,gamma=.8,rate=rate,seed=17)
    b=generate(q,labels,gamma=.8,rate=rate,seed=17)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_array_equal(a['feedback'],a['choices']==labels)
    response=replay(q,a['choices'],a['feedback'],gamma=.8,rate=rate,trace=True)
    np.testing.assert_array_equal(response['probabilities'],a['probabilities'])
    np.testing.assert_array_equal(response['final_associations'],a['final_associations'])


def test_future_categories_cannot_change_current_choice():
    q=np.tile(np.array([[[.1,.2,.3,.4],[.4,.3,.2,.1]]]),(12,1,1));labels=np.tile([1,2,3,4],3)
    a=generate(q,labels,gamma=.8,rate=2.,seed=17);labels[5:]=4
    b=generate(q,labels,gamma=.8,rate=2.,seed=17)
    np.testing.assert_array_equal(a['probabilities'][:6],b['probabilities'][:6])
    np.testing.assert_array_equal(a['choices'][:6],b['choices'][:6])


def test_invalid_categories_rejected():
    with pytest.raises(ValueError,match='condition-2'):
        generate(np.full((2,1,4),.25),np.array([1,5]),gamma=1.,rate=.5,seed=17)
