import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_region_association_diagnostic import replay


def test_future_choice_and_feedback_do_not_change_earlier_predictions():
    q=np.tile(np.array([[[.1,.2,.3,.4],[.4,.3,.2,.1]]]),(8,1,1))
    y=np.array([1,2,3,4,2,3,1,4]);f=np.array([1.,0.,1.,0.,1.,0.,0.,1.])
    before=replay(q,y,f,gamma=.8,rate=.5,trace=True)
    y2=y.copy();y2[4:]=1;f2=f.copy();f2[4:]=1-f[4:]
    after=replay(q,y2,f2,gamma=.8,rate=.5,trace=True)
    np.testing.assert_array_equal(before['probabilities'][:5],after['probabilities'][:5])
    assert not np.array_equal(before['probabilities'][5:],after['probabilities'][5:])
    np.testing.assert_array_equal(before['observed_probability'],before['probabilities'][np.arange(8),y-1])


def test_prefix_replay_is_identical_to_prefix_of_longer_replay():
    q=np.tile(np.array([[[.1,.2,.3,.4],[.4,.3,.2,.1]]]),(8,1,1))
    y=np.array([1,2,3,4,2,3,1,4]);f=np.array([1.,0.,1.,0.,1.,0.,0.,1.])
    full=replay(q,y,f,gamma=.3,rate=2.,trace=True)
    prefix=replay(q[:4],y[:4],f[:4],gamma=.3,rate=2.,trace=True)
    np.testing.assert_array_equal(prefix['probabilities'],full['probabilities'][:4])
    np.testing.assert_array_equal(prefix['associations'],full['associations'][:4])


def test_graded_feedback_is_not_treated_as_binary():
    with pytest.raises(ValueError,match='condition-2'):
        replay(np.full((2,1,4),.25),np.array([1,2]),np.array([1.,.5]),gamma=1.,rate=.5)
