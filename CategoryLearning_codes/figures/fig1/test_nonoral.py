"""Boundary geometry, lag boundaries and independent period definitions."""
import numpy as np
import pandas as pd
from CategoryLearning_codes.figures.fig1.nonoral import analyze


def test_distance_uses_current_branch_and_feedback_never_crosses_block_or_session():
    frame=pd.DataFrame({'condition':[3]*4,'iSub':[301]*4,'iSession':[1,1,1,2],
                        'iBlock':[1,1,2,1],'iTrial':[1,2,3,1],'choRT':[2,4,8,16],
                        'feedback':[0,.5,1,0],'correct':[0,0,1,0],
                        'feature1':[.2,.8,.2,.8], 'feature2':[.4,.5,.4,.5],
                        'feature3':[.5,.9,.5,.9]})
    trials,summary,*_=analyze(frame,window=2)
    np.testing.assert_allclose(trials.boundary_distance,[.1,.3,.1,.3])
    assert trials.previous_feedback.isna().tolist()==[True,False,True,True]
    assert trials.previous_feedback.iloc[1]==0
    assert summary.first_median_rt.iloc[0]==3
    assert summary.last_median_rt.iloc[0]==12
    short=analyze(frame.iloc[:3],window=2)[1]
    assert short.first_median_rt.isna().all()
    assert short.last_median_rt.isna().all()


def test_invalid_rt_does_not_remove_accuracy_observations():
    frame=pd.DataFrame({'condition':[1]*2,'iSub':[101]*2,'iSession':[1]*2,
                        'iBlock':[1]*2,'iTrial':[1,2],'choRT':[0,np.nan],
                        'feedback':[1,0],'correct':[1,0], 'feature1':[.6,.6],
                        'feature2':[.5,.5],'feature3':[.5,.5]})
    trials,summary,_,_,boundary=analyze(frame,window=1)
    assert len(trials)==2
    assert summary.invalid_rt_n.iloc[0]==2
    assert boundary.n.sum()==2
    assert boundary.accuracy.iloc[0]==.5
    np.testing.assert_allclose(trials.boundary_distance,.1)
