"""Generated process summaries preserve event types and trial boundaries."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_generated_process import process_summary


def test_process_summary_keeps_search_and_removal_channels_separate():
    a=dict(rule=np.array([[[1,-1]],[[1,2]],[[2,3]],[[1,3]]]),
        contribution=np.array([[[1.,np.nan]],[[.7,.3]],[[.4,.6]],[[.6,.4]]]),
        proposed=np.array([[1],[2],[3],[1]]),source=np.array([[0],[1],[1],[2]]),
        admitted=np.array([[1],[1],[0],[1]],bool),rejected=np.array([[0],[0],[1],[0]],bool),
        search_probability=np.array([[0.],[.2],[1.],[.3]]),
        capacity_dropped=np.array([[-1],[-1],[1],[-1]]),
        exits=np.array([[[-1,-1]],[[-1,-1]],[[-1,-1]],[[2,-1]]]),
        joint_dropped=np.array([[[-1,-1]],[[3,-1]],[[-1,-1]],[[-1,-1]]]))
    rows={r['phase']:r for r in process_summary(a,np.array([1]),2)}
    assert rows['training']['proposals']==1
    assert rows['test']['proposals']==2 and rows['test']['admissions']==1 and rows['test']['rejections']==1
    assert rows['test']['compatible_capacity_drops']==1 and rows['test']['compatible_exits']==0
    assert rows['training']['joint_drops']==1 and rows['test']['joint_drops']==0
    assert rows['test']['compatible_presence']==.5
    assert rows['test']['compatible_contribution']==pytest.approx(.3)
    assert rows['test']['search_near_one_fraction']==.5
