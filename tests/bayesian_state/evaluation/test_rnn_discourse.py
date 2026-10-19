"""Frozen text diagnostics require exact order, task boundary and coordinates."""
import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.model_0923_rnn_discourse import verify_trial_identity


def frame():
    return pd.DataFrame(dict(subject=107,state_trial=np.arange(192),iSession=1,iTrial=np.arange(1,193),
        choice=np.ones(192,int),phase=np.where(np.arange(192)<128,'training','later'),nonempty=True,
        has_contrast=False,named_or_collective_mask=1,before_mask=0,after_mask=0,encoded_mask=1))


def test_original_chronology_is_accepted_without_changing_text_table():
    table=frame(); before=table.copy(deep=True)
    verify_trial_identity(table,np.ones(192,int),128)
    pd.testing.assert_frame_equal(table,before)


@pytest.mark.parametrize('fault',['order','identity','subject','boundary','phase','choice','mask','flag'])
def test_changed_text_to_model_alignment_is_rejected(fault):
    table=frame();cut=128
    if fault=='order':table=table.iloc[::-1]
    if fault=='identity':table.loc[1,'iTrial']=1
    if fault=='subject':table.loc[0,'subject']=208
    if fault=='boundary':cut=127
    if fault=='phase':table.loc[127,'phase']='later'
    if fault=='choice':table.loc[0,'choice']=2
    if fault=='mask':table.loc[0,'after_mask']=16
    if fault=='flag':table['has_contrast']=table.has_contrast.astype(int)
    with pytest.raises((ValueError,AssertionError)):
        verify_trial_identity(table,np.ones(192,int),cut)
