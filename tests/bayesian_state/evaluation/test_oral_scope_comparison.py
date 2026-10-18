"""Check the scientific boundary between revised coverage and revised scores."""
import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_scope_comparison import compare_frames


def frames():
    common=dict(iSub=[308,308],arm=['BASE','BASE'],ordinal_trial=[1,2],iSession=[1,1],iTrial=[1,2],
                choice=[1,2],text=['脖子短','头长于一半'],training_trials=[1,1],mean_choice_probability=[.4,.5])
    old=pd.DataFrame(dict(common,scope=['complete','needs_review'],tokens=['neck','[]']))
    new=pd.DataFrame(dict(common,scope=['complete','complete'],tokens=['neck','head']))
    for p in ('','prior_'):
        for d in ('structural','functional'):
            for v,values in [('support',[.3,.8]),('conflict',[.1,.1]),('undetermined',[.6,.1])]:
                new[p+d+'_'+v]=values;old[p+d+'_'+v]=[values[0],np.nan]
    return old,new


def test_added_reports_are_separate_and_original_scores_stay_fixed():
    summary,added=compare_frames(*frames())
    assert len(added)==1 and added.ordinal_trial.iloc[0]==2
    s=summary.query("window=='whole' and domain=='structural'").set_index('scope')
    assert s.loc['original_complete','support']==pytest.approx(.3)
    assert s.loc['newly_complete','support']==pytest.approx(.8)
    assert s.loc['revised_complete','support']==pytest.approx(.55)


@pytest.mark.parametrize('column,value',[('choice',4),('structural_support',.4),('scope','needs_review')])
def test_changed_behavior_or_original_complete_score_is_rejected(column,value):
    old,new=frames();new.loc[0,column]=value
    with pytest.raises((ValueError,AssertionError)):compare_frames(old,new)
