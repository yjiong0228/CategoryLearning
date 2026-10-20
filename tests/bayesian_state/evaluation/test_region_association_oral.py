import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_region_association_oral import tokens_key,summarize


def test_tokens_key_is_content_only_and_rejects_non_tokens():
    assert tokens_key('["b","a","a"]')==('a','b')
    with pytest.raises(ValueError):tokens_key('[1]')


def test_summary_preserves_scopes_boundaries_and_feedback():
    rows=[]
    for t,(scope,feedback,support) in enumerate([('complete',1.,.2),('complete',0.,.4),('parsed_fragment',0.,.9),('needs_review',1.,np.nan)]):
        row=dict(reading='midpoint',iSub=207,arm='ROW',training_trials=2,state_trial=t,scope=scope,feedback=feedback)
        for timing in ['conditional','prechoice']:
            for domain in ['structural','functional']:
                row.update({timing+'_'+domain+'_support':support,timing+'_'+domain+'_conflict':0. if np.isfinite(support) else np.nan,
                    timing+'_'+domain+'_undetermined':1-support})
        rows.append(row)
    result=summarize(pd.DataFrame(rows));complete=result[(result.window=='training')&(result.scope=='complete')&(result.domain=='functional')]
    assert complete.set_index('feedback_stratum').reports.to_dict()=={'all':2,'correct':1,'incorrect':1}
    assert complete[complete.feedback_stratum=='all'].conditional_support.iloc[0]==pytest.approx(.3)
    later=result[(result.window=='test')&(result.feedback_stratum=='all')&(result.domain=='functional')]
    assert len(later)==1 and later.iloc[0].scope=='parsed_fragment' and later.iloc[0].reports==1
