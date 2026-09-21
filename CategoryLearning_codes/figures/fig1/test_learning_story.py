"""The behavior overview must retain short and non-crossing records."""
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig1.build_learning_story import behavior_table


def test_short_record_retained_without_inventing_shape_or_criterion():
    trials=pd.DataFrame({'iSub':[105]*64,'trial':np.arange(1,65),'correct':np.tile([0,1],32)})
    summary=pd.DataFrame({'iSub':[105],'condition':[1],'first_crossing64':[np.nan],
                          'last64_accuracy':[.5],'n_trials':[64]})
    result=behavior_table(trials,summary)
    assert len(result)==1
    assert result.preferred.iloc[0]=='unavailable'
    assert np.isnan(result.delta_bic.iloc[0])
    assert not result.criterion_reached.iloc[0]
    assert result.task.iloc[0]==1
