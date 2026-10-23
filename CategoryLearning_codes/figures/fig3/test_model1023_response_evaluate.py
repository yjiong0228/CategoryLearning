"""State plots preserve conditional precision and missing-rule intervals."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model1023_response_evaluate import conditional_state_blocks


def test_block_precision_conditions_on_presence_and_keeps_absent_blocks():
    presence=np.zeros(64);presence[:2]=[.1,.9]
    precision=np.full(64,np.nan);precision[:2]=[10.,20.]
    source=pd.DataFrame(dict(subject=205,arm='Q1',trial=np.arange(1,65),training_trials=32,
        target_present=presence,target_contribution=presence/2,
        target_precision_given_presence=precision))
    blocks=conditional_state_blocks(source)
    assert blocks.trial.tolist()==[32,64]
    assert blocks.iloc[0].target_present==pytest.approx(1/32)
    assert blocks.iloc[0].target_contribution==pytest.approx(1/64)
    assert blocks.iloc[0].target_precision_given_presence==pytest.approx(19.)
    assert np.isnan(blocks.iloc[1].target_precision_given_presence)
    missing=source.copy();missing.loc[1,'target_precision_given_presence']=np.nan
    with pytest.raises(ValueError,match='Missing precision'):conditional_state_blocks(missing)
    changed=source.copy();changed.loc[1,'target_precision_given_presence']=50.
    with pytest.raises(ValueError,match='declared model range'):conditional_state_blocks(changed)
