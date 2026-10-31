"""Check that fit diagnostics retain every trial across session boundaries."""

import numpy as np
import pandas as pd

from src.SUSTAIN.report import block_summary


def test_blocks_retain_partial_segments_and_never_cross_sessions():
    frame = pd.DataFrame(dict(iSession=[1]*5+[2]*4, global_trial=np.arange(1,10),
                              correct=[1,1,0,0,1,0,0,1,1], p_correct=np.arange(1,10)/10,
                              choice_nll=np.ones(9)))
    blocks = block_summary(frame, 3)
    assert blocks.n.tolist() == [3,2,3,1]
    assert blocks.session.tolist() == [1,1,2,2]
    assert blocks.first_trial.tolist() == [1,4,6,9]
    assert blocks.last_trial.tolist() == [3,5,8,9]
    assert blocks.n.sum() == 9
    assert blocks.choice_nll.sum() == 9
    np.testing.assert_allclose(np.average(blocks.observed_accuracy,weights=blocks.n),frame.correct.mean())
