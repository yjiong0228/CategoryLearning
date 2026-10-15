"""Whole-path deletion retains the empirical distribution scoring convention."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_h_generation_uncertainty import path_jackknife
from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se


def test_identical_paths_have_zero_simulation_error():
    scores=path_jackknife(np.tile([.1,.9],(4,1)),np.array([.2,.7]))
    assert jackknife_se(scores['energy'])==pytest.approx(0.)
    assert jackknife_se(scores['crps'])==pytest.approx(0.)


def test_single_block_deletions_have_known_empirical_energy_scores():
    scores=path_jackknife(np.array([[0.],[1.],[1.]]),np.array([0.]))
    np.testing.assert_allclose(scores['energy'],[1.,.25,.25])
    np.testing.assert_allclose(scores['crps'],scores['energy'])
    assert jackknife_se(scores['energy'])==pytest.approx(.5)


def test_screen_contrasts_preserve_pairing_and_detect_changed_paths(tmp_path):
    import json
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_h_validation import score_screen
    from src.Bayesian_state.evaluation.model_0923_h_generation_uncertainty import screen_comparisons
    source=tmp_path/'screen';directory=source/'participant';directory.mkdir(parents=True)
    manifest=dict(status='complete',smoke=False,records=[dict(id='participant',subject=101)],
        settings=dict(variants={'B':{},'JF':{}},comparisons=[['JF','B']]))
    (source/'simulation_manifest.json').write_text(json.dumps(manifest))
    pd.DataFrame(dict(choice=[1]*32,category=[1]*32)).to_csv(directory/'observed_trials.csv',index=False)
    for model in ('B','JF'):
        (directory/model).mkdir()
        np.savez_compressed(directory/model/'correct_paths.npz',seeds=[4,5,6],
            realized=np.tile([0.,1.,1.],(32,1)),correct_probability=np.tile([0.,1.,1.],(32,1)))
    score_screen(source,tmp_path/'scores')
    screen_comparisons(source,tmp_path/'scores',tmp_path/'uncertainty')
    row=pd.read_csv(tmp_path/'uncertainty/participants.csv').iloc[0]
    assert row.delta_energy==0 and row.delta_energy_mc_se==0
    assert row.delta_crps==0 and row.delta_crps_mc_se==0
    with (directory/'JF/correct_paths.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='paths changed'):
        screen_comparisons(source,tmp_path/'scores',tmp_path/'invalid')
