"""Temporal selection uses complete inner folds and original training updates."""
import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_rnn_temporal_validation import fold_bounds, choose_epochs


@pytest.mark.parametrize('length,expected', [(128,[(64,96),(96,128)]),
    (672,[(336,504),(504,672)]),(384,[(192,288),(288,384)]),(512,[(256,384),(384,512)])])
def test_forward_folds_remain_inside_each_original_prefix(length, expected):
    assert fold_bounds(length,[.5,.75,1.]) == expected


@pytest.mark.parametrize('length,fractions', [(3,[.5,.75,1.]),(128,[.75,.5,1.]),
    (128,[0,.75,1.]),(128,[.5,.75,1.1]),(128,[.5,np.nan,1.])])
def test_invalid_folds_rejected(length, fractions):
    with pytest.raises(ValueError): fold_bounds(length,fractions)


def scores():
    return pd.DataFrame([dict(subject=107,seed=seed,fold=fold,epochs=epoch,
        validation_trials=10 if fold==0 else 20,
        validation_nll=(.1 if fold==0 else 1.) if epoch==1 else .6)
        for seed in (11,12) for fold in (0,1) for epoch in (1,51)])


def test_epoch_selection_pools_trials_and_all_initializations():
    table = scores(); summary, selected = choose_epochs(table,[107],[11,12],[1,51])
    assert selected == {107:51}
    assert summary.set_index('epochs').loc[1,'validation_nll'] == pytest.approx(.7)
    table['outer_nll'] = 0.; table['oral_agreement'] = 1.
    assert choose_epochs(table,[107],[11,12],[1,51])[1] == selected


def test_shorter_exact_tie_wins_without_selecting_a_lucky_seed():
    table = scores(); table['validation_nll'] = np.where(table.seed==11,.3,.9)
    assert choose_epochs(table,[107],[11,12],[1,51])[1] == {107:1}


@pytest.mark.parametrize('fault',['missing','duplicate','nan','unequal_trials','negative_nll'])
def test_incomplete_or_invalid_selection_rejected(fault):
    table = scores()
    if fault=='missing': table = table.iloc[:-1]
    elif fault=='duplicate': table = pd.concat([table,table.iloc[:1]])
    elif fault=='nan': table.loc[0,'validation_nll'] = np.nan
    elif fault=='unequal_trials': table.loc[0,'validation_trials'] = 9
    else: table.loc[0,'validation_nll'] = -1
    with pytest.raises(ValueError): choose_epochs(table,[107],[11,12],[1,51])


def test_returned_original_checkpoint_equals_direct_shorter_training():
    torch = pytest.importorskip('torch')
    from src.Bayesian_state.workflows.analysis.model_0923_rnn_comparator import ROOT,RNN,rnn_module,replay
    torch.set_num_threads(1)
    module = rnn_module(ROOT/RNN,2)
    arrays = dict(stimulus=np.array([[.2,.3,.8,.1],[.7,.3,.4,.2],[.4,.8,.1,.7],[.9,.2,.3,.1]]),
                  choices=np.array([1,2,1,2]),feedback=np.array([1.,0.,1.,1.]))
    torch.manual_seed(23)
    _,_,snapshots = module.train(arrays['stimulus'],arrays['choices'],arrays['feedback'],H=16,epochs=6,lr=.001)
    torch.manual_seed(23)
    shorter,_,_ = module.train(arrays['stimulus'],arrays['choices'],arrays['feedback'],H=16,epochs=3,lr=.001)
    for key,value in shorter.state_dict().items():
        np.testing.assert_array_equal(value.detach().numpy(),snapshots[2].state_dict()[key].detach().numpy())
    np.testing.assert_array_equal(replay(module,shorter,arrays)['probabilities'],
                                  replay(module,snapshots[2],arrays)['probabilities'])
