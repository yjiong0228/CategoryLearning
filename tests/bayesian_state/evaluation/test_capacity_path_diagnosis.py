import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_capacity_path_diagnosis import path_windows, summarize_paths


def fixture():
    present = np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    return dict(present=present, contribution=present*np.array([[.5], [0], [.9]]),
                beta=present*3, target_exited=np.zeros_like(present),
                target_capacity_dropped=np.array([[0, 0, 1, 0, 0], [0]*5, [0]*5]),
                seeds=np.arange(3))


def test_censoring_and_post_discovery_absence_partition_all_paths():
    rows = path_windows(fixture(), 2).query("window == 'whole'").set_index('seed')
    np.testing.assert_allclose(rows.pre_discovery_absence+rows.post_discovery_absence+rows.presence, 1)
    assert rows.loc[0, 'first_presence_censored'] == 2
    assert rows.loc[0, 'post_discovery_absence'] == .4
    assert rows.loc[0, 'entries'] == 2
    assert rows.loc[1, 'first_presence_censored'] == 6
    assert rows.loc[1, 'ever_discovered'] == 0
    assert rows.loc[1, 'pre_discovery_absence'] == 1
    assert rows.loc[1, 'post_discovery_absence'] == 0


def test_conditional_contribution_pools_present_trials_and_retains_empty_paths():
    rows = path_windows(fixture(), 2).query("window == 'whole'")
    values, deletes = summarize_paths(rows)
    assert values['contribution_when_present'] == pytest.approx(5.5/7)
    assert values['presence']*values['contribution_when_present'] == pytest.approx(values['contribution'])
    assert deletes['contribution_when_present'][0] == pytest.approx(.9)
    assert values['first_presence_censored'] == 3


@pytest.mark.parametrize('key', ['present', 'target_exited', 'target_capacity_dropped'])
def test_invalid_flags_rejected(key):
    arrays = fixture(); arrays[key][0, 0] = 2
    with pytest.raises(ValueError): path_windows(arrays, 2)


def test_unrecorded_loss_rejected():
    arrays = fixture(); arrays['target_capacity_dropped'][:] = 0
    with pytest.raises(ValueError, match='cause'): path_windows(arrays, 2)


def test_contribution_while_absent_rejected():
    arrays = fixture(); arrays['contribution'][1, 2] = .1
    with pytest.raises(ValueError, match='zero while absent'): path_windows(arrays, 2)


def test_no_present_trials_has_no_conditional_estimate():
    arrays = fixture()
    for key in ('present', 'contribution', 'beta', 'target_capacity_dropped'): arrays[key][:] = 0
    rows = path_windows(arrays, 2).query("window == 'whole'")
    values, deletes = summarize_paths(rows)
    assert np.isnan(values['contribution_when_present'])
    assert np.isnan(deletes['contribution_when_present']).all()
    assert values['ever_discovered'] == 0
    assert values['pre_discovery_absence'] == 1
