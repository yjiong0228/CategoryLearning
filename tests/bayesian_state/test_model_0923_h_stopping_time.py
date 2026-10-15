"""The horizon category retains every non-passing simulated trajectory."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_h_stopping_time import capped_first_pass_times
from src.Bayesian_state.evaluation.model_0923_h_validation import distribution_scores


def test_event_at_horizon_and_non_event_remain_distinct():
    result = capped_first_pass_times(np.array([1, 20, 20]), np.array([True, True, False]), 20)
    np.testing.assert_array_equal(result, [1., 20., 21.])


@pytest.mark.parametrize('lengths,events,horizon', [
    ([19], [False], 20), ([21], [True], 20), ([0], [True], 20),
    ([1.5], [True], 20), ([np.nan], [True], 20), ([1], [1], 20),
    ([1], [True, False], 20), ([], [], 20), ([1], [True], True),
    ([True], [True], 20), ([1], [True], 0), ([1], [True], 20.5),
])
def test_incomplete_or_invalid_timing_is_rejected(lengths, events, horizon):
    with pytest.raises(ValueError):
        capped_first_pass_times(np.array(lengths), np.array(events), horizon)


def test_crps_penalizes_early_and_late_predictions_and_retains_censoring():
    target = np.array([3.])
    values = []
    for blocks, passed in [([2, 2], [True, True]), ([4, 4], [True, True]),
                           ([2, 4], [True, True]), ([20, 20], [False, False])]:
        samples = capped_first_pass_times(np.array(blocks), np.array(passed), 20)
        values.append(distribution_scores(samples[:, None], target)['crps'])
    np.testing.assert_allclose(values, [1., 1., .5, 18.], atol=1e-12, rtol=0)
