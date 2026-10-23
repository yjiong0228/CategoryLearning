"""Keep training windows chronological and retain independent simulation units."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_1022_belief_training_windows as diagnostic


def test_unequal_windows_cover_training_once_at_each_resolution():
    windows = diagnostic.training_windows(11)
    for prefix in ('half', 'quarter'):
        trials = [t for name, start, stop in windows if name.startswith(prefix)
                  for t in range(start, stop)]
        assert trials == list(range(11))


def test_late_training_slice_uses_its_own_choices_but_keeps_complete_paths():
    categories = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    observed = categories.copy()
    paths = np.stack([categories, 3-categories, np.array([2, 1, 2, 1, 1, 2, 1, 2])])
    probabilities = np.array([[.2]*4+[.8]*4, [.4]*4+[.6]*4])
    rows = diagnostic.describe(paths, categories, observed, probabilities,
                               dict(block_width=2, temporal_weight=.5), 2)
    late = next(row for row in rows if row['window'] == 'half_2')
    assert late['generated_accuracy'] == pytest.approx(2/3)
    assert late['observed_accuracy'] == 1
    assert late['nll'] == pytest.approx(-np.log(.7))
    assert late['paths'] == 3
    assert late['filters'] == 2
    # A change strictly before this window cannot change its descriptive score.
    paths[:, :4] = 1
    probabilities[:, :4] = .01
    changed = diagnostic.describe(paths, categories, observed, probabilities,
                                  dict(block_width=2, temporal_weight=.5), 2)
    assert changed[2] == late


def test_rejects_future_probabilities_mixed_into_training():
    with pytest.raises(ValueError, match='align'):
        diagnostic.describe(np.ones((3, 8)), np.ones(8), np.ones(8), np.full((2, 9), .5),
                            dict(block_width=2, temporal_weight=.5), 2)
