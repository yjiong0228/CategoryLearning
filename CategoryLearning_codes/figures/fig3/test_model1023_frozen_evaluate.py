"""Check parameter identity and precision denominators in sensitivity figures."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model1023_frozen_evaluate import ordered_points, state_blocks


def test_selected_reference_alias_retains_two_other_candidates():
    table = pd.DataFrame([dict(subject=108, candidate=key, selected=flag, reference=flag)
                          for key, flag in [('c', False), ('a', True), ('b', False)]])
    points = ordered_points(table, 108)
    assert [p['candidate'] for p in points] == ['a', 'b', 'c']
    assert [p['label'] for p in points] == ['Selected / reference', 'Alternative 1', 'Alternative 2']


def test_block_precision_conditions_on_presence_and_preserves_empty_tail():
    trials = pd.DataFrame(dict(subject=[108]*3, candidate=['a']*3, trial=[1, 2, 3],
        training_trials=[2]*3, target_present=[.1, .9, 0], target_contribution=[.05, .5, 0],
        target_precision_given_presence=[9, 1, np.nan]))
    result = state_blocks(trials, 2)
    np.testing.assert_allclose(result.target_present, [.5, 0])
    np.testing.assert_allclose(result.target_precision_given_presence, [1.8, np.nan], equal_nan=True)
    assert result.trial.to_list() == [2, 3]
    trials['training_trials'] = 1
    with pytest.raises(ValueError, match='training boundary'):
        state_blocks(trials, 2)
