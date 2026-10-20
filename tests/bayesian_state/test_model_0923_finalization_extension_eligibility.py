import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_finalization_extension_eligibility import select_participants


def test_selection_keeps_lowest_id_and_requires_both_phases():
    rows = pd.DataFrame([
        dict(subject=111, condition=1, task_valid=True, training_common_complete=90, later_common_complete=90),
        dict(subject=110, condition=1, task_valid=True, training_common_complete=32, later_common_complete=32),
        dict(subject=209, condition=2, task_valid=True, training_common_complete=90, later_common_complete=31),
        dict(subject=210, condition=2, task_valid=True, training_common_complete=40, later_common_complete=32),
        dict(subject=309, condition=3, task_valid=False, training_common_complete=100, later_common_complete=100),
        dict(subject=310, condition=3, task_valid=True, training_common_complete=40, later_common_complete=40),
    ])
    result = select_participants(rows, 32)
    assert result[result.selected].subject.tolist() == [110, 210, 310]
    assert len(result) == 6
    assert result.set_index("subject").loc[111, "eligible"]
    assert not result.set_index("subject").loc[209, "eligible"]


def test_duplicate_identity_is_rejected():
    row = dict(subject=110, condition=1, task_valid=True, training_common_complete=50, later_common_complete=50)
    with pytest.raises(ValueError, match="unique"):
        select_participants(pd.DataFrame([row, row]), 32)
