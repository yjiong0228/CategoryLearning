import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_reason_chronology import label_templates, pooled_series


def test_templates_preserve_unmatched_qualifiers_and_missing_text():
    result = label_templates(pd.Series(['脖子短，尾巴长。', '脖子很短，尾巴长。', '', '尾巴长，脖子短。']),
                             {'脖子短，尾巴长。': 2})
    assert result.iloc[0] == 2
    assert result.iloc[1:].isna().all()


def test_current_choice_pooling_and_whole_filter_deletion():
    probability = np.array([[.1, .9], [.9, .1], [.5, .5]])
    joint = probability * np.array([[1., .2], [0., .8], [.3, .7]])
    point, deleted = pooled_series(joint, probability)
    np.testing.assert_allclose(point, [1/6, .61/1.5])
    for i in range(3):
        keep = np.arange(3) != i
        np.testing.assert_allclose(deleted[i], joint[keep].sum(axis=0) / probability[keep].sum(axis=0))
    assert not np.allclose(point, (joint/probability).mean(axis=0))


@pytest.mark.parametrize('joint,probability', [
    ([[.2]], [[.5]]),
    ([[.6], [.1]], [[.5], [.5]]),
    ([[.1], [.1]], [[0.], [.5]]),
    ([[np.nan], [.1]], [[.5], [.5]]),
])
def test_invalid_state_masses_are_rejected(joint, probability):
    with pytest.raises(ValueError):
        pooled_series(joint, probability)
