"""Exact column reuse compared with the unchanged all-rule base evaluator."""
import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.observation_model import ContinuousPartition
from src.Bayesian_state.hypothesis_space.observation_model.base_partition import BasePartition


@pytest.mark.parametrize('n_cats', [2, 4])
@pytest.mark.parametrize('mode', ['prototype', 'boundary'])
@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('feedback_mode', ['category_feedback', 'legacy', 'legacy_category_feedback', 'bernoulli_choice'])
def test_batch_matches_independent_base_loop(n_cats, mode, normalized, feedback_mode):
    partition = ContinuousPartition(4, n_cats)
    hypos = list(range(partition.length))[::-1] + [0, 0]
    x = np.array([[0., 0., 0., 0.], [.4, .3, .7, .5], [1., 1., 1., 1.]])
    mixed = np.zeros(len(hypos))
    mixed[:3] = [1., 1e-12, 3.]
    for beta in (0., -0., mixed, [0.], np.array([0.]), 1e-12):
        for responses in ([1., 0., 1.], [.5, 0., 1.]):
            data = (x, [1, n_cats, 1], responses)
            kwargs = dict(beta=beta, distance_mode=mode, normalized=normalized,
                          feedback_likelihood_mode=feedback_mode, feedback_lapse=.15)
            if .5 in responses and feedback_mode in ('category_feedback', 'legacy'):
                # The short legacy alias is ordinary category feedback; only
                # the explicit historical mode permits its overlapping weights.
                with pytest.raises(ValueError, match='Partial feedback'):
                    BasePartition.calc_likelihood(partition, hypos, data, **kwargs)
                with pytest.raises(ValueError, match='Partial feedback'):
                    partition.calc_likelihood(hypos, data, **kwargs)
                continue
            expected = BasePartition.calc_likelihood(partition, hypos, data, **kwargs)
            actual = partition.calc_likelihood(hypos, data, **kwargs)
            np.testing.assert_array_equal(actual, expected)


def test_repeated_zero_columns_reuse_formula_but_keep_every_column(monkeypatch):
    partition = ContinuousPartition(2, 2)
    data = ([[.2, .7]], [1], [0.])
    calls = []
    original = partition.calc_likelihood_entry
    def entry(hypo, *args, **kwargs):
        calls.append(hypo)
        return original(hypo, *args, **kwargs)
    monkeypatch.setattr(partition, 'calc_likelihood_entry', entry)
    result = partition.calc_likelihood([2, 0, 1, 0], data, [0., 1e-12, 0., 0.])
    assert calls == [2, 0]
    assert result.shape == (1, 4)
    partition.zero_beta_likelihood_batch = False
    calls.clear()
    np.testing.assert_array_equal(result, partition.calc_likelihood([2, 0, 1, 0], data, [0., 1e-12, 0., 0.]))
    assert calls == [2, 0, 1, 0]


@pytest.mark.parametrize('changes', [
    {'hypos': [0, 999]}, {'data': ([[.2, .3]], [3], [1.])},
    {'data': ([[.2, .3, .4]], [1], [1.])},
    {'distance_mode': 'bad'}, {'feedback_likelihood_mode': 'bad'},
    {'feedback_lapse': 1.}, {'feedback_lapse': np.nan},
])
def test_validation_matches_base_exception(changes):
    partition = ContinuousPartition(2, 2)
    args = dict(hypos=[0, 1], data=([[.2, .3]], [1], [1.]), beta=0.)
    args.update(changes)
    with pytest.raises(Exception) as expected:
        BasePartition.calc_likelihood(partition, **args)
    with pytest.raises(type(expected.value)) as actual:
        partition.calc_likelihood(**args)
    assert str(actual.value) == str(expected.value)


def test_empty_inputs_and_subclass_keep_base_contract():
    class CustomPartition(ContinuousPartition):
        def calc_likelihood_entry(self, hypo, *args, **kwargs):
            return np.array([.2 + .1 * hypo])
    partition = CustomPartition(2, 2)
    data = ([[.2, .3]], [1], [1.])
    np.testing.assert_array_equal(partition.calc_likelihood([0, 1], data, 0.),
                                  BasePartition.calc_likelihood(partition, [0, 1], data, 0.))
    plain = ContinuousPartition(2, 2)
    with np.errstate(invalid='ignore'):
        np.testing.assert_array_equal(plain.calc_likelihood([], data, 0.),
                                      BasePartition.calc_likelihood(plain, [], data, 0.))
    with pytest.raises(ValueError, match='must be a boolean'):
        ContinuousPartition(2, 2, zero_beta_likelihood_batch=1)
