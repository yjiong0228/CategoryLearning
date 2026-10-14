"""Independent enumerated reference for finite-capacity subset probabilities."""
from itertools import combinations

import numpy as np
import pytest
from scipy.special import softmax

from src.Bayesian_state.model.modules.hypothesis_transition.candidate_retention import (
    sample_candidate_set, subset_distribution,
)


@pytest.mark.parametrize('scores', [[0.,0.,0.],[-3.,.2,2.],[1000.,-1000.,1.]])
@pytest.mark.parametrize('capacity', [1,2,3])
def test_partition_and_inclusion_match_independent_enumeration(scores,capacity):
    sets=[s for k in range(1,capacity+1) for s in combinations(range(3),k)]
    probabilities=softmax([sum(scores[i] for i in s) for s in sets])
    _,cardinality,marginal=subset_distribution(np.array(scores),capacity)
    expected=[sum(p for s,p in zip(sets,probabilities) if len(s)==k) for k in range(1,capacity+1)]
    np.testing.assert_allclose(cardinality,expected,atol=1e-12,rtol=1e-12)
    assert marginal==pytest.approx(sum(p for s,p in zip(sets,probabilities) if 0 in s),abs=1e-12)


def test_sampled_set_frequencies_and_reproducibility():
    scores=np.array([-.7,.4,1.2]);capacity=2
    sets=[s for k in range(1,capacity+1) for s in combinations(range(3),k)]
    probabilities=softmax([scores[list(s)].sum() for s in sets])
    rng=np.random.default_rng(20261015)
    counts={s:0 for s in sets}
    for _ in range(6000):
        chosen,_=sample_candidate_set(scores,capacity,rng)
        counts[tuple(chosen)]+=1
    np.testing.assert_allclose([counts[s]/6000 for s in sets],probabilities,atol=.025,rtol=0)
    a,b=np.random.default_rng(27),np.random.default_rng(27)
    for _ in range(20):
        np.testing.assert_array_equal(sample_candidate_set(scores,capacity,a)[0],sample_candidate_set(scores,capacity,b)[0])


@pytest.mark.parametrize('scores,capacity', [([],2),([np.nan],1),([1.],0),([1.],True),([1.],1.5)])
def test_invalid_subset_inputs(scores,capacity):
    with pytest.raises(ValueError):subset_distribution(np.array(scores),capacity)
