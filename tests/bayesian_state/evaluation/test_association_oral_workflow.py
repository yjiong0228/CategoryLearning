import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_association_oral import content_key,query_id,component_labels


def test_content_identity_ignores_order_but_keeps_geometry_region():
    a=content_key('["b", "a", "a"]');b=content_key('["a", "b"]')
    assert a==b and query_id(a,0)==query_id(b,0)
    assert len({query_id(a,g) for g in range(4)})==4
    with pytest.raises(ValueError):content_key('[1]')
    with pytest.raises(ValueError):query_id((),0)


def test_component_table_keeps_every_real_component_and_padding():
    counts=np.array([[2,1,1,1],[1,1,2,1]])
    rows=[dict(rule=0,component=0,structural='support',functional='support'),
          dict(rule=0,component=1,structural='undetermined',functional='support'),
          dict(rule=1,component=0,structural='conflict',functional='conflict')]
    labels=component_labels(rows,counts,0)
    np.testing.assert_array_equal(labels,[[[0,0],[2,0]],[[1,1],[-1,-1]]])
    for bad in [rows[:-1],rows+[rows[0]],rows+[dict(rows[0],component=2)]]:
        with pytest.raises(ValueError):component_labels(bad,counts,0)
    with pytest.raises(ValueError):component_labels([dict(r,structural='missing') for r in rows],counts,0)
