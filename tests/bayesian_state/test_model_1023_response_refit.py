"""Parameter comparisons preserve their declared scientific scope."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_1023_response_refit import precision_bank


def test_precision_grid_varies_only_two_coordinates_and_retains_anchor():
    beta='engine.modules.beta_mod.kwargs.beta_init';rate='engine.modules.beta_mod.kwargs.learning_rate'
    space={beta:[.1,2.,25.],rate:[0.,.03,1.], 'search_probability':[.1,.5]}
    anchor={beta:2.,rate:.03,'search_probability':.1};original=deepcopy(anchor)
    rows=precision_bank(space,anchor,[beta,rate])
    assert len(rows)==9 and any(r['hyperparams']==anchor for r in rows)
    assert all(r['hyperparams']['search_probability']==.1 for r in rows)
    assert {(r['hyperparams'][beta],r['hyperparams'][rate]) for r in rows}=={
        (b,r) for b in space[beta] for r in space[rate]}
    assert anchor==original
    with pytest.raises(ValueError):precision_bank(space,anchor,[beta,'search_probability'])
