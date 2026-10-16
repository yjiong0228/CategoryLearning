"""Particle restoration tolerates rounding while preserving strict validation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule as Memory
from src.Bayesian_state.model.modules.pairing_snapshot_roundoff import pairing_snapshot_roundoff_guard


def memory():
    engine=SimpleNamespace(set_size=2,prior=np.ones(2)/2,hypotheses_mask=np.ones(2))
    result=Memory(engine,forgetting_mode='factorized')
    result.joint=np.ones((2,3))/6
    result.feedback_evidence=np.array([.5,1.])
    result.feedback_pairing_weights=np.ones((2,3))/3
    return result


def test_guard_preserves_valid_snapshot_exactly_and_restores_method():
    a,b=memory(),memory();state=a.state_dict();original=Memory.load_state_dict
    with pairing_snapshot_roundoff_guard(Memory) as corrections:
        b.load_state_dict(state)
        assert corrections==[]
        for key,value in state.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,b.state_dict()[key])
            else:assert value==b.state_dict()[key]
    assert Memory.load_state_dict is original


def test_roundoff_is_bounded_logged_and_does_not_mutate_shared_snapshot():
    a=memory();state=a.state_dict();state['feedback_evidence'][1]=np.nextafter(1.,2.)
    unchanged=deepcopy(state)
    with pytest.raises(ValueError,match='feedback evidence'):a.load_state_dict(state)
    with pairing_snapshot_roundoff_guard(Memory) as corrections:
        a.load_state_dict(state)
        np.testing.assert_array_equal(a.feedback_evidence,[.5,1.])
        assert corrections==[dict(phase='pairing_snapshot_evidence',maximum=np.nextafter(1.,2.),entries=1)]
    np.testing.assert_array_equal(state['feedback_evidence'],unchanged['feedback_evidence'])


@pytest.mark.parametrize('invalid',[np.nan,-1e-16,1.00000001])
def test_real_probability_errors_still_fail(invalid):
    a=memory();state=a.state_dict();state['feedback_evidence'][0]=invalid
    with pairing_snapshot_roundoff_guard(Memory),pytest.raises(ValueError):a.load_state_dict(state)


def test_other_snapshot_invariants_are_preserved():
    a=memory();state=a.state_dict();state['feedback_evidence']=np.ones(3)
    with pairing_snapshot_roundoff_guard(Memory),pytest.raises(ValueError,match='feedback evidence'):a.load_state_dict(state)
