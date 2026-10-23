"""Continuation summaries keep posterior clouds separate from generated branches."""
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_1023_response_dynamics import assemble
from src.Bayesian_state.workflows.analysis.model_1023_response_dynamics import write,digest


def test_prefix_assembly_preserves_filter_clusters_and_rejects_partial_schema(tmp_path):
    jobs=[]
    candidate=dict(id='fixed',hyperparams={'beta':2})
    for seed in (11,22):
        file=tmp_path/f'{seed}.npz'
        np.savez_compressed(file,categories=[1,2],filter_seed=seed,terminal_weights=[.1,.2,.7],
            endpoint_beta=[2,3,4],prefix_probabilities=np.ones((4,2))/2,
            branch_seeds=[seed+100,seed+200],ancestors=[2,1],choices=[[1,2],[2,1]])
        write(file.with_suffix('.json'),dict(status='complete',candidate=candidate,filter_seed=seed,
            predictions_sha256=digest(file)))
        jobs.append(dict(target=str(file.with_suffix('')),candidate=candidate,seeds=[seed]))
    arrays,_=assemble(jobs,'prefix')
    assert arrays['terminal_weights'].shape==(2,3)
    assert arrays['choices'].shape==(4,2)
    np.testing.assert_array_equal(arrays['branch_filter_seed'],[11,11,22,22])
    # A valid hash cannot make an incomplete numerical schema acceptable.
    file=tmp_path/'33.npz'
    np.savez_compressed(file,categories=[1,2],filter_seed=33,branch_seeds=[133,233],choices=[[1,2],[2,1]])
    write(file.with_suffix('.json'),dict(status='complete',candidate=candidate,filter_seed=33,predictions_sha256=digest(file)))
    with pytest.raises(ValueError,match='schemas'):
        assemble(jobs+[dict(target=str(file.with_suffix('')),candidate=candidate,seeds=[33])],'prefix')
    (tmp_path/'11.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):assemble(jobs,'prefix')
