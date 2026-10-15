"""Forecast assembly preserves filter membership for Monte Carlo inference."""
import numpy as np

from src.Bayesian_state.workflows.analysis.model_0923_h_prefix_forecast import combine, validation


def test_assembled_branches_keep_their_original_filter_membership(tmp_path):
    v = validation(); jobs = []
    for index, seed in enumerate([11, 22]):
        file = tmp_path / f'chunk_{index}'
        np.savez_compressed(file.with_suffix('.npz'), categories=[1, 2], filter_seed=seed,
            terminal_weights=[.25, .75], prefix_observed_probability=[.4, .8],
            ancestors=[0, 1], branch_seeds=[seed+100, seed+200], choices=[[1, 2], [2, 2]])
        v.write(file.with_suffix('.json'), dict(paths=2, cpu_seconds=1., seconds=2.,
            predictions_sha256=v.digest(file.with_suffix('.npz'))))
        jobs.append(dict(target=str(file), filter_seed=seed))
    row = combine(v, jobs, tmp_path / 'combined')
    assert row['paths'] == 4 and row['filters'] == 2
    with np.load(tmp_path / 'combined.npz') as saved:
        np.testing.assert_array_equal(saved['filter_seed'], [11, 22])
        np.testing.assert_array_equal(saved['branch_filter_seed'], [11, 11, 22, 22])
        assert saved['terminal_weights'].shape == (2, 2)
        assert saved['prefix_observed_probability'].shape == (2, 2)
