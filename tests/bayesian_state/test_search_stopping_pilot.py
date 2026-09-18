"""Scientific decision and covariance contracts of the stopping pilot."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.optimization.model_0826 import REACTIVE_PROFILE_KEY, extract_model_0826_parameters
from src.Bayesian_state.workflows.analysis.pilot_model_0826_search_stopping import (
    axis_values, candidate, influence_diagnostics, merge_bank, ray_proposals, stopping_advice,
)
from tests.bayesian_state.test_adaptive_effort_pilot import configuration


def test_rays_respect_primitive_coordinates_and_derived_error_probability():
    _, space, anchor = configuration()
    values = space[REACTIVE_PROFILE_KEY]
    rays = axis_values(REACTIVE_PROFILE_KEY, anchor[REACTIVE_PROFILE_KEY], values)
    original = extract_model_0826_parameters({REACTIVE_PROFILE_KEY: anchor[REACTIVE_PROFILE_KEY]})
    assert len(rays) == 2
    for ray in rays:
        for value in ray:
            named = extract_model_0826_parameters({REACTIVE_PROFILE_KEY: value})
            assert sum(abs(named[k]-original[k]) > 1e-9 for k in ('E_C', 'delta_E')) <= 1
            assert named['E_E'] >= named['E_C'] - 1e-12


def test_proposals_cover_every_block_before_exhausting_one_and_are_unique():
    _, space, anchor = configuration()
    a = candidate(anchor, 'base')
    rows = ray_proposals([a], space, {a['id']}, 48, 19, 'challenge')
    assert rows == ray_proposals([a], space, {a['id']}, 48, 19, 'challenge')
    assert len(rows) == len({r['id'] for r in rows}) <= 48
    changed = set()
    for row in rows:
        p = row['hyperparams']
        keys = [k for k in p if p[k] != anchor[k]]
        assert len(keys) == 1
        changed.update(keys)
        assert all(p[k] in space[k] for k in p)
    assert changed == set(space)


def test_stopping_requires_plateau_and_bounded_challenger_improvement():
    assert stopping_advice(False, True, 0, .001, .005) == 'calibrate_score_precision'
    assert stopping_advice(True, False, 0, .001, .005) == 'budget_cap_without_plateau'
    assert stopping_advice(True, True, .006, .012, .005) == 'expand_search_challenger_improved'
    assert stopping_advice(True, True, -.001, .012, .005) == 'challenge_inconclusive'
    assert stopping_advice(True, True, -.003, .004, .005) == 'provisional_stop_within_tested_scope'


def test_influence_keeps_temporal_covariance_and_segments_sum_to_total():
    q = np.repeat(np.array([.2, .3, .7, .8])[:, None], 8, axis=1)
    arrays = {'probabilities': np.stack([q, 1-q], axis=-1),
              'observed': np.zeros(8, dtype=int), 'mask': np.ones(8, dtype=bool)}
    d, segments = influence_diagnostics(arrays, 2)
    assert d['temporal_covariance_variance_ratio'] == pytest.approx(8.)
    assert d['delta_method_mcse'] > d['independent_trial_mcse_diagnostic']
    assert sum(r['score_variance_covariance_share'] for r in segments) == pytest.approx(1.)
    constant = deepcopy(arrays)
    constant['probabilities'][:] = .5
    d, _ = influence_diagnostics(constant, 2)
    assert d['delta_method_mcse'] == 0.


def test_shared_candidates_keep_both_source_labels():
    a = candidate({'x': 1}, 'base')
    bank = merge_bank([('base', [a]), ('challenge', [a])])
    assert len(bank) == 1
    assert bank[0]['sources'] == ['base', 'challenge']
