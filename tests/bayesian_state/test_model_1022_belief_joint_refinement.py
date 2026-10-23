"""Bound candidate expansion and preserve training-only paired guard decisions."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_1022_belief_joint_refinement as study


def test_factorial_expansion_keeps_every_old_point_and_only_changes_declared_axes():
    space = dict(retention=[.3, .99], learning=[.001, .01, .1], capacity=[4, 6], search=[.05, .35])
    anchor = dict(retention=.3, learning=.001, capacity=6, search=.35)
    old = [study.candidate(dict(anchor, search=.05), 'old')]
    before = deepcopy((anchor, old))
    bank = study.expand_bank(old, anchor, {k: space[k] for k in ('retention', 'learning', 'capacity')}, space)
    assert len(bank) == 13
    assert old[0]['id'] in {r['id'] for r in bank}
    new = [r['hyperparams'] for r in bank if r['id'] != old[0]['id']]
    assert {p['search'] for p in new} == {.35}
    assert len({(p['retention'], p['learning'], p['capacity']) for p in new}) == 12
    assert (anchor, old) == before


@pytest.mark.parametrize('axes', [dict(x=[3]), dict(x=[1, 1]), dict(y=[1])])
def test_factorial_expansion_rejects_undeclared_support(axes):
    with pytest.raises(ValueError, match='axes'):
        study.expand_bank([], dict(x=1), axes, dict(x=[1, 2]))


def saved_point(tmp_path, identity, probability, seeds=(11, 12)):
    path = tmp_path / (identity + '.json')
    point = dict(id=identity, hyperparams=dict(x=identity), score_file=str(path))
    np.savez_compressed(path.with_suffix('.npz'), observed_probability=np.asarray(probability), seeds=seeds)
    path.write_text(json.dumps(dict(subject=306, candidate=point,
        predictions_sha256=study.digest(path.with_suffix('.npz')))))
    return point


def test_guard_rejects_energy_proposal_that_exceeds_training_likelihood_cost(tmp_path):
    old = saved_point(tmp_path, 'old', [[.8, .8], [.8, .8]])
    new = saved_point(tmp_path, 'new', [[.7, .7], [.7, .7]])
    bank = {(306, 'SLOT'): [old, new]}
    proposal = [dict(subject=306, arm='SLOT', original='old', proposed='new')]
    decisions, nominees = study.guarded_nominees(bank, bank, proposal,
        dict(training_nll_tolerance=.02, guard_mcse_multiplier=2.), {})
    assert not decisions[0]['accepted']
    assert nominees[0]['candidate']['id'] == 'old'
    assert len(nominees[0]['finalists']) == 2


def test_guard_rejects_different_filter_seed_identities_even_when_rehashed(tmp_path):
    old = saved_point(tmp_path, 'old', [[.8, .8], [.8, .8]])
    new = saved_point(tmp_path, 'new', [[.8, .8], [.8, .8]], seeds=(12, 11))
    bank = {(306, 'SLOT'): [old, new]}
    with pytest.raises(AssertionError):
        study.guarded_nominees(bank, bank, [dict(subject=306, arm='SLOT', original='old', proposed='new')],
                               dict(training_nll_tolerance=.02, guard_mcse_multiplier=2.), {})


def test_generation_worker_must_use_requested_arm_and_seed_order(tmp_path):
    job = dict(model='SLOT', mechanism=dict(capacity='evidence'), seeds=[11, 12, 13])
    saved = tmp_path / 'worker.json'
    expected = dict(job, model='COMPACT', comparison_arm='SLOT')
    saved.write_text(json.dumps(expected))
    inputs = {}
    study.verify_generation_job(job, saved, inputs)
    assert inputs[str(saved.resolve())] == study.digest(saved)
    for change in (dict(comparison_arm='SUPPORT'), dict(seeds=[12, 11, 13])):
        saved.write_text(json.dumps(dict(expected, **change)))
        with pytest.raises(ValueError, match='worker'):
            study.verify_generation_job(job, saved, {})
