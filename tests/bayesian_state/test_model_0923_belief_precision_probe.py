"""Keep a local parameter interaction separate from refitting and later data."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis import model_0923_belief_precision_probe as probe


def anchors_and_space():
    space = {probe.GAMMA: [0., .6, 1.], probe.RATE: [.003, .03, .06, .3, 1.],
             probe.TRANSFER: [.01], 'fixed_other_coordinate': [1., 4.]}
    first = {key: values[0] for key, values in space.items()}
    first[probe.RATE] = .3
    second = dict(first, **{probe.GAMMA: 1., probe.RATE: .06, 'fixed_other_coordinate': 4.})
    return [candidate(first, 'first'), candidate(second, 'second')], space


def test_full_factorial_keeps_other_coordinates_and_exact_anchor_outside_rate_grid():
    anchors, space = anchors_and_space(); before = deepcopy(anchors)
    gammas, rates = [0., .6, 1.], [.003, .03, .3, 1.]
    bank = probe.build_bank(anchors, gammas, rates, space)
    assert anchors == before and len(bank) == 25
    assert {a['id'] for a in anchors} <= {p['id'] for p in bank}
    for anchor in anchors:
        own = [point for point in bank if anchor['id'] in point['anchor_ids']]
        expected = {(g, r) for g in gammas for r in rates}
        expected.add((anchor['hyperparams'][probe.GAMMA], anchor['hyperparams'][probe.RATE]))
        assert {(p['hyperparams'][probe.GAMMA], p['hyperparams'][probe.RATE]) for p in own} == expected
        for p in own:
            assert {k: v for k, v in p['hyperparams'].items() if k not in (probe.GAMMA, probe.RATE)} == {
                k: v for k, v in anchor['hyperparams'].items() if k not in (probe.GAMMA, probe.RATE)}


def test_duplicate_anchor_contexts_deduplicate_numerical_work():
    anchors, space = anchors_and_space()
    bank = probe.build_bank([anchors[0], deepcopy(anchors[0])], [0., 1.], [.3], space)
    assert len(bank) == 2
    assert all(p['anchor_ids'] == [anchors[0]['id']] for p in bank)


@pytest.mark.parametrize('fault', ['gamma', 'rate', 'repeated', 'identity', 'coordinate'])
def test_rejects_unsupported_or_unbound_grid(fault):
    anchors, space = anchors_and_space(); gammas, rates = [0., 1.], [.3]
    if fault == 'gamma': gammas = [.9]
    elif fault == 'rate': rates = [.9]
    elif fault == 'repeated': gammas = [0., 0.]
    elif fault == 'identity': anchors[0]['id'] = 'changed'
    elif fault == 'coordinate': anchors[0]['hyperparams']['extra'] = 1.
    with pytest.raises(ValueError): probe.build_bank(anchors, gammas, rates, space)


def execution(mode):
    fixed = {probe.TRANSFER: .01}
    point = candidate(dict(fixed, **{probe.GAMMA: 1.,
        'engine.modules.hypo_transitions_mod.kwargs.intervention.decay': .8}), 'test')
    job = dict(phase='training', model='SEARCH', candidate=point, seeds=[2, 3], particles=4, smoke=True,
        training_trials=384, diagnostic='belief_precision_training_grid', probe_mechanism=mode,
        belief_intervention=probe.MODES[mode], online_belief=mode == 'ONLINE',
        fixed_coordinates=fixed, retain_precision=True, structural_intervention='retained_rule_precision')
    arrays = dict(seeds=np.array([2, 3]), observed_probability=np.full((2, 8), .3), correct_probability=np.full((2, 8), .4))
    result = dict(role='training', subject=208, condition=2, trials=8, candidate=deepcopy(point),
        model='SEARCH', seeds=[2, 3], particles=4, numeric_pools=[{'num_threads': 1}], mean_nll=-np.log(.3))
    precision = dict(structural_intervention='retained_rule_precision', retained_precision=True, transitions=64)
    belief = dict(enabled=True, belief_intervention=probe.MODES[mode])
    if mode == 'ONLINE': belief.update(fixed_coordinates=fixed, exact_transition_timing=True,
        evidence_matches_retained_state=True, reconstructions=56)
    else: belief.update(gamma=1., delta=[.8], completed_feedback_timing_exact=True,
                         belief_matches_saved_state=True, updates=56)
    return job, result, arrays, belief, precision


@pytest.mark.parametrize('mode', ['ONLINE', 'SEPARATE'])
def test_training_worker_evidence_and_precision_counts(mode):
    probe.check_result(*execution(mode))


@pytest.mark.parametrize('mode', ['ONLINE', 'SEPARATE'])
@pytest.mark.parametrize('fault', ['evaluation', 'window', 'seed', 'update', 'precision', 'candidate', 'subject', 'threads'])
def test_rejects_changed_training_scope_or_execution(mode, fault):
    job, result, arrays, belief, precision = execution(mode)
    if fault == 'evaluation': job['phase'] = 'evaluate'
    elif fault == 'window': result['trials'] = 384
    elif fault == 'seed': arrays['seeds'] = arrays['seeds'][::-1]
    elif fault == 'update': belief['updates' if mode == 'SEPARATE' else 'reconstructions'] += 1
    elif fault == 'precision': precision['transitions'] += 1
    elif fault == 'candidate': result['candidate']['id'] = 'changed'
    elif fault == 'subject': result['subject'] = 207
    elif fault == 'threads': result['numeric_pools'][0]['num_threads'] = 2
    with pytest.raises((ValueError, AssertionError)):
        probe.check_result(job, result, arrays, belief, precision)
