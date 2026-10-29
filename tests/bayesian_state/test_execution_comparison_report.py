"""Independent numerical checks and conservative treatment of missing evidence."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.metrics.autonomous_fit import energy_u
from src.Bayesian_state.workflows.reports.model_1024_execution_comparison_report import (
    KEYS, fit_complete, group_averages, numerical_direction, paired_energy_difference,
    recovery_audit, stage_coverage, truth_events,
)


def test_paired_energy_matches_explicit_leave_one_path_out():
    a = np.array([[0., 1.], [1., 1.], [2., 0.], [2., 2.], [1., 0.]])
    b = np.array([[0., 0.], [1., 2.], [2., 1.], [3., 2.], [1., 1.]])
    y = np.array([.7, 1.3])
    actual = paired_energy_difference(a, b, y)
    deleted = np.array([energy_u(np.delete(a, i, axis=0), y)['value']
                        - energy_u(np.delete(b, i, axis=0), y)['value'] for i in range(len(a))])
    se = np.sqrt((len(a)-1)/len(a) * np.square(deleted-deleted.mean()).sum())
    assert actual['mcse'] == pytest.approx(se)
    assert actual['difference'] == pytest.approx(energy_u(a, y)['value'] - energy_u(b, y)['value'])
    same = paired_energy_difference(a, a, y)
    assert same['difference'] == same['mcse'] == 0


@pytest.mark.parametrize('complete,precise,expected', [
    (False, True, 'incomplete'), (True, False, 'numerically_unresolved'),
    (True, True, 'free_lower'),
])
def test_numerical_gain_does_not_override_missing_checks(complete, precise, expected):
    assert numerical_direction(-.1, .001, complete=complete, precise=precise) == expected


def test_identical_zero_selection_still_retains_precision_failure():
    assert numerical_direction(0, 0, complete=True, precise=False, identical=True) == 'numerically_unresolved'
    assert numerical_direction(0, 0, complete=True, precise=True, identical=True) == 'same_zero_nominee'


def test_missing_stage_and_one_missing_repeat_are_detected():
    case = dict(id='one', group='prefix')
    policy = {'search': {'prefix': dict(initial_pairs=1, rounds=0, finalists_per_boundary=1),
                         'audit_per_boundary': 1}}
    jobs, ledger = [], {}
    for role in ('screen_0', 'confirm', 'audit'):
        for cid in ('zero', 'positive'):
            for i in range(2):
                name = f'{role}_{cid}_{i}'
                jobs.append(dict(name=name, comparison_case='one', comparison_role=role, candidate={'id': cid}))
                ledger[name] = {'status': 'complete'}
    assert fit_complete(stage_coverage(case, policy, jobs, ledger))
    incomplete = deepcopy(ledger)
    del incomplete['audit_positive_1']
    assert not fit_complete(stage_coverage(case, policy, jobs, incomplete))
    assert not fit_complete(stage_coverage(case, policy, jobs[:8], ledger))


def test_unresolved_recovery_is_not_counted_as_a_wrong_class():
    case = dict(id='R0', subject=110, cut=2)
    point = dict(id='truth', hyperparams={KEYS['inertia']: 2})
    comparison = dict(free_kappa=0, suffix_direction='numerically_unresolved', fit_jobs_complete=True)
    frozen = dict(status='frozen', free={'candidate': point}, zero={'candidate': point})
    row = recovery_audit(None, case, frozen, {'point': point}, [], comparison)
    assert row['recovered_class'] == 'unresolved'
    assert row['class_agreement'] is None


def test_missing_optimization_check_keeps_recovery_unresolved():
    case = dict(id='R0', subject=110, cut=2)
    point = dict(id='truth', hyperparams={KEYS['inertia']: 2})
    comparison = dict(free_kappa=2, suffix_direction='free_lower', fit_jobs_complete=True,
                      fit_numerical_resolution=True, suffix_numerical_resolution=True)
    frozen = dict(status='frozen', free={'candidate': point}, zero={'candidate': point})
    row = recovery_audit(None, case, frozen, {'point': point}, [], comparison)
    assert row['recovered_class'] == 'optimization_unchecked'
    assert row['class_agreement'] is None


def test_better_generating_point_marks_search_problem_separately():
    class SavedArrays:
        def arrays(self, jobs, key):
            return np.full((4, 4), .5 if jobs == 'fit' else .8)

    case = dict(id='R0', subject=110, cut=2)
    truth = dict(id='truth', hyperparams={KEYS['inertia']: 2})
    fit = dict(id='fit', hyperparams={KEYS['inertia']: 2})
    frozen = dict(status='frozen', free={'candidate': fit}, zero={'candidate': fit})
    checked = [dict(candidate=truth, jobs='truth'), dict(candidate=fit, jobs='fit')]
    comparison = dict(free_kappa=2, suffix_direction='free_lower', fit_jobs_complete=True,
                      fit_numerical_resolution=True, suffix_numerical_resolution=True)
    row = recovery_audit(SavedArrays(), case, frozen, {'point': truth}, checked, comparison)
    assert row['generating_point_better_than_search']
    assert row['recovered_class'] == 'search_unresolved'
    assert row['class_agreement'] is None


def test_group_mean_never_silently_drops_missing_participants():
    rows = [dict(group='prefix', fit_jobs_complete=True, suffix_difference=-.02, suffix_mcse=.001,
                 fit_numerical_resolution=True, suffix_numerical_resolution=True)]
    summary = next(r for r in group_averages(rows, 6) if r['group'] == 'prefix' and r['metric'] == 'suffix')
    assert summary['available_mean'] == -.02
    assert summary['all_subject_mean'] is None
    assert summary['subjects'] == 1 and summary['expected_subjects'] == 6


def test_truth_persistence_counts_availability_and_full_dwell_runs():
    class TruthArrays:
        def arrays(self, jobs, key):
            return {'executed': np.array([[1, 1, 2, 2, 2]]),
                    'rule': np.array([[[1, 2], [1, 2], [2, 3], [2, 3], [2, 3]]]),
                    'forced': np.array([[True, False, True, False, False]])}[key]

    row = truth_events(TruthArrays(), dict(generation_jobs=[], path_index=0))
    assert row['truth_switches'] == row['truth_forced_switches'] == 1
    assert row['truth_previous_rule_available'] == 3
    assert row['truth_same_rule_given_available'] == 1
    assert row['truth_longest_execution_run'] == 3
