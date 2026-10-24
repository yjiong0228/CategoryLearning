"""Recovery scores against independent, exactly specified latent examples."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1023_execution_recovery_score import (
    binary_scores, pool_states, state_metrics)
from src.Bayesian_state.workflows.analysis.model_1023_execution_summary import trajectory_diagnostics


def test_binary_metrics_preserve_ties_and_expose_no_positive_events():
    metrics = binary_scores(np.array([.2, .8, .5, .5]), np.array([0, 1, 0, 1]))
    assert metrics['auc'] == pytest.approx(.875)
    assert metrics['brier'] == pytest.approx(.145)
    assert metrics['positives'] == 2
    absent = binary_scores(np.array([0., .1]), np.zeros(2))
    assert np.isnan(absent['auc']) and absent['prevalence_brier'] == 0.


def test_pool_current_choice_states_uses_joint_mass_not_mean_of_conditionals():
    parts = []
    for seed, chance, state in [(1, .2, .1), (2, .8, .9)]:
        parts.append(dict(choices=np.array([1]), categories=np.array([1]), target_rules=np.array([0]),
            seeds=np.array([seed]), prediction=np.array([[chance, 1-chance]]),
            filtered_target_present=np.array([state])))
    pooled = pool_states(parts)
    np.testing.assert_allclose(pooled['filtered_target_present'], [.74])
    np.testing.assert_allclose(pooled['prediction'], [[.5, .5]])


def test_oracle_recovers_three_distinct_error_sources_and_separate_onsets():
    raw = dict(rule=np.array([[1, -1], [0, 1], [0, 1]]),
        belief=np.array([[1., 0.], [.8, .2], [.9, .1]]),
        contribution=np.array([[1., 0.], [0., 1.], [1., 0.]]), executed=np.array([1, 1, 0]))
    state = dict(choices=np.array([2, 2, 2]), categories=np.ones(3, dtype=int), target_rules=np.array([0]),
                 conditional_decision_sources=np.eye(3))
    for timing in ('predictive', 'filtered'):
        state.update({timing+'_belief': np.array([[0., 1.], [.8, .2], [.9, .1]]),
            timing+'_execution': np.array([[0., 1.], [0., 1.], [1., 0.]]),
            timing+'_target_present': np.array([0., 1., 1.]),
            timing+'_target_high_belief': np.array([0., 1., 1.]),
            timing+'_target_execution': np.array([0., 0., 1.]),
            timing+'_available_other_execution': np.array([0., 1., 0.]),
            timing+'_high_belief_other_execution': np.array([0., 1., 0.])})
    scores, onsets = state_metrics(state, raw, 'STICKY')
    error, = [row for row in scores if row['state'] == 'error_source']
    assert error['brier'] == 0 and error['accuracy'] == 1 and error['log_loss'] == 0
    assert [error[k] for k in ('absent_errors', 'available_other_errors', 'target_execution_errors')] == [1, 1, 1]
    first = {r['state']: r for r in onsets if r['timing'] == 'filtered'}
    assert first['present']['true_first_trial'] == first['present']['first_detected_trial'] == 2
    assert first['target_execution']['true_first_trial'] == first['target_execution']['first_detected_trial'] == 3


def test_path_diagnostics_detect_known_change_and_keep_full_low_window():
    choices = np.array([[2]*128 + [1]*64])
    values = trajectory_diagnostics(choices, np.ones(192, dtype=int))
    assert values['max_adjacent_gain'][0] == 1.
    assert values['jump_after_trial'][0] == 128
    assert values['minimum_window_accuracy'][0] == 0.
    assert values['full_accuracy'][0] == pytest.approx(1/3)
    assert values['maintenance_observed'][0] and values['maintenance_accuracy'][0] == 1.
    short = trajectory_diagnostics(choices[:, :160], np.ones(160, dtype=int))
    assert not short['maintenance_observed'][0] and np.isnan(short['maintenance_accuracy'][0])


def test_low_accuracy_diagnostic_accounts_for_response_and_category_marginals():
    # Always answering category 4 scores below uniform chance here because
    # category 4 is rare; the marginal baseline exposes that explanation.
    values = trajectory_diagnostics(np.full((1, 128), 4),
        np.array([1]*112 + [4]*16), n_categories=4)
    assert values['minimum_window_start_trial'][0] == 1
    assert values['minimum_window_accuracy'][0] == .125
    assert values['minimum_window_marginal_chance'][0] == .125
    assert values['minimum_window_accuracy_minus_marginal_chance'][0] == 0.
