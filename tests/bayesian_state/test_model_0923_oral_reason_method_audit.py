"""Counterexamples fix what marginal oral evidence can and cannot establish."""
from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_oral_reason_method_audit as audit


def test_saved_states_cannot_identify_simultaneous_rule_use():
    single, state_a, a = audit.contribution_example('uncertain_single_rule')
    mixed, state_b, b = audit.contribution_example('simultaneous_mixture')
    for key in single:
        np.testing.assert_array_equal(single[key], mixed[key])
    assert a['whole_functional_support'] == b['whole_functional_support'] == 0.
    assert a['mean_predicate_functional_support'] == b['mean_predicate_functional_support'] == .5
    assert a['probability_both_effects_present_within_state'] == 0.
    assert b['probability_both_effects_present_within_state'] == 1.
    assert a['mean_effective_rule_count'] == 1. and b['mean_effective_rule_count'] == 2.
    assert a['both_rules_active_probability'] == b['both_rules_active_probability'] == 1.
    # Even querying every binary combination cannot distinguish these frozen
    # marginal response functions. The difference is inside the latent state.
    for neck, head in product((.2, .8), repeat=2):
        rule_probability = np.array([.9 if neck < .5 else .1, .9 if head < .5 else .1])
        responses = [s['choice_weights'][:, [0, 2]] @ rule_probability for s in (state_a, state_b)]
        np.testing.assert_allclose(state_a['weights'] @ responses[0], state_b['weights'] @ responses[1], rtol=0, atol=1e-14)
    assert not np.array_equal(state_a['choice_weights'], state_b['choice_weights'])


def test_reverse_entailment_and_stimulus_truth_credit_an_unused_reason():
    rows = audit.logical_examples().set_index('example')
    unused, joint = rows.loc['unused_extra_reported_reason'], rows.loc['joint_rule_with_two_reasons']
    assert unused.report_true_at_current_stimulus and joint.report_true_at_current_stimulus
    assert unused.report_implies_rule == joint.report_implies_rule == 'support'
    assert unused.dimension2_flip_effect == 0.
    assert joint.dimension2_flip_effect == .8
    assert unused.rule_implies_report == 'undetermined'


def test_description_scope_and_feature_use_are_separate():
    row = audit.logical_examples().set_index('example').loc['narrower_current_description']
    assert row.rule_implies_report == 'undetermined' and row.report_implies_rule == 'support'
    assert row.report_true_at_current_stimulus and row.dimension0_flip_effect == .8


def test_single_removal_misses_redundant_reasons():
    row = audit.redundant_reasons()
    assert row['remove_first_effect'] == row['remove_second_effect'] == 0.
    assert row['remove_both_effect'] == .8


def test_method_output_keeps_synthetic_scope_and_existing_results(tmp_path):
    output = tmp_path / 'method'
    result = audit.run(output)
    assert result['scientific_scope'] == 'synthetic_method_diagnostic'
    assert result['new_scientific_score'] is False and result['human_reports_used'] is False
    assert result['parameters_fitted'] is False
    assert result['identical_array_elements'] > 0
    rows = pd.read_csv(output / 'state_mixture_examples.csv')
    np.testing.assert_array_equal(rows.probability_both_effects_present_within_state, [0., 1.])
    with pytest.raises(FileExistsError):
        audit.run(output)


def test_unknown_synthetic_mechanism_rejected():
    with pytest.raises(ValueError):
        audit.contribution_example('some_other_policy')
