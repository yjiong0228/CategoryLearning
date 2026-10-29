"""Protect quantity-specific scope and the bounded focus-only escalation."""
from pathlib import Path

import yaml

from src.Bayesian_state.workflows.analysis.model_1024_state_interpretation import (
    MEASURES, escalation_cases, measure_checks)


def policy():
    return yaml.safe_load(Path('configs/shared/model1024_state_interpretation.yaml').read_text())


def row(subject=205):
    p = policy()
    cid = next(c for c in p['pilot_cases'] if c.startswith(f'S{subject}_'))
    result = dict(case=cid, subject=subject, particles=4096, systems=16, bank_complete=True,
                  confirmation=False, full_ess_min=8., full_maximum_weight=.25)
    for field in MEASURES:
        for check in ('half', 'comparison'):
            result[f'full_{check}_{field}_mean'] = .01
            result[f'full_{check}_{field}_q95'] = .04
    return result


def test_passing_aggregate_cannot_hide_one_unstable_quantity():
    r = row(); r.update(full_half_mean=.02, full_half_q95=.09)
    r['full_half_belief_q95'] = .21
    assert measure_checks(r, policy(), 'full', 'target_present')['usable']
    assert not measure_checks(r, policy(), 'full', 'belief')['usable']
    assert escalation_cases([r], policy()) == [r['case']]


def test_escalation_is_focus_only_and_does_not_repeat_at_8192():
    a, b = row(), row(306)
    assert escalation_cases([a, b], policy()) == []
    a['full_comparison_target_belief_mean'] = .06
    b['full_comparison_target_belief_mean'] = .06
    assert escalation_cases([a, b], policy()) == [a['case']]
    a['particles'] = 8192
    assert escalation_cases([a, b], policy()) == []


def test_incomplete_and_weight_concentrated_banks_remain_explicit():
    r = row(); r['bank_complete'] = False
    checks = measure_checks(r, policy(), 'full', 'target_execution')
    assert checks['repeat'] and checks['resolution'] and not checks['usable']
    r['bank_complete'] = True; r['full_ess_min'] = 3.9
    assert not measure_checks(r, policy(), 'full', 'target_execution')['usable']


def test_single_candidate_is_not_evidence_for_parameter_robustness():
    import numpy as np
    from src.Bayesian_state.workflows.analysis.model_1024_state_interpretation_summary import (
        candidate_difference, sensitivity_status)
    assert candidate_difference(np.ones((1, 10)), False) is None
    assert sensitivity_status(1, True, 0., 0., policy()) == 'single_candidate'
    assert sensitivity_status(4, True, 0., 0., policy(), False) == 'diagnostic_parameters_only'


def test_parameter_disagreement_and_numerical_disagreement_are_separate():
    import numpy as np
    from src.Bayesian_state.workflows.analysis.model_1024_state_interpretation_summary import (
        candidate_difference, sensitivity_status)
    values=np.array([[[.9,.1],[.2,.8]],[[.1,.9],[.3,.7]],[[.5,.5],[.2,.8]]])
    np.testing.assert_allclose(candidate_difference(values, True), [.8,.1])
    assert sensitivity_status(3, False, .45, .8, policy()) == 'numerically_limited'
    assert sensitivity_status(3, True, .45, .8, policy()) == 'parameter_sensitive'
    assert sensitivity_status(3, True, .02, .05, policy()) == 'close_within_checked_set'


def test_blocks_retain_short_last_block_and_all_trials():
    from src.Bayesian_state.workflows.analysis.model_1024_state_interpretation_summary import block_slices
    blocks=block_slices(130,64)
    assert blocks == [(0,64),(64,128),(128,130)]
    assert sum(hi-lo for lo,hi in blocks)==130


def test_behavioral_predictions_use_evidence_before_the_current_choice():
    import numpy as np
    from src.Bayesian_state.workflows.analysis.model_1024_state_interpretation_summary import conditional_predictions
    probabilities=np.array([[[.9,.1],[.8,.2]],[[.1,.9],[.2,.8]]])
    prediction,logz=conditional_predictions(probabilities,np.array([1,1]))
    np.testing.assert_allclose(prediction[:,0],[.5,.74])
    np.testing.assert_allclose(np.prod(prediction[:,0]),np.exp(logz[:,-1]).mean())
    changed,_=conditional_predictions(probabilities,np.array([1,2]))
    np.testing.assert_allclose(changed,prediction)


def test_live_completion_repair_cannot_reuse_old_terminal_record(tmp_path):
    import json
    from src.Bayesian_state.workflows.analysis.model_1024_state_interpretation_summary import calculation_record
    original = dict(jobs_complete=287)
    (tmp_path/'calculation_completion.json').write_text(json.dumps(original))
    assert calculation_record(tmp_path) == original
    repair = tmp_path/'completion_repair_v1'
    repair.mkdir()
    (repair/'registration.json').write_text('{}')
    assert calculation_record(tmp_path) is None
    final = dict(jobs_complete=288)
    (tmp_path/'final_calculation_completion.json').write_text(json.dumps(final))
    assert calculation_record(tmp_path) == final
    assert json.loads((tmp_path/'calculation_completion.json').read_text()) == original
