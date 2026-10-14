"""Guard cohort denominators, equivalent-rule replacement and censoring."""
import numpy as np
import yaml

from src.Bayesian_state.evaluation.model_0923_cohort_paths import set_residences, summarize_trace
from src.Bayesian_state.workflows.analysis.model_0923_cohort_path_diagnosis import records_from_config
from src.Bayesian_state.workflows.analysis.model_0923_path_diagnosis import PathRecorder


def test_set_residence_censoring_and_eligibility():
    a = np.array([[1, 0], [1, 1], [0, 1], [1, 1], [1, 1]], bool)
    rows = set_residences(a, 3)
    assert [(r['path'], r['duration'], r['right_censored'], r['retention_eligible'], r['retained']) for r in rows] == [
        (0, 2, False, True, False), (0, 2, True, False, False), (1, 4, True, True, True)]


def trace():
    ids = np.array([[[2, -1]], [[3, -1]], [[3, -1]], [[4, -1]]])
    a = {k: np.where(ids >= 0, 1., np.nan) for k in
         ('prior', 'posterior', 'forgotten', 'contribution', 'beta_pre', 'rule_correct')}
    a.update(rule=ids, categories=np.ones(4, int), prediction=np.tile([.8, .2], (4, 1, 1)),
        choice=np.ones((4, 1), int), feedback=np.ones((4, 1)), proposed=np.array([[-1], [3], [-1], [4]]),
        exits=np.array([[[-1, -1]], [[2, -1]], [[-1, -1]], [[3, -1]]]),
        capacity_dropped=np.full((4, 1), -1), pairing_pre_true=np.full(ids.shape, np.nan))
    return a


def test_equivalent_rule_replacement_preserves_set_but_records_member_exit():
    paths, episodes, exits, risk, _, _, _ = summarize_trace(trace(), [2, 3], 2, .5, 2)
    assert len(episodes) == 1 and episodes[0]['duration'] == 3
    assert [e['set_lost'] for e in exits] == [False, True]
    assert len(risk) == 3  # final trial is never in the transition denominator
    assert paths[0]['late_active'] == .5
    assert paths[0]['useful_proposed'] == 1


def test_empty_compatible_set_keeps_paths_and_has_no_exit_exposure():
    paths, ep, exits, risk, _, _, _ = summarize_trace(trace(), [], 2, .5, 2)
    assert paths[0]['ever_active'] is False
    assert paths[0]['late_active'] == 0
    assert ep == exits == risk == []


def test_primary_sample_has_fifteen_people_and_seventeen_fits():
    from pathlib import Path
    settings = yaml.safe_load(Path('configs/shared/model0923_cohort_path_diagnosis.yaml').read_text())
    records = records_from_config(settings)
    assert len(records) == 17
    assert sum(r['primary'] for r in records) == 15
    assert {r['subject'] for r in records if not r['primary']} == {307, 335}


def test_binary_recorder_has_two_choices_and_no_pairing_values():
    recorder = PathRecorder(np.array([1]), 1, 2, category_count=2, has_pairing=False)
    assert recorder.arrays['prediction'].shape == (1, 1, 2)
    assert np.isnan(recorder.arrays['pairing_pre_true']).all()
