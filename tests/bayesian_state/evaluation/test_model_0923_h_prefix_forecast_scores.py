"""Score uncertainty keeps branches from one filter in one MC cluster."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_h_prefix_forecast import (
    clustered_mean_se, filter_cluster_jackknife, resolve_summary,
)
from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se


def test_cluster_deletion_matches_known_three_filter_distribution():
    samples = np.array([[0.], [0.], [1.], [1.], [1.], [1.]])
    labels, scores = filter_cluster_jackknife(samples, np.array([0.]), np.array([11, 11, 22, 22, 33, 33]))
    np.testing.assert_array_equal(labels, [11, 22, 33])
    np.testing.assert_allclose(scores['energy'], [1., .25, .25])
    np.testing.assert_array_equal(scores['energy'], scores['crps'])
    assert jackknife_se(scores['energy']) == pytest.approx(.5)


@pytest.mark.parametrize('labels', [[1, 1, 1], [1, 1, 2], [1, 2]])
def test_invalid_filter_groups_are_rejected(labels):
    with pytest.raises(ValueError):
        filter_cluster_jackknife(np.zeros((3, 1)), np.zeros(1), np.asarray(labels))


def test_repeated_branches_do_not_inflate_independent_information():
    original = clustered_mean_se(np.array([0., 1., 1.]), np.array([1, 2, 3]))
    repeated = clustered_mean_se(np.array([0., 0., 1., 1., 1., 1.]), np.array([1, 1, 2, 2, 3, 3]))
    assert original == pytest.approx(1/3)
    assert repeated == original


def test_recalled_summary_requires_same_frozen_nomination(tmp_path):
    import json
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest
    fit, evaluation, summary = [tmp_path / name for name in ('fit', 'evaluation', 'summary')]
    for path in (fit, evaluation, summary):path.mkdir()
    (fit / 'nominees.json').write_text('{}')
    sha = digest(fit / 'nominees.json')
    for name in ('protocol', 'completion'):
        (evaluation / (name + '.json')).write_text(json.dumps(dict(status='complete', nomination_sha256=sha)))
    manifest = dict(status='complete', fit=str(fit), evaluation=str(evaluation), reference_model='POOL')
    (summary / 'manifest.json').write_text(json.dumps(manifest))
    result, source = resolve_summary(summary)
    assert result['nomination_sha256'] == sha
    assert result['conditional'] == str(evaluation / 'conditional')
    assert source == summary / 'manifest.json'
    (fit / 'nominees.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='same completed frozen fit'):
        resolve_summary(summary)


def test_recalled_score_pipeline_keeps_filter_uncertainty_and_target_identity(tmp_path):
    import json
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_h_prefix_forecast import summarize
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest

    def write(file, value):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(value))

    fit, evaluation, summary, forecast = [tmp_path / name for name in ('fit', 'evaluation', 'summary', 'forecast')]
    write(fit / 'nominees.json', {})
    sha = digest(fit / 'nominees.json')
    for name in ('protocol', 'completion'):
        write(evaluation / (name + '.json'), dict(status='complete', nomination_sha256=sha))
    write(summary / 'manifest.json', dict(status='complete', fit=str(fit), evaluation=str(evaluation), reference_model='POOL'))
    candidate = dict(id='fixture', hyperparams={})
    common = dict(subject=207, condition=2, model='POOL', candidate=candidate, training_trials=2)
    future_choices = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    present = np.array([[True, False], [True, True], [False, False], [False, False]])
    categories = np.array([1, 2, 1, 2])
    conditional = dict(choices=categories, categories=categories, correct_probability=np.ones((2, 4)))
    autonomous = dict(choices=np.concatenate([np.ones((4, 2), int), future_choices], axis=1), categories=categories,
                      correct_probability=np.full((4, 4), .5), seeds=np.arange(101, 105),
                      present=np.concatenate([np.zeros((4, 2), bool), present], axis=1),
                      contribution=np.concatenate([np.zeros((4, 2)), present.astype(float)], axis=1))
    for part, arrays in [('conditional', conditional), ('autonomous', autonomous)]:
        file = evaluation / part / 'evaluate/subject_207/POOL/fixture.npz'
        file.parent.mkdir(parents=True)
        np.savez(file, **arrays)
        write(evaluation / part / 'evaluation.json', dict(status='complete', rows=[
            dict(common, predictions_sha256=digest(file), target_rule=0)]))
    forecast.mkdir()
    file = forecast / 'paths.npz'
    np.savez(file, choices=future_choices, categories=categories[2:], probabilities=np.full((4, 2, 2), .5),
             branch_seeds=np.arange(201, 205), branch_filter_seed=[11, 11, 22, 22], filter_seed=[11, 22],
             rule=np.where(present, 0, 99)[..., None], contribution=np.ones((4, 2, 1)),
             search_probability=np.zeros((4, 2)))
    write(forecast / 'forecast_protocol.json', dict(nomination_sha256=sha, smoke=False))
    record = dict(common, forecast_trials=2, target=str(file.with_suffix('')), predictions_sha256=digest(file))
    write(forecast / 'forecast.json', dict(status='complete', smoke=False, rows=[record]))
    summarize(summary, forecast, tmp_path / 'scores')
    table = pd.read_csv(tmp_path / 'scores/scores.csv')
    row = table[table.start == 'filtered_prefix'].iloc[0]
    assert row.late_generated_accuracy == .5
    assert row.late_generated_accuracy_mc_se == .25
    assert row.late_target_present == .375
    assert row.late_target_present_mc_se == pytest.approx(.375)
    assert row.late_target_contribution == .375
    assert row.first_future_target_present == .5
    assert row.future_target_ever_present_paths == 2
    assert row.paths_reaching_observed_late_accuracy == 1
    assert row.independent_units == 2
    record['training_trials'] = 1
    write(forecast / 'forecast.json', dict(status='complete', smoke=False, rows=[record]))
    with pytest.raises(ValueError, match='chronological boundaries'):
        summarize(summary, forecast, tmp_path / 'wrong_boundary')
