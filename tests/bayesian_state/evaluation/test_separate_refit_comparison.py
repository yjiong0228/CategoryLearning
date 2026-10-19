"""Check mechanism labels, paired-path uncertainty and scientific audit binding."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation import model_0923_separate_refit_comparison as comparison


def test_feedback_scores_keep_actual_mechanism_names_and_loss_direction():
    old = dict(seeds=np.array([1, 2, 3]), choices=np.array([1, 2, 3, 4]),
               observed_probability=np.array([[.1, .9, .2, .3], [.9, .1, .4, .6], [.2, .8, .3, .9]]))
    new = dict(old, observed_probability=np.array([[.3, .8, .7, .3], [.7, .4, .9, .4], [.5, .6, .8, .8]]))
    table, trials = comparison.feedback_scores(old, new, np.array([0., .5, 1., 1.]), 2)
    assert not any(name.startswith('retained') for frame in (table, trials) for name in frame)
    np.testing.assert_array_equal(trials.online_probability, old['observed_probability'].mean(0))
    np.testing.assert_array_equal(trials.separate_probability, new['observed_probability'].mean(0))
    whole = table[(table.phase == 'whole') & (table.feedback_group == 'all')].iloc[0]
    assert whole.online_choice_nll == pytest.approx(-np.log(old['observed_probability'].mean(0)).mean())
    assert whole.separate_choice_nll == pytest.approx(-np.log(new['observed_probability'].mean(0)).mean())
    assert whole.delta_choice_nll == pytest.approx(whole.separate_choice_nll - whole.online_choice_nll)


def path_tables():
    fields = ('late_accuracy', 'target_ever', 'target_late_present', 'target_late_contribution',
              'target_exits', 'target_capacity_drops', 'search_events', 'target_proposals')
    old = pd.DataFrame(dict(seed=[1, 2, 3], **{key: [.2, .5, .7] for key in fields}))
    new = pd.DataFrame(dict(seed=[3, 1, 2], **{key: [.6, .4, .8] for key in fields}))
    return old, new


def test_paths_pair_by_seed_and_error_deletes_complete_paths():
    old, new = path_tables(); scores = comparison.path_contrasts(old, new)
    delta = np.array([.2, .3, -.1])
    deleted = np.array([np.delete(delta, i).mean() for i in range(3)])
    error = np.sqrt(2 / 3 * ((deleted - deleted.mean()) ** 2).sum())
    np.testing.assert_allclose(scores.delta, delta.mean())
    np.testing.assert_allclose(scores.paired_mc_se, error)
    np.testing.assert_allclose(scores.online, old.late_accuracy.mean())
    np.testing.assert_allclose(scores.separate, new.late_accuracy.mean())
    identical = comparison.path_contrasts(old, old)
    assert (identical.delta == 0).all() and (identical.paired_mc_se == 0).all()


@pytest.mark.parametrize('fault', ['duplicate', 'different', 'single', 'nonfinite'])
def test_path_summary_rejects_invalid_pairing(fault):
    old, new = path_tables()
    if fault == 'duplicate': new.loc[1, 'seed'] = 3
    elif fault == 'different': new.loc[1, 'seed'] = 4
    elif fault == 'single': old, new = old.iloc[:1], new.iloc[:1]
    elif fault == 'nonfinite': new.loc[1, 'late_accuracy'] = np.nan
    with pytest.raises((ValueError, AssertionError)): comparison.path_contrasts(old, new)


def audit_fixture(tmp_path):
    def save(path, value): path.write_text(json.dumps(value))
    source = tmp_path / 'summary'; source.mkdir()
    manifest, audits = {}, {}
    for role in ('fit', 'evaluation'):
        run = tmp_path / role; (run / 'jobs').mkdir(parents=True)
        job = run / 'jobs' / 'worker.json'; save(job, dict(role=role))
        manifest[role] = str(run)
        audits[role] = dict(status='complete', smoke=False, jobs=1, rows=[dict(role=role)],
                            inputs_sha256={str(job): comparison.digest(job)})
    save(source / 'manifest.json', manifest)
    save(source / 'execution_audit.json', audits)
    summary_file = Path(comparison.summary.__file__)
    extra = dict(status='complete', belief_intervention=comparison.refit.INTERVENTION,
        generic_manifest_sha256=comparison.digest(source / 'manifest.json'),
        execution_audit_sha256=comparison.digest(source / 'execution_audit.json'),
        jobs=dict(fit=1, evaluation=1), inputs_sha256={},
        source_sha256={str(summary_file.relative_to(comparison.refit.ROOT)): comparison.digest(summary_file)})
    save(source / 'separate_manifest.json', extra)
    return source, manifest, audits, extra, save


def test_binds_existing_scientific_summary_and_executed_inputs(tmp_path):
    source, manifest, _, _, _ = audit_fixture(tmp_path)
    inputs = comparison.audited_inputs(source, manifest)
    assert str(source / 'execution_audit.json') in inputs
    assert str(Path(manifest['evaluation']) / 'jobs' / 'worker.json') in inputs


@pytest.mark.parametrize('fault', ['summary', 'audit', 'job', 'missing_job', 'extra_job',
                                  'smoke', 'count', 'wrong_mechanism', 'auditor', 'row_count'])
def test_rejects_changed_or_incomplete_execution_audit(tmp_path, fault):
    source, manifest, audits, extra, save = audit_fixture(tmp_path)
    if fault == 'summary': save(source / 'manifest.json', dict(changed=True))
    elif fault == 'audit': save(source / 'execution_audit.json', dict(changed=True))
    elif fault == 'job': save(Path(manifest['fit']) / 'jobs' / 'worker.json', dict(changed=True))
    elif fault == 'extra_job': save(Path(manifest['fit']) / 'jobs' / 'extra.json', dict(changed=True))
    elif fault == 'wrong_mechanism': extra['belief_intervention'] = 'online_retention_evidence'
    elif fault == 'auditor': extra['source_sha256'] = {}
    else:
        if fault == 'smoke': audits['fit']['smoke'] = True
        elif fault == 'count': audits['fit']['jobs'] = 2
        elif fault == 'row_count': audits['fit']['rows'] = []
        elif fault == 'missing_job': audits['fit']['inputs_sha256'] = {}
        save(source / 'execution_audit.json', audits)
        extra['execution_audit_sha256'] = comparison.digest(source / 'execution_audit.json')
    save(source / 'separate_manifest.json', extra)
    with pytest.raises(ValueError): comparison.audited_inputs(source, manifest)
