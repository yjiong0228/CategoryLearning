"""Compare all frozen recovery points using their original audit filters.

Use --study COMPLETE_RECOVERY_PHASE --audit-summary COMPLETE_AUDIT --output NEW.
The original reference, selected estimate, alternative and generating point use
the same independent seeds. Existing target-state definitions are retained.
Paired whole-filter jackknife errors describe numerical precision, with a fixed
common trial set for conditional precision. No filters or searches are added.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_target_state_recovery import summaries
from .model_0923_h_validation import jackknife_scores, jackknife_se, nll
from ..workflows.analysis.model_1023_estimation_reliability import ROOT, read, write, digest
from ..workflows.analysis.model_1021_simplification_recovery import candidate

PHASES = ('predictive', 'filtered')
MOMENTS = ('presence_probability', 'contribution_mean', 'conditional_beta_mean')


def state_traces(bank: dict, truth: dict, phase: str) -> tuple:
    """Reuse the established weighted state summaries for every replicate."""
    curves, scores = summaries(bank, truth, phase)
    traces = [{k: curves[k] for k in MOMENTS}]
    for seed in range(len(next(iter(bank.values())))):
        own, _ = summaries({k: np.delete(v, seed, axis=0) for k, v in bank.items()}, truth, phase)
        traces.append({k: own[k] for k in MOMENTS})
    return curves, scores, {k: np.stack([t[k] for t in traces]) for k in MOMENTS}


def paired_changes(a: dict, b: dict, truth: dict) -> dict:
    """Candidate minus reference errors, fixing scored trials across filters."""
    result = {}
    for moment, metric, target, root in (
        ('presence_probability', 'membership_brier', 'present', False),
        ('contribution_mean', 'contribution_rmse', 'contribution', True),
        ('conditional_beta_mean', 'conditional_beta_rmse', 'beta', True),
    ):
        x, y = a[moment], b[moment]
        if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 3:
            raise ValueError('Require matching full estimates and whole-filter replicates')
        mask = np.isfinite(x[0]) & np.isfinite(y[0])
        if target == 'beta':
            mask &= np.asarray(truth['present'], dtype=bool)
        result[metric+'_trials'] = int(mask.sum())
        available = mask.any() and np.isfinite(x[:, mask]).all() and np.isfinite(y[:, mask]).all()
        if not mask.any():
            result[metric+'_delta'] = result[metric+'_mc_se'] = np.nan
            continue
        xx = ((x[:, mask]-np.asarray(truth[target])[mask])**2).mean(1)
        yy = ((y[:, mask]-np.asarray(truth[target])[mask])**2).mean(1)
        difference = (np.sqrt(xx)-np.sqrt(yy)) if root else xx-yy
        result[metric+'_delta'] = difference[0]
        result[metric+'_mc_se'] = jackknife_se(difference[1:]) if available else np.nan
    return result


def load_point(folder: Path, case: dict, point: dict, smoke: bool, bind) -> tuple:
    count, particles = (2, 2) if smoke else (8, 1024)
    trials = min(8, case['trials']) if smoke else case['trials']
    files = sorted((folder/'state_capture').glob(f'confirmation_S{case["subject"]}_COMPACT_{point["id"]}_C*.json'))
    if len(files) != count:
        raise ValueError('Incomplete original recovery captures')
    loaded = []
    for file in files:
        record = read(bind(file))
        if (record['status'] != 'complete' or record['smoke'] is not smoke
                or record['candidate']['hyperparams'] != point['hyperparams']
                or record['target_rule'] != case['settings']['recovery_target_rule']
                or record['particles'] != particles or record['trials'] != trials
                or record['additional_filters'] != 0 or not record['exact_original_filter_capture']
                or len(record['seeds']) != 1):
            raise ValueError('Recovery capture identity or numerical budget differs')
        job = read(bind(record['original_job'], record['original_job_sha256']))
        if (job['seeds'] != record['seeds'] or job['candidate']['hyperparams'] != point['hyperparams']
                or job['target_rule'] != record['target_rule']
                or Path(job['folder']).resolve() != (ROOT/case['settings']['data']/f'subject_{case["subject"]}').resolve()):
            raise ValueError('Recovery capture uses different behavior or parameters')
        for source, sha in record['capture_source_sha256'].items():
            bind(ROOT/source, sha)
        with np.load(bind(file.with_suffix('.npz'), record['array_sha256'])) as saved:
            np.testing.assert_array_equal(saved['seeds'], record['seeds'])
            arrays = {k: saved[k].copy() for k in saved.files if k != 'seeds'}
        expected = {p+'_'+k for p in PHASES for k in ('weights', 'present', 'contribution', 'active_beta')}
        if set(arrays) != expected or any(v.shape != (trials, particles) for v in arrays.values()):
            raise ValueError('Recovery state fields or shapes changed')
        with np.load(bind(record['score_file'], record['score_sha256'])) as saved:
            np.testing.assert_array_equal(saved['seeds'], record['seeds'])
            probability = saved['observed_probability'][0].copy()
        if probability.shape != (trials,) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError('Invalid captured behavior probability')
        loaded.append((record['seeds'][0], arrays, probability))
    loaded.sort(key=lambda item: item[0])
    seeds = np.array([r[0] for r in loaded])
    np.testing.assert_array_equal(seeds, sorted(case['independent_seed_banks']['post_freeze_audit'][:count]))
    if len(set(seeds)) != count or case['generation_seed'] in seeds:
        raise ValueError('Recovery filter seeds overlap or repeat')
    return ({k: np.stack([r[1][k] for r in loaded]) for k in loaded[0][1]},
            np.stack([r[2] for r in loaded]))


def summarize(study: Path, audit: Path, output: Path, allow_smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery-candidate directory')
    done, audited = read(study/'completion.json'), read(audit/'completion.json')
    if (done['status'] != 'complete' or audited['status'] != 'complete'
            or done['phase'] == 'human' or done['phase'] != audited['phase']
            or Path(audited['study']).resolve() != study.resolve() or done['smoke'] is not audited['smoke']
            or done['smoke'] and not allow_smoke):
        raise ValueError('Require a completed matching recovery phase and budget audit')
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Recovery comparison input changed: '+str(path))
        inputs[str(path)] = sha
        return path

    for file in (study/'completion.json', audit/'completion.json', Path(__file__),
                 Path(inspect.getfile(summaries)), Path(inspect.getfile(jackknife_se)),
                 Path(inspect.getfile(candidate))):
        bind(file)
    for file, sha in audited['inputs_sha256'].items():
        bind(file, sha)
    cases = read(bind(study/'cases.json'))['cases']
    preparation = read(bind(study/'audit_prepared/completion.json'))
    truths = {r['case_id']: r for r in read(bind(study/'audit_prepared/truths.json', preparation['truths_sha256']))['rows']}
    scores, trials, changes = [], [], []
    for case in cases:
        case_id = case['case_id']
        original = read(bind(study/'audit_fit'/case_id/'nominees.json'))['nominee']
        actual = truths[case_id]
        with np.load(bind(actual['states_truth'], actual['states_truth_sha256'])) as saved:
            truth = {k: saved[k][:min(8, case['trials']) if done['smoke'] else case['trials']].copy()
                     for k in ('present', 'contribution', 'beta')}
        points = [(p, study/'audit_fit'/case_id, False) for p in original['finalists']]
        points.append((candidate(actual['point'], 'generating_point_after_freeze'), study/'recovery_scores'/case_id, True))
        traces, probabilities, tags = {}, {}, {}
        for point, folder, generating in points:
            identity = point['id']+('_generating' if generating else '')
            tag = dict(case_id=case_id, subject=case['subject'], candidate=point['id'],
                selected=not generating and point['id'] == original['candidate']['id'],
                reference=not generating and point['id'] == original['reference_id'], generating=generating)
            bank, probability = load_point(folder, case, point, done['smoke'], bind)
            tags[identity], probabilities[identity] = tag, probability
            for phase in PHASES:
                curve, metrics, trace = state_traces(bank, truth, phase)
                traces[identity, phase] = trace
                scores.append(dict(tag, phase=phase, choice_nll=nll(probability), filters=len(probability), **metrics))
                trials.extend(dict(tag, phase=phase, trial=t+1,
                    **{'truth_'+k: v[t] for k, v in truth.items()}, **{k: v[t] for k, v in curve.items()})
                    for t in range(len(truth['present'])))
            del bank
        reference, = [identity for identity, tag in tags.items() if tag['reference']]
        for identity, tag in tags.items():
            probability, baseline = probabilities[identity], probabilities[reference]
            for phase in PHASES:
                changes.append(dict(tag, phase=phase, reference_candidate=tags[reference]['candidate'],
                    nll_delta=nll(probability)-nll(baseline),
                    nll_mc_se=jackknife_se(jackknife_scores(probability)-jackknife_scores(baseline)),
                    **paired_changes(traces[identity, phase], traces[reference, phase], truth)))
    output.mkdir(parents=True)
    for name, rows in [('scores', scores), ('trials', trials), ('changes_to_reference', changes)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'), index=False)
    write(output/'completion.json', dict(status='complete', smoke=done['smoke'], study=str(study.resolve()),
        cases=len(cases), additional_filters=0, inputs_sha256=inputs,
        scope='All fixed audited points and the generating point; one recorded target rule per synthetic sequence',
        uncertainty='Paired whole-filter numerical error; finite-sequence descriptive recovery, no trait confidence interval',
        state_pooling='Original equal-filter weighted summaries, with precision conditional on target presence',
        conditional_precision_comparison='Common truth-present trials fixed across jackknife replicates; undefined values retained',
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit-summary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    summarize(args.study, args.audit_summary, args.output, args.allow_smoke)
