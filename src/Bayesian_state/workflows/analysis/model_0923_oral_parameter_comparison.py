"""Compare frozen parameter points against the same oral reports.

Use --original ALIGNMENT --revised ALIGNMENT --subject ID --arm ARM --output NEW.
Both inputs must have complete behavior-only state replays, identical oral coding,
task inputs, mechanism and numerical budgets. Recompute the existing oral scores
and pair delete-one-filter estimates for common seeds by default. For disjoint
seed banks, --seed-relation independent combines the two delete-filter variances.
Both modes require equal filter budgets. Parameter labels stay separate from model
labels. This is retrospective development; oral content does not select parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_0923_decision_basis_alignment as alignment
from ...evaluation.oral.decision_basis_scoring import DOMAINS, LABELS
from ...evaluation.oral.information_controls import jackknife_error

IDENTITY = ['iSub', 'arm', 'iSession', 'iTrial', 'state_trial', 'choice', 'text', 'tokens',
            'query_id', 'scope', 'catalogue_version', 'training_trials', 'phase']


def paired_scores(original: tuple, revised: tuple, *, seed_relation: str = 'paired') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match reports; estimate MCSE using the declared relation of filter banks."""
    if seed_relation not in ('paired', 'independent'):
        raise ValueError('Seed relation must be paired or independent')
    old_trials, old_rows, old_estimates = original
    new_trials, new_rows, new_estimates = revised
    old_trials, new_trials = (table.sort_values('state_trial').reset_index(drop=True)
                             for table in (old_trials, new_trials))
    if old_trials.state_trial.duplicated().any() or new_trials.state_trial.duplicated().any():
        raise ValueError('Repeated trial identity')
    pd.testing.assert_frame_equal(old_trials[IDENTITY], new_trials[IDENTITY])
    keys = ['window', 'domain', 'subset']
    left, right = (pd.DataFrame(rows).set_index(keys).sort_index() for rows in (old_rows, new_rows))
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError('Repeated oral score window')
    count_columns = ['subject', 'condition', 'arm', 'window_trials', 'reports', 'full_reports',
                     'encoded_fragments', 'filter_seeds']
    pd.testing.assert_frame_equal(left[count_columns], right[count_columns])
    if set(left.index) != set(old_estimates) or set(left.index) != set(new_estimates):
        raise ValueError('Incomplete paired oral estimates')
    rows = []
    for key in left.index:
        a, b = old_estimates[key], new_estimates[key]
        for estimate in (a, b):
            seeds = estimate['seeds']
            if len(seeds) < 2 or len(set(seeds)) != len(seeds):
                raise ValueError('Need distinct complete filter seeds within each bank')
            if (len(seeds) != left.loc[key, 'filter_seeds']
                    or np.asarray(estimate['values']).shape != (len(seeds), len(LABELS))):
                raise ValueError('Incomplete delete-filter estimates')
        if seed_relation == 'paired':
            np.testing.assert_array_equal(a['seeds'], b['seeds'])
        elif set(a['seeds']) & set(b['seeds']):
            raise ValueError('Independent comparison requires disjoint filter seed banks')
        np.testing.assert_allclose(a['point'], left.loc[key, list(LABELS)].to_numpy(float), atol=1e-12, rtol=0)
        np.testing.assert_allclose(b['point'], right.loc[key, list(LABELS)].to_numpy(float), atol=1e-12, rtol=0)
        if seed_relation == 'paired':
            error = jackknife_error(b['values']-a['values'])
        else:
            # Disjoint complete-filter repetitions contribute independent variance.
            error = np.hypot(jackknife_error(a['values']), jackknife_error(b['values']))
        row = dict(zip(keys, key), **left.loc[key, count_columns].to_dict())
        for k, name in enumerate(LABELS):
            row['original_'+name] = a['point'][k]
            row['revised_'+name] = b['point'][k]
            row[name+'_difference'] = b['point'][k]-a['point'][k]
            row[name+'_difference_mcse'] = error[k]
        rows.append(row)
    trials = old_trials[IDENTITY].copy()
    for domain in DOMAINS:
        for label in LABELS:
            name = domain+'_'+label
            trials['original_'+name] = old_trials[name]
            trials['revised_'+name] = new_trials[name]
            trials[name+'_difference'] = new_trials[name]-old_trials[name]
    return pd.DataFrame(rows), trials


def load_group(directory: Path, subject: int, arm: str) -> tuple[tuple, dict, dict]:
    """Reproduce the saved scores while binding each state to its task and point."""
    read, digest = alignment.read, alignment.digest
    manifest = read(directory/'manifest.json')
    if (manifest['status'] != 'complete' or manifest['smoke'] or not manifest['full_replay_complete']
            or subject not in manifest['subjects'] or arm not in manifest['arms']):
        raise ValueError('Need a complete scientific alignment for the requested participant and model')
    source, audit = Path(manifest['source']), Path(manifest['oral_audit'])
    if (digest(source/'protocol.json') != manifest['source_protocol_sha256']
            or digest(audit/'manifest.json') != manifest['oral_manifest_sha256']
            or read(source/'completion.json')['status'] != 'complete'):
        raise ValueError('Replay or oral audit changed')
    files = [directory/'manifest.json', source/'protocol.json', source/'completion.json', audit/'manifest.json']
    for folder, hashes in [(directory, manifest['artifacts']), (audit, read(audit/'manifest.json')['artifacts'])]:
        for name, expected in hashes.items():
            if digest(folder/name) != expected: raise ValueError('Alignment or oral artifact changed')
            files.append(folder/name)
    protocol = read(source/'protocol.json')
    records, identity = [], None
    for file in sorted((source/'states').glob('*.json')):
        record = read(file)
        if (record['subject'], record['arm']) != (subject, arm): continue
        array = file.with_suffix('.npz'); job_file = source/'jobs'/file.name; job = read(job_file)
        if (manifest['state_arrays'].get(str(array.resolve())) != record['array_sha256']
                or job['candidate']['hyperparams'] != record['candidate']['hyperparams']
                or job['seeds'] != record['seeds'] or len(record['seeds']) != 1
                or digest(Path(job['original_job'])) != job['original_job_sha256']
                or digest(Path(job['reference'])) != job['reference_sha256']):
            raise ValueError('State, parameter or seed identity changed')
        folder = Path(job['folder'])
        task_files = [folder/name for name in ('training.npz', 'sealed_test.npz', 'engine.json')]
        current = dict(candidate=record['candidate'],
            scientific={name: record[name] for name in ('particles', 'trials', 'training_trials',
                        'condition', 'catalogue_signature', 'component_policy', 'timing')},
            mechanism=job['mechanism'], arm_settings=job['arm_settings'], archived_pf_sha256=job['archived_pf_sha256'],
            model_sources=job['source_sha256'], task_sha256={p.name:digest(p) for p in task_files})
        if identity is not None and current != identity: raise ValueError('Seed groups use different scientific settings')
        identity = current; records.append((file, record))
        files.extend([file, array, job_file, Path(job['original_job']), Path(job['reference']),
                      folder/'prepared.json', *task_files])
    if (len(records) != protocol['settings']['seeds_per_arm']
            or len({r['seeds'][0] for _, r in records}) != len(records)):
        raise ValueError('Incomplete or duplicate filter seed group')
    scored = alignment.group_scores(source, records, pd.read_csv(audit/'trials.csv'), pd.read_csv(audit/'component_matches.csv'))
    old_trials = pd.read_csv(directory/'trials.csv')
    old_trials = old_trials[old_trials.iSub.eq(subject) & old_trials.arm.eq(arm)].reset_index(drop=True)
    pd.testing.assert_frame_equal(scored[0].reset_index(drop=True), old_trials, check_dtype=False, atol=1e-12, rtol=0)
    old_summary = pd.read_csv(directory/'summary.csv')
    old_summary = old_summary[old_summary.subject.eq(subject) & old_summary.arm.eq(arm)].reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.DataFrame(scored[1]), old_summary, check_dtype=False, atol=1e-12, rtol=0)
    identity.update(oral_manifest_sha256=manifest['oral_manifest_sha256'],
                    parameter_estimation_scope=manifest.get('parameter_estimation_scope', 'original_evaluation_protocol'),
                    parameter_candidate_policy=manifest.get('parameter_candidate_policy', 'original_evaluation_protocol'))
    return scored, identity, {str(p.resolve()):digest(p) for p in files}


def run(original: Path, revised: Path, output: Path, subject: int, arm: str,
        *, seed_relation: str = 'paired') -> None:
    if output.exists(): raise FileExistsError('Use a new oral parameter comparison directory')
    old, before, inputs = load_group(original, subject, arm)
    new, after, more = load_group(revised, subject, arm); inputs.update(more)
    for key in ('scientific', 'mechanism', 'arm_settings', 'archived_pf_sha256', 'task_sha256', 'oral_manifest_sha256'):
        if before[key] != after[key]: raise ValueError('Parameter comparison requires the same task, mechanism, oral coding and budget')
    if any(after['model_sources'].get(name) != sha for name, sha in before['model_sources'].items()):
        raise ValueError('Revised parameter replay changes an original executing dependency')
    scores, trials = paired_scores(old, new, seed_relation=seed_relation)
    output.mkdir(parents=True)
    scores.to_csv(output/'scores.csv', index=False); trials.to_csv(output/'trials.csv', index=False)
    display = scores[scores.subset.eq('complete') & scores.domain.eq('structural')]
    error_description = ('模拟标准误按共同种子逐次删除一个完整滤波重复计算。' if seed_relation == 'paired' else
                         '两组滤波种子互不重叠，分别删除一个完整滤波重算分数；差值的模拟标准误为两组方差之和的平方根。')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>参数变化与口述表征</title>'
            '<style>body{max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
            '<h1>同一模型在两组冻结参数下的口述表征</h1>'
            '<p>两组参数均由行为数据确定，口述编码、试次与完整报告分母保持一致。'
            '差值为修订参数减原参数。'+error_description+
            '原有training／test标记表示时间范围。本次为回顾性开发比较，参数和被试不确定性另行评价。</p>'
            +display.to_html(index=False, float_format=lambda x:f'{x:.6f}')+
            '<p><a href="scores.csv">全部时间范围与两种口径</a> · <a href="trials.csv">全部逐题质量差值</a></p>')
    (output/'index.html').write_text(page)
    for name, expected in inputs.items():
        if alignment.digest(Path(name)) != expected: raise ValueError('Comparison input changed during scoring')
    sources = [Path(__file__), Path(alignment.__file__),
               alignment.ROOT/'src/Bayesian_state/evaluation/oral/decision_basis.py',
               alignment.ROOT/'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py',
               alignment.ROOT/'src/Bayesian_state/evaluation/oral/information_controls.py']
    result = dict(status='complete', subject=subject, arm=arm, original=str(original.resolve()), revised=str(revised.resolve()),
        original_parameters=before, revised_parameters=after, original_scores_reproduced=True,
        revised_scores_reproduced=True, seed_relation=seed_relation,
        uncertainty=('paired_delete_one_complete_filter_seed_mcse' if seed_relation == 'paired' else
                     'independent_sum_of_delete_one_complete_filter_variances_mcse'),
        data_use='retrospective development; both parameter points behavior-selected; oral evaluation only',
        input_sha256=inputs, source_sha256={str(p.relative_to(alignment.ROOT)):alignment.digest(p) for p in sources},
        artifacts={p.name:alignment.digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'revised', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--seed-relation', choices=('paired', 'independent'), default='paired',
                        help='Use paired errors for common seeds, or independent errors for disjoint banks')
    args = parser.parse_args()
    run(args.original, args.revised, args.output, args.subject, args.arm, seed_relation=args.seed_relation)
