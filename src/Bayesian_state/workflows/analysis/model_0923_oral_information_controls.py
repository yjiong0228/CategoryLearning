"""Audit current-choice weighting and static-prefix oral support controls.

Run --alignment COMPLETED_ALIGNMENT --output NEW_DIRECTORY, optionally
--subjects IDS. Read already frozen behavior-only states; perform no fitting.
All controls use the observed category to define local regions. Recompute the
prefix control in each delete-filter replicate. Save support, conflict and
undetermined mass, paired differences, numerical errors and provenance.
"""
from __future__ import annotations

import argparse
import html
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from . import model_0923_decision_basis_alignment as alignment
from ...evaluation.oral.decision_basis_scoring import DOMAINS, LABELS
from ...evaluation.oral.information_controls import METRICS, information_controls, jackknife_error

ROOT = alignment.ROOT


def run(aligned: Path, output: Path, subjects: list[int] | None = None) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral-information directory')
    read, digest = alignment.read, alignment.digest
    manifest = read(aligned/'manifest.json')
    if manifest['status'] != 'complete' or manifest['smoke'] or not manifest['full_replay_complete']:
        raise ValueError('Require complete scientific alignment and state replay')
    scope_metadata, scope_note = alignment.estimation_context(manifest)
    selected = subjects if subjects is not None else manifest['subjects']
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(manifest['subjects']):
        raise ValueError('Subjects must be a nonempty unique aligned subset')
    source, audit = Path(manifest['source']), Path(manifest['oral_audit'])
    protocol = read(source/'protocol.json')
    if (digest(source/'protocol.json') != manifest['source_protocol_sha256']
            or digest(audit/'manifest.json') != manifest['oral_manifest_sha256']
            or read(source/'completion.json')['status'] != 'complete'):
        raise ValueError('Original replay or oral audit changed')
    dependencies = [aligned/'manifest.json', source/'protocol.json', source/'completion.json', audit/'manifest.json']
    for directory, hashes in [(aligned, manifest['artifacts']), (audit, read(audit/'manifest.json')['artifacts'])]:
        for name, expected in hashes.items():
            if digest(directory/name) != expected:
                raise ValueError('Alignment or oral-audit artifact changed')
            dependencies.append(directory/name)
    oral = pd.read_csv(audit/'trials.csv')
    matches = pd.read_csv(audit/'component_matches.csv')
    old = pd.read_csv(aligned/'trials.csv')
    groups = {}
    for file in sorted((source/'states').glob('*.json')):
        record = read(file)
        if record['subject'] in selected and record['arm'] in manifest['arms']:
            groups.setdefault((record['subject'], record['arm']), []).append((file, record))
    expected = {(subject, arm) for subject in selected for arm in manifest['arms']}
    if set(groups) != expected:
        raise ValueError('Incomplete subject/arm groups')
    for records in groups.values():
        for file, _ in records:
            dependencies.extend([file, file.with_suffix('.npz'), source/'jobs'/file.name])
        folder = Path(read(source/'jobs'/records[0][0].name)['folder'])
        dependencies.extend(folder/name for name in ('prepared.json', 'engine.json', 'training.npz', 'sealed_test.npz'))
    inputs = {str(p.resolve()): digest(p) for p in dependencies}
    source_files = [Path(__file__), ROOT/'src/Bayesian_state/evaluation/oral/information_controls.py',
        Path(alignment.__file__), ROOT/'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py']
    sources = {str(file.relative_to(ROOT)): digest(file) for file in source_files}
    output.mkdir(parents=True)
    for file in source_files:
        destination = output/'source'/file.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(file, destination)
    tables, summaries, contrasts, saved, estimates = [], [], [], {}, {}
    for (subject, arm), records in sorted(groups.items()):
        records.sort(key=lambda item: item[1]['seeds'][0])
        seeds = [r['seeds'][0] for _, r in records]
        if len(seeds) != protocol['settings']['seeds_per_arm'] or len(set(seeds)) != len(seeds):
            raise ValueError('Incomplete independent filter bank')
        first_file, first = records[0]
        job_file = source/'jobs'/first_file.name
        job = read(job_file)
        trials, cut = alignment.align_identities(oral, job, first['trials'])
        banks = {name: [] for name in ('choice_component_joint_mass', 'choice_probability', 'prior_component_mass')}
        labels = None
        for file, record in records:
            array = file.with_suffix('.npz')
            if (not record['exact_behavior_replay'] or record['catalogue_signature'] != first['catalogue_signature']
                    or record['trials'] != len(trials) or digest(array) != record['array_sha256']
                    or manifest['state_arrays'].get(str(array.resolve())) != record['array_sha256']):
                raise ValueError('Frozen state identity or array changed')
            with np.load(array) as z:
                np.testing.assert_array_equal(z['choices'], trials.choice.to_numpy())
                for name in banks: banks[name].append(z[name])
                if labels is None:
                    labels = alignment.make_labels(trials, matches, *z['prior_component_mass'].shape[1:])
        ordinal = np.arange(len(trials))
        complete = trials.scope.eq('complete').to_numpy()
        masks = dict(test=complete & (ordinal >= cut),
                     last64=complete & (ordinal >= max(cut, len(trials)-64)))
        result = information_controls(*[np.asarray(banks[name]) for name in banks],
            labels, trials.choice.to_numpy(), cut, masks)
        previous = old.loc[old.iSub.eq(subject) & old.arm.eq(arm)].sort_values('state_trial')
        np.testing.assert_array_equal(previous[['iSession', 'iTrial']].to_numpy(), trials[['iSession', 'iTrial']].to_numpy())
        table = trials[['iSub', 'iSession', 'iTrial', 'state_trial', 'scope']].copy()
        table['arm'] = arm
        for m, metric in enumerate(METRICS):
            for d, domain in enumerate(DOMAINS):
                for k, label in enumerate(LABELS):
                    values = result['trials'][m, :, d, k]
                    table[f'{metric}_{domain}_{label}'] = values
                    if m < 3:
                        old_key = ('', 'prior_', 'prefix_static_')[m]+f'{domain}_{label}'
                        np.testing.assert_allclose(values, previous[old_key], atol=1e-12, rtol=0, equal_nan=True)
        tables.append(table)
        estimates[subject, arm] = dict(seeds=seeds, result=result['windows'])
        for window, values in result['windows'].items():
            saved[f'S{subject}_{arm}_{window}_jackknife'] = values['jackknife']
            saved[f'S{subject}_{arm}_{window}_seeds'] = np.asarray(seeds)
            for m, metric in enumerate(METRICS):
                for d, domain in enumerate(DOMAINS):
                    row = dict(subject=subject, arm=arm, window=window, domain=domain, metric=metric,
                        filter_seeds=len(seeds), requested_reports=values['requested_reports'],
                        reports=values['reports'], excluded_reports=values['excluded_reports'])
                    for k, label in enumerate(LABELS):
                        row[label] = values['point'][m, d, k]
                        row[label+'_mcse'] = values['mcse'][m, d, k]
                    summaries.append(row)
        print(f'{subject} {arm}: all original trial scores reproduced; controls recalculated in each jackknife', flush=True)
    reference = manifest['reference_model']
    for (subject, arm), group in estimates.items():
        if arm == reference: continue
        base = estimates[subject, reference]
        np.testing.assert_array_equal(group['seeds'], base['seeds'])
        for window, value in group['result'].items():
            other = base['result'][window]
            np.testing.assert_array_equal(value['mask'], other['mask'])
            delta = value['point']-other['point']
            error = jackknife_error(value['jackknife']-other['jackknife'])
            for m, metric in enumerate(METRICS):
                for d, domain in enumerate(DOMAINS):
                    row = dict(subject=subject, arm=arm, reference=reference, window=window,
                               domain=domain, metric=metric, reports=value['reports'])
                    for k, label in enumerate(LABELS):
                        row[label+'_difference'] = delta[m, d, k]
                        row[label+'_difference_mcse'] = error[m, d, k]
                    contrasts.append(row)
    pd.concat(tables, ignore_index=True).to_csv(output/'trials.csv', index=False)
    summary = pd.DataFrame(summaries); summary.to_csv(output/'summary.csv', index=False)
    difference = pd.DataFrame(contrasts); difference.to_csv(output/'contrasts.csv', index=False)
    with (output/'jackknife.npz').open('xb') as stream: np.savez_compressed(stream, **saved)
    display = summary.loc[summary.window.eq('test') & summary.domain.eq('structural'),
        ['subject', 'arm', 'metric', 'reports', 'support', 'support_mcse']]
    note = ('所有对照均按实际所选类别确定局部区域。unweighted仅去掉当前选择似然的权重更新；'
            '历史行为条件推断仍保留。prefix_static冻结前段各所选类别的平均局部依据。'
            '动态与静态差值同时涉及状态变化和刺激相关读出。支持质量用于表征检验。'
            '模拟标准误逐次删除完整滤波种子，并重建静态基线；参数和被试抽样不确定性另行评价。')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>口述支持的信息对照</title>'
            '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}td,th{padding:6px}</style>'
            '<h1>口述支持的信息对照</h1><p>'+note+'</p>'+scope_note+display.to_html(index=False, float_format=lambda x:f'{x:.5f}')+
            '<ul>'+''.join(f'<li><a href="{html.escape(n)}">{n}</a></li>' for n in
                ('trials.csv','summary.csv','contrasts.csv','jackknife.npz'))+'</ul></html>')
    (output/'index.html').write_text(page)
    for name, expected_hash in inputs.items():
        if digest(Path(name)) != expected_hash: raise ValueError('Input changed during information audit')
    for name, expected_hash in sources.items():
        if digest(ROOT/name) != expected_hash: raise ValueError('Analysis source changed during information audit')
    result = dict(status='complete', subjects=selected, arms=manifest['arms'], groups=len(groups),
        alignment=str(aligned.resolve()), state_source=str(source), oral_audit=str(audit),
        source_sha256=sources, input_sha256=inputs, original_scores_reproduced=True,
        python=sys.version, versions={name:version(name) for name in ('numpy', 'pandas', 'scipy')},
        data_use='development; frozen behavior-only parameters and states; oral used only for evaluation',
        interpretation=note, artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()}, **scope_metadata)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subjects', type=int, nargs='+')
    args = parser.parse_args()
    run(args.alignment, args.output, args.subjects)
