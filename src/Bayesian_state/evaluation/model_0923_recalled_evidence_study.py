"""Summarize every fixed-parameter recalled-evidence path and paired contrast.

Use --source STUDY --output NEW_DIRECTORY. Curve scores reuse the established
32-trial empirical Energy definition; whole-path jackknife errors measure finite
simulation. No conditional choice likelihood or refitted-model rank is inferred
from these autonomous paths. Never-discovered rules remain in all denominators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import block_paths, distribution_scores, jackknife_se
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis.model_0923_recalled_evidence_study import digest, read, write


def summarize(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recalled-evidence summary directory')
    protocol = read(source / 'protocol.json'); completion = read(source / 'completion.json')
    if completion['status'] != 'complete':
        raise ValueError('Generation is incomplete')
    groups, inputs = {}, {}
    for file in sorted((source / 'jobs').glob('*.json')):
        job = read(file); target = Path(job['target']); record = read(target.with_suffix('.json'))
        validation = read(target.with_suffix('.validation.json'))
        if job['model'] == 'POOL' and validation['exact_control_replay'] is not True:
            raise ValueError('Control paths did not exactly replay the reference')
        if digest(target.with_suffix('.npz')) != record['predictions_sha256']:
            raise ValueError('Saved path changed')
        with np.load(target.with_suffix('.npz')) as saved:
            arrays = {key: saved[key] for key in saved.files}
        np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
        groups.setdefault((record['subject'], job['model']), []).append((job, record, arrays))
        inputs[str(target.with_suffix('.npz').resolve())] = record['predictions_sha256']
    settings = protocol['settings']; subjects = protocol['subjects']
    if set(groups) != {(s, m) for s in subjects for m in settings['arms']}:
        raise ValueError('Missing declared mechanism cell')
    score_rows, path_rows, curve_rows, banks = [], [], [], {}
    for (subject, arm), chunks in sorted(groups.items()):
        job, record, first = chunks[0]
        categories = first['categories']
        for _, _, part in chunks[1:]:
            np.testing.assert_array_equal(part['categories'], categories)
        arrays = {key: np.concatenate([part[key] for _, _, part in chunks]) for key in first if key != 'categories'}
        order = np.argsort(arrays['seeds']); arrays = {k: v[order] for k, v in arrays.items()}
        count = 2 if protocol['smoke'] else settings['paths']
        if len(arrays['seeds']) != count or len(set(arrays['seeds'])) != count:
            raise ValueError('Incomplete or duplicate path seeds')
        folder = Path(job['folder']); meta = read(folder / 'prepared.json')
        human_parts = []
        for name in ('training.npz', 'sealed_test.npz'):
            if digest(folder / name) != meta['array_sha256'][name]:
                raise ValueError('Prepared behavior changed')
            inputs[str((folder / name).resolve())] = meta['array_sha256'][name]
            with np.load(folder / name) as original:
                human_parts.append({k: original[k] for k in ('choices', 'categories')})
        original = {key: np.concatenate([part[key] for part in human_parts]) for key in human_parts[0]}
        n = len(categories); cut = min(meta['training_trials'], n)
        np.testing.assert_array_equal(categories, original['categories'][:n])
        observed = original['choices'][:n] == categories
        generated = arrays['choices'] == categories[None, :]
        late = max(0, n - 128)
        per_path = dict(late_accuracy=generated[:, late:].mean(axis=1),
                        late_correct_probability=arrays['correct_probability'][:, late:].mean(axis=1),
                        target_ever=arrays['present'].any(axis=1).astype(float),
                        target_proposed=(arrays['proposed'] == record['target_rule']).any(axis=1).astype(float),
                        target_late_present=arrays['present'][:, late:].mean(axis=1),
                        target_late_contribution=arrays['contribution'][:, late:].mean(axis=1),
                        search_events=arrays['search'].sum(axis=1),
                        target_proposal_count=(arrays['proposed'] == record['target_rule']).sum(axis=1))
        for i, seed in enumerate(arrays['seeds']):
            found = np.flatnonzero(arrays['present'][i])
            path_rows.append(dict(subject=subject, arm=arm, seed=int(seed),
                first_target_trial=int(found[0] + 1) if len(found) else np.nan,
                **{name: float(value[i]) for name, value in per_path.items()}))
        for phase, start, stop in [('whole', 0, n), ('training', 0, cut), ('test', cut, n)]:
            if start == stop:
                continue
            samples = block_paths(generated, start, stop, settings['block_width'])
            target = block_paths(observed[None, :], start, stop, settings['block_width'])[0]
            metrics = distribution_scores(samples, target)
            jackknife = path_jackknife(samples, target)
            banks[subject, arm, phase] = dict(seeds=arrays['seeds'], metrics=metrics,
                                              jackknife=jackknife, path_metrics=per_path)
            score_rows.append(dict(subject=subject, arm=arm, phase=phase, trials=stop-start, paths=count,
                observed_late=float(observed[late:].mean()), **metrics,
                energy_mcse=jackknife_se(jackknife['energy']), crps_mcse=jackknife_se(jackknife['crps']),
                **{name: float(value.mean()) for name, value in per_path.items()}))
            if phase == 'whole':
                lower, upper = np.quantile(samples, [.1, .9], axis=0)
                for b in range(samples.shape[1]):
                    curve_rows.append(dict(subject=subject, arm=arm, trial=min((b+1)*settings['block_width'], n),
                        observed=float(target[b]), mean=float(samples[:, b].mean()),
                        lower=float(lower[b]), upper=float(upper[b]), paths=count))
    contrasts = []
    for (subject, arm, phase), values in banks.items():
        control = banks[subject, 'POOL', phase]
        np.testing.assert_array_equal(values['seeds'], control['seeds'])
        row = dict(subject=subject, arm=arm, phase=phase, reference='POOL', paths=len(values['seeds']))
        for metric in ('energy', 'crps'):
            row[f'delta_{metric}'] = values['metrics'][metric] - control['metrics'][metric]
            row[f'delta_{metric}_mcse'] = jackknife_se(values['jackknife'][metric] - control['jackknife'][metric])
        for name, value in values['path_metrics'].items():
            difference = value - control['path_metrics'][name]
            row[f'delta_{name}'] = float(difference.mean())
            row[f'delta_{name}_mcse'] = float(difference.std(ddof=1)/np.sqrt(len(difference)))
        contrasts.append(row)
    output.mkdir(parents=True)
    tables = dict(scores=pd.DataFrame(score_rows), paths=pd.DataFrame(path_rows),
                  curves=pd.DataFrame(curve_rows), contrasts=pd.DataFrame(contrasts))
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    shown = tables['scores'].loc[tables['scores'].phase.eq('whole'),
        ['subject', 'arm', 'paths', 'energy', 'energy_mcse', 'late_accuracy', 'observed_late',
         'target_ever', 'target_late_present', 'target_late_contribution']]
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>历史证据机制对照</title>'
            '<style>body{font:16px sans-serif;max-width:1250px;margin:30px auto}td,th{padding:5px}</style>'
            '<h1>历史证据参与搜索与信念形成</h1><p>四方案共用原POOL参数、刺激日程和路径种子。'
            'POOL为原流程，PRIOR在新规则入集时重建信念，SEARCH引导候选身份，BOTH同时开启。'
            '口述和正确规则身份不参与学习；正确规则身份用于路径诊断。</p>'
            '<p>Energy评价32题分组的完整自主正确率曲线分布。模拟误差以完整路径为单位计算，'
            '被试抽样和参数不确定性另需检验。路径使用固定真实刺激日程；参数尚未按新机制重拟合。</p>'
            + ('<p><strong>16题软件短验证，结果不用于科学机制排序。</strong></p>' if protocol['smoke'] else '')
            + shown.to_html(index=False, float_format=lambda x: f'{x:.4f}') + '</html>')
    (output / 'index.html').write_text(page)
    paths = [Path(__file__), Path(__file__).with_name('model_0923_h_validation.py'),
             Path(__file__).with_name('model_0923_h_generation_uncertainty.py')]
    write(output / 'manifest.json', dict(status='complete', smoke=protocol['smoke'], source=str(source.resolve()),
        protocol_sha256=digest(source / 'protocol.json'), completion_sha256=digest(source / 'completion.json'),
        inputs=inputs, source_sha256={str(p.resolve()): digest(p) for p in paths},
        artifacts={p.name: digest(p) for p in output.iterdir()},
        interpretation='fixed-parameter process intervention; paired whole-path simulation error; development participants'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.output)


if __name__ == '__main__':
    main()
