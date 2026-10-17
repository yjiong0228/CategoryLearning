"""Plot all autonomous paths once that part of a frozen-fit evaluation finishes.

Run --fit FIT --evaluation EVALUATION --output NEW_DIRECTORY. Every declared
autonomous chunk must be complete; conditional filtering may still be running.
The six-panel PNG retains every equally weighted path, the observed curve and
the fitting boundary. Scores reuse the existing Energy and path-jackknife
definitions. Parameters were estimated from the training prefix only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.Bayesian_state.evaluation import model_0923_h_validation as metrics
from src.Bayesian_state.evaluation import model_0923_h_generation_uncertainty as uncertainty

ARMS = ('POOL', 'SEARCH')
COLORS = {'POOL': '#68778B', 'SEARCH': '#277F8E'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def collect(fit: Path, evaluation: Path) -> tuple[dict, dict, dict]:
    """Require every generated path and its exact frozen parameter identity."""
    done = read(fit / 'completion.json'); frozen = read(fit / 'nominees.json')
    protocol = read(evaluation / 'protocol.json')
    if (done['status'] != 'complete' or done['smoke'] or done['test_choices_used']
            or frozen['smoke'] or frozen['status'] != 'frozen_before_test'
            or done['nomination_sha256'] != digest(fit / 'nominees.json')
            or protocol['nomination_sha256'] != done['nomination_sha256']
            or protocol['source_sha256'] != frozen['source_sha256']
            or Path(protocol['fit']).resolve() != fit.resolve()):
        raise ValueError('Require a completed scientific fit and its frozen evaluation')
    settings = frozen['settings']; subjects = settings['subjects']
    if tuple(settings['arms']) != ARMS or len(subjects) != 3 or len(set(subjects)) != 3:
        raise ValueError('This figure requires three participants and POOL/SEARCH')
    width, count = settings['features']['block_width'], settings['autonomous_paths']
    if count < 2 or width < 1:
        raise ValueError('Need multiple paths and a positive block width')
    nominees = {(r['subject'], r['model']): r['candidate'] for r in frozen['nominees']}
    if len(nominees) != 6 or set(nominees) != {(s, arm) for s in subjects for arm in ARMS}:
        raise ValueError('Incomplete frozen nominees')
    inputs = {str(p.resolve()): digest(p) for p in
              (fit / 'completion.json', fit / 'nominees.json', evaluation / 'protocol.json')}
    groups = defaultdict(list)
    jobs = [job for job in protocol['jobs'] if job['phase'] == 'autonomous']
    if len({job['target'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate autonomous task target')
    for job in jobs:
        target = Path(job['target']); record = read(target.with_suffix('.json'))
        key = record['subject'], job['model']; candidate = nominees[key]
        if (record['model'] != job['model'] or record['candidate']['id'] != candidate['id']
                or job['source_sha256'] != protocol['source_sha256']
                or job['arm_settings'] != settings['arms'][job['model']]
                or job['mechanism'] != settings['mechanism']
                or record['candidate']['hyperparams'] != candidate['hyperparams']
                or job['candidate']['id'] != candidate['id']
                or job['candidate']['hyperparams'] != candidate['hyperparams']
                or digest(target.with_suffix('.npz')) != record['predictions_sha256']):
            raise ValueError('Generated path differs from its frozen nominee or saved hash')
        with np.load(target.with_suffix('.npz')) as saved:
            arrays = {name: saved[name].copy() for name in
                      ('seeds', 'categories', 'choices', 'probabilities', 'present', 'contribution')}
        np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
        if record['paths'] != len(job['seeds']) or any(p['num_threads'] != 1 for p in record['numeric_pools']):
            raise ValueError('Unexpected path count or nested numerical threads')
        groups[key].append((job, record, arrays))
        for path in (target.with_suffix('.json'), target.with_suffix('.npz')):
            inputs[str(path.resolve())] = digest(path)
    if set(groups) != set(nominees):
        raise ValueError('Incomplete autonomous comparison')

    scores, curves, blocks, paths, bank = [], [], [], [], {}
    for (subject, arm), chunks in sorted(groups.items()):
        job, record, first = chunks[0]; folder = Path(job['folder'])
        meta = read(folder / 'prepared.json'); inputs[str((folder / 'prepared.json').resolve())] = digest(folder / 'prepared.json')
        if meta['subject'] != subject:
            raise ValueError('Participant folder differs from the generated paths')
        human = []
        for name in ('training.npz', 'sealed_test.npz'):
            file = folder / name
            if digest(file) != meta['array_sha256'][name]:
                raise ValueError('Original trial data changed')
            inputs[str(file.resolve())] = digest(file)
            with np.load(file) as saved:
                human.append({k: saved[k].copy() for k in ('choices', 'categories')})
        original = {k: np.concatenate([part[k] for part in human]) for k in human[0]}
        categories = original['categories']; n = len(categories); cut = meta['training_trials']
        if not 0 < cut < n:
            raise ValueError('Need distinct training and subsequent-trial segments')
        for own_job, own_record, part in chunks:
            if Path(own_job['folder']).resolve() != folder.resolve() or own_record['trials'] != n:
                raise ValueError('Task sequence differs between chunks')
            np.testing.assert_array_equal(part['categories'], categories)
        arrays = {k: np.concatenate([part[k] for _, _, part in chunks]) for k in first if k != 'categories'}
        order = np.argsort(arrays['seeds']); arrays = {k: value[order] for k, value in arrays.items()}
        if len(arrays['seeds']) != count or len(set(arrays['seeds'])) != count:
            raise ValueError('Missing or duplicated autonomous path seeds')
        probability = arrays['probabilities']
        if (arrays['choices'].shape != (count, n) or probability.ndim != 3
                or probability.shape[:2] != (count, n) or probability.shape[-1] < 2
                or not np.isfinite(probability).all() or np.any(probability < 0)):
            raise ValueError('Invalid generated choices or probabilities')
        np.testing.assert_allclose(probability.sum(axis=-1), 1., atol=1e-12, rtol=0)
        labels = np.arange(1, probability.shape[-1] + 1)
        if not np.isin(arrays['choices'], labels).all() or not np.isin(original['choices'], labels).all():
            raise ValueError('Choice labels differ from the category probability axis')
        for name in ('present', 'contribution'):
            value = arrays[name]
            if value.shape != (count, n) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1 + 1e-12)):
                raise ValueError('Invalid target-rule process trace')
        if not np.isin(arrays['present'], [0, 1]).all():
            raise ValueError('Autonomous presence must describe each realized path')
        np.testing.assert_allclose(arrays['contribution'][~arrays['present'].astype(bool)], 0., atol=1e-12, rtol=0)
        observed = original['choices'] == categories
        generated = arrays['choices'] == categories[None, :]
        samples = metrics.block_paths(generated, 0, n, width)
        target = metrics.block_paths(observed[None, :], 0, n, width)[0]
        score = metrics.distribution_scores(samples, target)
        jackknife = uncertainty.path_jackknife(samples, target)
        late = max(0, n - 128); late_accuracy = generated[:, late:].mean(axis=1)
        tag = dict(subject=subject, model=arm)
        scores.append(dict(tag, phase='whole', candidate=nominees[subject, arm]['id'],
            training_trials=cut, trials=n, paths=count, **score,
            energy_mc_se=metrics.jackknife_se(jackknife['energy']),
            crps_mc_se=metrics.jackknife_se(jackknife['crps']),
            late_accuracy=float(late_accuracy.mean()), observed_late=float(observed[late:].mean()),
            late_reaches_observed=float((late_accuracy >= observed[late:].mean()).mean()),
            target_ever=float(arrays['present'].any(axis=1).mean()),
            target_late_present=float(arrays['present'][:, late:].mean()),
            target_late_contribution=float(arrays['contribution'][:, late:].mean())))
        bank[subject, arm] = dict(seeds=arrays['seeds'], scores=score, jackknife=jackknife)
        for index in range(samples.shape[1]):
            trial = min((index + 1) * width, n)
            curves.append(dict(tag, trial=trial, training_trials=cut, observed=float(target[index]),
                               autonomous=float(samples[:, index].mean())))
            blocks.extend(dict(tag, seed=int(seed), trial=trial, accuracy=float(samples[i, index]))
                          for i, seed in enumerate(arrays['seeds']))
        for i, seed in enumerate(arrays['seeds']):
            found = np.flatnonzero(arrays['present'][i])
            paths.append(dict(tag, seed=int(seed), late_accuracy=float(late_accuracy[i]),
                target_ever=bool(len(found)), first_target=int(found[0] + 1) if len(found) else np.nan,
                target_late_present=float(arrays['present'][i, late:].mean()),
                target_late_contribution=float(arrays['contribution'][i, late:].mean())))
    contrasts = []
    for subject in subjects:
        base, alternative = [bank[subject, arm] for arm in ARMS]
        np.testing.assert_array_equal(base['seeds'], alternative['seeds'])
        row = dict(subject=subject, model='SEARCH', reference='POOL', phase='whole')
        for metric in ('energy', 'crps'):
            row['delta_' + metric] = alternative['scores'][metric] - base['scores'][metric]
            row['delta_' + metric + '_mc_se'] = metrics.jackknife_se(
                alternative['jackknife'][metric] - base['jackknife'][metric])
        contrasts.append(row)
    tables = {name: pd.DataFrame(rows) for name, rows in
              dict(scores=scores, curves=curves, path_blocks=blocks, paths=paths, contrasts=contrasts).items()}
    return tables, inputs, dict(subjects=subjects, arms=list(ARMS), paths_per_cell=count,
                               block_width=width, autonomous_tasks=len(jobs))


def plot(fit: Path, evaluation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    with threadpool_limits(limits=1):
        tables, inputs, details = collect(fit, evaluation)
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': .6})
    fig, axes = plt.subplots(3, 2, figsize=(183 / 25.4, 180 / 25.4), squeeze=False)
    scores = tables['scores'].set_index(['subject', 'model'])
    for row, subject in enumerate(details['subjects']):
        for column, arm in enumerate(ARMS):
            ax = axes[row, column]
            frame = tables['curves'].query('subject == @subject and model == @arm').sort_values('trial')
            full = tables['path_blocks'].query('subject == @subject and model == @arm')
            matrix = full.pivot(index='seed', columns='trial', values='accuracy')
            ax.plot(matrix.columns, matrix.to_numpy().T, color='#777777', alpha=.10, linewidth=.35, zorder=1)
            ax.axvline(frame.training_trials.iloc[0], color='#9B6D46', linestyle='--', linewidth=.8, zorder=2)
            ax.plot(frame.trial, frame.autonomous, color=COLORS[arm], linewidth=1.7, zorder=3)
            ax.plot(frame.trial, frame.observed, color='#171717', linewidth=1., marker='o',
                    markersize=2, markerfacecolor='white', markeredgewidth=.55, zorder=4)
            score = scores.loc[(subject, arm), :]
            ax.text(.025, .035, f"Energy {score.energy:.3f}; target found {round(score.target_ever * score.paths)}/{int(score.paths)}",
                    transform=ax.transAxes, fontsize=6, bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1.5))
            ax.set(title=f'Participant {subject} | {arm}', xlim=(0, frame.trial.max()), ylim=(0, 1.02), yticks=[0, .5, 1])
            ax.text(-.095, 1.035, chr(ord('a') + row * 2 + column), transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='left', va='bottom', clip_on=False)
            ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, width=.6)
            if column == 0:
                ax.set_ylabel(f"Accuracy / {details['block_width']} trials")
            if row == 2:
                ax.set_xlabel('Trial')
    fig.suptitle('Autonomous predictions from frozen prefix fits', fontsize=9, y=.986)
    fig.text(.5, .953, f"All {details['paths_per_cell']} equally weighted paths per cell; observed stimulus schedule", ha='center', fontsize=7)
    handles = [Line2D([], [], color='#171717', marker='o', markersize=3, markerfacecolor='white', label='Observed'),
               Line2D([], [], color=COLORS['SEARCH'], linewidth=1.7, label='Generated mean'),
               Line2D([], [], color='#777777', linewidth=.5, label='Individual paths'),
               Line2D([], [], color='#9B6D46', linestyle='--', label='Fit boundary')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .006))
    fig.subplots_adjust(left=.08, right=.985, bottom=.095, top=.905, hspace=.38, wspace=.20)
    fig.savefig(output / 'autonomous_paths.png', dpi=300); plt.close(fig)
    caption = ('每格保留全部自主路径，黑色曲线为被试，彩色曲线为模型均值。竖虚线标出参数估计的训练段末尾。'
               '参数根据虚线前的真实选择估计，每条自主路径从第一题连续学习。'
               '每条路径在原刺激顺序上根据自身回答获得反馈；路径等权，口述未参与参数估计或作图。'
               f"曲线采用互不重叠的{details['block_width']}题分组。路径分散程度描述固定参数下的生成分布；参数不确定性另需检验。"
               '这些被试均用于机制开发。此页只要求自主任务完整，条件预测及口述检验另行汇总。')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>冻结参数的自主学习路径</title>'
            '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}img{width:100%}td,th{padding:5px}</style>'
            '<h1>冻结参数的自主学习路径</h1><p>' + caption + '</p><img src="autonomous_paths.png" alt="全部自主路径与被试曲线">'
            + tables['scores'][['subject', 'model', 'energy', 'late_accuracy', 'observed_late', 'target_ever']].to_html(index=False, float_format=lambda x: f'{x:.4f}')
            + '<p><a href="path_blocks.csv">全部路径绘图数据</a> · <a href="scores.csv">过程评分</a> · <a href="contrasts.csv">配对差值与模拟误差</a></p></html>')
    (output / 'index.html').write_text(page)
    contract = dict(backend='python', formats=['PNG'], dpi=300, width_mm=183, **details,
        archetype='faceted temporal trajectories',
        template_compatibility='existing Fig3 line-plot style extended with every path and the fitting boundary',
        claim='Show the full autonomous path distributions and their mismatch with observed learning after training-prefix estimation.',
        sample_role='three development participants; behavior-only prefix estimation; original task schedule',
        participant_n=3, simulation_replicates=f"{details['paths_per_cell']} computational paths per participant/model; separate from participant sampling",
        center='arithmetic mean of path accuracies in each nonoverlapping trial block',
        comparison='SEARCH minus POOL with common path seeds; whole-path jackknife Monte Carlo SE in contrasts.csv',
        hypothesis_tests='none', multiple_comparison_correction='not applicable',
        panels='three participant rows by two mechanism columns; every path retained with equal weight',
        uncertainty='individual generated paths and paired whole-path jackknife MCSE; parameter and participant uncertainty separate',
        limitations=['conditional evaluation may still be running', 'no oral input', 'future prediction after prefix fitting'],
        export_exceptions=['PNG only by standing repository instruction; discussion figures use 300 dpi'])
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    source_files = [Path(__file__), Path(metrics.__file__), Path(uncertainty.__file__)]
    archive = output / 'source'; archive.mkdir()
    for file in source_files:
        shutil.copy2(file, archive / file.name)
    manifest = dict(status='complete_autonomous_part', fit=str(fit.resolve()), evaluation=str(evaluation.resolve()),
        inputs_sha256=inputs, source_sha256={str(file.resolve()): digest(file) for file in source_files},
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'threadpoolctl')},
        artifacts={str(file.relative_to(output)): digest(file) for file in output.rglob('*') if file.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); plot(args.fit, args.evaluation, args.output)
