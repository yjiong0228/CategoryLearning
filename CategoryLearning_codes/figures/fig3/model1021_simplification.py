"""Draw the frozen simplification comparison, retaining every autonomous path.

Figure contract: show whether fewer mechanisms preserve behavior and oral
compatibility, with separate evidence for conditional prediction, autonomous
learning distributions and noisy representation support. Quantitative grids
use Python/matplotlib, 183-mm PNG exports and the complete source tables.
Colors and typography follow the existing report style; score calculations
come from the shared analysis. All 128 paths appear as background trajectories.
Error bars show two Monte Carlo standard errors at fixed fitted parameters.
They do not include between-person or parameter-estimation uncertainty.

Run --source FOLLOWTHROUGH --output NEW_DIRECTORY after all four stages finish.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import (
    block_paths, checked_arrays, digest, jackknife_scores, jackknife_se, load, nll,
)

COLORS = dict(REFERENCE='#527697', COMPACT='#3B8C85')
NAMES = dict(REFERENCE='Reference', COMPACT='Compact')
MASS = dict(support='#538B84', conflict='#C58B60', undetermined='#D9DDE2')


def selection_candidates(fit: Path, frozen: dict, bind) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all selection points to the subsequently confirmed nominee.

    Every difference uses the same complete filter seeds, including candidates
    omitted from confirmation when the warm-start anchor occupies its second slot.
    These training scores describe finite-budget selection, not validation.
    """
    stage = load(bind(fit / 'selection_completion.json'))
    if stage['status'] != 'complete' or stage['stage'] != 'selection':
        raise ValueError('Require complete selection scores')
    settings = frozen['settings']
    budget = settings['budgets']['selection']
    nominees = {(r['subject'], r['model']): r for r in frozen['nominees']}
    groups = {(g['subject'], g['model']): g['rows'] for g in stage['groups']}
    if len(groups) != len(stage['groups']) or set(groups) != set(nominees):
        raise ValueError('Selection and frozen subject-by-model cells differ')
    scores, parameters = [], []
    for (subject, model), points in groups.items():
        nominee = nominees[subject, model]
        chosen = nominee['candidate']['id']
        prepared = load(bind(Path(frozen['data']) / f'subject_{subject}' / 'prepared.json'))
        trials = prepared['training_trials']
        if len(points) != settings['selection_candidates'] or len({p['id'] for p in points}) != len(points):
            raise ValueError('Missing or duplicated selection candidates')
        estimates, seeds = {}, None
        for point in points:
            file = bind(Path(point['score_file']))
            record = load(file)
            if (record['subject'], record['model'], record['candidate']['id']) != (subject, model, point['id']):
                raise ValueError('Selection candidate identity changed')
            if record['particles'] != budget['particles'] or record['candidate']['hyperparams'] != point['hyperparams']:
                raise ValueError('Selection budget or parameters changed')
            with np.load(bind(file.with_suffix('.npz'), record['predictions_sha256'])) as z:
                probability, own_seeds = z['observed_probability'], z['seeds']
            if record['trials'] != trials or probability.shape != (budget['seeds'], trials) or len(np.unique(own_seeds)) != budget['seeds']:
                raise ValueError('Selection filter count or trial boundary differs')
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError('Invalid selection probability')
            np.testing.assert_array_equal(own_seeds, record['seeds'])
            if seeds is not None:
                np.testing.assert_array_equal(own_seeds, seeds)
            seeds = own_seeds
            value = nll(probability)
            np.testing.assert_allclose(value, [point['mean_nll'], record['mean_nll']], atol=1e-12, rtol=0)
            estimates[point['id']] = value, jackknife_scores(probability)
            parameters.extend(dict(subject=subject, model=model, candidate=point['id'], parameter=k, value=v)
                              for k, v in point['hyperparams'].items())
        if chosen not in estimates:
            raise ValueError('Confirmed nominee is absent from selection bank')
        if next(p for p in points if p['id'] == chosen)['hyperparams'] != nominee['candidate']['hyperparams']:
            raise ValueError('Confirmed nominee parameters changed')
        reference, deleted = estimates[chosen]
        for rank, point in enumerate(sorted(points, key=lambda p: (estimates[p['id']][0], p['id'])), 1):
            value, own_deleted = estimates[point['id']]
            scores.append(dict(subject=subject, model=model, candidate=point['id'], selection_rank=rank,
                selected=point['id'] == chosen, warm_start=point['id'] == nominee['reference_id'],
                nll=value, delta_nll=value-reference, paired_mc_se=jackknife_se(own_deleted-deleted),
                particles=budget['particles'], filter_seeds=budget['seeds']))
    return pd.DataFrame(scores), pd.DataFrame(parameters)


def load_data(source: Path) -> dict:
    if load(source / 'completion.json')['status'] != 'complete':
        raise ValueError('Require the complete behavior and two-reading oral comparison')
    summary = source / 'behavior_summary'
    done = load(summary / 'completion.json')
    fit, evaluation = Path(done['fit']), Path(done['evaluation'])
    frozen, protocol = load(fit / 'nominees.json'), load(fit / 'protocol.json')
    subjects = frozen['settings']['subjects']
    tables, bindings = {}, {}

    def bind(path, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Figure input changed: ' + str(path))
        bindings[str(path.resolve())] = actual
        return path

    for file in (source / 'completion.json', summary / 'completion.json', fit / 'nominees.json', fit / 'protocol.json'):
        bind(file)
    for name, sha in done['artifacts_sha256'].items():
        tables[Path(name).stem] = pd.read_csv(bind(summary / name, sha))
    tables['selection_candidates'], tables['selection_parameters'] = selection_candidates(fit, frozen, bind)
    oral_tables = []
    for reading in ('midpoint', 'relative'):
        folder = source / ('oral_' + reading)
        meta = load(bind(folder / 'completion.json'))
        if meta['status'] != 'complete' or meta['subjects'] != subjects:
            raise ValueError('Oral comparison does not match the behavioral cohort')
        for name, sha in meta['artifacts_sha256'].items():
            bind(folder / name, sha)
        oral_tables.append(pd.read_csv(folder / 'summary.csv').assign(reading=reading))
    oral = pd.concat(oral_tables, ignore_index=True)
    complete = oral[(oral.window == 'test') & (oral.subset == 'complete')]
    if complete.duplicated(['subject', 'arm', 'domain', 'reading']).any():
        raise ValueError('Duplicate oral score')
    values = complete[list(MASS)].to_numpy()
    available = complete.reports.to_numpy() > 0
    if not np.isfinite(values[available]).all() or not np.isnan(values[~available]).all():
        raise ValueError('Invalid oral masses or undefined report scores')
    np.testing.assert_allclose(values[available].sum(1), 1., atol=1e-12, rtol=0)
    for _, rows in complete.groupby(['subject', 'domain', 'reading']):
        if set(rows.arm) != set(COLORS) or rows.reports.nunique() != 1:
            raise ValueError('Oral denominators differ across models')
    paths = {}
    rows = load(bind(evaluation / 'autonomous/evaluation.json'))['rows']
    for row in rows:
        key = row['subject'], row['model']
        arrays = checked_arrays(evaluation / 'autonomous', row)
        file = evaluation / 'autonomous/evaluate' / f'subject_{key[0]}' / key[1] / (row['candidate']['id'] + '.npz')
        bind(file, row['predictions_sha256'])
        blocks = block_paths(arrays['choices'] == arrays['categories'][None, :], 0, len(arrays['categories']),
                            frozen['settings']['features']['block_width'])
        own = tables['curves'][(tables['curves'].subject == key[0]) & (tables['curves'].model == key[1])].sort_values('trial')
        if len(blocks) != frozen['settings']['autonomous_paths'] or blocks.shape[1] != len(own):
            raise ValueError('Missing autonomous paths or trials')
        np.testing.assert_allclose(blocks.mean(0), own.autonomous, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.quantile(blocks, [.1, .9], axis=0), own[['lower', 'upper']].to_numpy().T, atol=1e-12, rtol=0)
        paths[key] = blocks, own, arrays['seeds']
    if set(paths) != {(s, m) for s in subjects for m in COLORS}:
        raise ValueError('Incomplete subject-by-model matrix')
    counts = {(r['subject'], r['model']): r['variable_coordinates'] for r in protocol['parameter_spaces']}
    return dict(subjects=subjects, tables=tables, oral=oral, paths=paths, counts=counts, bindings=bindings)


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output / name, dpi=300, facecolor='white')
    plt.close(fig)


def trajectories(data: dict, output: Path, *, include_conditional: bool = True) -> None:
    """Draw autonomous paths, optionally before conditional evaluation completes."""
    subjects = data['subjects']
    fig, axes = plt.subplots(len(subjects), 2, figsize=(183/25.4, 172/25.4), sharey=True)
    fig.subplots_adjust(left=.08, right=.985, bottom=.105, top=.86, hspace=.38, wspace=.16)
    for row, subject in enumerate(subjects):
        for col, model in enumerate(COLORS):
            ax = axes[row, col]
            samples, curve, _ = data['paths'][subject, model]
            cut = int(curve.training_trials.iloc[0])
            ax.axvspan(cut, curve.trial.max(), color='#F3F4F6', zorder=0)
            ax.plot(curve.trial, samples.T, color=COLORS[model], alpha=.075, lw=.35, zorder=1)
            ax.plot(curve.trial, curve.autonomous, color=COLORS[model], lw=1.6, zorder=3)
            if include_conditional:
                ax.plot(curve.trial, curve.conditional, color=COLORS[model], lw=1., ls='--', zorder=4)
            ax.plot(curve.trial, curve.observed, color='#222222', marker='o', ms=2.1, lw=1.05, zorder=5)
            ax.axvline(cut, color='#7B8087', ls=':', lw=.7)
            ax.set(xlim=(0, curve.trial.max()), ylim=(0, 1.02), yticks=[0, .5, 1.])
            ax.set_title(f'S{subject} · {NAMES[model]} · {data["counts"][subject, model]} parameters', fontsize=8)
            if col == 0:
                ax.set_ylabel('Accuracy')
            if row == len(subjects)-1:
                ax.set_xlabel('Trial (32-trial blocks)')
    handles = [Line2D([], [], color='#222222', marker='o', ms=2, lw=1, label='Participant'),
        Line2D([], [], color='#527697', lw=1.6, label='Autonomous mean'),
        Line2D([], [], color='#527697', lw=.5, alpha=.4, label='All 128 autonomous paths')]
    if include_conditional:
        handles.insert(2, Line2D([], [], color='#527697', lw=1, ls='--', label='Conditional prediction'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .96), ncol=2, fontsize=7)
    fig.text(.5, .025, 'Shading: later trials excluded from parameter fitting. Three development participants.', ha='center', fontsize=7)
    save(fig, output, 'autonomous_and_conditional.png')


def behavior(data: dict, output: Path) -> None:
    contrast = data['tables']['contrasts']
    contrast = contrast[contrast.phase == 'test'].set_index('subject').loc[data['subjects']]
    mean = data['tables']['aggregate'].query('phase == "test"').iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 85/25.4), sharey=True)
    fig.subplots_adjust(left=.16, right=.97, bottom=.29, top=.81, wspace=.25)
    labels = [f'S{s}' for s in data['subjects']] + ['Equal-person\nmean']
    y = np.arange(len(labels))
    for ax, metric, title, tolerance in zip(axes, ['choice_nll', 'energy'], ['Choice NLL / trial', 'Autonomous Energy score'], [.02, .03]):
        values = np.r_[contrast['delta_' + metric], mean['delta_' + metric]]
        error = 2*np.r_[contrast['delta_' + metric + '_mc_se'], mean['delta_' + metric + '_mc_se']]
        ax.axvline(0, color='#80868E', lw=.8)
        ax.axvline(tolerance, color='#B08D65', lw=.8, ls=':')
        ax.errorbar(values, y, xerr=error, fmt='o', color=COLORS['COMPACT'], ms=4, capsize=2, lw=1.)
        ax.set_yticks(y, labels)
        ax.set_ylim(len(labels)-.5, -.5)
        ax.set_title(f'{title}\nWorking tolerance: +{tolerance:.2f}', fontsize=8)
        ax.set_xlabel('Compact − reference\nLower favors compact')
    fig.text(.5, .06, 'Bars: ±2 paired Monte Carlo SE at fitted parameters; uncertainty across people is not shown.', ha='center', fontsize=7)
    save(fig, output, 'later_behavior_comparison.png')


def oral(data: dict, output: Path) -> None:
    subjects = data['subjects']
    selected = data['oral'].query('window == "test" and subset == "complete"')
    fig, axes = plt.subplots(len(subjects), 2, figsize=(183/25.4, 164/25.4), sharex=True)
    fig.subplots_adjust(left=.22, right=.98, top=.91, bottom=.1, hspace=.45, wspace=.75)
    order = [(reading, model) for reading in ('midpoint', 'relative') for model in COLORS]
    for row, subject in enumerate(subjects):
        for col, domain in enumerate(('functional', 'structural')):
            ax = axes[row, col]
            table = selected[(selected.subject == subject) & (selected.domain == domain)].set_index(['reading', 'arm'])
            if set(table.index) != set(order):
                raise ValueError('Missing fixed-reading oral row')
            table = table.loc[order]
            offset = np.zeros(len(order))
            for label, color in MASS.items():
                values = table[label].to_numpy()
                ax.barh(np.arange(len(order)), np.nan_to_num(values), left=offset, color=color, height=.65)
                offset += np.nan_to_num(values)
            labels = [f'{NAMES[model]} · {reading}\n{int(table.loc[(reading, model), "reports"])} / '
                      f'{int(table.loc[(reading, model), "window_trials"])} reports' for reading, model in order]
            ax.set_yticks(range(len(order)), labels, fontsize=6.4)
            ax.invert_yaxis()
            ax.set(xlim=(0, 1), xticks=[0, .5, 1.])
            ax.set_title(f'S{subject} · {domain.capitalize()}', fontsize=8)
            if row == len(subjects)-1:
                ax.set_xlabel('Content compatibility mass')
            for i, count in enumerate(table.reports):
                if count == 0:
                    ax.text(.5, i, 'No complete reports', va='center', ha='center', fontsize=7)
    fig.legend(handles=[Patch(facecolor=c, label=k.capitalize()) for k, c in MASS.items()],
               loc='upper center', ncol=3, bbox_to_anchor=(.6, .995), fontsize=8)
    fig.text(.5, .025, 'Complete reports; identical denominators within each reading. Oral reports are noisy external evidence.', ha='center', fontsize=7)
    save(fig, output, 'oral_compatibility.png')


def candidates(data: dict, output: Path) -> None:
    subjects = data['subjects']
    fig, axes = plt.subplots(len(subjects), 2, figsize=(183/25.4, 161/25.4))
    fig.subplots_adjust(left=.21, right=.975, top=.94, bottom=.15, hspace=.55, wspace=.78)
    for row, subject in enumerate(subjects):
        for col, model in enumerate(COLORS):
            ax = axes[row, col]
            points = data['tables']['selection_candidates']
            points = points[(points.subject == subject) & (points.model == model)].sort_values('selection_rank')
            labels = []
            for i, point in enumerate(points.itertuples()):
                role = 'Chosen' if point.selected else f'Candidate {point.selection_rank}'
                labels.append(role + (' · start' if point.warm_start else ''))
                ax.errorbar(point.delta_nll, i, xerr=2*point.paired_mc_se, fmt='o',
                            color=COLORS[model], markerfacecolor=COLORS[model] if point.selected else 'white',
                            ms=4, capsize=2, lw=.9)
            ax.axvline(0, color='#80868E', lw=.7)
            ax.set_yticks(range(len(points)), labels, fontsize=7)
            ax.set_ylim(len(points)-.5, -.5)
            ax.set_title(f'S{subject} · {NAMES[model]}', fontsize=8)
            ax.tick_params(axis='x', labelsize=7)
            if row == len(subjects)-1:
                ax.set_xlabel('NLL / trial − chosen candidate', fontsize=7)
    fig.text(.5, .065, 'All four selection candidates; bars: ±2 paired MCSE. Each panel has its own x scale.', ha='center', fontsize=7)
    fig.text(.5, .025, 'Chosen: final confirmation nominee. Start: warm-start anchor. Training scores; fixed parameters.', ha='center', fontsize=7)
    save(fig, output, 'parameter_candidates.png')


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure output directory')
    data = load_data(source)
    output.mkdir(parents=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .65, 'legend.frameon': False}):
        trajectories(data, output)
        behavior(data, output)
        oral(data, output)
        candidates(data, output)
    inputs = output / 'source_data'
    inputs.mkdir()
    for name, frame in data['tables'].items():
        frame.to_csv(inputs / (name + '.csv'), index=False)
    data['oral'].to_csv(inputs / 'oral_scores.csv', index=False)
    path_rows = []
    for (subject, model), (samples, curve, seeds) in data['paths'].items():
        path_rows.extend(dict(subject=subject, model=model, seed=int(seed), trial=int(trial), accuracy=float(value))
                         for seed, path in zip(seeds, samples) for trial, value in zip(curve.trial, path))
    pd.DataFrame(path_rows).to_csv(inputs / 'autonomous_path_blocks.csv', index=False)
    images = ['autonomous_and_conditional.png', 'later_behavior_comparison.png', 'oral_compatibility.png',
              'parameter_candidates.png']
    page = ('<!doctype html><meta charset="utf-8"><title>H机制简化比较</title>'
            '<style>body{font:16px sans-serif;max-width:1100px;margin:32px auto}img{width:100%}</style>'
            '<h1>H机制简化比较</h1><p>三名开发被试；参数使用原训练段行为估计。细线包含全部自主路径。'
            '误差条表示固定参数下的蒙特卡洛误差。口述保留噪声、歧义和未确定质量。</p>'
            '<p>候选参数图保留选择阶段的全部四个点，用同一组四次滤波比较其训练分数。'
            '基准为随后独立确认选中的参数；零线左侧表示该候选在选择阶段得分更好。'
            '确认阶段比较两个点，其中一个位置固定保留既有起点。图中的误差条反映数值误差，'
            '参数的不确定性还需结合候选取值与恢复检验。</p>'
            + ''.join(f'<p><a href="{html.escape(n)}"><img src="{html.escape(n)}"></a></p>' for n in images))
    (output / 'index.html').write_text(page)
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete', source=str(source.resolve()),
        subjects=data['subjects'], input_sha256=data['bindings'], source_sha256=digest(Path(__file__)),
        exclusions=[], all_autonomous_paths_retained=True, export='PNG at 300 dpi, 183 mm width',
        uncertainty='Two MCSE at fixed fitted parameters; no population equivalence inference',
        artifacts_sha256={n: digest(output / n) for n in images}), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
