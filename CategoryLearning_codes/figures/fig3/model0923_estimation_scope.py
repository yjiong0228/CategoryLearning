"""Plot candidate selection, all autonomous paths and learning distributions.

Use --summary COMPLETE_SCOPE_SUMMARY --output NEW_DIRECTORY. All confirmed
candidates and all original/prefix/full trajectory labels remain visible.
Identical points share data and labels. --smoke permits the explicitly marked
short pipeline fixture. Outputs are PNG discussion figures and an HTML review.
--selection-stage displays provisional full-objective nominees against original
parameters using a completed selection stage, while confirmation continues.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import block_paths, digest, load

SCOPES = ('original', 'prefix', 'full')
LABELS = {'original': 'O: original', 'prefix': 'P: prefix', 'full': 'F: full'}
COLORS = {'original': '#9B9B9B', 'prefix': '#357698', 'full': '#B67842'}
ARMS = ('POOL', 'SEARCH')
ARM_COLORS = {'POOL': '#68778B', 'SEARCH': '#468578'}


def ecdf(ax, values: np.ndarray, upper: float = 1., **kwargs) -> None:
    """Keep every path, including paths with no discovery by the last trial."""
    x = np.sort(np.asarray(values, float))
    # Infinite discovery times are right-censored; the visible CDF stays below 1.
    finite = np.isfinite(x)
    y = np.arange(1, len(x)+1) / len(x)
    end = y[finite][-1] if finite.any() else 0.
    ax.step(np.r_[0., x[finite], upper], np.r_[0., y[finite], end], where='post', **kwargs)


def selection_figure(subject: int, tables: dict, output: Path, smoke: bool,
                     selection_stage: bool = False) -> None:
    stage = 'selection' if selection_stage else 'confirmation'
    scores = tables['scores']; candidates = tables['candidate_scores']; matched = tables['matched_mechanisms']
    own = scores[scores.subject.eq(subject) & scores.phase.eq('whole')]
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 135/25.4), layout='constrained')
    for arm, ax in zip(ARMS, axes[0]):
        points = candidates[candidates.subject.eq(subject) & candidates.model.eq(arm)]
        x = points[points.phase.eq('prefix')].set_index('candidate_id')
        y = points[points.phase.eq('whole')].set_index('candidate_id').loc[x.index]
        ax.errorbar(x.choice_nll, y.choice_nll, xerr=x.choice_nll_mc_se, yerr=y.choice_nll_mc_se,
                    fmt='o', color=ARM_COLORS[arm], ms=3, elinewidth=.6, alpha=.7)
        selected = own[own.model.eq(arm)]
        for identity, group in selected.groupby('candidate_id'):
            label = '/'.join(LABELS[name][0] for name in SCOPES if name in set(group.objective))
            ax.annotate(label, (x.loc[identity, 'choice_nll'], y.loc[identity, 'choice_nll']),
                        xytext=(4, 5), textcoords='offset points', fontsize=6, weight='bold')
        ax.set(title=f'{arm}: every {stage} candidate', xlabel='Prefix choice NLL', ylabel='Full choice NLL')
        ax.margins(.18)
    ax = axes[1, 0]
    values = matched[matched.subject.eq(subject) & matched.phase.eq('whole')].sort_values('candidate_id')
    ax.axvline(0, color='#888888', lw=.7)
    ax.errorbar(values.delta_choice_nll, np.arange(len(values)), xerr=values.delta_choice_nll_mc_se,
                fmt='o', color=ARM_COLORS['SEARCH'], ms=3, elinewidth=.7)
    ax.set(yticks=np.arange(len(values)), yticklabels=[x[:6] for x in values.candidate_id],
           xlabel='Full choice NLL: SEARCH − POOL', ylabel='Identical parameter point',
           title='Mechanism effect at matched parameters')
    ax.tick_params(axis='y', labelsize=5)
    ax = axes[1, 1]
    for (arm, identity), group in own.groupby(['model', 'candidate_id']):
        row = group.iloc[0]
        # Labels for a shared point denote one simulation bank, not replicates.
        label = '/'.join(LABELS[name][0] for name in SCOPES if name in set(group.objective))
        ax.errorbar(row.choice_nll, row.energy, xerr=row.choice_nll_mc_se, yerr=row.energy_mc_se,
                    fmt='o' if arm == 'POOL' else '^', color=ARM_COLORS[arm], ms=4, elinewidth=.7)
        ax.annotate(label, (row.choice_nll, row.energy), xytext=(4, -10 if arm == 'POOL' else 5),
                    textcoords='offset points', fontsize=6, color=ARM_COLORS[arm])
    ax.set(xlabel='Full choice NLL', ylabel='Full autonomous Energy score', title='Choice fit and autonomous learning')
    ax.margins(.22)
    ax.legend(handles=[Line2D([], [], marker=marker, ls='', color=ARM_COLORS[arm], label=arm)
                       for arm, marker in zip(ARMS, ['o', '^'])], fontsize=6)
    for label, ax in zip('abcd', axes.flat):
        title = ax.get_title(); ax.set_title('')
        ax.set_title(label + '  ' + title, loc='left')
    title = f'Subject {subject}: why parameter selection changes'
    prefix = 'PROVISIONAL SELECTION — ' if selection_stage else ('SMOKE FIXTURE — ' if smoke else '')
    fig.suptitle(prefix + title, fontsize=9)
    labels = 'O: original; F: full-selected.' if selection_stage else 'O: original; P: prefix-selected; F: full-selected.'
    fig.supxlabel(labels + ' Bars: 1 simulation SE. Lower scores are better.', fontsize=6)
    fig.savefig(output / f'subject_{subject}_selection.png', dpi=300)
    plt.close(fig)


def trajectory_figures(subject: int, tables: dict, arrays: dict, output: Path, width: int, smoke: bool,
                       selection_stage: bool = False) -> None:
    scopes = ('original', 'full') if selection_stage else SCOPES
    prefix = 'PROVISIONAL SELECTION — ' if selection_stage else ('SMOKE FIXTURE — ' if smoke else '')
    own = tables['curves'][tables['curves'].subject.eq(subject)]
    fig, axes = plt.subplots(2, len(scopes), figsize=(183/25.4, 135/25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.20, hspace=.42, wspace=.15)
    for row, arm in enumerate(ARMS):
        for column, scope in enumerate(scopes):
            ax = axes[row, column]
            curve = own[own.model.eq(arm) & own.objective.eq(scope)].sort_values('first_trial')
            key = subject, arm, curve.candidate_id.iloc[0]
            data = arrays[key]; n = int(curve.last_trial.max())
            paths = block_paths(data['correct_probability'], 0, n, width)
            np.testing.assert_allclose(paths.mean(axis=0), curve.probability_mean, atol=1e-12, rtol=0)
            x = (curve.first_trial.to_numpy()+curve.last_trial.to_numpy())/2
            marker = '.' if len(x) == 1 else None
            ax.fill_between(x, curve.generated_q10, curve.generated_q90, color=COLORS[scope], alpha=.15)
            ax.plot(x, paths.T, color=COLORS[scope], alpha=.12, lw=.4, marker=marker, ms=2)
            ax.plot(x, curve.probability_mean, color=COLORS[scope], lw=1.5, marker=marker, ms=4)
            ax.plot(x, curve.conditional, color='#666666', lw=.8, ls='--', marker=marker, ms=3)
            ax.plot(x, curve.observed, color='#222222', lw=.8, marker='o', ms=2)
            ax.axvline(curve.prefix_trials.iloc[0]+.5, color='#AAAAAA', lw=.7, ls=':')
            ax.set(title=f'{arm} | {LABELS[scope]} | n={len(paths)}', ylim=(-.02, 1.03), xlim=(.5, n+.5), yticks=[0, .5, 1])
        axes[row, 0].set_ylabel('Correct probability / rate')
    for ax in axes[-1]: ax.set_xlabel('Trial')
    handles = [Line2D([], [], color='#555555', lw=1.5, label='Autonomous mean'),
               Line2D([], [], color='#999999', lw=.5, label='Every autonomous probability path'),
               Line2D([], [], color='#666666', lw=.8, ls='--', label='Sequential conditional mean'),
               Line2D([], [], color='#222222', marker='o', ms=2, lw=.8, label='Observed'),
               Patch(facecolor='#999999', alpha=.15, label='80% sampled-choice interval')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .045), ncol=2, fontsize=6)
    fig.suptitle(prefix+f'Subject {subject}: all autonomous learning paths', fontsize=9)
    fig.text(.09, .018, f'{width}-trial display blocks. Dotted line: original time split. Full selection uses both sides.', fontsize=6)
    fig.savefig(output / f'subject_{subject}_trajectories.png', dpi=300); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(183/25.4, 110/25.4), layout='constrained', sharey=True)
    paths = tables['paths'][tables['paths'].subject.eq(subject)]
    for row, arm in enumerate(ARMS):
        for scope in scopes:
            part = paths[paths.model.eq(arm) & paths.objective.eq(scope)]
            for column, field in enumerate(['late_accuracy', 'first_target_trial', 'target_late_contribution']):
                values = part[field].to_numpy()
                if field == 'first_target_trial': values = np.where(np.isnan(values), np.inf, values)
                ecdf(axes[row, column], values, upper=n if column == 1 else 1.,
                     color=COLORS[scope], lw=1.1, ls={'original':'-', 'prefix':'--', 'full':':'}[scope],
                     label=LABELS[scope])
        observed = tables['scores'].query('subject == @subject and model == @arm and phase == "whole"').observed_late_accuracy.unique()
        if len(observed) != 1: raise ValueError('Observed late accuracy differs across selection scopes')
        axes[row, 0].axvline(observed[0], color='#222222', lw=.8, ls='--')
        axes[row, 0].set_ylabel(f'{arm}\nCumulative path fraction')
        for column, title in enumerate(['Late sampled accuracy', 'First correct-rule discovery', 'Late correct-rule contribution']):
            axes[row, column].set(title=title, ylim=(0, 1.02), xlim=(0, n if column == 1 else 1))
            axes[row, column].set_xlabel('Trial' if column == 1 else 'Proportion')
        axes[row, 1].legend(fontsize=6, loc='upper left')
    fig.suptitle(prefix+f'Subject {subject}: learning outcome distributions', fontsize=9)
    fig.supxlabel('Late = last min(128, sequence length) trials. Dashed line: observed accuracy. Undiscovered paths stay censored.', fontsize=6)
    fig.savefig(output / f'subject_{subject}_distributions.png', dpi=300); plt.close(fig)


def plot(summary: Path, output: Path, smoke: bool = False, selection_stage: bool = False) -> None:
    if output.exists(): raise FileExistsError('Use a new estimation-scope figure directory')
    manifest = load(summary / 'manifest.json')
    if manifest['status'] != 'complete' or manifest['smoke'] != smoke:
        raise ValueError('Need a complete summary; fixtures require --smoke')
    stage = 'selection' if selection_stage else 'confirmation'
    if manifest.get('selection_phase', 'confirmation') != stage or (smoke and selection_stage):
        raise ValueError('Selection-stage diagnostics require --selection-stage and a full sequence')
    scopes = ('original', 'full') if selection_stage else SCOPES
    for name, expected in manifest['inputs_sha256'].items():
        if digest(Path(name)) != expected: raise ValueError('Scope summary input changed')
    for name, expected in manifest['output_sha256'].items():
        if digest(summary / name) != expected: raise ValueError('Scope score table changed')
    tables = {name: pd.read_csv(summary / (name + '.csv')) for name in
              ['scores', 'curves', 'paths', 'candidate_scores', 'matched_mechanisms', 'parameters', 'contrasts']}
    generation = load(Path(manifest['generation']) / 'generation.json')
    arrays, array_hashes = {}, {}
    for row in generation['rows']:
        file = Path(row['target']).with_suffix('.npz')
        if digest(file) != row['predictions_sha256']: raise ValueError('Autonomous plotting paths changed')
        with np.load(file) as data:
            arrays[row['subject'], row['model'], row['candidate_id']] = {'correct_probability': data['correct_probability']}
        array_hashes[str(file)] = digest(file)
    subjects = sorted(tables['scores'].subject.unique().tolist())
    expected = {(s, arm, scope, phase) for s in subjects for arm in ARMS for scope in scopes
                for phase in ('prefix', 'later', 'whole')}
    keys = tables['scores'][['subject', 'model', 'objective', 'phase']]
    if keys.duplicated().any() or set(map(tuple, keys.to_numpy())) != expected:
        raise ValueError('Missing or repeated comparison rows for the requested selection phase')
    output.mkdir(parents=True)
    contract = dict(question='Does full-sequence likelihood selection improve autonomous learning and rule use?',
        archetype='quantitative grid', backend='python', formats=['PNG'], dpi=300, width_mm=183,
        subjects=subjects, smoke=smoke, data_use='retrospective development diagnostic',
        selection_phase=stage, diagnostic_status=manifest.get('diagnostic_status', 'confirmed_selection'),
        panels=[f'all {stage} candidates and matched-parameter effects', 'all autonomous paths', 'learning outcome CDFs'],
        sample_size='all saved independent filter seeds and autonomous paths at each fixed parameter point',
        sample_counts=tables['scores'][tables['scores'].phase.eq('whole')][
            ['subject', 'model', 'objective', 'filter_seeds', 'particles', 'paths']].to_dict('records'),
        uncertainty='1 Monte Carlo SE for scores; central 80% sampled-choice intervals for trajectories',
        inference='parameter and population uncertainty excluded; no significance test or prospective test claim',
        reuse='structural adaptation of the existing scope diagnostic; unchanged metrics and all saved paths',
        row_counts={name: len(table) for name, table in tables.items()},
        input_sha256={str(summary / 'manifest.json'): digest(summary / 'manifest.json'),
                      **{str(summary / name): sha for name, sha in manifest['output_sha256'].items()}, **array_hashes},
        source_preflight_exceptions={'EXPORT-VECTOR': 'Repository PNG-only instruction.',
            'EXPORT-RASTER': 'TIFF omitted under PNG-only instruction.', 'RASTER-DPI': '300 dpi discussion figure.'})
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7,
        'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 6, 'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    for subject in subjects:
        selection_figure(subject, tables, output, smoke, selection_stage)
        trajectory_figures(subject, tables, arrays, output, manifest['block_width'], smoke, selection_stage)
    scope_text = ('O表示原参数，F表示完成selection阶段后按完整行为序列目标选定的暂定参数。'
                  '当前展示selection阶段的全部候选；更高精度的confirmation阶段继续计算。' if selection_stage else
                  'O表示原参数，P表示本轮前段目标选定参数，F表示完整行为序列目标选定参数。'
                  '全部确认候选来自已有参数库。')
    candidate_text = 'selection阶段候选' if selection_stage else '确认候选'
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>选参范围与自主学习</title>',
        '<style>body{max-width:1050px;margin:30px auto;font:16px/1.6 sans-serif}img{max-width:100%}td,th{padding:5px}</style>',
        '<h1>选参范围、机制效果与自主学习</h1>',
        '<p>'+scope_text+'相同参数共享模拟数据。本轮筛选使用完整行为，属于回顾性开发诊断。</p>',
        '<p>第一张图给出'+candidate_text+'的前段／全程选择负对数似然、同参数的机制差值，以及选定参数的自主Energy评分。'
        '分数越低越好；误差棒为固定参数下的1个模拟标准误。第二张图保留全部自主概率路径、'
        '其均值、模拟作答正确率中央80%区间、逐题条件预测均值和实测正确率。</p>',
        '<p>第三张图展示末段正确率、首次发现正确规则的时间和末段作答贡献的完整分布。'
        '尚未发现正确规则的路径计入分母，其发现时间在任务末尾右删失。口述表征另由冻结参数后的状态重放评价。</p>']
    if smoke: page.append('<p><strong>短序列流程验证，数值仅用于检查绘图实现。</strong></p>')
    if selection_stage: page.append('<p><strong>暂定候选的自主诊断；最终参数选择等待confirmation阶段完成。</strong></p>')
    for subject in subjects:
        for kind in ('selection', 'trajectories', 'distributions'):
            name = f'subject_{subject}_{kind}.png'; page.append(f'<img src="{name}" alt="{subject} {kind}">')
    page.append('<h2>全程评分</h2>'+tables['scores'][tables['scores'].phase.eq('whole')].to_html(index=False, float_format=lambda x:f'{x:.4f}'))
    (output / 'index.html').write_text(''.join(page))
    (output / 'manifest.json').write_text(json.dumps(dict(status='complete', smoke=smoke, selection_phase=stage,
        source_sha256=digest(Path(__file__)), summary=str(summary.resolve()),
        artifacts={p.name:digest(p) for p in output.iterdir()}), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--selection-stage', action='store_true', help='Plot provisional full-objective selection nominees')
    args = parser.parse_args(); plot(args.summary, args.output, args.smoke, args.selection_stage)
