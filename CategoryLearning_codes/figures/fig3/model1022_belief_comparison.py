"""Plot every participant and all paths in finite-workspace belief comparisons.

Figure contract: compare conditional prediction, autonomous learning and target
rule use after separate bounded fits. Quantitative grid, one row per participant,
32-trial blocks; all simulated paths contribute. Color and line style denote models,
black denotes observed behavior. This is a development report, exported as PNG.
Optional --reference adds the frozen SUPPORT from a completed earlier study.
Its observed trials and split must match, and its simulation seeds must be
independent. Every reference path is retained. Gray dotted curves distinguish
this fixed benchmark from SUPPORT selected in the current study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1022_belief_summary import read, load_case, digest
from src.Bayesian_state.evaluation.model_1022_belief_fixed_references import validate_comparison

COLORS = {'SUPPORT': '#417CA8', 'BELIEF': '#D58A35', 'SLOT': '#8565A5'}
LABELS = {'SUPPORT': 'Support', 'BELIEF': 'Belief: uniform entry', 'SLOT': 'Belief: slot transfer'}
STYLES = {'SUPPORT': '-', 'BELIEF': '--', 'SLOT': '-.'}
REFERENCE = 'SUPPORT_REFERENCE'


def blocks(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] % 32:
        raise ValueError('This pilot preserves complete 32-trial blocks')
    return values.reshape(*values.shape[:-1], -1, 32).mean(axis=-1)


def frozen_inputs(study: Path) -> tuple[list[dict], list[dict]]:
    """Require complete, unchanged scientific nominations before drawing curves."""
    nomination = read(study/'nominees.json')
    if (read(study/'completion.json')['status'] != 'complete' or nomination['smoke']
            or nomination['status'] != 'frozen_before_test'
            or read(study/'summary/completion.json')['nominee_sha256'] != digest(study/'nominees.json')):
        raise ValueError('Figure inputs must be complete studies with intact frozen nominations')
    return nomination['nominees'], read(study/'autonomous_jobs.json')['jobs']


def run(study: Path, output: Path, reference: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    nominees, jobs = frozen_inputs(study)
    subjects = sorted({n['subject'] for n in nominees})
    present_arms = {n['arm'] for n in nominees}
    if not present_arms <= COLORS.keys() or 'SUPPORT' not in present_arms:
        raise ValueError('Require SUPPORT and declared belief comparison arms')
    arms = [arm for arm in COLORS if arm in present_arms]
    if len(nominees) != len(subjects)*len(arms):
        raise ValueError('Every participant must have every declared comparison arm')
    cases, provenance = {}, []
    for nominee in nominees:
        key = nominee['subject'], nominee['arm']
        if key in cases:
            raise ValueError('Duplicate participant/model case')
        cases[key] = load_case(study, nominee, jobs)
        provenance.append(dict(subject=key[0], arm=key[1], source=str(study),
                               candidate=nominee['candidate']['id'], nomination_sha256=digest(study/'nominees.json')))
    colors, labels, styles = dict(COLORS), dict(LABELS), dict(STYLES)
    if reference is not None:
        if reference.resolve() == study.resolve():
            raise ValueError('Frozen reference must come from a distinct study')
        previous, previous_jobs = frozen_inputs(reference)
        for sid in subjects:
            matched = [n for n in previous if n['subject'] == sid and n['arm'] == 'SUPPORT']
            if len(matched) != 1:
                raise ValueError('Frozen SUPPORT reference must cover every plotted participant')
            case = load_case(reference, matched[0], previous_jobs)
            validate_comparison(case, cases[sid, 'SUPPORT'])
            cases[sid, REFERENCE] = case
            provenance.append(dict(subject=sid, arm=REFERENCE, source=str(reference),
                candidate=matched[0]['candidate']['id'], nomination_sha256=digest(reference/'nominees.json')))
        arms.append(REFERENCE)
        colors[REFERENCE], styles[REFERENCE] = '#555555', ':'
        labels.update(SUPPORT='Support: current fit', SUPPORT_REFERENCE='Support: frozen reference')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .7, 'legend.frameon': False})
    extra = .5 if reference is not None else 0.
    fig, axes = plt.subplots(len(subjects), 2, figsize=(10, max(4., 2.5*len(subjects)+1.)+extra),
                             squeeze=False, sharey=True)
    process, state_axes = plt.subplots(len(subjects), 2, figsize=(10, max(4., 2.5*len(subjects)+.5)+extra),
                                       squeeze=False, sharey=True)
    records = []
    for row, sid in enumerate(subjects):
        for arm in arms:
            case = cases[sid, arm]
            meta, obs, pred, auto = [case[k] for k in ('meta', 'observed', 'prediction', 'autonomous')]
            x = np.arange(16, meta['trials'], 32)
            observed = blocks((obs['choices'] == obs['categories']).astype(float))
            conditional = blocks(pred['correct_probability'].mean(axis=0))
            generated = blocks((auto['choices'] == obs['categories'][None, :]).astype(float))
            present, contribution = [blocks(auto[k]).mean(axis=0) for k in ('present', 'contribution')]
            axes[row, 0].plot(x, conditional, color=colors[arm], ls=styles[arm], lw=1.8)
            axes[row, 1].plot(x, generated.T, color=colors[arm], lw=.35, alpha=.05 if arm == REFERENCE else .10)
            axes[row, 1].plot(x, generated.mean(axis=0), color=colors[arm], ls=styles[arm], lw=2)
            for ax, values in zip(state_axes[row], (present, contribution)):
                ax.plot(x, values, color=colors[arm], ls=styles[arm], lw=1.8)
            for i in range(len(x)):
                records.append(dict(subject=sid, arm=arm, block=i+1, trial_center=x[i], observed_accuracy=observed[i],
                    conditional_correct_probability=conditional[i], autonomous_accuracy=generated[:, i].mean(),
                    target_presence=present[i], target_choice_mass=contribution[i], paths=len(generated)))
            if arm == 'SUPPORT':
                for ax in axes[row]:
                    ax.plot(x, observed, color='#222222', lw=1.5, marker='o', ms=2.7, zorder=5)
                axes[row, 0].set_ylabel(f'S{sid} | condition {meta["condition"]}\nCorrect response')
                state_axes[row, 0].set_ylabel(f'S{sid} | condition {meta["condition"]}\nProbability / mass')
                for ax in [*axes[row], *state_axes[row]]:
                    ax.axvline(meta['training_trials'], color='#777777', ls='--', lw=.8)
                    ax.set(xlim=(0, meta['trials']), ylim=(-.02, 1.02))
                    ax.set_yticks([0, .25, .5, .75, 1])
        axes[row, 0].set_title('Conditional on observed history', loc='left', fontsize=10)
        axes[row, 1].set_title('Autonomous from task start', loc='left', fontsize=10)
        state_axes[row, 0].set_title('Target rule in the workspace', loc='left', fontsize=10)
        state_axes[row, 1].set_title('Target rule contribution to choice', loc='left', fontsize=10)
    handles = [Line2D([0], [0], color='#222222', marker='o', ms=3, label='Participant')]
    handles += [Line2D([0], [0], color=colors[a], ls=styles[a], lw=2, label=labels[a]) for a in arms]
    for canvas, grid in ((fig, axes), (process, state_axes)):
        for ax in grid[-1]:
            ax.set_xlabel('Trial (32-trial blocks)')
        canvas.legend(handles=handles if canvas is fig else handles[1:], loc='upper center',
                      ncol=3 if reference is not None else len(handles),
                      bbox_to_anchor=(.5, 1.005))
        height = canvas.get_figheight()
        canvas.subplots_adjust(left=.10, right=.98, top=1-(.85 if reference is not None else .55)/height,
                               bottom=.75/height, hspace=.40, wspace=.20)
    counts = sorted({r['paths'] for r in records})
    count_text = '/'.join(str(n) for n in counts)
    fig.text(.10, .025, f'Development evaluation. Thin lines: all {count_text} autonomous paths per model. '
             'Dashed line: end of parameter-fitting prefix.', fontsize=8)
    process.text(.10, .025, f'Means over all {count_text} autonomous paths, including paths that never discover the target rule. '
                 'Dashed line: training boundary.', fontsize=8)
    fig.savefig(output/'behavior.png', dpi=300)
    process.savefig(output/'rule_process.png', dpi=300)
    plt.close(fig)
    plt.close(process)
    pd.DataFrame(records).to_csv(output/'source_curves.csv', index=False)
    for item in provenance:
        case = cases[item['subject'], item['arm']]
        item.update(filters=len(case['prediction']['seeds']), paths=len(case['autonomous']['seeds']))
    (output/'figure_sources.json').write_text(json.dumps(dict(cases=provenance,
        figure_source_sha256=digest(Path(__file__)), reference_matching='same observed trials and split; independent seeds'),
        indent=2)+'\n')
    reference_note = ('灰色点线表示较早研究中冻结的支持版，蓝色表示本轮选参的支持版。'
                      '参照只取当前图中被试的 SUPPORT，保留这些被试的全部模拟路径。\n\n'
                      if reference is not None else '')
    (output/'README.md').write_text(
        '# 规则信念对照图\n\n'
        'behavior.png：左列为条件于此前真实行为的预测，右列为从任务起点自主学习。'
        '黑色为真实行为，蓝色为支持版，橙色为均匀入场信念版，紫色为信念份额转移版；'
        '实际比较的模型见图例。所有曲线按 32 题聚合；'
        f'每版 {count_text} 条自主路径全部绘出并参与均值。虚线为原训练边界。\n\n'
        + reference_note +
        'rule_process.png：目标规则在场概率及作答贡献，均由全部自主路径计算。'
        '这些内部过程来自自主模拟，未用口述作验证。\n\n'
        '图中路径差异描述模拟变异。该图用于开发评价，人群差异尚待检验。'
        '图稿为 Python 生成的 PNG；source_curves.csv 保存全部汇总曲线。'
        'figure_sources.json 记录各曲线的参数身份、来源、滤波与路径数量。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Optional completed study supplying frozen SUPPORT curves')
    args = parser.parse_args()
    run(args.study, args.output, args.reference)
