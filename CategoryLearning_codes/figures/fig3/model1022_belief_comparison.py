"""Plot every participant and all paths in the finite-workspace belief pilot.

Figure contract: compare conditional prediction, autonomous learning and target
rule use after separate bounded fits. Quantitative grid, three participants,
32-trial blocks; all simulated paths contribute. Blue/orange denote models,
black denotes observed behavior. This is a development report, exported as PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1022_belief_summary import read, load_case

COLORS = {'SUPPORT': '#417CA8', 'BELIEF': '#D58A35'}
LABELS = {'SUPPORT': 'Support', 'BELIEF': 'Belief'}


def blocks(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] % 32:
        raise ValueError('This pilot preserves complete 32-trial blocks')
    return values.reshape(*values.shape[:-1], -1, 32).mean(axis=-1)


def run(study: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    nominees = read(study/'nominees.json')['nominees']
    jobs = read(study/'autonomous_jobs.json')['jobs']
    subjects = sorted({n['subject'] for n in nominees})
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .7, 'legend.frameon': False})
    fig, axes = plt.subplots(len(subjects), 2, figsize=(10, 8.5), squeeze=False, sharey=True)
    process, state_axes = plt.subplots(len(subjects), 2, figsize=(10, 8), squeeze=False, sharey=True)
    records = []
    for row, sid in enumerate(subjects):
        for arm in COLORS:
            nominee, = [n for n in nominees if (n['subject'], n['arm']) == (sid, arm)]
            case = load_case(study, nominee, jobs)
            meta, obs, pred, auto = [case[k] for k in ('meta', 'observed', 'prediction', 'autonomous')]
            x = np.arange(16, meta['trials'], 32)
            observed = blocks((obs['choices'] == obs['categories']).astype(float))
            conditional = blocks(pred['correct_probability'].mean(axis=0))
            generated = blocks((auto['choices'] == obs['categories'][None, :]).astype(float))
            present, contribution = [blocks(auto[k]).mean(axis=0) for k in ('present', 'contribution')]
            axes[row, 0].plot(x, conditional, color=COLORS[arm], lw=1.8)
            axes[row, 1].plot(x, generated.T, color=COLORS[arm], lw=.35, alpha=.10)
            axes[row, 1].plot(x, generated.mean(axis=0), color=COLORS[arm], lw=2)
            for ax, values in zip(state_axes[row], (present, contribution)):
                ax.plot(x, values, color=COLORS[arm], lw=1.8)
            for i in range(len(x)):
                records.append(dict(subject=sid, arm=arm, block=i+1, trial_center=x[i], observed_accuracy=observed[i],
                    conditional_correct_probability=conditional[i], autonomous_accuracy=generated[:, i].mean(),
                    target_presence=present[i], target_choice_mass=contribution[i], paths=len(generated)))
            if arm == 'SUPPORT':
                for ax in axes[row]:
                    ax.plot(x, observed, color='#222222', lw=1.5, marker='o', ms=2.7)
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
    handles += [Line2D([0], [0], color=COLORS[a], lw=2, label=LABELS[a]) for a in COLORS]
    for canvas, grid in ((fig, axes), (process, state_axes)):
        for ax in grid[-1]:
            ax.set_xlabel('Trial (32-trial blocks)')
        canvas.legend(handles=handles if canvas is fig else handles[1:], loc='upper center', ncol=3,
                      bbox_to_anchor=(.5, 1.005))
        canvas.subplots_adjust(left=.10, right=.98, top=.94, bottom=.09, hspace=.40, wspace=.20)
    fig.text(.10, .025, 'Three development participants. Thin lines: all 48 autonomous paths per model. '
             'Dashed line: end of parameter-fitting prefix.', fontsize=8)
    process.text(.10, .025, 'Means over all 48 autonomous paths, including paths that never discover the target rule. '
                 'Dashed line: training boundary.', fontsize=8)
    fig.savefig(output/'behavior.png', dpi=300)
    process.savefig(output/'rule_process.png', dpi=300)
    plt.close(fig)
    plt.close(process)
    pd.DataFrame(records).to_csv(output/'source_curves.csv', index=False)
    (output/'README.md').write_text(
        '# 规则信念小规模对照图\n\n'
        'behavior.png：左列为条件于此前真实行为的预测，右列为从任务起点自主学习。'
        '黑色为真实行为，蓝色为支持版，橙色为信念版。所有曲线按 32 题聚合；'
        '每版 48 条自主路径全部绘出并参与均值。虚线为原训练边界。\n\n'
        'rule_process.png：目标规则在场概率及作答贡献，均由全部自主路径计算。'
        '这些内部过程来自自主模拟，未用口述作验证。\n\n'
        '三人均为开发被试；图中路径差异描述模拟变异，不是人群置信区间。'
        '图稿为 Python 生成的 PNG；source_curves.csv 保存全部汇总曲线。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)
