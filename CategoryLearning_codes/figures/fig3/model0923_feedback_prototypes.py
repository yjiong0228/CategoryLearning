"""Frozen-point prototype comparisons, complete curves and path distributions.

Python/matplotlib discussion figures at 183 mm, PNG only, 300 dpi. Every declared
participant, parameter point, arm and path is retained in the output tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_0923_h_validation import read, write, digest


ORDER = ['BASE', 'P', 'A', 'PA', 'FAST', 'WIDE', 'LOG', 'PA_LOG']
COLORS = dict(zip(ORDER, ['#64748B', '#428A99', '#91A886', '#2F6959', '#C58B47', '#BEA68D', '#9274AA', '#9B4C74']))
STYLES = dict(zip(ORDER, ['-', '-', '--', ':', '-', '--', '-', ':']))


def run(source: Path, output: Path):
    if output.exists(): raise FileExistsError('Use a new figure output')
    manifest = read(source / 'manifest.json')
    for name, sha in manifest['summary_sha256'].items():
        if digest(source / name) != sha: raise ValueError('Source summary changed')
    summary, curves, paths = [pd.read_csv(source / (n+'.csv')) for n in ('summary', 'curves', 'paths')]
    output.mkdir(parents=True)
    for name in manifest['summary_sha256']: shutil.copy2(source / name, output / name)
    write(output / 'figure_contract.json', dict(
        conclusion='Higher late accuracy can accompany worse learning curves and choice likelihood; feedback-guided H gains require numerical confirmation.',
        evidence=['all-point distribution score changes', 'isolated precision intervention',
                  'training choice likelihood', 'complete learning curves', 'all late-accuracy path distributions'],
        archetype='quantitative grid', backend='python', width_mm=183, dpi=300, formats=['PNG'],
        inclusion='all declared participants, frozen parameter points, arms, trajectories and filter seeds',
        data_use='development data; oracle precision uses task-correct rule after training boundary',
        statistics='path means and empirical distributions; oracle bars +/- 2 Monte Carlo SE; no population p-values',
        limits='fixed-parameter screening; original fit parameters are not reoptimized for these mechanisms'))
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':7, 'axes.titlesize':7, 'axes.labelsize':7, 'xtick.labelsize':6,
        'ytick.labelsize':6, 'legend.fontsize':6, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.6, 'svg.fonttype':'none', 'pdf.fonttype':42})
    files = []
    def save(fig, name):
        fig.savefig(output / name, dpi=300, facecolor='white'); plt.close(fig); files.append(name)

    natural = summary[summary.stage == 'autonomous']
    arms = [a for a in ORDER if a != 'BASE' and a in natural.arm.unique()]
    points = natural[['subject','model']].drop_duplicates().sort_values(['subject','model'])
    indexed = natural.set_index(['subject', 'model', 'arm'])
    matrix = np.array([[indexed.loc[(s, m, a), 'delta_energy']
                        for a in arms] for s,m in points.itertuples(index=False, name=None)])
    fig, ax = plt.subplots(figsize=(183/25.4, 185/25.4))
    bound = max(abs(matrix).max(), .01)
    im = ax.imshow(matrix, aspect='auto', cmap='BrBG_r', vmin=-bound, vmax=bound)
    ax.set(xticks=range(len(arms)), xticklabels=arms, yticks=range(len(points)),
           yticklabels=[f'S{s} / {m}' for s,m in points.itertuples(index=False, name=None)])
    for (i,j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:+.3f}', ha='center', va='center', fontsize=6,
                color='white' if abs(value) > .65*bound else '#20242A')
    ax.set_title('Autonomous curve distribution: change in Energy score\nLower values favor the prototype; 32 paths per cell', loc='left')
    fig.colorbar(im, ax=ax, shrink=.5, pad=.03, label='Energy score change from BASE')
    fig.tight_layout(); save(fig, 'autonomous_energy.png')

    oracle = summary[summary.stage == 'oracle']
    if len(oracle):
        subjects = sorted(oracle.subject.unique()); models = ['FREE', 'CUMULATIVE', 'SHARED']
        fig, axes = plt.subplots(3, len(subjects), figsize=(183/25.4, 145/25.4), squeeze=False)
        selected = ['BASE', 'FAST', 'WIDE', 'LOG']
        for r,m in enumerate(models):
            for c,s in enumerate(subjects):
                ax = axes[r,c]; group = oracle.query('subject == @s and model == @m').set_index('arm').loc[selected]
                ax.errorbar(np.arange(4), group.late_accuracy, yerr=2*group.late_mc_se,
                            fmt='none', color='#64748B', capsize=2)
                ax.scatter(np.arange(4), group.late_accuracy, c=[COLORS[a] for a in selected], s=18, zorder=3)
                ax.axhline(group.observed_late.iloc[0], color='#20242A', ls='--', lw=.8)
                ax.set(xticks=range(4), xticklabels=selected, ylim=(.4, 1.02), title=f'S{s} / {m}')
                if c == 0: ax.set_ylabel('Late accuracy')
        fig.suptitle('Correct rule only: regenerated feedback and precision learning\n32 paths; mean ± 2 Monte Carlo SE; dashed = participant', fontsize=8)
        fig.tight_layout(rect=(0,0,1,.93)); save(fig, 'oracle_precision.png')

    filtered = summary[summary.stage == 'score']
    if len(filtered):
        subjects = sorted(filtered.subject.unique())
        indexed = filtered.set_index(['subject', 'arm'])
        matrix = np.array([[indexed.loc[(s, a), 'delta_nll'] for a in arms] for s in subjects])
        fig, ax = plt.subplots(figsize=(183/25.4, 95/25.4))
        bound = max(abs(matrix).max(), .01)
        im = ax.imshow(matrix, aspect='auto', cmap='BrBG_r', vmin=-bound, vmax=bound)
        ax.set(xticks=range(len(arms)), xticklabels=arms, yticks=range(len(subjects)), yticklabels=[f'S{s}' for s in subjects])
        for (i,j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{value:+.3f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(value) > .65*bound else '#20242A')
        ax.set_title('Training choice NLL: frozen FREE parameters\nLower values favor the prototype; preliminary particle-filter screen', loc='left')
        fig.colorbar(im, ax=ax, shrink=.7, pad=.03, label='NLL change per trial')
        fig.tight_layout(); save(fig, 'training_nll.png')

    confirmed = summary[summary.stage == 'score_confirm']
    if len(confirmed):
        fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 115/25.4))
        for ax, subject in zip(axes.flat, sorted(confirmed.subject.unique())):
            selected = ['P', 'A', 'PA']
            fine = confirmed[confirmed.subject == subject].set_index('arm').loc[selected]
            coarse = filtered[filtered.subject == subject].set_index('arm').loc[selected]
            ax.axvline(0, color='#64748B', lw=.8)
            ax.scatter(coarse.delta_nll, np.arange(3)+.12, marker='x', s=20, color='#A89A8B', label='64 particles × 2 seeds')
            ax.errorbar(fine.delta_nll, np.arange(3)-.12, xerr=2*fine.delta_nll_mc_se,
                        fmt='o', ms=4, color='#428A99', capsize=2, label='256 particles × 8 seeds')
            ax.set(yticks=range(3), yticklabels=selected, xlabel='NLL change from BASE (lower is better)', title=f'S{subject}')
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5,.90))
        fig.suptitle('Independent numerical confirmation of H changes\nCircles: pooled NLL difference ± 2 paired filter-seed jackknife SE', fontsize=8)
        fig.tight_layout(rect=(0,0,1,.86)); save(fig, 'nll_confirmation.png')

    for model in ['FREE', 'CUMULATIVE', 'SHARED']:
        subjects = sorted(natural.subject.unique())
        fig, axes = plt.subplots(3, 3, figsize=(183/25.4, 155/25.4))
        distribution, axs = plt.subplots(3, 3, figsize=(183/25.4, 145/25.4))
        handles = []
        for ax, ecdf, subject in zip(axes.flat, axs.flat, subjects):
            group = curves.query('stage == "autonomous" and subject == @subject and model == @model')
            for arm in ORDER:
                block = group[group.arm == arm].sort_values('trial')
                line, = ax.plot(block.trial, block['mean'], color=COLORS[arm], ls=STYLES[arm], lw=.9, label=arm)
                if subject == subjects[0]: handles.append(line)
                if arm == 'BASE': ax.fill_between(block.trial, block.lower, block.upper, color=COLORS[arm], alpha=.12, linewidth=0)
                samples = np.sort(paths.query('stage == "autonomous" and subject == @subject and model == @model and arm == @arm').late_accuracy)
                ecdf.step(samples, np.arange(1, len(samples)+1)/len(samples), where='post', color=COLORS[arm], ls=STYLES[arm], lw=.9)
            observed = group[group.arm == 'BASE'].sort_values('trial')
            ax.plot(observed.trial, observed.observed, color='#17191C', lw=1.1, label='Participant')
            late = natural.query('subject == @subject and model == @model and arm == "BASE"').observed_late.iloc[0]
            ecdf.axvline(late, color='#17191C', lw=.8, ls='--')
            ax.set(title=f'S{subject}', ylim=(0,1.05), xlabel='Trial', ylabel='Accuracy')
            ecdf.set(title=f'S{subject}', xlim=(0,1.02), ylim=(0,1.03), xlabel='Late accuracy', ylabel='Path fraction ≤ x')
        fig.legend(handles, ORDER, loc='upper center', ncol=8, bbox_to_anchor=(.5,.96))
        fig.suptitle(f'{model}: full autonomous curves; black = participant; BASE band = 10–90% of paths', fontsize=8)
        fig.tight_layout(rect=(0,0,1,.91)); save(fig, f'curves_{model}.png')
        distribution.legend(handles, ORDER, loc='upper center', ncol=8, bbox_to_anchor=(.5,.96))
        distribution.suptitle(f'{model}: all 32 late-accuracy paths per arm; dashed vertical = participant', fontsize=8)
        distribution.tight_layout(rect=(0,0,1,.91)); save(distribution, f'paths_{model}.png')
    content = '<!doctype html><meta charset="utf-8"><title>反馈与精度原型比较</title><style>body{font-family:sans-serif;max-width:1100px;margin:32px auto;line-height:1.65}img{max-width:100%}</style>'
    content += '<h1>反馈与精度原型：固定参数比较</h1><p>九人均为开发资料。P：上一题反馈引导候选；A：候选以反馈证据参与准入；PA：二者结合。FAST：β步长乘10、上限25；WIDE：原步长、上限100；LOG：在log β上更新、上限100；PA_LOG：组合。默认模型保持原设置。</p>'
    counts = natural.groupby('arm').delta_energy.apply(lambda values: int((values < 0).sum()))
    content += f'<p><strong>后期更准确与整段学习更相似，需要分别检验。</strong>在{len(points)}组固定参数中，FAST有{counts["FAST"]}组、LOG有{counts["LOG"]}组的全程曲线分布评分改善；PA为{counts["PA"]}组。所有方向和模拟误差均保留在源表中。这些计数描述本批参数点。</p>'
    content += '<p>自主曲线Energy score和训练选择NLL越低越好。正确率越高是否更合适，需结合真实曲线判断。全部路径进入评分和经验累积分布；颜色相同代表同一方案。固定正确规则实验从原训练边界开始干预，具有额外任务信息。初筛为64粒子、两个种子；四人的H方案另用256粒子、八个独立种子复核。误差反映有限模拟，参数重新估计与外部验证仍待完成。</p>'
    content += '<p><a href="summary.csv">完整分数</a> · <a href="curves.csv">区块曲线</a> · <a href="paths.csv">全部路径</a> · <a href="filters.csv">各滤波种子</a></p>'
    for name in files: content += f'<h2>{name}</h2><img src="{name}">'
    (output / 'index.html').write_text(content)
    write(output / 'qa_numeric.json', dict(summary_rows=len(summary), path_rows=len(paths), curve_rows=len(curves),
        source_manifest_sha256=digest(source / 'manifest.json'), image_sha256={name:digest(output/name) for name in files},
        export_exception='User requests PNG only; report drafts at 300 dpi. Vector/TIFF export intentionally omitted.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.output)
