"""Adaptive-task stopping distributions for all frozen H trajectories.

Figure contract: adaptive stimulation does not account for the short observed
learning times under the frozen fits. Quantitative grid; every subject, model,
policy and path retained. Each curve is the fraction of all 64 paths that first
passed by a block, with administrative censoring explicit. Fixed schedules end
at the observed length. Python PNG, 183 mm / 300 dpi, development review.
Style-only inheritance: existing H palette and typography; new stopping metric.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .model0923_h_validation import COLORS
from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load

POLICY_NAMES = {'fixed': 'Observed stimulus schedule', 'balanced': 'Resampled balanced categories',
                'adaptive': 'Response-adaptive categories', 'adaptive_repeat': 'Adaptive; repeated session innovations'}
LINE_STYLES = ['-', '--', ':', '-.', (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]


def stopping_curve(group: pd.DataFrame, horizon: int) -> np.ndarray:
    """Unconditional empirical first-pass CDF; failed paths stay in denominator."""
    if group.seed.duplicated().any() or group.empty:
        raise ValueError('Need distinct nonempty whole-path seeds')
    return np.array([(group.passed & (group.blocks <= block)).mean() for block in range(horizon + 1)])


def build(source: Path, output: Path) -> None:
    if output.exists() or (source / 'review.html').exists():
        raise FileExistsError('Use new adaptive-task figure destinations')
    protocol, completion = load(source / 'protocol.json'), load(source / 'completion.json')
    if completion['status'] != 'complete' or completion['smoke']:
        raise ValueError('Need completed scientific generation')
    paths = pd.read_csv(source / 'paths.csv'); summary = pd.read_csv(source / 'summary.csv')
    if len(paths) != completion['paths'] or paths.duplicated(['subject', 'model', 'policy', 'seed']).any():
        raise ValueError('Path coverage differs from completed generation')
    models = paths.model.drop_duplicates().tolist(); subjects = sorted(paths.subject.unique())
    if len(models) > len(LINE_STYLES):
        raise ValueError('Need a distinct line style for every displayed model')
    policies = protocol['settings']['policies']; n = protocol['settings']['paths']
    expected = {(s, m, p) for s in subjects for m in models for p in policies}
    if set(paths.groupby(['subject','model','policy']).groups) != expected or not (paths.groupby(['subject','model','policy']).size() == n).all():
        raise ValueError('Unbalanced participant/model/policy coverage')
    # Validate the saved stopping outcomes against every complete path array.
    for row in paths.itertuples():
        file = Path(row.path)
        if digest(file) != row.path_sha256:
            raise ValueError('Path changed since generation')
        with np.load(file) as saved:
            passed = saved['block_passed']
            if len(passed) != row.blocks or bool(passed[-1]) != row.passed or passed[:-1].any():
                raise ValueError('Saved trajectory did not stop at its first passing block')
            if not np.isfinite(saved['probabilities']).all():
                raise ValueError('Nonfinite saved probability')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7, 'axes.titlesize':8, 'axes.spines.right':False, 'axes.spines.top':False,
        'axes.linewidth':.6, 'legend.frameon':False, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    curves = []
    for subject in subjects:
        observed = int(paths.loc[paths.subject == subject, 'observed_blocks'].iloc[0])
        panel_rows = (len(policies)+1)//2
        fig, axes = plt.subplots(panel_rows, 2, figsize=(183/25.4, (65*panel_rows+15)/25.4),
            layout='constrained', sharex=True, sharey=True, squeeze=False)
        display_horizon = max(observed, protocol['settings']['max_blocks'])
        for index, (ax, policy) in enumerate(zip(axes.flat, policies)):
            horizon = observed if policy == 'fixed' else protocol['settings']['max_blocks']
            x = np.arange(horizon + 1)
            for model, style in zip(models, LINE_STYLES):
                group = paths[(paths.subject == subject) & (paths.model == model) & (paths.policy == policy)]
                y = stopping_curve(group, horizon)
                ax.step(x, y, where='post', color=COLORS[model], linestyle=style, lw=1.3, label=model)
                curves.extend(dict(subject=int(subject), model=model, policy=policy, block=int(b), first_pass_fraction=float(v), paths=n)
                              for b, v in zip(x, y))
            ax.axvline(observed, color='#222222', lw=.8, ls=':')
            ax.set(title=POLICY_NAMES[policy], ylim=(0,1.03), xlim=(0,display_horizon), yticks=[0,.25,.5,.75,1])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.text(-.06, 1.07, chr(97+index), transform=ax.transAxes, fontweight='bold', fontsize=8)
            if policy == 'fixed':
                ax.text(.98,.05,f'Administrative end: block {observed}',transform=ax.transAxes,ha='right',fontsize=6)
        for ax in axes.flat[len(policies):]:ax.set_visible(False)
        for ax in axes[:,0]:
            ax.set_ylabel('Fraction reaching criterion')
        for ax in axes[-1,:]:
            ax.set_xlabel('Completed 64-trial blocks')
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=6,fontsize=7)
        fig.suptitle(f'Participant {subject}: first pass; all {n} paths per model and task policy\nDotted line: observed completion at block {observed}',fontsize=9)
        fig.savefig(output / f'subject_{subject}.png',dpi=300); plt.close(fig)
    pd.DataFrame(curves).to_csv(output / 'stopping_curves.csv',index=False)
    # Paired differences use common cognitive and task seed labels, whole paths.
    contrasts = []
    for subject in subjects:
        for policy in policies:
            baseline = paths[(paths.subject==subject)&(paths.model=='B')&(paths.policy==policy)].sort_values('seed')
            for model in models:
                other = paths[(paths.subject==subject)&(paths.model==model)&(paths.policy==policy)].sort_values('seed')
                np.testing.assert_array_equal(baseline.seed, other.seed)
                for metric in ('passed_by_observed_horizon','passed'):
                    difference = other[metric].to_numpy(dtype=float)-baseline[metric].to_numpy(dtype=float)
                    contrasts.append(dict(subject=int(subject),model=model,policy=policy,metric=metric,
                        difference=float(difference.mean()),paired_mc_se=float(difference.std(ddof=1)/np.sqrt(n))))
    pd.DataFrame(contrasts).to_csv(output / 'paired_contrasts.csv',index=False)
    aggregate = summary.groupby(['model','policy'],sort=False)[['pass_by_observed_fraction','pass_fraction','mean_blocks_to_stop_or_censor']].mean()
    aggregate.to_csv(output / 'participant_equal_summary.csv')
    adaptive = aggregate.xs('adaptive',level='policy')
    timing_summary = (f'按被试等权平均，各模型在实际完成时长内的达标比例为'
        f'{adaptive.pass_by_observed_fraction.min():.3f}至{adaptive.pass_by_observed_fraction.max():.3f}；'
        f'延长至{protocol["settings"]["max_blocks"]}组时为{adaptive.pass_fraction.min():.3f}至{adaptive.pass_fraction.max():.3f}。逐人曲线保留模型之间的差异。')
    task_checks = len(load(source/'task_policy_audit.json')['checks'])
    reserved = protocol.get('data_use_status') == 'reserved_validation'
    cohort_label = '预先保留的验证被试' if reserved else '开发被试'
    boundary_text = ('机制、参数搜索预算及本项评价在打开这些被试的行为资料前确定。'
                     if reserved else '来源数据已在此前检验中查看，本报告属于开发诊断。')
    relative = lambda p: escape(os.path.relpath(p, source))
    html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>H模块：自适应实验与达标时间</title>',
        '<style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#222}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:4px 8px;border-bottom:1px solid #ddd}.scroll{overflow:auto}</style>',
        '<h1>H模块：自适应实验与达标时间</h1>',
        f'<p>{timing_summary}</p>',
        f'<p>{len(subjects)}名{cohort_label}、{len(models)}种模型、{len(policies)}种任务流程，每组{n}条路径，共{len(paths)}条。参数保持为原训练前缀拟合值。任务每64题检查一次：排除模糊刺激后，每个类别正确率都达到0.9即停止。条件3按物种完全正确计算。</p>',
        f'<p>曲线分母始终包含全部{n}条路径。固定刺激版本的截尾点为被试实际长度；重新采样版本最多{protocol["settings"]["max_blocks"]}组。虚线标出被试完成时间；曲线之外的未达标路径属于截尾。0/{n}表示本次有限模拟未观察到达标。首次达标即停止记录学习路径。</p>',
        '<p>按实验规则更新类别比例时，使用模型自己的选择。均衡版本保持每种物种16题；两种自适应版本分别使用会话间独立与重复随机创新。首组刺激及顺序固定为被试的实际首组，模型状态跨会话保持。NumPy采样复现任务分布，MATLAB显示时长等步骤的随机数没有重放。</p>',
        f'<p>类别配额与原实验{task_checks}次可核对的转移全部一致；{completion["exact_fixed_replay_paths"]:,}条固定刺激路径在首次达标前与已保存路径逐值一致。{boundary_text}</p>',
        '<h2>逐人等权平均</h2><p>pass_by_observed_fraction表示在各人实际完成时长内达标的比例；pass_fraction表示在该流程截尾前达标的比例。mean_blocks_to_stop_or_censor同时包含停止与截尾时间，其均值不能解释为成功路径的平均学习时长。</p>',
        '<div class="scroll">'+aggregate.to_html(float_format=lambda x:f'{x:.3f}')+'</div>']
    for subject in subjects:
        html.append(f'<h2>被试{subject}</h2><img src="{relative(output/f"subject_{subject}.png")}" alt="被试{subject}所有模型和任务流程的达标曲线">')
    html += ['<p>配对Monte Carlo标准误仅衡量固定参数下有限路径数的计算波动；参数估计和人群不确定性需另行评估。</p>',
        '<ul>'+''.join(f'<li><a href="{relative(p)}">{escape(p.name)}</a></li>' for p in [source/'paths.csv',source/'summary.csv',
            source/'protocol.json',source/'task_policy_audit.json',output/'stopping_curves.csv',output/'paired_contrasts.csv'])+'</ul></html>']
    with (source/'review.html').open('x') as handle:
        handle.write('\n'.join(html))
    with (output/'qa.json').open('x') as handle:
        json.dump(dict(backend='python',format='PNG',dpi=300,width_mm=183,archetype='quantitative_grid',
            purpose='stopping_distribution_review',data_use_status=protocol.get('data_use_status','development'),source_sha256=digest(Path(__file__)),
            inputs_sha256={str(p):digest(p) for p in [source/'paths.csv',source/'summary.csv',source/'protocol.json']},
            all_paths=len(paths),exclusions=[],data_unit='whole simulated path within participant/model/policy',
            uncertainty='paired whole-path Monte Carlo SE; no population test or p-values',
            adaptation='style-only palette/typography; stopping CDF constructed anew',visual_qa='pending',
            export_exceptions='PNG-only repository instruction; 300 dpi development review'),handle,indent=2)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();build(args.source,args.output)


if __name__=='__main__':main()
