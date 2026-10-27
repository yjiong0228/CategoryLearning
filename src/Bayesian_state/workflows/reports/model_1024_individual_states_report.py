"""Render individual-state reconstruction, numerical checks and oral review.

Figure contract: show which internal explanations vary across supported
parameter candidates and numerical repetitions, then compare their reportable
content with provisional oral coding. Quantitative grids use Python and PNG
only. All trials and candidates remain in source arrays; state curves are
unsmoothed and the behavioral context uses a trailing 32-trial mean.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/model1024_states_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.model_1024_individual_states import read, write, digest
from ..analysis.model_1024_individual_state_summary import TARGETS
from ..analysis.model_1024_individual_oral_alignment import SCOPE_LABELS

COLORS = ['#287ca0', '#bd6b35', '#6e63a8', '#43866e']
STATE_COLORS = ['#287ca0', '#6e63a8', '#bd6b35']
STATE_KEYS = ['target_present', 'target_belief', 'target_execution']
STATE_NAMES = ['正确规则在场概率', '正确规则的平均信念', '正确规则执行概率']


def configure():
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.unicode_minus': False})


def save(fig, output: Path, name: str):
    fig.savefig(output/name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_arrays(file: Path) -> dict:
    with np.load(file) as z:
        return {k: z[k] for k in z.files}


def behavior(case: dict) -> dict:
    values = []
    for name in ('training.npz', 'sealed_test.npz'):
        with np.load(Path(case['template']['folder'])/name) as z:
            values.append({k: z[k] for k in ('choices', 'categories', 'feedback')})
    return {k: np.concatenate([v[k] for v in values]) for k in values[0]}


def parameter_sensitivity(cases: list[dict], arrays: dict) -> list[dict]:
    """Describe candidate disagreement; a singleton cannot establish robustness."""
    rows = []
    for sid in sorted(set(c['subject'] for c in cases)):
        own = [c for c in cases if c['subject']==sid]
        for mode in ('filtered', 'full'):
            for field in (*TARGETS, 'belief', 'execution'):
                values = np.stack([arrays[c['id']][mode+'_'+field] for c in own])
                if len(own)==1:
                    mean, q95 = np.nan, np.nan
                else:
                    if field in TARGETS:
                        difference = values.max(axis=0)-values.min(axis=0)
                    else:
                        difference = np.max(np.stack([.5*np.abs(values[i]-values[j]).sum(axis=1)
                            for i in range(len(own)) for j in range(i)]), axis=0)
                    mean, q95 = float(difference.mean()), float(np.quantile(difference, .95))
                rows.append(dict(subject=sid, mode=mode, measure=field, candidates=len(own),
                    mean_difference=mean, q95_difference=q95,
                    scope='single_candidate' if len(own)==1 else 'diagnostic_candidates' if sid==129 else 'finite_retained_candidates'))
    return rows


def overview(sid, cases, arrays, observed, output):
    fig = plt.figure(figsize=(12, 10), layout='constrained')
    gs = fig.add_gridspec(4, 2, height_ratios=[.8, 1, 1, 1])
    top = fig.add_subplot(gs[0, :]); n = len(observed['choices']); x = np.arange(1, n+1)
    correct = observed['choices'] == observed['categories']
    top.plot(x, pd.Series(correct.astype(float)).rolling(32, min_periods=1).mean(), color='#202020', lw=1.5)
    top.set(ylim=(0, 1.03), ylabel='真实选择正确率', title=f'S{sid} · 行为背景与内部状态（完整有效记录）')
    top.text(.99, .08, '黑线：最近32题正确率', ha='right', transform=top.transAxes)
    for row, (key, label) in enumerate(zip(STATE_KEYS, STATE_NAMES), 1):
        for col, mode in enumerate(('filtered', 'full')):
            ax = fig.add_subplot(gs[row, col])
            for i, case in enumerate(cases):
                ax.plot(x, arrays[case['id']][mode+'_'+key], color=COLORS[i], lw=.85, alpha=.9,
                        label=f'P{i+1} · {case["candidate"]["id"][:6]}')
            ax.set(ylim=(-.02, 1.02), xlim=(1, n), ylabel=label if col==0 else '',
                   xlabel='有效试次' if row==3 else '')
            if row==1:
                ax.set_title('过滤：使用截至本题的选择' if col==0 else '回看：使用完整行为记录')
                if col==1:
                    ax.legend(loc='lower right', fontsize=8)
    save(fig, output, f'S{sid}_overview.png')


def candidate_states(case, a, number, diag, output):
    h = a['full_belief'].shape[1]; n = len(a['full_belief'])
    importance = sum(a[mode+'_'+key].mean(axis=0) for mode in ('filtered', 'full')
                     for key in ('belief', 'execution'))
    target = case['target_rules']
    order = list(target)+[int(i) for i in np.argsort(importance)[::-1] if i not in target]
    order = order[:min(12, h)]
    fig, axes = plt.subplots(5, 2, figsize=(12, 13), layout='constrained', sharex=True)
    fig.suptitle(f'S{case["subject"]} · P{number} ({case["candidate"]["id"][:8]}) · 固定参数下的状态分布', fontsize=12)
    for col, mode in enumerate(('filtered', 'full')):
        ax = axes[0, col]
        for key, label, color in zip(STATE_KEYS, STATE_NAMES, STATE_COLORS):
            ax.plot(np.arange(1, n+1), a[mode+'_'+key], color=color, label=label, lw=.85)
        stable = bool(diag.get(mode+'_numerically_stable', False))
        ax.set(ylim=(-.02, 1.02), title=('过滤' if col==0 else '完整历史')+' · '+('数值检查通过' if stable else '数值受限'))
        if col==0:
            ax.legend(fontsize=8, loc='lower right')
        fields = [('active', '规则在场概率', 'Blues', 1), ('belief', '平均信念权重', 'Purples', 1),
                  ('execution', '执行规则概率', 'Oranges', 1), ('precision', '在场条件下的平均精度', 'viridis', 25)]
        for row, (key, label, cmap, vmax) in enumerate(fields, 1):
            ax = axes[row, col]
            if key=='precision':
                active = a[mode+'_active']
                value = np.divide(a[mode+'_active_precision'], active, out=np.full_like(active, np.nan), where=active>=.05)
            else:
                value = a[mode+'_'+key]
            cm = plt.get_cmap(cmap).copy(); cm.set_bad('#eeeeee')
            im = ax.imshow(value[:, order].T, aspect='auto', interpolation='nearest', origin='upper',
                           extent=[.5, n+.5, len(order)-.5, -.5], vmin=0, vmax=vmax, cmap=cm)
            ax.set_yticks(range(len(order)), [f'R{i}'+(' *' if i in target else '') for i in order])
            ax.set_ylabel(label if col==0 else '')
            fig.colorbar(im, ax=ax, shrink=.8, pad=.015)
        axes[-1, col].set_xlabel('有效试次')
    save(fig, output, f'{case["id"]}_states.png')
    return dict(case=case['id'], displayed_rules=order, total_rules=h,
        selection='target rules plus highest mean belief/execution across both conditioning modes',
        precision_display_minimum_presence=.05,
        precision_masked_cells={m: int((a[m+'_active'][:, order]<.05).sum()) for m in ('filtered', 'full')})


def numerical_figure(checks, output):
    frame = checks.loc[checks.bank.eq('resolution')].copy()
    labels = [f'S{r.subject} · {r.candidate[:6]}' for r in frame.itertuples()]
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), layout='constrained', sharey=True)
    for mode, color, offset, label in [('filtered', '#287ca0', -.13, '过滤'), ('full', '#bd6b35', .13, '完整历史')]:
        axes[0].scatter(frame[mode+'_ess_min'], y+offset, color=color, label=label, s=24)
        axes[1].scatter(frame[mode+'_half_q95'], y+offset, color=color, label=label, s=24)
    axes[0].axvline(4, color='#777777', ls='--', lw=1); axes[1].axvline(.1, color='#777777', ls='--', lw=1)
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis()
    axes[0].set(xlabel='独立过滤的最小有效数量', title='计算权重是否过度集中')
    axes[1].set(xlabel='两组独立计算差异的95%分位', title='同样设置重新计算是否一致')
    axes[0].legend(); save(fig, output, 'numerical_checks.png')


def oral_figure(metrics, output):
    frame = metrics.loc[metrics.reading.eq('midpoint') & metrics.domain.eq('functional') & metrics.scope.eq('complete')]
    subjects = sorted(frame.subject.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained', sharey=True)
    for ax, mode in zip(axes, ('filtered', 'full')):
        for x, sid in enumerate(subjects):
            own = frame.loc[frame.subject.eq(sid) & frame['mode'].eq(mode)]
            for metric, color, offset, marker, label in [('dynamic_support', '#287ca0', -.14, 'o', '动态状态'),
                ('static_support', '#bd6b35', .14, 'D', '整段静态内容')]:
                values = own[metric].to_numpy()
                ax.plot([x+offset]*2, [values.min(), values.max()], color=color, lw=1)
                ax.scatter(x+offset+np.linspace(-.04, .04, len(values)), values, color=color, marker=marker,
                           s=23, label=label if x==0 else None)
            ax.scatter(x, own.uniform_support.iloc[0], color='#555555', marker='_', s=110, label='目录平均基准' if x==0 else None)
        ax.set_xticks(range(len(subjects)), [f'S{s}' for s in subjects])
        ax.set(ylim=(-.02, 1.02), title='过滤状态的口述相容性' if mode=='filtered' else '完整历史状态的口述相容性')
    axes[0].set_ylabel('完整编码口述的平均相容质量'); axes[1].legend(fontsize=8)
    save(fig, output, 'oral_compatibility.png')


def oral_mass_figure(metrics, output):
    selected = metrics.loc[metrics.reading.eq('midpoint') & metrics['mode'].eq('full') & metrics.scope.eq('complete')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), layout='constrained', sharey=True)
    for ax, domain, title in zip(axes, ('structural', 'functional'), ('明确判断约束的对应', '整组口述内容的相容')):
        frame = selected.loc[selected.domain.eq(domain)]
        y = np.arange(len(frame)); left = np.zeros(len(frame))
        for key, color, label in [('support', '#287ca0', '相容'), ('conflict', '#bd6b35', '冲突'),
                                   ('undetermined', '#bac1c5', '无法判断')]:
            ax.barh(y, frame['dynamic_'+key], left=left, color=color, label=label)
            left += frame['dynamic_'+key].to_numpy()
        ax.set_yticks(y, [f'S{r.subject} · {r.candidate[:6]}' for r in frame.itertuples()])
        ax.set(xlim=(0, 1), xlabel='完整报告的平均概率质量', title=title)
    axes[0].invert_yaxis(); axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3)
    save(fig, output, 'oral_three_verdicts.png')


def oral_time_figure(sid, scores, representative, trials, output):
    own = scores.loc[scores.subject.eq(sid) & scores.reading.eq('midpoint') &
                     scores.domain.eq('functional') & scores.scope.eq('complete')].copy()
    own['block'] = (own.trial-1)//32
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), layout='constrained', sharex=True)
    for ax, mode in zip(axes, ('filtered', 'full')):
        selected = own.loc[own['mode'].eq(mode)]
        for key, color, label in [('support', '#287ca0', '相容'), ('conflict', '#bd6b35', '冲突'),
                                   ('undetermined', '#8c959a', '无法判断')]:
            grouped = selected.groupby(['candidate', 'block'])[key].mean().unstack('candidate')
            grouped = grouped.reindex(range((trials+31)//32))
            x = np.minimum(grouped.index.to_numpy()*32+16.5, trials)
            ax.fill_between(x, grouped.min(axis=1), grouped.max(axis=1), color=color, alpha=.12, linewidth=0)
            ax.plot(x, grouped[representative], color=color, lw=1.3, label=label)
        ax.set(ylim=(-.02, 1.02), ylabel='口述内容的概率质量',
               title=f'S{sid} · '+('过滤' if mode=='filtered' else '完整历史')+' · 每32题汇总完整报告')
    axes[0].legend(ncol=3, loc='center right'); axes[1].set_xlabel('有效试次')
    save(fig, output, f'S{sid}_oral_time.png')


def recovery_figure(scores, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout='constrained')
    mse_upper = min(1., max(.1, float(np.ceil(scores[
        ['target_present_mse', 'target_execution_mse']].to_numpy().max()*10)/10)))
    for ax, key, label in zip(axes, ['target_present_mse', 'target_execution_mse', 'belief_total_variation'],
                              ['正确规则在场概率误差', '正确规则执行概率误差', '完整规则信念分布距离']):
        frame = scores.pivot(index='case', columns='mode', values=key)
        for i, (_, row) in enumerate(frame.iterrows()):
            ax.plot([0, 1], [row.filtered, row.full], color='#b7c2c8', lw=.7)
            ax.scatter([0, 1], [row.filtered, row.full], color=['#6b7478', '#287ca0'], s=19)
        upper = 1. if key=='belief_total_variation' else mse_upper
        ax.set_xticks([0, 1], ['过滤', '完整历史']); ax.set(xlim=(-.3, 1.3), ylim=(0, upper), title=label)
    axes[0].set_ylabel('越低越接近模拟真值')
    save(fig, output, 'state_recovery.png')


def error_figure(frame, output):
    own = frame.loc[frame['mode'].eq('full')]
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    y = np.arange(len(own)); left = np.zeros(len(own))
    for key, color, label in [('target_absent', '#9ba7ac', '正确规则不在场'),
        ('target_available_other_rule', '#287ca0', '正确规则在场，执行其他规则'),
        ('target_rule', '#bd6b35', '执行正确规则，仍作答错误')]:
        ax.barh(y, own[key], left=left, color=color, label=label); left += own[key].to_numpy()
    ax.set_yticks(y, [f'S{r.subject} · P{r.number}' for r in own.itertuples()]); ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel='实际错误试次中的平均状态概率质量', title='错误选择对应的三种可能情况 · 完整历史重建')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3, fontsize=8)
    save(fig, output, 'error_sources.png')


def distribution_figure(case, a, distributions, number, output):
    n = case['trials']; target = case['target_rules'][0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout='constrained')
    fig.suptitle(f'S{case["subject"]} · P{number} · 正确规则的信念与精度分布', fontsize=12)
    for col, mode in enumerate(('filtered', 'full')):
        belief = distributions[mode+'_belief_histogram']
        precision = distributions[mode+'_precision_conditional_histogram'].copy()
        precision[a[mode+'_target_present']<.05] = np.nan
        for row, (value, upper, label) in enumerate([(belief, 1., '正确规则的信念值'),
                                                     (precision, 25., '正确规则在场时的精度')]):
            cm = plt.get_cmap('Blues').copy(); cm.set_bad('#eeeeee')
            im = axes[row, col].imshow(value.T, origin='lower', aspect='auto', interpolation='nearest',
                extent=[.5, n+.5, 0, upper], cmap=cm, vmin=0, vmax=1)
            fig.colorbar(im, ax=axes[row, col], label='该区间的概率质量', shrink=.85)
            axes[row, col].set_ylabel(label if col==0 else '')
            axes[row, col].set_xlabel('有效试次')
        axes[0, col].plot(np.arange(1, n+1), a[mode+'_target_belief'], color='#bd6b35', lw=.8)
        axes[0, col].set_title('过滤' if mode=='filtered' else '完整历史')
    save(fig, output, f'{case["id"]}_distributions.png')


def pct(value):
    return f'{100*float(value):.1f}%'


def span(values):
    return pct(np.min(values))+'—'+pct(np.max(values))


def points_span(values):
    return f'{100*np.min(values):+.1f} 至 {100*np.max(values):+.1f}'


def run(source: Path, summary: Path, oral: Path, output: Path, distributions: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError('Use a new report directory')
    output.mkdir(parents=True); figures = output/'figures'; figures.mkdir()
    configure()
    protocol = read(source/'protocol.json'); selected = [s for s in read(summary/'selected.json') if 'file' in s]
    case_map = {c['id']: c for c in protocol['cases']}
    cases = [case_map[s['case']] for s in selected]
    arrays = {s['case']: load_arrays(Path(s['file'])) for s in selected}
    checks = pd.read_csv(summary/'numerical_checks.csv')
    quantity_checks = []
    quantity_names = dict(zip((*TARGETS, 'belief', 'execution'), [
        '正确规则在场概率', '正确规则平均信念', '正确规则信念达到0.5的概率',
        '正确规则执行概率', '全部规则的信念分布', '全部规则的执行分布']))
    for row in checks.loc[checks.bank.eq('resolution')].to_dict('records'):
        for mode in ('filtered', 'full'):
            for comparison in ('half', 'resolution'):
                for measure in (*TARGETS, 'belief', 'execution'):
                    prefix = mode+'_'+comparison+'_'+measure+'_'
                    quantity_checks.append(dict(case=row['case'], subject=row['subject'],
                        candidate=row['candidate'], mode=mode, comparison=comparison,
                        measure=measure, measure_label=quantity_names[measure],
                        mode_label='截至本题' if mode=='filtered' else '完整历史',
                        comparison_label='两组独立重复' if comparison=='half' else '512与1024粒子',
                        mean_difference=row[prefix+'mean'],
                        q95_difference=row[prefix+'q95'], whole_mean_difference=row[prefix+'whole']))
    pd.DataFrame(quantity_checks).to_csv(output/'state_quantity_diagnostics.csv', index=False, encoding='utf-8-sig')
    recovery = pd.read_csv(summary/'recovery_scores.csv')
    metrics = pd.read_csv(oral/'compatibility_summary.csv')
    oral_scores = pd.read_csv(oral/'candidate_trial_scores.csv.gz')
    numerical_figure(checks, figures); oral_figure(metrics, figures); oral_mass_figure(metrics, figures)
    recovery_figure(recovery, figures)
    sensitivity = pd.DataFrame(parameter_sensitivity(cases, arrays))
    sensitivity.to_csv(output/'parameter_state_sensitivity.csv', index=False)
    parameter_rows = [dict(subject=c['subject'], candidate=c['candidate']['id'], status=c['parameter_status'],
                           **c['candidate']['hyperparams']) for c in cases]
    pd.DataFrame(parameter_rows).to_csv(output/'parameters.csv', index=False)
    write(output/'parameters.json', [dict(subject=c['subject'], candidate=c['candidate'],
        status=c['parameter_status']) for c in cases])
    source_rows, target_rows, displays, sections = [], [], [], []
    state_csv = []
    for sid in sorted(set(c['subject'] for c in cases)):
        own = [c for c in cases if c['subject']==sid]; observed = behavior(own[0])
        overview(sid, own, arrays, observed, figures)
        oral_time_figure(sid, oral_scores, own[0]['candidate']['id'], len(observed['choices']), figures)
        sections.extend([f'## S{sid}', '', f'本页包含{len(own)}组完整参数。正确规则为'+
                         '、'.join('R'+str(i) for i in own[0]['target_rules'])+'。', '',
                         f'![S{sid}状态概览](figures/S{sid}_overview.png)', '',
            '顶行黑线是实际选择的最近32题正确率。下面每种颜色代表一组参数，状态曲线保留逐题结果。'
            '左列只使用截至本题的选择；右列利用后续行为回看本题。三行依次显示正确规则是否在场、'
            '它得到多少信念、以及本题是否执行它。信念均值与执行概率分别计算。', ''])
        if sid==129:
            sections.extend(['S129的四组参数保持诊断用途，上一阶段的参数数值问题尚未解决。', ''])
        if sid==229:
            sections.extend(['S229仅有一个主候选；上一阶段对明显波动轨迹的生成不足继续保留。', ''])
        if sid==110:
            sections.extend(['S110的行为分类沿用“持续较高”。行为类别描述正确率形态，状态图描述模型中的规则变化。'
                '较高正确率可以由不同规则在不同阶段产生；替代规则的内容与口述对照提供进一步检查的入口。', ''])
        if len(own)>1:
            delta = sensitivity.loc[sensitivity.subject.eq(sid) & sensitivity['mode'].eq('full') & sensitivity.measure.eq('execution')].iloc[0]
            sections.extend([f'完整历史下，各候选执行规则分布的最大两两差异，逐题平均为{delta.mean_difference:.3f}，'
                f'95%分位为{delta.q95_difference:.3f}（总变差距离，0为相同、1为互不重叠）。'
                '这一差异同时需要结合本人的数值检查阅读。', ''])
        else:
            sections.extend(['当前仅一组候选，跨参数稳定性尚不能据此检验。', ''])
        sections.extend(['| 参数组 | 容量 | 执行维持 | 信念保持 | 初始精度 | 精度学习率 |',
                         '|---|---:|---:|---:|---:|---:|'])
        for number, case in enumerate(own, 1):
            h = case['candidate']['hyperparams']; prefix = 'engine.modules.'
            values = [h[prefix+key] for key in ['hypo_transitions_mod.kwargs.max_capacity',
                'hypo_transitions_mod.kwargs.execution_inertia', 'hypo_transitions_mod.kwargs.intervention.decay',
                'beta_mod.kwargs.beta_init', 'beta_mod.kwargs.learning_rate']]
            sections.append(f'| P{number} | '+' | '.join(f'{v:g}' for v in values)+' |')
        sections.extend(['', '表中列出五个主要参数；候选组合的各项设置见[参数表](parameters.csv)，'
            f'计算设置和输入来源见[运行协议]({os.path.relpath(source/"protocol.json", output)})。', ''])
        for number, case in enumerate(own, 1):
            a = arrays[case['id']]
            diag = checks.loc[checks.case.eq(case['id']) & checks.bank.eq('resolution')].iloc[0].to_dict()
            displays.append(candidate_states(case, a, number, diag, figures))
            if distributions is not None:
                distribution_figure(case, a, load_arrays(distributions/(case['id']+'.npz')), number, figures)
            wrong = observed['choices'] != observed['categories']
            for mode in ('filtered', 'full'):
                source_mass = a[mode+'_source'][wrong].mean(axis=0)
                source_rows.append(dict(subject=sid, number=number, candidate=case['candidate']['id'], mode=mode,
                    error_trials=int(wrong.sum()), target_absent=source_mass[0],
                    target_available_other_rule=source_mass[1], target_rule=source_mass[2]))
                target_rows.append(dict(subject=sid, number=number, candidate=case['candidate']['id'], mode=mode,
                    **{key: float(a[mode+'_'+key].mean()) for key in TARGETS}))
                for key in TARGETS:
                    state_csv.append(pd.DataFrame(dict(subject=sid, candidate=case['candidate']['id'], mode=mode,
                        measure=key, trial=np.arange(1, len(a[mode+'_'+key])+1), probability=a[mode+'_'+key])))
            state = '过滤通过' if diag['filtered_numerically_stable'] else '过滤受限'
            state += '；完整历史通过' if diag['full_numerically_stable'] else '；完整历史受限'
            sections.extend([f'### P{number} · {case["candidate"]["id"]}', '', state+'。', '',
                f'1024粒子下实际完成{int(diag["systems"])}次独立计算，计划{int(diag["expected_systems"])}次。', '',
                f'完整历史的整段平均状态量，两组独立计算的最大差异为{diag["full_half_whole_maximum"]:.3f}；'
                f'512与1024粒子的最大差异为{diag["full_resolution_whole_maximum"]:.3f}。'
                '这些描述量帮助区分逐题变化与整段占比的计算差异，原数值验收门槛保持不变。', '',
                f'![P{number}完整规则状态](figures/{case["id"]}_states.png)', '',
                '顶行是正确规则摘要；下面依次是各规则在场概率、信念、执行概率和在场条件下的平均精度。'
                '标有星号的是正确规则。每列显示相同的最多12条规则，按平均信念与执行质量排序，正确规则保留。'
                '灰色精度格表示该规则在场概率低于0.05；完整数组仍保留这些值。'
                '在场概率允许多条规则同时较高；全部规则的信念、执行概率各自合计为1。', ''])
            if distributions is not None:
                sections.extend([f'[查看P{number}的信念与精度概率分布](figures/{case["id"]}_distributions.png)', '',
                    '分布图上行把信念值按0.05分组，蓝色深浅表示每组的概率质量；橙线为平均信念。'
                    '平均信念为0.5时，可能对应较集中的中等信念，也可能对应信念接近0和1的两种解释。'
                    '下行把在场条件下的精度按1分组；在场概率低于0.05的题显示为灰色。', ''])
        sections.extend([f'![S{sid}口述随试次的对照](figures/S{sid}_oral_time.png)', '',
            '实线表示P1下的口述相容、冲突和无法判断质量；阴影为该被试全部候选的最小值至最大值。'
            '每个点汇总连续32道有效题中完整编码的报告；没有完整报告的块留空。此处按既有中点读法展示。', '',
            f'[S{sid}逐条口述复核CSV]({os.path.relpath(oral/f"S{sid}_review.csv", output)})', ''])
    errors = pd.DataFrame(source_rows); errors.to_csv(output/'error_sources.csv', index=False)
    pd.DataFrame(target_rows).to_csv(output/'target_state_means.csv', index=False)
    pd.concat(state_csv, ignore_index=True).to_csv(output/'target_state_curves.csv.gz', index=False, compression='gzip')
    error_figure(errors, figures)
    human = checks.loc[checks.bank.eq('resolution')]
    pass_filter = int(human.filtered_numerically_stable.sum()); pass_full = int(human.full_numerically_stable.sum())
    incomplete_banks = int((~human.bank_complete).sum())
    completion_note = (f'其中{incomplete_banks}组尚未完成全部预定重复次数，按登记规则计入计算受限。'
                       if incomplete_banks else '全部参数组均完成预定重复次数。')
    run_status = read(source/'progress.json'); oral_status = read(oral/'completion.json')
    failed_jobs = [read(f) for f in (source/'ledger').glob('*.json') if read(f)['status']=='failed']
    at_wall_limit = run_status['elapsed_hours'] >= protocol['policy']['budget']['wall_hours']
    budget_terminated = [j for j in failed_jobs if at_wall_limit and j['returncode']==-15]
    runtime_failures = len(failed_jobs)-len(budget_terminated)
    reported_stop = ('wall_budget' if budget_terminated and not runtime_failures else run_status['stop'])
    write(output/'computation_audit.json', dict(raw_stop_reason=run_status['stop'],
        reported_stop_reason=reported_stop, budget_terminated_jobs=budget_terminated,
        runtime_failure_count=runtime_failures, elapsed_hours=run_status['elapsed_hours'],
        cpu_hours_charged=run_status['cpu_hours'],
        classification='SIGTERM exits at the registered wall deadline are budget terminations; original ledgers retained'))
    failure_note = (f'{len(budget_terminated)}次按时间上限终止，{runtime_failures}次运行失败；'
                    if budget_terminated else f'{run_status["failed"]}次失败；')
    recovery_pass = recovery.groupby('mode').numerically_stable.sum()
    intro = ['# Model 1024：个体内部状态与口述外部对照', '',
        '本轮使用六人的完整行为记录，分别在13组保留参数和S129的4组诊断参数下重建内部状态。'
        '口述用于外部内容对照，尚待人工最终检查。全部真人案例继续作为开发资料。', '',
        f'当前运行账本记录{run_status["completed"]}次成功过滤，'+failure_note+
        f'累计计入{run_status["cpu_hours"]:.2f} CPU小时。完整记录的精确复现与已知模拟状态检查分别保存。', '',
        f'按登记门槛，{len(human)}组参数中，过滤结果有{pass_filter}组通过数值检查，'
        f'完整历史结果有{pass_full}组通过。'+completion_note+
        '受限结果可用于观察不同解释，精确概率与改善时点的结论仍需保留计算不确定性。', '',
        f'[口述逐条复核页面]({os.path.relpath(oral/"review.html", output)}) · '
        f'[全部口述CSV]({os.path.relpath(oral/"all_subjects_review.csv", output)}) · '
        f'[规则内容目录]({os.path.relpath(oral/"rule_content_catalogue.csv", output)})', '',
        '## 先读这些图', '',
        '这些状态图以实际发生的选择与反馈为条件。概率较高表示该状态更能解释这份行为记录。'
        '每组参数保留独立结果；不同参数之间的范围表示有限候选的分歧。数值检查另外评价随机计算的分歧。', '',
        '图中的“规则精度”是模型状态中的判断精度，决定规则区分类别的程度。'
        '“计算精度”指用有限粒子估计状态概率时的数值误差；增加粒子数用于检查这一误差。'
        '本轮各组参数的初始判断精度和学习率均沿用上一阶段。', '',
        '“正确规则”指在任务几何审计中对全部记录刺激都能正确分类的规则。其他规则也可能答对许多题，'
        '因此较高正确率与较低的正确规则在场概率可以同时出现。各条替代规则的内容可在规则目录中查阅。', '',
        '![数值检查](figures/numerical_checks.png)', '',
        '左图看独立重复的权重是否集中：有效数量越小，说明结果越依赖少数计算。右图把同样设置的独立重复分成两组，'
        '比较状态结果；差异越小越一致。虚线是本轮工作门槛。图外还检查了最大权重、平均差异及512/1024粒子差异，'
        '完整结果见数值检查表。通过这些门槛说明计算较稳定，准确性继续由模拟真值检验。', '',
        '验收范围为正确规则的四项摘要，以及全部规则的信念和执行分布。'
        '其他规则的在场概率和判断精度图用于描述；它们各自的数值稳定性与概率校准仍需单独检验。', '',
        f'[数值检查表]({os.path.relpath(summary/"numerical_checks.csv", output)})', '',
        '[各项状态量的计算差异](state_quantity_diagnostics.csv)另外列出正确规则摘要和完整规则分布各自的差异。'
        '例如，对正确规则是否在场的判断可以较接近，同时对其他规则身份仍有分歧。'
        '这些分项描述帮助定位差异，原验收门槛和追加规则保持不变。', '',
        '## 模拟真值能否找回', '',
        '固定使用六人各两条既有模拟路径，共12条。推断使用真实生成参数，生成状态仅用于事后评分。'
        '这些点覆盖本轮候选的有限部分，状态恢复与参数恢复分别解释。', '',
        f'恢复任务各使用512粒子、8次独立计算。重复计算的稳定性检查：过滤通过{int(recovery_pass["filtered"])}/12条，'
        f'完整历史通过{int(recovery_pass["full"])}/12条。恢复任务本轮只检查这一档粒子数；'
        '未通过组的恢复误差仍包含计算不稳定的影响，需要与状态本身的推断难度一起考虑。', '',
        '![状态恢复](figures/state_recovery.png)', '',
        '每条连线是一条完整模拟路径。左、中图是概率与真实事件的平方误差；右图是完整规则信念分布的总变差距离。'
        '三者均越低越好，线向下表示利用完整历史后更接近真值。'
        '前两图共用从零开始的平方误差刻度，右图的距离刻度单列。', '']
    for key, name in [('target_present_mse', '正确规则在场'), ('target_execution_mse', '正确规则执行'),
                      ('belief_total_variation', '完整规则信念分布')]:
        means = recovery.groupby('mode')[key].mean()
        intro.append(f'- {name}：12条路径等权平均，过滤{means["filtered"]:.4f}，完整历史{means["full"]:.4f}。')
    whole_error = recovery.loc[recovery['mode'].eq('full')]
    intro.extend(['', '完整历史对整段正确规则在场比例的平均绝对误差为'
        f'{pct(np.abs(whole_error.target_present_inferred_mean-whole_error.target_present_truth_mean).mean())}，'
        '该数值按12条路径等权汇总。'])
    rare = recovery.loc[recovery['mode'].eq('filtered')]
    total_rare = int(rare.source_available_other_count.sum())
    rare_means = {}
    for mode in ('filtered', 'full'):
        own = recovery.loc[recovery['mode'].eq(mode) & recovery.source_available_other_count.gt(0)]
        rare_means[mode] = float(np.average(own.source_available_other_probability_when_true,
            weights=own.source_available_other_count)) if len(own) else np.nan
    intro.extend(['', f'“正确规则已在场但执行其他规则”在{int(rare.source_available_other_count.gt(0).sum())}条路径中'
        f'出现，共{total_rare}题。在这些真实发生的题上，模型赋予该状态的平均概率为：'
        f'过滤{pct(rare_means["filtered"])}，完整历史{pct(rare_means["full"])}。'
        '这一项单独检查先前较难恢复的状态，避免总体误差被大量其他试次主导。'])
    intro.extend(['', f'[逐路径恢复分数]({os.path.relpath(summary/"recovery_scores.csv", output)})', '',
        '## 口述外部对照：暂定结果', '',
        f'共保留{oral_status["reports"]}条口述记录。编码状态为：'+
        '；'.join(f'{SCOPE_LABELS.get(k, k)} {v}条' for k, v in oral_status['scopes'].items())+'。', '',
        '主分母包含全部完整编码报告，目录没有支持项的报告也保留。解析片段单列；待复核、缺失或编码约束矛盾的记录继续保留。'
        '图中采用功能相容：模型当前执行内容能推出口述中的整组约束。结构相容要求每条约束明确出现在规则描述中，两种分数均已导出。', '',
        '![口述相容性](figures/oral_compatibility.png)', '',
        '蓝点为各候选参数的动态状态；橙点使用同一选择类别的整段平均内容；灰色短横线在目录中平均分配质量。'
        '竖线仅连接候选参数的最小值和最大值。相容质量评价内容兼容程度，口述句子的生成概率尚未建立。', '',
        '![相容、冲突与无法判断](figures/oral_three_verdicts.png)', '',
        '每一行保留同一组完整报告。蓝色表示支持口述的状态质量，橙色表示冲突，灰色表示现有内容不足以判断。'
        '大量灰色提示当前目录或编码缺少判别信息，需要与橙色的明确冲突分别检查。', '',
        '| 被试 | 完整报告数 | 目录无支持项 | 完整历史相容范围 | 相对静态基准（百分点） | 相对目录基准（百分点） |',
        '|---|---:|---:|---:|---:|---:|'])
    frame = metrics.loc[metrics.reading.eq('midpoint') & metrics.domain.eq('functional') & metrics.scope.eq('complete') & metrics['mode'].eq('full')]
    for sid, group in frame.groupby('subject'):
        intro.append(f'| S{sid} | {int(group.n.iloc[0])} | {int(group.no_catalogue_support_count.iloc[0])} | {span(group.dynamic_support)} | '
                     f'{points_span(group.dynamic_minus_static_support)} | {points_span(group.dynamic_minus_uniform_support)} |')
    intro.extend(['', '“目录无支持项”表示无论怎样分配内部状态概率，该条口述的完整约束都无法得到正的相容质量。'
        '此时先核对整句包含的判断、刺激描述与目录覆盖。冲突和无法判断继续分别显示，低相容分数的来源可据此区分。', '',
        '表中变化为百分点。两种参照解释均保持原编码流程；相对长短解释的结果与差异保存在完整表格和逐条复核页中。'
        '口述没有参与本轮参数选择或状态更新。部分口述此前参与模型开发，因此本次对照按开发验证解释。', '',
        f'[完整口述汇总]({os.path.relpath(oral/"compatibility_summary.csv", output)})', '',
        '## 实际错误对应的可能状态', '', '![错误来源](figures/error_sources.png)', '',
        '每行只汇总该被试实际答错的试次。三段分别表示：正确规则不在场；正确规则在场但执行另一规则；'
        '执行正确规则仍产生错误选择。后一情况可包含知觉和判断精度等影响。这是模型条件解释，数值受限的参数组保留同样限制。', ''])
    ending = ['## 计算、资料与复现边界', '',
        f'主运行累计{run_status["elapsed_hours"]:.2f}小时，剩余未启动任务{run_status["pending"]}次。'
        '停止状态：'+{'cpu_budget': '达到CPU预算的派发边界', 'wall_budget': '达到主运行时间上限',
            'repeated_worker_failure': '工作进程累计失败达到停止门槛', None: '计划队列完成'}.get(reported_stop, str(reported_stop))+'。'
        '未达到预定重复次数的候选会标记计算受限。', '',
        '[计算停止审计](computation_audit.json)保留原始停止标签，并区分截止时刻收到终止信号的任务与运行失败。'
        '被终止任务按单任务CPU上限计入原队列账本，该记账量可高于实际CPU使用量。', '',
        '所有状态对应选择后、当前反馈前。过滤使用截至本题的行为，完整历史使用整段行为。'
        '两者均使用整次任务估计的固定参数，属于拟合后的过程重建。', '',
        '数值检查与已知参数的状态恢复分别报告；连续参数区域的边界、完整参数恢复及总体概率校准仍是后续工作。'
        '候选集合与轨迹分类沿用上一阶段，本轮保持原参数判定。', '',
        '本轮交付规则在场、信念、精度与执行分布。知觉样本、搜索控制量和配对矩阵的完整分布验证留待单独研究。', '',
        '原始口述和数据保持原样。逐条复核CSV预留人工修正文本、预期含义和备注列。'
        '明天可先检查长短参照、躯干比较、整句与片段范围，以及多个特点是否都在描述当前作答依据。', '',
        '[状态曲线数据](target_state_curves.csv.gz) · [状态均值](target_state_means.csv) · [错误来源数据](error_sources.csv) · '
        '[跨参数差异](parameter_state_sensitivity.csv) · [完整参数表](parameters.csv)', '']
    if distributions is not None:
        ending.extend([f'[完整历史占比分布]({os.path.relpath(distributions/"whole_history_functionals.csv", output)})'
            '保存正确规则在场占比、平均信念以及执行占比的条件均值和标准差。'
            '在场占比与平均信念另给出条件分位数；这些分位数的统计覆盖尚未校准。', ''])
    (output/'report.md').write_text('\n'.join(intro+sections+ending))
    (output/'interpretation.md').write_text('\n'.join([
        '# 阅读入口', '', '先打开[完整报告](report.md)，依次看数值检查、模拟恢复和口述总览，再看每个被试。', '',
        '本轮的问题是：给定一个人真实发生的全部选择和反馈，模型认为他每题可能考虑哪些规则、相信到什么程度，'
        '以及可能按哪条规则作答。上一阶段保留的参数组合各自完成一遍推断，因此可以看到同一行为对应的不同内部解释。', '',
        '这次使用全部有效记录。模型从首题到末题持续根据反馈学习，参数保持固定。'
        '左列“过滤”在推断本题状态时使用截至本题的选择；右列“完整历史”再利用后续行为回看本题。'
        '两列的参数都来自整次任务的估计，均属于开发资料上的过程重建。', '',
        '每人的概览图，黑线是实际选择的最近32题正确率；下面各条彩色线代表不同参数组合下的状态概率。'
        '每组参数的详细图把规则在场、信念、执行和精度分开。', '',
        '- 规则在场：这条规则是否处于当前考虑和学习的集合中。多条规则可以同时在场。',
        '- 信念：学习者在当前规则集合中分给各规则的相对权重，各规则合计为1。',
        '- 执行：本题实际采用哪条规则的概率，各规则合计为1。实际选择也会帮助判断本题的执行身份。',
        '- 精度：规则区分类别的程度。精度图同时依赖规则在场概率，较少在场的规则需要结合在场图阅读。', '',
        '这里的规则精度属于模型中的学习者状态；数值检查评价的是研究者计算状态概率时的误差。'
        '增加粒子数用于改善数值计算，各组参数的初始判断精度和学习率沿用原值。', '',
        '研究者对状态的不确定性与学习者本身的信念分开表示。例如，平均信念0.5可以来自两种互不相同的内部解释：'
        '一半路径的信念接近1，另一半接近0。分布细节图可以看出这一点。', '',
        '概率曲线的变化有两种来源：模型内部状态发生变化，以及新的选择使研究者重新判断哪条潜在过程更可能。'
        '判断规则切换需要结合完整内部路径及其不确定性。', '',
        f'过滤数值检查通过{pass_filter}/{len(human)}组，完整历史通过{pass_full}/{len(human)}组。'
        '各参数组的图注明确标记限制。数值检查中的“两组”指两组独立随机计算，每次计算都使用完整行为记录。'
        '通过数值检查表示计算较稳定；12条模拟路径的真值检查再说明哪些状态较容易或较难找回。', '',
        f'口述逐条检查请打开[复核页面]({os.path.relpath(oral/"review.html", output)})。'
        '记录包含原话、编码解释、参数范围与列示候选下最可能的规则内容。口述没有参与参数选择和状态更新。', '',
        '口述图的蓝色是内容相容，橙色是明确冲突，灰色是现有内容无法判断。“完整编码”表示现有语法解析了整句话，'
        '其含义仍需人工确认。如果整句话在模型目录中没有任何支持项，零相容分数就缺少区分状态结果的能力。'
        '报告已单独列出这种情况，也保留中点解释与相对长短解释的差异。', '',
        '明天可先确认原话中哪些特点表达当前作答依据、哪些属于一般刺激描述，再核对长短参照、躯干比较和未解析片段。'
        'CSV末尾预留修正文本、预期含义和备注列，原始数据保持不变。', '']))
    write(output/'figure_qa.json', dict(backend='python', format='PNG', dpi=300,
        purpose='development report; quantitative grids with explicit numerical and parameter uncertainty',
        behavior_window='trailing 32 valid trials, min_periods=1', state_smoothing='none',
        oral_display_window='nonoverlapping 32 valid trials; complete encoded reports only; empty blocks remain missing',
        recovery_axes='MSE panels share a zero-based range covering all values; total variation uses 0 to 1',
        data_exclusion='none; heatmap row selection is display-only and recorded', displays=displays,
        n='six development subjects, 17 fixed candidates, 12 fixed simulation paths',
        intervals='candidate ranges, not confidence intervals', tests='no significance testing',
        export_contract='repository instruction: PNG only; large on-screen research report, 300 dpi; vector/TIFF journal checks waived',
        source_sha256=digest(Path(__file__))))
    write(output/'completion.json', dict(status='complete', subjects=len(set(c['subject'] for c in cases)),
        candidates=len(cases), figures=len(list(figures.glob('*.png'))),
        summary=str(summary.resolve()), oral=str(oral.resolve()),
        filtering_numerical_pass=pass_filter, full_history_numerical_pass=pass_full))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'summary', 'oral', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--distributions', type=Path)
    args = parser.parse_args(); run(args.source, args.summary, args.oral, args.output, args.distributions)
