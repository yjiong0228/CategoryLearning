"""PNG-only evidence panels for an audited execution-comparison study.

Each human remains visible; missing scores are labeled. Error bars are twice
the Monte Carlo standard error, not participant-sampling confidence intervals.
Recovery displays all registered paths, including ungenerated or unresolved ones.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

BLUE = '#39738C'
GRAY = '#6B7280'


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ''):
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f'Nonfinite plotted value: {key}')
    return result


def flag(row: dict, key: str) -> bool:
    return str(row.get(key)).lower() == 'true'


def style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': .7, 'legend.frameon': False, 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'pdf.fonttype': 42})


def marker_status(row: dict, metric: str) -> tuple[bool, bool]:
    complete = flag(row, 'fit_jobs_complete')
    precise = flag(row, 'fit_numerical_resolution')
    if metric == 'suffix':
        precise = precise and flag(row, 'suffix_numerical_resolution')
    if metric == 'suffix_energy':
        complete = complete and all(flag(row, k + '_paths_complete') for k in ('free', 'zero'))
    return complete, precise


def human_figure(rows: list[dict], subjects: list[int], output: Path, smoke: bool) -> dict:
    # Three linked quantitative panels; suffix prediction has the widest panel.
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 103 / 25.4),
                             gridspec_kw={'width_ratios': [1, 1.2, 1]}, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.25, top=.80, wspace=.24)
    specifications = [('full', 'fit', 'Full-record fit', 'NLL difference / trial'),
                      ('prefix', 'suffix', 'Suffix prediction', 'NLL difference / trial'),
                      ('prefix', 'suffix_energy', 'Autonomous suffix', 'Energy difference')]
    source_rows, nll_extents = [], [0.]
    for axis, (group, metric, title, xlabel), panel in zip(axes, specifications, 'abc'):
        lookup = {int(row['subject']): row for row in rows if row['group'] == group}
        if len(lookup) != sum(row['group'] == group for row in rows):
            raise ValueError('Duplicate subject rows in audited human data')
        if set(lookup) - set(subjects):
            raise ValueError('Human data contain unregistered participants')
        axis.axvline(0, color='#B5B9BF', linewidth=.7, zorder=0)
        axis.set_title(title, loc='left', pad=13)
        axis.text(-.12, 1.11, panel, transform=axis.transAxes, fontsize=9, weight='bold')
        extents = [0.]
        for index, sid in enumerate(subjects):
            row = lookup.get(sid, {})
            value, se = number(row, metric + '_difference'), number(row, metric + '_mcse')
            complete, precise = marker_status(row, metric)
            source_rows.append(dict(subject=sid, group=group, metric=metric, difference=value,
                                    mcse=se, registered_work_complete=complete, precision_pass=precise))
            if value is None or se is None:
                axis.text(.97, index, 'not available', transform=axis.get_yaxis_transform(),
                          ha='right', va='center', fontsize=7, color=GRAY)
                continue
            if se < 0:
                raise ValueError('Negative standard error')
            marker = 'o' if complete else 'x'
            axis.errorbar(value, index, xerr=2 * se, fmt=marker, markersize=4.5,
                          markerfacecolor=BLUE if precise and complete else 'white',
                          markeredgecolor=BLUE if precise else GRAY,
                          color=BLUE if precise else GRAY, linewidth=.9, capsize=2, zorder=2)
            extents.extend([value - 2 * se, value + 2 * se])
        if metric in ('fit', 'suffix'):
            nll_extents.extend(extents)
        low, high = min(extents), max(extents)
        padding = .12 * max(high - low, .01)
        axis.set_xlim(low - padding, high + padding)
        axis.set_xlabel(xlabel, labelpad=9)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.tick_params(axis='y', length=0)
        axis.grid(axis='y', color='#EDF0F2', linewidth=.5, zorder=0)
    low, high = min(nll_extents), max(nll_extents)
    padding = .12 * max(high - low, .01)
    for axis in axes[:2]:
        axis.set_xlim(low - padding, high + padding)
    axes[0].set_yticks(range(len(subjects)), labels=[f'S{sid}' for sid in subjects])
    axes[0].set_ylim(len(subjects) - .5, -.5)
    legend = [Line2D([], [], marker='o', color=BLUE, linestyle='', markersize=4.5, label='Checks passed'),
              Line2D([], [], marker='o', markerfacecolor='white', color=GRAY, linestyle='', markersize=4.5,
                     label='Numerically unresolved'),
              Line2D([], [], marker='x', color=GRAY, linestyle='', markersize=4.5, label='Incomplete work')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .095), ncol=3, handletextpad=.5)
    fig.suptitle('SMOKE VALIDATION — not scientific evidence' if smoke else 'Execution persistence: six development participants',
                 fontsize=9, x=.09, y=.965, ha='left')
    fig.text(.09, .897, 'Free persistence minus zero inertia; negative values favor free persistence.', fontsize=7)
    fig.text(.09, .035, 'Bars: ±2 Monte Carlo SE. All paths start at trial 1; suffix Energy is unconditional.', fontsize=7)
    fig.savefig(output / 'human_comparison.png', dpi=600)
    plt.close(fig)
    return dict(rows=source_rows, participants_expected=len(subjects), records_plotted=sum(
                r['difference'] is not None and r['mcse'] is not None for r in source_rows))


def recovery_figure(rows: list[dict], policy: dict, output: Path, smoke: bool) -> dict:
    kappas = policy['recovery']['kappas']
    expected = len(policy['recovery']['subjects']) * policy['recovery']['replicates']
    labels = ['Zero', 'Positive', 'Unseparated /\nuncertain', 'Search\nunresolved', 'Missing work /\nchecks']
    if len({row['case'] for row in rows}) != len(rows):
        raise ValueError('Duplicate recovery cases')
    if any(number(row, 'generating_kappa') not in kappas for row in rows):
        raise ValueError('Recovery data contain an unregistered generating value')
    counts = np.zeros((len(kappas), len(labels)), dtype=int)
    for i, kappa in enumerate(kappas):
        own = [r for r in rows if number(r, 'generating_kappa') == kappa]
        if len(own) > expected:
            raise ValueError('Recovery count exceeds the registered path count')
        counts[i, -1] = expected - len(own)
        for row in own:
            if not flag(row, 'fit_complete'):
                column = 4
            else:
                column = {'zero': 0, 'positive': 1, 'unresolved': 2,
                          'search_unresolved': 3}.get(row.get('recovered_class'), 4)
            counts[i, column] += 1
    assert np.all(counts.sum(axis=1) == expected)
    fig, axis = plt.subplots(figsize=(183 / 25.4, 74 / 25.4))
    fig.subplots_adjust(left=.13, right=.98, bottom=.28, top=.72)
    axis.imshow(counts, cmap='Blues', vmin=0, vmax=max(1, expected), aspect='auto')
    for (i, j), value in np.ndenumerate(counts):
        axis.text(j, i, str(value), ha='center', va='center', fontsize=9,
                  color='white' if value > expected * .6 else '#202A33')
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(kappas)), labels=[f'κ = {k:g}' for k in kappas])
    axis.set_ylabel('Generating value')
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.suptitle('SMOKE VALIDATION — recovery workflow only' if smoke else 'Mechanism recovery: all registered synthetic paths',
                 fontsize=9, x=.13, y=.955, ha='left')
    fig.text(.13, .83, f'Counts across {expected} paths per generating value; no unresolved path is omitted.', fontsize=7)
    fig.text(.13, .055, 'Classes use frozen estimates, suffix prediction and numerical/search checks.\n'
             'Counts describe the registered backgrounds and replicates; they are not calibrated recovery rates.', fontsize=7)
    fig.savefig(output / 'recovery_counts.png', dpi=600)
    plt.close(fig)
    return dict(generating_kappas=kappas, labels=labels, counts=counts.tolist(), expected_per_value=expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Figure output must be new')
    protocol = json.loads((args.study / 'protocol.json').read_text())
    style()
    args.output.mkdir(parents=True)
    human_rows = read_csv(args.report / 'comparison.csv')
    humans = human_figure(human_rows, protocol['policy']['subjects'], args.output, protocol['smoke'])
    recovery = recovery_figure(read_csv(args.report / 'recovery.csv'), protocol['policy'], args.output, protocol['smoke'])
    inputs = {}
    for name in ('comparison.csv', 'recovery.csv'):
        source = args.report / name
        if source.exists():
            inputs[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            shutil.copy2(source, args.output / name)
    qa = dict(backend='python', matplotlib=matplotlib.__version__, output_format='PNG only',
        physical_width_mm=183, dpi=600, source_sha256=inputs,
        protocol_sha256=hashlib.sha256((args.study / 'protocol.json').read_bytes()).hexdigest(),
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        human=humans, recovery=recovery, observations_excluded=0,
        exclusions='Human and synthetic observations occupy separate figures. Missing values remain labeled.',
        inference='MCSE describes computation. Humans are development cases. No hypothesis test, p value or population CI.',
        export_exception='Repository instructions request PNG only; no redundant vector exports.',
        visual_review='pending; inspect the actual PNG outputs before final delivery')
    (args.output / 'figure_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    def counts(group: str, key: str) -> str:
        values = {int(number(row, key)) for row in human_rows
                  if row['group'] == group and number(row, key) is not None}
        return '、'.join(str(x) for x in sorted(values)) if values else '尚无已完成结果'

    captions = ['# 比较图说明', '',
        '**图1｜执行维持的逐人拟合与预测比较。** a，完整记录拟合的每题负对数似然（NLL）差。'
        'b，仅用前段估计并冻结参数后的条件后段NLL差。c，同一前段参数下，自主路径后段的Energy差。'
        '全部差值为κ自由模型减κ=0模型；负值表示自由模型评分较低。真人均为已查看的开发资料。', '',
        '每点对应一名被试。误差线为正负两倍蒙特卡洛标准误：NLL以完整独立粒子系统为刀切单位，'
        'Energy以同种子的完整配对路径为刀切单位。独立系统数和路径数逐点保存在comparison.csv。'
        '这些误差线描述数值计算误差；本图未进行人群假设检验或多重比较检验。', '',
        f'登记真人样本量n={len(protocol["policy"]["subjects"])}。'
        f'a栏每点独立粒子系统数为{counts("full", "fit_repeats")}；'
        f'b栏为{counts("prefix", "suffix_repeats")}；'
        f'c栏自由模型每点路径数为{counts("prefix", "free_paths")}，'
        f'零维持模型为{counts("prefix", "zero_paths")}。缺失被试在对应行保留。', '',
        '实心点表示该栏适用的登记检查通过：a检查拟合数值精度，b同时检查拟合和后段数值精度，'
        'c检查拟合数值精度。空心点保留数值未确定结果；叉号表示拟合计划或所需自主路径未完整执行。'
        '缺失分数标注not available。自主路径从任务首题开始生成，其后段分布未以真人前段状态为条件。', '',
        '**图2｜指定生成条件下的机制恢复计数。** 每行对应一个生成κ，汇总所有登记路径。'
        '列分别表示零维持、非零维持、未确定、搜索不足以及任务或检查缺失。'
        '已生成和未生成路径均保留在登记总数内；各行计数与生成计划一致。'
        '分类使用冻结参数、条件后段评分及拟合和评价的数值检查。生成点评分优于搜索结果时标记搜索不足。'
        '本轮每个生成参数点的重复数有限，计数用于描述指定背景下的结果。', '',
        f'恢复登记包含{len(protocol["policy"]["recovery"]["subjects"])}个任务背景，'
        f'每个生成参数点{protocol["policy"]["recovery"]["replicates"]}条独立路径；'
        f'每个κ合计{recovery["expected_per_value"]}条。每条路径为一个恢复案例。', '',
        '两图均由已保存的审查CSV生成，未删减被试或选择有利模拟路径。原CSV及其哈希与PNG一并保留。'
        '小规模流程的图仅用于计算与绘图验收，图题有SMOKE VALIDATION标识。', '']
    (args.output / 'captions.md').write_text('\n'.join(captions))


if __name__ == '__main__':
    main()
