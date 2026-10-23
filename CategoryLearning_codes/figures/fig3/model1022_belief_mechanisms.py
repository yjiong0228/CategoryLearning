"""Plot the complete five-mechanism, two-parameter-point development probe.

Figure contract: combining feedback-dependent retention and entry can improve
later behavior while costing training likelihood. A quantitative 2-by-3 grid
shows training NLL, later NLL and autonomous Energy for both fixed parameter
points. Every mechanism, complete filter and autonomous path is retained.
Style follows the existing Python comparison figures; the score layout is new.
This is a development figure, exported as PNG under the repository policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_1022_belief_fixed_references import scores
from src.Bayesian_state.workflows.analysis.model_1022_belief_summary import (
    digest, jackknife_se, load_case, nll, read,
)

MECHANISMS = ('SUPPORT', 'SLOT', 'OUTCOME', 'ENTRY', 'ADAPT')
LABELS = ('Support', 'Belief: slot transfer', 'Belief + retention',
          'Belief + entry', 'Belief + both changes')
COLORS = ('#417CA8', '#8565A5', '#A6A6A6', '#737373', '#2B8C6F')
POINTS = ('REFERENCE', 'JOINT')
ROOT = Path(__file__).resolve().parents[3]


def prepare(study: Path) -> tuple[pd.DataFrame, dict]:
    """Recompute all plotted estimates and check them against the saved diagnosis."""
    done, protocol = read(study / 'completion.json'), read(study / 'protocol.json')
    nomination = read(study / 'nominees.json')
    diagnosis = read(study / 'diagnostics_v2/completion.json')
    if (done['status'] != 'complete' or done['smoke'] or protocol['parameters_refitted']
            or nomination['status'] != 'frozen_before_test' or nomination['smoke']
            or nomination['later_choices_used'] or nomination['oral_used']
            or diagnosis['status'] != 'complete' or diagnosis['mechanism_selection_changed']
            or done['summary_sha256'] != digest(study / 'summary/summary.csv')):
        raise ValueError('Require a completed fixed-parameter scientific probe')
    jobs = read(study / 'autonomous_jobs.json')['jobs']
    diagnostic = pd.read_csv(study / 'diagnostics_v2/summary.csv').set_index(['cell', 'window'])
    nominees = {row['arm']: row for row in nomination['nominees']}
    cells = [m if (m, p) == ('SUPPORT', 'REFERENCE') else f'{m}_AT_{p}'
             for p in POINTS for m in MECHANISMS]
    if (len(nominees) != len(nomination['nominees']) or set(nominees) != set(cells)
            or len(diagnostic) != 20 or diagnostic.index.has_duplicates):
        raise ValueError('Require all ten unique cells and both scoring windows')
    rows, metadata = [], []
    for point in POINTS:
        candidate = None
        for mechanism in MECHANISMS:
            cell = mechanism if (mechanism, point) == ('SUPPORT', 'REFERENCE') else f'{mechanism}_AT_{point}'
            row = nominees[cell]
            if row['mechanism'] != mechanism:
                raise ValueError('Mechanism label differs from the frozen nomination')
            values = row['candidate']['hyperparams']
            if candidate is not None and values != candidate:
                raise ValueError('Mechanisms must share identical parameters within a row')
            candidate = values
            case = load_case(study, row, jobs)
            cut = case['meta']['training_trials']
            p = case['prediction']['observed_probability'][:, :cut]
            later = scores(case)
            train = dict(nll=nll(p), nll_mcse=jackknife_se(
                np.array([nll(np.delete(p, i, 0)) for i in range(len(p))])))
            for metric, window, field in (('training_nll', 'training', 'nll'),
                                           ('later_nll', 'test', 'nll'),
                                           ('later_energy', 'test', 'energy')):
                result = train if window == 'training' else later
                value, se = result[field], result[field + '_mcse']
                saved = diagnostic.loc[(cell, window), :]
                np.testing.assert_allclose([value, se], [saved[field], saved[field + '_mcse']],
                                           rtol=1e-11, atol=1e-13)
                if not np.isfinite([value, se]).all() or se < 0:
                    raise ValueError('Invalid plotted estimate or numerical standard error')
                rows.append(dict(subject=row['subject'], point=point, mechanism=mechanism, cell=cell,
                                 candidate=row['candidate']['id'], metric=metric, value=value, mcse=se))
            metadata.append(dict(subject=row['subject'], cell=cell, candidate=row['candidate']['id'],
                                 filters=later['filters'], particles=read(study / 'evaluate' /
                                 f"subject_{row['subject']}" / cell / (row['candidate']['id']+'.json'))['particles'],
                                 paths=later['paths'], training_trials=cut, test_trials=case['meta']['test_trials']))
    if len({r['subject'] for r in metadata}) != 1:
        raise ValueError('The probe figure requires one development participant')
    bound = ('completion.json', 'protocol.json', 'nominees.json', 'autonomous_jobs.json',
             'diagnostics_v2/summary.csv', 'diagnostics_v2/completion.json')
    return pd.DataFrame(rows), dict(cases=metadata, study=str(study),
        inputs_sha256={name: digest(study / name) for name in bound},
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
            (Path(__file__).resolve(), ROOT / 'src/Bayesian_state/evaluation/model_1022_belief_fixed_references.py',
             ROOT / 'src/Bayesian_state/workflows/analysis/model_1022_belief_summary.py',
             ROOT / 'src/Bayesian_state/metrics/autonomous_fit.py')})


def run(study: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new mechanism-figure directory')
    table, provenance = prepare(study)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.3), sharex='col', sharey=True)
    metrics = ('training_nll', 'later_nll', 'later_energy')
    titles = ('Training choice NLL', 'Later choice NLL', 'Later autonomous Energy')
    for r, point in enumerate(POINTS):
        for c, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[r, c]
            group = table[(table.point == point) & (table.metric == metric)].set_index('mechanism')
            ax.axvline(group.loc['SUPPORT', 'value'], ls=':', color='#888888', lw=.8, zorder=0)
            for y, mechanism in enumerate(MECHANISMS):
                item = group.loc[mechanism]
                ax.errorbar(item.value, y, xerr=2*item.mcse, fmt='o', color=COLORS[y],
                            ms=5, capsize=3, elinewidth=1.2)
            ax.set_yticks(np.arange(5), LABELS)
            ax.set_ylim(4.6, -.7)
            ax.tick_params(axis='y', length=0)
            if r == 0:
                ax.set_title(title, fontsize=10, pad=31)
            ax.set_xlabel('Lower is better')
            ax.text(-.04, 1.04, chr(ord('a') + 3*r+c), transform=ax.transAxes, fontweight='bold')
        origin = 'Support-selected parameters' if point == 'REFERENCE' else 'Belief-selected parameters'
        axes[r, 0].text(0, 1.15, origin, transform=axes[r, 0].transAxes, fontsize=10)
    sid = provenance['cases'][0]['subject']
    counts = sorted({(x['filters'], x['particles'], x['paths']) for x in provenance['cases']})
    if len(counts) != 1:
        raise ValueError('The displayed replicate budget must be common to every cell')
    filters, particles, paths = counts[0]
    fig.subplots_adjust(left=.20, right=.98, bottom=.19, top=.84, hspace=.55, wspace=.25)
    fig.suptitle(f'S{sid}: fixed-parameter mechanism comparison', y=.98, fontsize=12)
    fig.text(.20, .045, f'Bars: ±2 Monte Carlo SE; {filters} filters × {particles} particles; '
             f'{paths} autonomous paths per cell.\nDotted lines: support at the same parameter point. '
             'Development results; all five mechanisms retained.', fontsize=8)
    output.mkdir(parents=True)
    fig.savefig(output / 'mechanism_scores.png', dpi=300)
    plt.close(fig)
    table.to_csv(output / 'source_scores.csv', index=False)
    (output / 'figure_sources.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (output / 'README.md').write_text(
        '# 反馈调节机制的固定参数对照\n\n'
        '每行采用同一组固定参数，比较支持版及四种信念机制。左列为训练选择负对数似然（NLL），'
        '中列为后段选择 NLL，右列为自主行为的 Energy 距离；三个指标均为越低越好。'
        'Energy 比较每 32 题的正确率和各类别条件下的反应频率，两类特征各占一半权重。'
        '所有滤波重复与自主路径参与评分。\n\n'
        '紫色为替换时转移信念份额的模型，两行灰色分别加入反馈调节保持率、失败后的新规则试用份额，'
        '绿色同时采用两项修改。点线为同参数支持版。组合修改的后段表现伴随训练 NLL 的增加，'
        '其独立重新拟合另行评价。两组参数的编号保存在 source_scores.csv。\n\n'
        f'每个组合使用 {filters} 次完整滤波（每次 {particles} 个粒子）与 {paths} 条完整自主路径。'
        '误差棒为两倍模拟标准误，按完整滤波或完整路径计算，表示固定参数下的数值不确定性。'
        '一名开发被试及其后段已用于提出机制；这些误差棒不包含被试差异或选参不确定性。\n\n'
        '绘图从原始保存概率及路径重新计算全部 30 个分数和标准误，逐项核对已有诊断。'
        '源表共 20 行，图中选取训练 NLL、后段 NLL 与后段 Energy；全部 10 个机制／参数组合保留。'
        '原始文件保持不变。使用 Python，输出为 300 dpi PNG，供开发分析阅读。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)
