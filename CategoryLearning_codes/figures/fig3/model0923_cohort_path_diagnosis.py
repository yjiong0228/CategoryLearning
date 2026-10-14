"""Render the cohort's complete autonomous path distributions and mechanisms.

Figure contract: compare discovery/retention and generated behavior across all
available participants before prioritizing mechanisms. Quantitative grid: the
hero figure retains every path; a companion compares fit-level mechanism metrics.
Python only; screen-review PNGs (12 x 10 inches, 300 dpi), source CSVs and hashes.
15 participants / 17 fitted points / 64 paths per point. Paths are stochastic
replicates at fixed parameters. No population tests or confidence intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
    'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
BLUE, ORANGE, GREY = '#3274A1', '#C47D33', '#B5BDC5'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source: Path, output: Path, report_name: str = 'index.html') -> None:
    if Path(report_name).name != report_name or not report_name.endswith('.html'):
        raise ValueError('report_name must be an HTML filename')
    if output.exists() or (source / report_name).exists():
        raise FileExistsError('Choose a new figure output and an analysis without a report')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['smoke']:
        raise ValueError('A complete full-sequence cohort analysis is required')
    summary = pd.read_csv(source / 'subjects.csv')
    curves = pd.read_csv(source / 'curves.csv')
    paths = pd.read_csv(source / 'paths.csv')
    for name in ('subjects.csv', 'curves.csv', 'paths.csv', *[f'{i}/correct_paths.npz' for i in summary.id]):
        if digest(source / name) != manifest['artifact_sha256'][name]:
            raise ValueError(f'Analysis artifact changed: {name}')
    output.mkdir(parents=True)
    summary.to_csv(output / 'mechanisms_plot_source.csv', index=False)
    curves.to_csv(output / 'curves_plot_source.csv', index=False)
    realized = paths.groupby('id').late_realized.quantile([.1, .5, .9]).unstack()
    realized.columns = ['realized_q10', 'realized_q50', 'realized_q90']
    realized = realized.join(summary.set_index('id')[['primary', 'late_observed']])
    realized.to_csv(output / 'realized_accuracy_plot_source.csv')
    background = []
    figures = []
    for primary, name in ((True, 'primary_trajectories'), (False, 'version_sensitivity_trajectories')):
        rows = summary.loc[summary.primary == primary].sort_values(['dataset', 'condition', 'subject'], ascending=[True, True, True])
        ncols = 4 if primary else 2
        nrows = int(np.ceil(len(rows)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.4*nrows+1), squeeze=False)
        for ax, row in zip(axes.flat, rows.itertuples()):
            frame = curves.loc[curves.id == row.id]
            with np.load(source / row.id / 'correct_paths.npz') as z:
                correct = z['correct_probability']
            x = (frame.first_trial.to_numpy()+frame.last_trial.to_numpy())/2
            blocked = np.stack([correct[int(r.first_trial)-1:int(r.last_trial)].mean(axis=0) for r in frame.itertuples()])
            ax.plot(x, blocked, color=GREY, alpha=.3, linewidth=.45, zorder=1)
            for index, r in enumerate(frame.itertuples()):
                for path in range(correct.shape[1]):
                    background.append(dict(id=row.id, path=path, first_trial=r.first_trial,
                                           last_trial=r.last_trial, expected_correct=blocked[index, path]))
            ax.plot(x, frame.autonomous, color=BLUE, linewidth=2, zorder=3)
            ax.plot(x, frame.observed, color='#202020', linewidth=1.5, zorder=4)
            if frame.conditional.notna().any():
                ax.plot(x, frame.conditional, color=ORANGE, linestyle='--', linewidth=1.3, zorder=3)
            ax.axhline(.5 if row.condition == 1 else .25, color='#888888', linestyle=':', linewidth=.6)
            ax.set(title=f"S{row.subject} | {row.dataset} C{row.condition}"+(' | initial' if not primary else ''),
                   ylim=(0, 1.02), xlim=(1, row.trials), xlabel='Valid trial', ylabel='Correct probability / accuracy')
            ax.set_yticks([0, .5, 1])
            ax.tick_params(labelsize=8)
        for ax in axes.flat[len(rows):]:
            ax.set_visible(False)
        fig.legend(handles=[Line2D([], [], color=GREY, label='All autonomous paths (64 per fit)'),
            Line2D([], [], color=BLUE, lw=2, label='Autonomous mean'),
            Line2D([], [], color='#202020', label='Observed accuracy'),
            Line2D([], [], color=ORANGE, ls='--', label='Conditional prediction, when saved')],
            loc='upper center', ncol=2, bbox_to_anchor=(.5, .98))
        fig.subplots_adjust(left=.07, right=.98, top=.88 if primary else .70, bottom=.07 if primary else .19, hspace=.65, wspace=.40)
        file = output / (name+'.png')
        fig.savefig(file, dpi=300, facecolor='white')
        plt.close(fig)
        figures.append(file)
    pd.DataFrame(background).to_csv(output / 'all_paths_plot_source.csv', index=False)

    rows = summary.sort_values(['dataset', 'condition', 'subject', 'primary'], ascending=[True, True, True, False]).reset_index(drop=True)
    y = np.arange(len(rows))
    labels = [f"{r.dataset} C{r.condition}  S{r.subject}"+(' *initial' if not r.primary else '') for r in rows.itertuples()]
    fig, axes = plt.subplots(1, 4, figsize=(12, 9), sharey=True)
    axes[0].scatter(rows.ever_active, y-.13, color=BLUE, s=24, label='Ever present')
    axes[0].scatter(rows.late_active, y+.13, color=ORANGE, s=22, marker='s', label='Late presence')
    axes[0].set_title('a  Discovery and presence', loc='left', fontsize=10)
    axes[1].scatter(rows.retention_fraction, y, color=BLUE, s=24, label='Retained >= 8 trials')
    axes[1].set_title('b  Set retention', loc='left', fontsize=10)
    axes[2].hlines(y, rows.late_correct_q10, rows.late_correct_q90, color=BLUE, linewidth=1.4)
    axes[2].scatter(rows.late_correct, y, color=BLUE, s=22, label='Autonomous: mean, 10–90%')
    axes[2].scatter(rows.late_observed, y, color='#202020', marker='|', s=65, label='Observed')
    axes[2].scatter(rows.late_conditional, y, color=ORANGE, marker='x', s=24, label='Conditional')
    axes[2].set_title('c  Late performance', loc='left', fontsize=10)
    axes[3].scatter(rows.high_support_exit_hazard, y-.13, color=BLUE, s=24, label='Posterior >= 0.5')
    axes[3].scatter(rows.low_support_exit_hazard, y+.13, color=ORANGE, s=22, marker='s', label='Posterior < 0.5')
    axes[3].set_title('d  Next-trial exit risk', loc='left', fontsize=10)
    axes[0].set_yticks(y, labels)
    for ax in axes:
        ax.invert_yaxis() if not ax.yaxis_inverted() else None
        ax.set_xlim(-.03, 1.03)
        ax.set_xticks([0, .5, 1])
        ax.grid(axis='y', color='#eeeeee', linewidth=.6)
        ax.set_axisbelow(True)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.06), fontsize=8)
        ax.set_xlabel('Proportion')
    axes[3].set_xlim(-.02, max(.2, rows[['high_support_exit_hazard', 'low_support_exit_hazard']].max().max()*1.15))
    axes[3].set_xticks(np.arange(0, axes[3].get_xlim()[1], .1))
    fig.suptitle('Rule discovery, persistence and autonomous performance across fitted participants', fontsize=12, y=.975)
    fig.subplots_adjust(left=.21, right=.98, top=.92, bottom=.16, wspace=.24)
    file = output / 'mechanism_comparison.png'
    fig.savefig(file, dpi=300, facecolor='white')
    plt.close(fig)
    figures.append(file)

    panels = ''.join(f'<h2>{html.escape(p.stem)}</h2><img src="{html.escape(os.path.relpath(p, source))}">' for p in figures)
    columns = ['id', 'primary', 'trials', 'paths', 'rules', 'ever_active', 'late_active', 'retention_fraction',
               'late_correct', 'late_observed', 'late_conditional', 'autonomous_exits', 'supported_autonomous_exits',
               'positive_feedback_support_exits', 'capacity_exits']
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Model 0923 全被试路径诊断</title>
<style>body{font:16px/1.65 sans-serif;margin:32px auto;max-width:1500px;padding:0 24px;color:#222}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}section{overflow:auto}a{color:#28688b}</style>
<h1>Model 0923：15名被试、17份拟合的自主路径诊断</h1>
<p>每份拟合64条路径，共1088条。主汇总每人保留一份拟合，307与335的初始拟合作为版本敏感性检查。exp123三种反馈条件与MEG分别汇总。所有路径在原参数、原代码下执行。</p>
<p>原12人的768条自主路径逐条重放；Plus两人的128条轨迹复用已核验记录；旧MEG三人的192条路径在原参数下补生成。此分析没有重拟合。</p>
<p>灰线显示全部自主路径，每32个有效试次计算一次平均正确概率；蓝线为路径均值，黑线为被试正确率，橙色虚线为已保存的conditional prediction。旧MEG拟合未保存相同协议的conditional数据，因此这三份结果的该项为空。</p>
<p>任务相容规则按原始物理刺激上的最高概率类别命中率≥0.8事后确定。规则集合在相容规则相互替换时仍视为存在；保留率使用有足够后续观察的进入事件。末64题定义后期。退出风险的分母是可观察下一次转换的规则—试次，按前一题反馈后的信念是否≥0.5分层。</p>
<p>分布宽度表示固定参数下的随机路径差异。窄分布的价值取决于其位置和对数据的覆盖。跨被试均值按人等权，路径数与规则退出次数均不充当被试样本量。初始搜索和未解决的Plus拟合状态原样保留；机制因果效果与参数可解释性仍须对照及恢复检验。</p>
<p>统一证据提示三个应分别检验的环节：合适规则的发现、已发现规则的持续保留、容量满时的新规则准入。S206的目标规则发现率98.4%，后期存在率2.3%；S222自主退出参数为0，仍有30次目标规则的容量淘汰。S335 Plus与S336的发现率分别为7.8%与17.2%。这些现象确定了干预位置，各项机制修改的效果仍需受控比较。</p>
<p><a href="subjects.csv">逐拟合指标</a> · <a href="strata.csv">分层汇总</a> · <a href="threshold_sensitivity.csv">规则阈值敏感性</a> · <a href="inventory.csv">纳入清单</a> · <a href="validation.json">重放校验</a></p>'''
    note = Path('src/Bayesian_state/docs/model_architecture/model_0923_cohort_path_diagnosis.md').resolve()
    page += f'<p><a href="{html.escape(os.path.relpath(note, source))}">机制诊断结论与下一步方案</a></p>'
    page += panels+'<h2>逐拟合数值</h2><section>'+rows[columns].to_html(index=False, float_format=lambda x:f'{x:.4f}', na_rep='未定义 / 未保存')+'</section>'
    page += '<h2>实际作答正确率的路径分布</h2><p>此表包含模型采样作答的变异；上图的细线表示各路径的平均正确概率。判断被试正确率是否落在生成范围内时，应同时查看此表。这里报告描述性分位数，尚未进行留出校准检验。</p><section>'
    page += realized.to_html(float_format=lambda x:f'{x:.4f}')+'</section></html>'
    with (source / report_name).open('x', encoding='utf-8') as stream:
        stream.write(page)
    contract = dict(backend='python', archetype='quantitative_grid', export='PNG only; repository instruction',
        conclusion='Compare the distribution of autonomous learning and rule lifecycle across every available participant.',
        participants=15, fit_records=17, paths_per_fit=64, intervals='10–90% fixed-parameter path variation; not confidence intervals',
        aggregation='One primary fitted point per participant; dataset and feedback condition remain separate',
        source_sha256={str(source / name): digest(source / name) for name in ('manifest.json', 'subjects.csv', 'curves.csv')},
        script_sha256=digest(Path(__file__)), artifact_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()})
    with (output / 'figure_contract.json').open('x') as stream:
        json.dump(contract, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-name', default='index.html', help='New HTML filename inside the analysis directory')
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve(), args.report_name)
