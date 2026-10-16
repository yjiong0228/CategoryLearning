"""Explain frozen parameter selection using every common final candidate.

Run as a module from the repository root with --fit, --features and a fresh
--output directory. Reads training confirmations only; produces Chinese PNG
review figures, an HTML reading guide, source CSVs and a provenance manifest.
The existing pooled-probability NLL and feature-energy definitions are reused.
No model, parameter, train/test boundary or selection rule is changed.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy

from src.Bayesian_state.evaluation.model_0923_h_objective_comparison import feature_jackknife
from src.Bayesian_state.evaluation.model_0923_h_validation import (
    block_paths, digest, jackknife_scores, jackknife_se, load, nll,
)
from src.Bayesian_state.metrics.autonomous_fit import behavior_features, energy_u

ARMS = ('NLL', 'GEN')
COLORS = {'NLL': '#0072B2', 'GEN': '#D55E00'}
CONTRACT = dict(
    conclusion='候选胜出取决于预先指定的目标；评分优势与数值近似并列需要区分。',
    archetype='quantitative grid', backend='python', width_mm=183,
    panels={'a': 'all candidates: paired NLL differences',
            'b': 'same candidates: paired autonomous feature-energy differences',
            'c': 'cumulative pre-choice log-score advantage of NLL selection',
            'd': 'all autonomous paths summarized as block means and pointwise quantiles'},
    reuse_level='style-only inheritance from model0923_h_objective_comparison.py; new layout',
    exclusions='Earlier stages have different budgets. Every common final candidate and every saved path is retained.',
    selection_scope='Finite-budget final candidate set; training scores used for selection, not independent validation.',
    uncertainty='Score bars: +/- 2 paired Monte Carlo SE. Bands: pointwise 10th--90th path percentiles.',
    independent_units='4 complete filter runs for NLL; 1024 complete autonomous paths for GEN, per candidate.',
    tests='No population significance test. Monte Carlo error is not parameter uncertainty or particle-bias estimation.',
    export='PNG only, 300 dpi scientific review; editable Python source retained.',
    static_export_exceptions=['Vector/TIFF omitted under repository PNG-only instruction.',
                              'These are review figures; journal submission export compliance is not claimed.'],
)


def cumulative_advantage(reference: np.ndarray, alternative: np.ndarray) -> np.ndarray:
    """Pool independent filters before logging, as in the frozen NLL protocol.

    Positive values mean the reference has accumulated a higher choice log
    score. Averaging filter log scores would define a different objective.
    """
    if reference.shape != alternative.shape or reference.ndim != 2 or not reference.size:
        raise ValueError('Aligned, nonempty filter-by-trial probabilities required')
    for probability in (reference, alternative):
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError('Probabilities must be finite and between zero and one')
    floor = np.finfo(float).tiny
    return np.cumsum(np.log(np.maximum(reference.mean(axis=0), floor))
                     - np.log(np.maximum(alternative.mean(axis=0), floor)))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def configure_style() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    name = next((name for name in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei')
                 if name in available), None)
    if name is None:
        raise RuntimeError('A Chinese sans-serif font is required for these review figures')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [name, 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'legend.fontsize': 6,
        'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    return name


def read_candidates(fit: Path, features: Path) -> tuple[dict, dict, pd.DataFrame, dict]:
    """Check frozen sources and independently reproduce all final candidate scores."""
    frozen = load(fit/'nominees.json')
    protocol = load(fit/'search_protocol.json')
    completion = load(features/'completion.json')
    recorded_inputs = {Path(name).resolve(): sha for name, sha in completion['inputs_sha256'].items()}
    assert frozen['status'] == 'frozen_before_test' and not frozen['smoke']
    assert load(fit/'completion.json')['status'] == completion['status'] == 'complete'
    assert completion['nomination_sha256'] == digest(fit/'nominees.json')
    table = pd.read_csv(features/'confirmation_candidates.csv')
    nominees = {(r['subject'], r['model']): r['candidate']['id'] for r in frozen['nominees']}
    data = Path(frozen['data'])
    assert digest(data/'preparation_manifest.json') == protocol['data_manifest_sha256']
    inputs = {str(p): digest(p) for p in (fit/'nominees.json', fit/'search_protocol.json',
              features/'completion.json', features/'confirmation_candidates.csv', data/'preparation_manifest.json')}
    subjects = {}
    for subject in protocol['subjects']:
        meta = load(data/f'subject_{subject}/prepared.json')
        training_file = data/f'subject_{subject}/training.npz'
        assert digest(training_file) == meta['array_sha256']['training.npz']
        inputs[str(training_file)] = digest(training_file)
        inputs[str(training_file.with_name('prepared.json'))] = digest(training_file.with_name('prepared.json'))
        with np.load(training_file) as saved:
            observed, categories = saved['choices'], saved['categories']
        frame = table[table.subject == subject]
        ordered = frame[frame.model == 'NLL'].sort_values(['objective_score', 'candidate']).candidate.tolist()
        assert set(ordered) == set(frame[frame.model == 'GEN'].candidate)
        assert len(frame) == 2*len(ordered) and len(ordered) == len(set(ordered))
        assert ordered[0] == nominees[(subject, 'NLL')]
        candidates = []
        other_index = 0
        for index, candidate_id in enumerate(ordered):
            point = dict(id=candidate_id, letter=chr(65+index), scores={}, deleted={}, seeds={},
                         selected_by=[arm for arm in ARMS if nominees[(subject, arm)] == candidate_id])
            if point['selected_by']:
                point['color'], point['style'] = COLORS[point['selected_by'][0]], '-'
            else:
                point['color'], point['style'] = [('#777777', '--'), ('#B0B0B0', ':')][other_index]
                other_index += 1
            for arm in ARMS:
                file = fit/'confirmation'/f'subject_{subject}'/arm/(candidate_id+'.json')
                row = load(file)
                assert recorded_inputs[file.resolve()] == digest(file)
                assert digest(file.with_suffix('.npz')) == row['predictions_sha256']
                inputs[str(file)] = digest(file)
                inputs[str(file.with_suffix('.npz'))] = row['predictions_sha256']
                assert row['trials'] == len(observed)
                assert (row['role'] == 'training' if arm == 'NLL' else row['data_boundary'] == 'training_only')
                if 'hyperparams' in point:
                    assert point['hyperparams'] == row['candidate']['hyperparams']
                point['hyperparams'] = row['candidate']['hyperparams']
                assert len(point['hyperparams']) == 14
                with np.load(file.with_suffix('.npz')) as saved:
                    arrays = {key: saved[key] for key in saved.files}
                point['seeds'][arm] = arrays['seeds']
                if arm == 'NLL':
                    assert arrays['observed_probability'].shape == (4, len(observed))
                    assert row['particles'] == 256
                    value = nll(arrays['observed_probability'])
                    deleted = jackknife_scores(arrays['observed_probability'])
                    np.testing.assert_allclose(value, row['mean_nll'], atol=1e-13, rtol=0)
                    point['probability'] = arrays['observed_probability']
                else:
                    assert arrays['choices'].shape == (1024, len(observed))
                    np.testing.assert_array_equal(arrays['categories'], categories)
                    kwargs = dict(n_categories=2 if meta['condition'] == 1 else 4, **frozen['settings']['features'])
                    np.testing.assert_allclose(behavior_features(observed[None, :], categories, **kwargs)[0],
                                               arrays['observed_features'], atol=1e-13, rtol=0)
                    np.testing.assert_allclose(behavior_features(arrays['choices'], categories, **kwargs),
                                               arrays['features'], atol=1e-13, rtol=0)
                    value = energy_u(arrays['features'], arrays['observed_features'])['value']
                    deleted = feature_jackknife(arrays['features'], arrays['observed_features'])
                    np.testing.assert_allclose(value, row['objective_score'], atol=1e-13, rtol=0)
                    point['paths'] = block_paths(arrays['choices'] == categories[None, :], 0, len(observed),
                                                width=frozen['settings']['features']['block_width'])
                point['scores'][arm], point['deleted'][arm] = value, deleted
                old = frame[(frame.model == arm) & (frame.candidate == candidate_id)].iloc[0]
                np.testing.assert_allclose([value, jackknife_se(deleted)], [old.objective_score, old.mc_se], atol=1e-13, rtol=0)
            candidates.append(point)
        for point in candidates:
            point['delta'], point['delta_se'] = {}, {}
            for arm in ARMS:
                best = next(p for p in candidates if arm in p['selected_by'])
                np.testing.assert_array_equal(point['seeds'][arm], best['seeds'][arm])
                delta = point['scores'][arm] - best['scores'][arm]
                se = jackknife_se(point['deleted'][arm] - best['deleted'][arm])
                point['delta'][arm], point['delta_se'][arm] = delta, se
                old = frame[(frame.model == arm) & (frame.candidate == point['id'])].iloc[0]
                np.testing.assert_allclose([delta, se], [old.delta_from_selected, old.paired_delta_mc_se], atol=1e-13, rtol=0)
            point['cumulative'] = cumulative_advantage(candidates[0]['probability'], point['probability'])
            np.testing.assert_allclose(point['cumulative'][-1], len(observed)*point['delta']['NLL'], atol=1e-10, rtol=0)
        subjects[subject] = dict(candidates=candidates, trials=len(observed),
            observed=block_paths((observed == categories)[None, :], 0, len(observed),
                                 width=frozen['settings']['features']['block_width'])[0])
    assert len(table) == sum(2*len(s['candidates']) for s in subjects.values())
    return subjects, inputs, table, frozen['settings']


def label(point: dict) -> str:
    role = ' / '.join(point['selected_by']) + '所选' if point['selected_by'] else '其他候选'
    return f"{point['letter']}：{role}"


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output/(name+'.png'), dpi=300, facecolor='white')
    plt.close(fig)


def plot_subject(subject: int, info: dict, output: Path) -> None:
    points = info['candidates']
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 161/25.4))
    fig.subplots_adjust(left=.095, right=.98, bottom=.23, top=.90, hspace=.48, wspace=.38)
    for ax, arm in zip(axes[0], ARMS):
        for i, point in enumerate(points):
            ax.errorbar(point['delta'][arm], i, xerr=2*point['delta_se'][arm],
                        fmt='o' if arm == 'NLL' else 's', ms=4, color=point['color'], capsize=2, lw=.8)
        ax.axvline(0, color='#333333', ls=':', lw=.7)
        ax.set(yticks=range(len(points)), yticklabels=[p['letter'] for p in points],
               ylim=(len(points)-.5, -.5), ylabel='整组参数候选',
               xlabel='与该目标所选候选的分差（越小越好）')
        ax.set_title('选择似然：平均 NLL' if arm == 'NLL' else '自主行为：特征能量分数 GEN', loc='left')
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.locator_params(axis='x', nbins=4)
    ax = axes[1, 0]
    for point in points[1:]:
        ax.plot(np.arange(1, info['trials']+1), point['cumulative'], color=point['color'],
                ls=point['style'], lw=1.1, label=point['letter'])
    ax.axhline(0, color=points[0]['color'], ls=':', lw=.8)
    ax.set(title='A 的似然优势如何累积', xlabel='训练试次', ylabel='累计 log p(A) − log p(其他候选)')
    ax.text(.03, .97, '高于 0：A 领先', transform=ax.transAxes, va='top', fontsize=6)
    ax = axes[1, 1]
    x = np.arange(1, len(info['observed'])+1)
    for point in points:
        paths = point['paths']
        low, high = np.quantile(paths, [.1, .9], axis=0)
        ax.fill_between(x, low, high, color=point['color'], alpha=.11, linewidth=0)
        ax.plot(x, paths.mean(axis=0), color=point['color'], ls=point['style'], lw=1.1)
    ax.plot(x, info['observed'], color='#222222', marker='.', ms=3, lw=1.1)
    ax.set(title='从任务起点自主生成', xlabel='训练区块（每块 32 试次）', ylabel='正确率', ylim=(-.03, 1.03))
    ax.locator_params(axis='x', integer=True, nbins=5)
    for i, ax in enumerate(axes.flat):
        ax.text(-.16, 1.06, 'abcd'[i], transform=ax.transAxes, fontsize=8, fontweight='bold')
    fig.suptitle(f'被试 {subject}｜同一批候选，按什么目标胜出？', y=.97, fontsize=10)
    handles = [Line2D([], [], color=p['color'], ls=p['style'], label=label(p)) for p in points]
    handles.append(Line2D([], [], color='#222222', marker='.', label='真实行为'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .105), ncol=3)
    fig.text(.5, .065, 'a、b：误差条为 ±2 配对 Monte Carlo SE；参照候选的分差恒为 0。', ha='center', fontsize=6)
    fig.text(.5, .041, 'd：每候选 1024 条路径；线为均值，色带为逐区块 10%–90% 分位数。', ha='center', fontsize=6)
    fig.text(.5, .017, '全部面板使用训练段。候选由同批评分选出；误差条反映计算波动，参数不确定性需另行估计。', ha='center', fontsize=6)
    save(fig, output, f'subject_{subject}')


def plot_overview(subjects: dict, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(183/25.4, 172/25.4))
    fig.subplots_adjust(left=.12, right=.96, bottom=.14, top=.91, hspace=.62, wspace=.40)
    for i, (subject, info) in enumerate(subjects.items()):
        ax = axes.flat[i]
        for point in info['candidates']:
            x, y = point['delta']['NLL'], point['delta']['GEN']
            ax.errorbar(x, y, xerr=2*point['delta_se']['NLL'], yerr=2*point['delta_se']['GEN'],
                        fmt='o', color=point['color'], ms=4, lw=.7, capsize=2)
            ax.annotate(point['letter'], (x, y), xytext=(4, 5 if not point['selected_by'] else -10),
                        textcoords='offset points', color=point['color'], fontsize=7)
        ax.axvline(0, color='#BBBBBB', lw=.5, ls=':')
        ax.axhline(0, color='#BBBBBB', lw=.5, ls=':')
        ax.set(title=f'被试 {subject}', xlabel='ΔNLL（相对 NLL 所选）', ylabel='ΔGEN（相对 GEN 所选）')
        ax.margins(.22)
        ax.locator_params(nbins=4)
        ax.ticklabel_format(style='plain', useOffset=False)
        ax.text(-.2, 1.07, 'abcdef'[i], transform=ax.transAxes, fontsize=8, fontweight='bold')
    fig.suptitle('全部 23 组候选｜两种目标如何取舍', y=.975, fontsize=10)
    fig.legend(handles=[Line2D([], [], marker='o', lw=0, color=c, label=t) for c, t in
               [(COLORS['NLL'], 'NLL 所选'), (COLORS['GEN'], 'GEN 所选'), ('#888888', '其他候选')]],
               loc='lower center', bbox_to_anchor=(.5, .055), ncol=3)
    fig.text(.5, .032, '越靠左，选择似然越好；越靠下，自主特征分布得分越好。各被试坐标尺度独立。', ha='center', fontsize=6)
    fig.text(.5, .011, '误差条：±2 配对 Monte Carlo SE；反映同批选择评分的计算波动。A–D 为各被试内部候选编号。', ha='center', fontsize=6)
    save(fig, output, 'overview')


def export_sources(subjects: dict, output: Path) -> pd.DataFrame:
    scores, trials, trajectories, parameters, margins = [], [], [], [], []
    for subject, info in subjects.items():
        for point in info['candidates']:
            for arm in ARMS:
                scores.append(dict(subject=subject, letter=point['letter'], candidate=point['id'], objective=arm,
                    selected=arm in point['selected_by'], value=point['scores'][arm],
                    delta=point['delta'][arm], paired_mc_se=point['delta_se'][arm]))
            pooled = point['probability'].mean(axis=0)
            trials.extend(dict(subject=subject, letter=point['letter'], trial=t+1,
                observed_choice_probability=float(pooled[t]), cumulative_nll_reference_advantage=float(v))
                for t, v in enumerate(point['cumulative']))
            low, high = np.quantile(point['paths'], [.1, .9], axis=0)
            trajectories.extend(dict(subject=subject, letter=point['letter'], block=b+1,
                observed=info['observed'][b], mean=value, q10=low[b], q90=high[b], paths=len(point['paths']))
                for b, value in enumerate(point['paths'].mean(axis=0)))
            parameters.extend(dict(subject=subject, letter=point['letter'], candidate=point['id'],
                selected_by='/'.join(point['selected_by']), parameter=name, value=value)
                for name, value in point['hyperparams'].items())
        gen = next(p for p in info['candidates'] if 'GEN' in p['selected_by'])
        margins.append(dict(subject=subject, trials=info['trials'], candidates=len(info['candidates']),
            nll_selected=info['candidates'][0]['id'], gen_selected=gen['id'],
            gen_minus_nll_score=gen['delta']['NLL'], paired_mc_se=gen['delta_se']['NLL'],
            total_log_score_advantage=gen['cumulative'][-1],
            alternatives_with_nll_gap_within_2_mc_se=sum(p['delta']['NLL'] <= 2*p['delta_se']['NLL']
                                                       for p in info['candidates'][1:])))
    for name, rows in [('candidate_scores', scores), ('trial_log_scores', trials),
                       ('autonomous_accuracy', trajectories), ('candidate_parameters', parameters), ('selection_margins', margins)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'), index=False)
    return pd.DataFrame(margins)


def report(subjects: dict, margins: pd.DataFrame, output: Path) -> None:
    parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Model 0923 个体参数为何胜出</title>
<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#222}
img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:5px 10px;border:1px solid #ddd}
code{overflow-wrap:anywhere}h2{margin-top:44px}a{color:#0072B2}</style>
<h1>个体参数估计：胜出目标与证据强度</h1>
<p><strong>建议以整段选择序列的边缘似然为主要选参目标，配合自主生成检验、留出预测和参数恢复。</strong>
边缘似然把可能的内部状态积分掉。逐试次预测真实回答的概率在读入本次回答前计算，其负对数之和就是 NLL。
当前比较使用每试次平均 NLL；同一被试内它与总 NLL 的排名相同。</p>
<p>GEN 用自主路径的区块正确率和按真实类别分组的回答频率计算特征能量分数。
这能约束整体行为分布，但压缩了逐试次回答信息。GEN 的改善应作为机制诊断证据；当前恢复结果尚不足以支持用它替代 NLL 解释个体特质。</p>
<p>本页重用既有共同候选确认结果：6 名被试、23 组候选、14 个联合估计坐标。
NLL 每点 4 个独立滤波种子，每次 256 粒子；GEN 每点 1024 条独立自主路径。
候选间配对使用相同种子。较早阶段预算不同，全部最终候选在这里按共同预算比较。</p>
<p><strong>本页全部图使用训练段，所选候选也来自同批评分。</strong>
误差条为 ±2 配对 Monte Carlo 标准误差，显示有限次模拟的波动。
它不覆盖粒子数有限产生的系统偏差，也不估计参数不确定性；4 次滤波对误差的估计本身仍较粗。
这些图说明有限候选中的选择依据，整体机制和参数解释仍需独立验证。</p>
<h2>全部被试</h2><img src="overview.png" alt="六名被试共同候选的两种目标分差">
<p>各坐标都相对于本目标所选候选。原点代表同时达到两种已评候选最小值；本次两种目标选出的参数均不同。
同一被试内 A–D 按 NLL 从低到高编号，候选包含整组参数。</p>''']
    summary = margins[['subject', 'trials', 'candidates', 'gen_minus_nll_score', 'paired_mc_se',
                       'total_log_score_advantage', 'alternatives_with_nll_gap_within_2_mc_se']].copy()
    summary.columns = ['被试', '训练试次', '候选数', 'GEN所选减NLL所选：平均NLL', '配对MC SE', '累计log得分优势', 'NLL分差≤2MC SE的其他候选数']
    parts.append(summary.to_html(index=False, float_format=lambda x: f'{x:.6f}'))
    parts.append('<p>最后一列是数值分辨率提示。它不是显著性检验，也不构成参数置信集合。</p>')
    for subject, info in subjects.items():
        parts.append(f'<h2 id="subject_{subject}">被试 {subject}</h2><img src="subject_{subject}.png" alt="被试{subject}全部候选的选择证据">')
        parts.append('<p>a、b 比较同一批候选在两种目标下的排名；c 将 A 相对每个其他候选的逐试次 log 概率差累加。'
                     'd 从任务起点自主生成，以全部路径计算均值和逐区块分位数。分位色带不表示整条轨迹落入的概率；GEN 还使用回答频率特征。</p>')
        if subject == 207:
            parts.append('<p><strong>207：A 在 NLL 上胜出，D 在自主行为上胜出。</strong> A 对整段回答积累更高的选择 log 得分，'
                         '但 D 的自主学习曲线更贴近被试。这个分歧要求继续检验机制与似然计算精度。当前证据支持 A 的选择目标排名，'
                         '对 A 的学习过程解释仍有明显缺口。</p>')
        if subject == 305:
            parts.append('<p><strong>305：A 是本次 NLL 数值最小值，但其他候选的分差均未超过两倍配对计算标准误。</strong>'
                         '当前预算下排序分辨率不足。应追加独立种子、检查粒子数收敛，再报告参数范围或多个近优点。</p>')
        points = info['candidates']
        rows = [dict(编号=p['letter'], 候选ID=p['id'], 所选目标='/'.join(p['selected_by']) or '—',
                     NLL=p['scores']['NLL'], GEN=p['scores']['GEN']) for p in points]
        parts.append(pd.DataFrame(rows).to_html(index=False, float_format=lambda x: f'{x:.6f}'))
        parts.append('<details><summary>查看全部14个坐标</summary>')
        parameter_table = pd.DataFrame({p['letter']: p['hyperparams'] for p in points})
        parts.append(parameter_table.to_html(float_format=lambda x: f'{x:g}'))
        parts.append('</details>')
    parts.append('''<h2>下一步如何使用这些结果</h2><ol>
<li>保持训练选择 NLL 为主目标。对近优候选增加独立计算精度，并分别检查优化是否充分与粒子数是否收敛。</li>
<li>用留出选择、自主轨迹和任务停止时间检验机制。如果 NLL 良好但自主生成持续偏离，定位这一分歧；此时参数含义需要保留。</li>
<li>通过联合参数恢复、近优区域和重复数据检验可识别性。只有稳定可恢复的坐标才支持强个体特质解释。</li>
<li>若引入先验或层级估计，事先规定其形式与训练数据边界，再检查恢复和先验敏感性。当前图展示既有点估计。</li></ol>
<h2>复现与数据</h2><ul>''')
    for name in ('candidate_scores.csv', 'trial_log_scores.csv', 'autonomous_accuracy.csv', 'candidate_parameters.csv',
                 'selection_margins.csv', 'figure_contract.json', 'completion.json', 'plot_source.py'):
        parts.append(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>')
    parts.append('</ul></html>')
    (output/'index.html').write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing research outputs are retained')
    args.output.mkdir(parents=True)
    write_json(args.output/'figure_contract.json', CONTRACT)
    shutil.copy2(__file__, args.output/'plot_source.py')
    font = configure_style()
    subjects, inputs, table, settings = read_candidates(args.fit, args.features)
    for subject, info in subjects.items():
        plot_subject(subject, info, args.output)
    plot_overview(subjects, args.output)
    margins = export_sources(subjects, args.output)
    report(subjects, margins, args.output)
    write_json(args.output/'completion.json', dict(status='complete', subjects=list(subjects),
        candidates=sum(len(s['candidates']) for s in subjects.values()), scores_checked=len(table),
        source_sha256=digest(Path(__file__)), inputs_sha256=inputs, settings=settings,
        test_choices_used=False, new_simulations=0, checks='All scores/paired SE match frozen tables; all cumulative endpoints equal T times NLL differences; all feature arrays reproduce from choices.',
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      matplotlib=matplotlib.__version__, scipy=scipy.__version__), font=font))
    print(margins.to_string(index=False))
    print(f'Created {args.output}/index.html')


if __name__ == '__main__':
    main()
