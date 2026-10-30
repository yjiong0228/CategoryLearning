"""Plot saved provisional oral scores over trials; see adjacent figure contract.

All subjects and parameter candidates remain visible. Trailing windows use
original trial positions, explicit eligibility and recorded denominators.
Exports are PNG only under the repository's figure-review convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1024_oral_preliminary import METRICS

BLUE, ORANGE, GREY, RED = '#3274A1', '#C9873E', '#CDD2D7', '#BE6460'
COLORS = (BLUE, '#409C93', '#8D7DAF', '#B9874C')
STYLES = ('-', '--', '-.', ':')
STATUS_COLORS = ('#E3E5E7', '#E9BA71', '#81B8AE')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rolling(values: np.ndarray, valid: np.ndarray, width: int, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    """Average eligible observations within actual trial windows; never fill gaps."""
    values, valid = np.asarray(values, float), np.asarray(valid, bool)
    if values.shape != valid.shape or values.ndim != 1 or not 1 <= minimum <= width:
        raise ValueError('Expected aligned one-dimensional data and valid window limits')
    if not np.isfinite(values[valid]).all():
        raise ValueError('Eligible observations must have finite scores')
    series = pd.Series(np.where(valid, values, np.nan))
    count = pd.Series(valid.astype(int)).rolling(width, min_periods=1).sum().to_numpy(int)
    mean = series.rolling(width, min_periods=minimum).mean().to_numpy()
    return mean, count


def prepare(source: Path, width: int, minimum: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['status'] != 'provisional_content_analysis_complete':
        raise ValueError('Require completed provisional content scores')
    inputs = {str(source/'manifest.json'): digest(source/'manifest.json')}
    def verified(name: str) -> Path:
        path = source/name
        if digest(path) != manifest['artifacts'][name]:
            raise ValueError('Frozen oral input changed: '+name)
        inputs[str(path)] = digest(path)
        return path
    review = pd.read_csv(verified('trial_review.csv'))
    scores = pd.read_csv(verified('candidate_trial_scores.csv.gz'))
    # Filtering and retrospective states retain separate conditioning. This
    # visualization explicitly selects the registered primary filtering mode.
    selected = scores[scores['mode'].eq('filtered')].copy()
    if selected.duplicated(['subject', 'number', 'trial']).any():
        raise ValueError('Duplicated candidate trial')
    output, counts = [], []
    for sid, reports in review.groupby('subject', sort=True):
        reports = reports.sort_values('trial').reset_index(drop=True)
        n = len(reports)
        np.testing.assert_array_equal(reports.trial, np.arange(1, n+1))
        with np.load(verified(f'S{sid}_reference_scores.npz')) as archive:
            reference = archive['stimulus_choice'][:, METRICS.index('predicate_supported')]
            available = archive['available']
        if reference.shape != (n,) or available.shape != (n,):
            raise ValueError('Reference trial shape changed')
        semantic = reports.predicate_primary_eligible.to_numpy(bool)
        feature = reports.feature_eligible.to_numpy(bool)
        common = semantic & available
        basis_reference, common_count = rolling(reference, common, width, minimum)
        coverage = pd.Series(semantic.astype(float)).rolling(width, min_periods=1).mean().to_numpy()
        candidates = selected[selected.subject.eq(sid)]
        if candidates.empty:
            raise ValueError('A report subject has no state candidates')
        counts.append(dict(subject=int(sid), trials=n, candidates=int(candidates.number.nunique()),
            feature_reports=int(feature.sum()), semantic_reports=int(semantic.sum()),
            common_reports=int(common.sum()), reference_missing_trials=(np.flatnonzero(~available)+1).tolist(),
            insufficient_common_windows=int((common_count<minimum).sum()),
            parameter_status=','.join(candidates.parameter_status.unique())))
        for number, own in candidates.groupby('number', sort=True):
            own = own.sort_values('trial').reset_index(drop=True)
            np.testing.assert_array_equal(own.trial, reports.trial)
            np.testing.assert_array_equal(own.feature_eligible, feature)
            np.testing.assert_array_equal(own.predicate_primary_eligible, semantic)
            own['common_eligible'] = common
            own['reference_available'] = available
            own['reference_raw'] = reference
            own['reference_mean'] = basis_reference
            own['common_count'] = common_count
            own['semantic_fraction'] = coverage
            own['trial_status'] = np.where(semantic, 2, np.where(feature, 1, 0))
            own['window_start'] = np.maximum(1, own.trial-width+1)
            own['window_trials'] = np.minimum(own.trial, width)
            own['support_common'], _ = rolling(own.predicate_supported, common, width, minimum)
            for field in ('predicate_supported', 'predicate_conflicting', 'predicate_unexplained',
                          'reason_feature_coverage', 'basis_feature_coverage'):
                mask = feature if field in ('reason_feature_coverage', 'basis_feature_coverage') else semantic
                own[field+'_mean'], own['feature_count' if mask is feature else 'semantic_count'] = rolling(
                    own[field], mask, width, minimum)
            totals = own[['predicate_supported_mean', 'predicate_conflicting_mean',
                          'predicate_unexplained_mean']].sum(axis=1, min_count=3)
            np.testing.assert_allclose(totals[np.isfinite(totals)], 1, atol=2e-10)
            output.append(own)
    plotted = pd.concat(output, ignore_index=True)
    if len(plotted) != len(selected) or plotted['case'].nunique() != manifest['candidates']:
        raise ValueError('Every declared candidate and filtering trial must be retained')
    return plotted, pd.DataFrame(counts), inputs


def style() -> None:
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size':9,
        'axes.titlesize':10, 'axes.labelsize':9, 'xtick.labelsize':8, 'ytick.labelsize':8,
        'legend.fontsize':8, 'legend.frameon':False, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.65, 'svg.fonttype':'none', 'pdf.fonttype':42})


def axes_style(ax, trials: int) -> None:
    ax.set_xlim(.5, trials+.5)
    ax.set_ylim(0, 1.025)
    ax.set_yticks([0, .5, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(axis='y', color='#ECEEF0', lw=.55)
    ax.set_axisbelow(True)


def support_panel(ax, frame: pd.DataFrame, minimum: int) -> None:
    p1 = frame[frame.number.eq(1)]
    x = p1.trial.to_numpy()
    available = p1.common_eligible.to_numpy(bool)
    # P1 is a fixed candidate; dots retain every scored trial without decimation.
    ax.scatter(x[available], p1.predicate_supported.to_numpy()[available], s=2.5,
               color='#C3C8CD', alpha=.7, linewidths=0, zorder=1)
    ax.fill_between(x, 0, 1, where=p1.common_count.to_numpy()<minimum,
                    color='#F0F1F2', lw=0, zorder=0)
    for number in sorted(frame.number.unique(), reverse=True):
        own = frame[frame.number.eq(number)]
        ax.plot(own.trial, own.support_common, color=COLORS[(number-1)%4],
                ls=STYLES[(number-1)%4], lw=1.35, zorder=3)
    ax.plot(x, p1.reference_mean, color='#4B535A', ls=(0,(5,3)), lw=1.2, zorder=4)
    axes_style(ax, len(p1))


def support_legend(numbers: list[int]) -> list:
    return [Line2D([], [], color=COLORS[(number-1)%4], ls=STYLES[(number-1)%4],
                   lw=1.5, label=f'参数 P{number}') for number in numbers]+[
        Line2D([], [], color='#4B535A', ls=(0,(5,3)), label='当前刺激与选项参照'),
        Line2D([], [], marker='.', color='#C3C8CD', ls='', label='P1逐题值')]


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


def overview(data: pd.DataFrame, counts: pd.DataFrame, output: Path, width: int, minimum: int) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4))
    fig.subplots_adjust(left=.075, right=.975, bottom=.15, top=.83, hspace=.69, wspace=.18)
    fig.suptitle('口述中的具体判断，与模型内容的对应如何随试次变化？', fontsize=14, y=.985)
    fig.text(.5, .943, f'最近{width}题内的平均判断支持  |  每个窗口至少{minimum}条可评分口述', ha='center', fontsize=10)
    fig.legend(handles=support_legend(sorted(data.number.unique())), loc='upper center',
               bbox_to_anchor=(.5,.925), ncol=6)
    for i, (ax, row) in enumerate(zip(axes.flat, counts.itertuples(index=False))):
        own = data[data.subject.eq(row.subject)]
        support_panel(ax, own, minimum)
        status = '诊断参数，尚未确立' if row.parameter_status=='diagnostic' else '保留参数'
        # Some archived diagnostic statuses have a longer descriptive name.
        if row.subject==129:
            status = '诊断参数，尚未确立'
        ax.set_title(f'{chr(97+i)}  S{row.subject}  |  {row.candidates}组{status}\n'
                     f'完整判断可评分 {row.semantic_reports}/{row.trials} 题', loc='left', pad=8)
        ax.set_xlabel('试次')
        if i%2==0:
            ax.set_ylabel('判断支持比例')
    fig.text(.075,.04,'灰点保留P1逐题值；灰底表示窗口内可评分报告不足。曲线使用模型与参照共同有效的报告。\n'
             '状态：本题选择后、反馈前；参数由完整行为记录估计。口述待人工复核，数值稳定性沿用原报告限制。', fontsize=9)
    save(fig,output/'overview.png')


def detail(data: pd.DataFrame, count: pd.Series, output: Path, width: int, minimum: int) -> None:
    sid = int(count.subject)
    p1 = data[data.number.eq(1)]
    x = p1.trial.to_numpy()
    fig, axes = plt.subplots(4,1,figsize=(11,10.5),sharex=True,
                             gridspec_kw={'height_ratios':[1.35,1.05,1.05,.85]})
    fig.subplots_adjust(left=.085,right=.975,top=.835,bottom=.13,hspace=.82)
    status = '四组诊断参数，尚未确立' if sid==129 else f'{int(count.candidates)}组保留参数'
    fig.suptitle(f'S{sid}  |  口述与模型内容的逐题对照',fontsize=15,y=.982)
    fig.text(.085,.942,f'{int(count.trials)}题；完整判断可评分{int(count.semantic_reports)}题；'
             f'部位可评分{int(count.feature_reports)}题；{status}',fontsize=10)
    fig.text(.085,.915,f'曲线：截至当前题，最近{width}题内有效报告的平均；至少需要{minimum}条。P1为固定展示候选。',fontsize=9)
    support_panel(axes[0],data,minimum)
    axes[0].set_title('a  具体判断对应多少？',loc='left',pad=30,fontweight='bold')
    axes[0].legend(handles=support_legend(sorted(data.number.unique())),ncol=6,
                   loc='lower left',bbox_to_anchor=(0,1),borderaxespad=0)
    axes[0].set_ylabel('判断支持')
    for field,color,label in [('reason_feature_coverage',BLUE,'口述部位被规则涉及'),
                              ('basis_feature_coverage',ORANGE,'规则部位被口述提及')]:
        values = data.pivot(index='trial',columns='number',values=field+'_mean')
        axes[1].fill_between(x,values.min(axis=1),values.max(axis=1),color=color,alpha=.19,lw=0)
        axes[1].plot(x,p1[field+'_mean'],color=color,lw=1.55,label=label)
    axes_style(axes[1],len(p1))
    axes[1].set_title('b  双向看部位覆盖：实线为P1，色带为全部候选范围',loc='left',pad=29,fontweight='bold')
    axes[1].legend(loc='lower left',bbox_to_anchor=(0,1),ncol=2,borderaxespad=0)
    axes[1].set_ylabel('部位覆盖')
    bottom = np.zeros(len(p1))
    for field,color,label in [('predicate_supported',BLUE,'支持'),('predicate_unexplained',GREY,'未确定'),
                              ('predicate_conflicting',RED,'冲突')]:
        values = p1[field+'_mean'].to_numpy()
        axes[2].fill_between(x,bottom,bottom+values,where=np.isfinite(values),color=color,label=label,lw=0)
        bottom = bottom+values
    axes_style(axes[2],len(p1))
    axes[2].set_title('c  具体判断的三种结果（P1；全部可评分报告）',loc='left',pad=29,fontweight='bold')
    axes[2].legend(loc='lower left',bbox_to_anchor=(0,1),ncol=3,borderaxespad=0)
    axes[2].set_ylabel('判断构成')
    axes[3].plot(x,p1.semantic_fraction,color='#30383F',lw=1.3)
    axes_style(axes[3],len(p1))
    axes[3].imshow(p1.trial_status.to_numpy()[None,:],cmap=ListedColormap(STATUS_COLORS),
                   vmin=0,vmax=2,interpolation='nearest',aspect='auto',extent=(.5,len(p1)+.5,-.18,-.055))
    axes[3].set_ylim(-.19,1.03)
    axes[3].set_title('d  完整判断可评分的报告比例；底部色条保留每一题的编码状态',loc='left',pad=29,fontweight='bold')
    axes[3].legend(handles=[Patch(color=STATUS_COLORS[i],label=label) for i,label in
        [(2,'完整判断可评分'),(1,'仅部位可评分'),(0,'暂不可评分')]],loc='lower left',bbox_to_anchor=(0,1),ncol=3,borderaxespad=0)
    axes[3].set_ylabel('可评分比例')
    axes[3].set_xlabel('累计有效行为试次（保留原顺序）')
    fig.text(.085,.045,'选择后、反馈前的过滤状态；参数使用完整行为记录估计；口述仍待人工复核。\n'
             'a中模型与参照使用共同有效报告，c保留全部主语义报告；留空表示窗口内报告不足。\n'
             '色带仅表示已检查参数之间的差异；本图未估计置信区间或独立重复误差。',fontsize=9)
    save(fig,output/f'S{sid}.png')


def report_text(report: Path, source: Path, output: Path, counts: pd.DataFrame, width: int, minimum: int) -> None:
    link = lambda path: Path(os.path.relpath(path,report)).as_posix()
    text = f'''# 口述对齐随试次的变化

六人总览与每人四行详图均使用已有评分。原文、编码、参数及状态概率保持原样。
曲线为最近{width}题内有效报告的平均，每个内容窗口至少需要{minimum}条。
逐题值保留在第一行浅灰点和[绘图数据]({link(output/'plot_data.csv.gz')})中。

横轴为累计有效行为试次。状态对应本题选择之后、反馈之前，参数由完整行为记录估计。
这一展示属于待人工复核口述上的开发分析。

## 六人总览

每条彩线对应一组参数，灰色虚线为只看当前刺激和所选类别的参照。
每个窗口中，模型与参照使用同一批可评分报告。灰底表示不足{minimum}条，曲线留空。
多个参数结果接近时，彩线会重叠。S129四组仍属于诊断参数。

![六人口述判断支持随试次变化]({link(output/'overview.png')})

## 每人四行图怎么读

1. **第一行：具体判断是否对应。** 例如“腿短”和“腿长”会得到不同评分。彩线为各候选，浅灰点为P1逐题值。
2. **第二行：双方各自说到了多少部位。** 蓝线看口述部位被规则涉及多少，橙线看规则部位被口述提及多少。
   实线固定为P1，色带是全部已检查候选的最小值至最大值。色带没有置信区间含义。
3. **第三行：支持、冲突、未确定。** 固定展示P1，蓝、红、灰三部分在有效窗口内合计100%。
   “未确定”表示该局部规则没有充分限定相应口述判断。
4. **第四行：能评分的口述够不够。** 黑线看最近窗口内完整判断可评分的比例。
   底部色条保留每一题，绿色为完整判断可评分，黄色为仅部位可评分，灰色为暂不可评分。

窗口随实际试次移动，缺失记录保留原位置；最近{width}题中不足{minimum}条有效报告时，内容曲线留空。
开始阶段只使用实际已有试次。平均线会延缓或平缓突然变化，细节可结合浅灰逐题点和逐条原话检查。
第三行使用全部主语义报告；第一行仅在参照可用的共同试次上比较，逐题分母均已保存。

| 被试 | 记录总数 | 完整判断可评分 | 与参照共同可评分 | 参数组数 |
|---|---:|---:|---:|---:|
'''
    for row in counts.itertuples(index=False):
        text += f'| S{row.subject} | {row.trials} | {row.semantic_reports} | {row.common_reports} | {row.candidates} |\n'
    notes = {110:'重点看第二行两条线的距离：模型涉及的部位被说到，与口述里的部位全部被模型涉及，是两个不同问题。',
        129:'四组参数均保留诊断身份，当前图用于查看内容变化。',
        205:'将第一行后段变化与第四行可评分比例合看，避免把口述编码范围改变直接理解为模型状态改变。',
        229:'只有一组保留候选，第二行候选色带宽度为零，参数间差异尚无比较依据。',
        306:'同时看具体判断支持、简单参照及尚未确定的内容。',
        317:'后段3/4躯干参照尚未完整编码。第四行仍可出现黄色，而第一、三行在有效报告不足后留空。'}
    for sid in counts.subject:
        text += f'\n## S{sid}\n\n{notes.get(sid, "")}\n\n![S{sid}逐题对照]({link(output/f"S{sid}.png")})\n'
    text += f'\n[原分析报告]({link(source/"report.md")}) · [逐条原话对照]({link(source/"review.html")})\n'
    text += '\n数值稳定性及人工编码的限制沿用原报告；本轮只增加图形展示，没有新增粒子计算。\n'
    (report/'report.md').write_text(text)


def run(source: Path, output: Path, report: Path, width: int, minimum: int) -> None:
    if output.exists() or report.exists():
        raise FileExistsError('Use new figure and report directories')
    data, counts, inputs = prepare(source,width,minimum)
    output.mkdir(parents=True); report.mkdir(parents=True)
    style()
    overview(data,counts,output,width,minimum)
    for _, count in counts.iterrows():
        detail(data[data.subject.eq(count.subject)],count,output,width,minimum)
    data.to_csv(output/'plot_data.csv.gz',index=False,compression='gzip')
    counts.to_csv(output/'coverage.csv',index=False)
    report_text(report,source,output,counts,width,minimum)
    if any(digest(Path(name))!=expected for name,expected in inputs.items()):
        raise ValueError('An input changed during plotting')
    record = dict(status='rendered_pending_visual_review',backend='python',primary_mode='filtered',
        window_trials=width,minimum_reports=minimum,averaging='trailing, equal report weights; fixed candidate states',
        all_subjects=int(len(counts)),all_candidates=int(data['case'].nunique()),
        original_reports=int(counts.trials.sum()),plotted_candidate_rows=len(data),
        coverage=counts.to_dict('records'),no_new_filters=True,
        parameter_band='finite candidate min to max; no confidence or numerical interval',
        source={str(Path(__file__).relative_to(Path.cwd())):digest(Path(__file__))},inputs=inputs,
        versions=dict(python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__),
        figures={path.name:digest(path) for path in output.glob('*.png')},
        plot_data_sha256=digest(output/'plot_data.csv.gz'),report_sha256=digest(report/'report.md'))
    (output/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    (report/'source.json').write_text(json.dumps(dict(figures=str(output),manifest_sha256=digest(output/'manifest.json')),indent=2))
    print(json.dumps(dict(figures=len(record['figures']),subjects=len(counts),candidate_rows=len(data)),ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--window',type=int,default=32)
    parser.add_argument('--minimum-reports',type=int,default=8)
    args = parser.parse_args()
    run(args.alignment,args.output,args.report,args.window,args.minimum_reports)


if __name__=='__main__':
    main()
