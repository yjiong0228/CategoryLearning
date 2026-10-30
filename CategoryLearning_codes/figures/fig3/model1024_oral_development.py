"""PNG-only figures for provisional Model 1024 oral development.

Run --source RESEARCH_BUNDLE --output NEW_FIGURE_DIRECTORY. The companion
figure contract defines denominators, aggregation and interpretation limits.
All plotting uses matplotlib; every source table and rendered series is hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/model1024_oral_development_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

BLUE, ORANGE, GRAY, DARK = '#3B719F', '#CC823D', '#A7ADB4', '#303A44'


def rolling(values: np.ndarray, width: int = 32, minimum: int = 8) -> np.ndarray:
    """Trailing original-trial windows, with gaps and denominator preserved."""
    return pd.Series(np.asarray(values, float)).rolling(width, min_periods=minimum).mean().to_numpy()


def numeric_pair(token: str, first: int, second: int) -> bool:
    if not token.startswith('facet:'):
        return False
    rows = np.asarray(json.loads(token[6:]))
    return bool(len(rows) == 1 and set(np.flatnonzero(np.abs(rows[0, :4]) > 1e-12)) == {first, second}
        and abs(rows[0, :4].sum()) < 1e-12 and abs(rows[0, 4]) < 1e-12)


def run(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    inputs = {}

    def read(relative: str) -> pd.DataFrame:
        file = source/relative
        inputs[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
        return pd.read_csv(file)

    trials = read('diagnosis_v2/trials.csv')
    diagnostic = read('diagnosis_v2/summary.csv')
    readout = read('readout_v1/trial_scores.csv.gz')
    contrasts = read('content_v1/fixed_rule_contrasts.csv')
    content = read('content_v1/trials.csv')
    atoms = read('content_v1/atoms.csv.gz')
    anchor = read('anchor_v2/trial_scores.csv.gz')
    anchors = read('anchor_v2/anchors.csv')
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Noto Sans CJK JP', 'DejaVu Sans'],
        'font.size':10, 'axes.labelsize':10, 'axes.titlesize':11, 'legend.fontsize':9,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.8,
        'axes.unicode_minus':False, 'legend.frameon':False, 'svg.fonttype':'none', 'pdf.fonttype':42})
    output.mkdir(parents=True)
    series_rows, figure_names, overview_rows = [], [], []

    def save(fig, name):
        fig.savefig(output/(name+'.png'), dpi=300, facecolor='white')
        plt.close(fig)
        figure_names.append(name+'.png')

    def line(ax, sid, trial, values, name, *, smooth=True, **kwargs):
        transformed = rolling(values) if smooth else np.asarray(values)
        ax.plot(trial, transformed, label=name, **kwargs)
        series_rows.extend(dict(subject=sid, trial=int(t), series=name, raw=float(v), shown=float(s))
            for t,v,s in zip(trial, values, transformed))

    def block_means(frame, value):
        rows = frame.groupby('block64')[value].agg(['mean', 'count'])
        rows.loc[rows['count'] < 8, 'mean'] = np.nan
        return rows

    # Overview: every participant, common registered early-half comparison.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained', gridspec_kw={'height_ratios':[1, 1.15]})
    sids = sorted(trials.subject.unique())
    positions = np.arange(len(sids))
    main = diagnostic[diagnostic.window.eq('early_middle') & diagnostic.split.eq('all')].set_index('subject').loc[sids]
    axes[0].bar(positions-.22, main.atomic_ceiling, width=.22, color=GRAY, label='目录最多支持的比例')
    axes[0].bar(positions, main.P1_support, width=.22, color=BLUE, label='P1状态实际支持')
    axes[0].bar(positions+.22, main.stimulus_reference, width=.22, facecolor='white', edgecolor=DARK, label='刺激与选项参照')
    axes[0].set(ylim=(0, 1.04), ylabel='已编码判断支持比例', title='a  前半段：表达上限与模型实际分配的概率')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xticks(positions, [f'S{s}\n编码 {int(main.loc[s,"reports"])} 条' for s in sids])
    axes[0].legend(ncol=3, loc='upper left')
    early = readout[readout.selection_window.eq('early_middle') & readout.split.eq('assessment') & readout.early_middle & readout.eligible]
    for sid in sids:
        own = early[early.subject.eq(sid)]
        scores = own.groupby('branch').atom_log_score.mean()
        matched_trials = own[own.branch.eq('semantic_P1')].trial
        fixed = contrasts[contrasts.subject.eq(sid) & contrasts.trial.isin(matched_trials)].groupby('comparator').atom_log_score.mean()
        overview_rows.append(dict(subject=int(sid), reports=len(own[own.branch.eq('semantic_P1')]),
            versus_expression=scores['semantic_P1']-scores['expression_background'],
            versus_stimulus=scores['semantic_P1']-scores['semantic_stimulus_choice'],
            versus_correct=scores['semantic_P1']-fixed['correct_rule'],
            selected_versus_expression=scores['semantic_selected']-scores['expression_background']))
    ov = pd.DataFrame(overview_rows).set_index('subject').loc[sids]
    for delta, col, color, label in [(-.23,'versus_expression',BLUE,'相对刺激与用词习惯'),
        (0,'versus_correct',ORANGE,'相对只用正确任务规则'),(.23,'versus_stimulus',GRAY,'相对刺激选项规则参照')]:
        axes[1].bar(positions+delta, ov[col], width=.22, color=color, label=label)
    axes[1].axhline(0, color=DARK, lw=.8)
    axes[1].set_xticks(positions, [f'S{s}\n留出 {int(ov.loc[s,"reports"])} 条' for s in sids])
    axes[1].set(ylabel='平均对数概率增益\n（每条内平均判断，nats）', title='b  前半段留出口述：大于零表示P1状态提供额外信息')
    axes[1].legend(ncol=3, loc='upper left')
    fig.suptitle('哪些对应来自模型状态？六名开发被试的共同检查', fontsize=14)
    save(fig, 'overview')
    ov.reset_index().to_csv(output/'overview_data.csv', index=False)

    for sid, frame in trials.groupby('subject', sort=True):
        frame = frame.sort_values('trial')
        t = frame.trial.to_numpy()
        own_content = content[content.subject.eq(sid)].set_index('trial').reindex(t)
        metadata_file = source/'diagnosis_v2'/f'S{sid}_metadata.json'
        metadata = json.loads(metadata_file.read_text())
        state_file = Path(next(s for s in metadata['selected'] if s['number'] == 1)['file'])
        inputs[str(state_file)] = hashlib.sha256(state_file.read_bytes()).hexdigest()
        with np.load(state_file) as z:
            correct_execution = z['filtered_target_execution']
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
        line(axes[0], sid, t, frame.feedback.eq(1).to_numpy(float), '最近32题完全正确率', color=DARK, lw=1.5)
        line(axes[0], sid, t, correct_execution, '正确规则的执行概率', color=BLUE, lw=1.5)
        axes[0].set(ylabel='概率 / 正确率', ylim=(-.02, 1.03), title='a  学习表现与模型中的正确规则')
        axes[0].legend(loc='upper left', ncol=2)
        line(axes[1], sid, t, frame.atomic_ceiling.to_numpy(), '目录表达上限', color=GRAY, lw=1.7)
        line(axes[1], sid, t, frame.P1_support.to_numpy(), 'P1实际判断支持', color=BLUE, lw=1.7)
        line(axes[1], sid, t, frame.stimulus_reference.to_numpy(), '刺激选项参照', color=DARK, lw=1., ls='--')
        axes[1].scatter(t[frame.split.eq('assessment')], frame.loc[frame.split.eq('assessment'), 'P1_support'], s=6, color=BLUE, alpha=.2)
        axes[1].set(ylabel='判断支持比例', ylim=(-.02, 1.03), title='b  原有严格语义评分；淡点为逐题留出值')
        axes[1].legend(loc='upper left', ncol=3)
        scores = readout[readout.subject.eq(sid) & readout.selection_window.eq('early_middle') &
            readout.split.eq('assessment') & readout.eligible].pivot(index='trial', columns='branch', values='atom_log_score')
        for comparator, color, label in [('expression_background', BLUE, '相对刺激与用词习惯'),
            ('semantic_P1_shuffled', GRAY, '相对同选项同阶段打乱状态')]:
            temp = pd.DataFrame({'gain':scores.semantic_P1-scores[comparator]})
            temp['block64'] = (temp.index.to_numpy()-1)//64+1
            blocks = block_means(temp, 'gain')
            axes[2].plot((blocks.index-.5)*64, blocks['mean'], 'o-', color=color, ms=4, lw=1.2, label=label)
            for block, row in blocks.iterrows():
                series_rows.append(dict(subject=int(sid), trial=float((block-.5)*64), series=label,
                    raw=row['mean'], shown=row['mean'], reports=int(row['count'])))
        axes[2].axhline(0, color=DARK, lw=.8)
        axes[2].set(ylabel='留出对数概率增益', title='c  每64题汇总留出报告；表达参数用前半段调整块确定')
        axes[2].legend(loc='upper left', ncol=2)
        line(axes[3], sid, t, own_content.support_from_correct.to_numpy(), '来自正确规则的支持', color=GRAY, lw=1.5)
        line(axes[3], sid, t, own_content.alternative_specific_support.to_numpy(), '其他规则支持、正确规则未要求的内容', color=ORANGE, lw=1.7)
        axes[3].set(ylabel='全部理由判断中的质量', xlabel='有效试次（原顺序）', ylim=(-.02, 1.03),
            title='d  区分共同任务内容与其他规则特有的内容')
        axes[3].legend(loc='upper left', ncol=2)
        for ax in axes:
            ax.axvline(len(frame)/2, color=GRAY, ls=':', lw=.9)
            ax.set_xlim(1, len(frame))
        for ax in (axes[0], axes[1], axes[3]):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle(f'S{sid}：全程保留，分别检查表达、状态和具体内容', fontsize=14)
        save(fig, f'S{sid}_timecourse')

    # All mentions of the head/neck relation are retained. The short/long-leg
    # distinction is derived from the physical stimulus, not from good scores.
    pair = atoms[atoms.subject.eq(306) & atoms.atom.apply(lambda s:numeric_pair(s, 1, 2))].copy()
    stimulus = trials[trials.subject.eq(306)].set_index('trial')
    pair['leg_short'] = [float(stimulus.loc[t, 'stimulus_description'].split('；')[0].split('=')[1]) < .5 for t in pair.trial]
    pair['support'] = pair.support_from_correct+pair.support_from_other
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, layout='constrained')
    for ax, short in zip(axes, [True, False]):
        group = pair[pair.leg_short.eq(short)]
        for split, marker, color, label in [('calibration','o',GRAY,'调整块'), ('buffer','x',GRAY,'缓冲题'), ('assessment','o',BLUE,'留出块')]:
            use = group[group.split.eq(split)]
            ax.scatter(use.trial, use.support, marker=marker, color=color, s=25 if split == 'assessment' else 12,
                alpha=1 if split == 'assessment' else .45, label=label)
        ax.set(ylabel='头—脖子比较的支持概率', ylim=(-.03, 1.03),
            title=f'{"a" if short else "b"}  刺激中的腿{"短" if short else "长"}：全部明确编码的头—脖子方向比较')
        ax.axvline(256, color=ORANGE, ls=':', lw=1)
        ax.text(610, .35, '后段主编码中未再出现\n这组明确方向比较', color=DARK, ha='center', fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(ncol=3, loc='center right')
    axes[1].set(xlabel='有效试次', xlim=(1, 768))
    fig.suptitle('S306：错误判断的对应集中在短腿分支，随后消失', fontsize=14)
    save(fig, 'S306_head_neck')
    pair.to_csv(output/'S306_head_neck_data.csv', index=False)

    for sid in [205, 306]:
        frame = anchor[anchor.subject.eq(sid)]
        all_bank = frame[frame.bank.eq('all')]
        base = all_bank[all_bank.variant.eq('behavior')].set_index('trial')
        assisted = all_bank[all_bank.variant.eq('oral_temperature_2')].set_index('trial')
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
        t = base.index.to_numpy()
        line(axes[0], sid, t, base.support.to_numpy(), '同段末行为重建', color=BLUE, lw=1.8, ls='--')
        line(axes[0], sid, t, assisted.support.to_numpy(), '每64题加入一条口述', color=ORANGE, lw=1.5)
        for a in anchors[anchors.subject.eq(sid)].anchor.dropna():
            axes[0].axvline(a, color=ORANGE, alpha=.25, lw=.7)
        axes[0].set(ylabel='判断支持比例', ylim=(-.02, 1.03), title='a  橙色竖线为事先按位置选定的口述锚点')
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(ncol=2, loc='upper left')
        for bank, style, color, label in [('all','o-',ORANGE,'全部16次'),('first_half','--',GRAY,'前8次'),('second_half',':',DARK,'后8次')]:
            own = frame[frame.bank.eq(bank) & frame.split.eq('assessment') & frame.eligible]
            pivot = own.pivot(index='trial', columns='variant', values='subset_log_probability')
            delta = pd.DataFrame({'gain':pivot.oral_temperature_2-pivot.behavior})
            delta['block64'] = (delta.index.to_numpy()-1)//64+1
            means = block_means(delta, 'gain')
            axes[1].plot((means.index-.5)*64, means['mean'], style, color=color, lw=1.3, ms=4, label=label)
        axes[1].axhline(0, color=DARK, lw=.8)
        axes[1].set(ylabel='新增口述带来的增益\n（每条完整编码理由，nats）', title='b  仅评价后32题的预留口述；两半独立计算并列')
        axes[1].legend(ncol=3, loc='upper right')
        paths = {}
        for name in ['behavior', 'oral_temperature_2']:
            file = source/'anchor_v2'/f'S{sid}_{name}_pooled.npz'
            inputs[str(file.relative_to(source))] = hashlib.sha256(file.read_bytes()).hexdigest()
            with np.load(file) as z: paths[name] = z['component']
        tv = .5*np.abs(paths['behavior']-paths['oral_temperature_2']).sum((1, 2))
        axes[2].plot(t, tv, color=ORANGE, lw=1)
        axes[2].set(ylabel='整个局部规则分布的变化', xlabel='有效试次', title='c  新旧分布的总变差：0表示相同，1表示完全分开', ylim=(-.01, max(.05, float(tv.max())*1.1)))
        axes[2].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes:
            ax.set_xlim(1, len(t))
        fig.suptitle(f'S{sid}：一条口述对相邻状态与留出理由的影响', fontsize=14)
        save(fig, f'S{sid}_anchor')
    pd.DataFrame(series_rows).to_csv(output/'line_data.csv.gz', index=False)
    result = dict(status='rendered', backend='python', dpi=300, images=figure_names, inputs_sha256=inputs,
        displayed_series_rows=len(series_rows), primary_reports=int(trials.eligible.sum()), total_reports=len(trials),
        aggregation='trailing 32 original trials; minimum 8 eligible; blocks use assessment rows only',
        uncertainty='No confidence intervals or significance tests; fixed half-system curves are numerical sensitivity',
        user_export_override='PNG only; internal report canvas 11 inches rather than journal column width',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir() if f.is_file()})
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.output)
    print({k:v for k,v in result.items() if k not in ('inputs_sha256', 'artifacts')})
