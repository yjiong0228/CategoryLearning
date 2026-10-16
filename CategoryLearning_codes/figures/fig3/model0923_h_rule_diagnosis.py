"""PNG diagnostics of privileged rule interventions at frozen fitted parameters.

All nine subject/parameter combinations, seven arms and every simulated path
are retained. Panels describe development diagnostics on a fixed task schedule.
Run with --source SUMMARY --output NEW; Python/matplotlib, 183 mm, 300 dpi PNG.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_0923_h_validation import digest, read, write


LABELS = dict(BASE='Baseline', ONCE='Provide once', ONCE_EQUAL='Once + equal prior',
              KEEP='Provide + retain', KEEP_EQUAL='Retain + equal prior', ONLY='Correct rule only', REJECT='Allow rejection')
COLORS = dict(BASE='#64748B', ONCE='#58A4B0', ONCE_EQUAL='#89B7D2', KEEP='#C97836',
              KEEP_EQUAL='#D6AC70', ONLY='#76588F', REJECT='#4A8768')
STYLES = dict(BASE='-', ONCE='-', ONCE_EQUAL='--', KEEP='-', KEEP_EQUAL='--', ONLY='-', REJECT=':')


def run(source: Path, output: Path, readout_source: Path | None = None) -> None:
    if output.exists(): raise FileExistsError('Use a new figure directory')
    manifest = read(source / 'manifest.json')
    for name, sha in manifest['summary_sha256'].items():
        if digest(source / name) != sha: raise ValueError('Summary changed')
    summary, curves, paths = [pd.read_csv(source / (name+'.csv')) for name in ('summary', 'curves', 'paths')]
    subjects, models, arms = [manifest[key] for key in ('subjects', 'models', 'arms')]
    output.mkdir(parents=True)
    for name in manifest['summary_sha256']: shutil.copy2(source / name, output / name)
    write(output / 'figure_contract.json', dict(
        conclusion='Discovery and retention failures differ across fits; residual errors remain with exclusive correct-rule use.',
        evidence=['late accuracy with Monte Carlo errors', 'all future blocks', 'all path distributions', 'membership versus contribution'],
        archetype='quantitative grid', backend='python', width_mm=183, dpi=300, formats=['PNG'],
        units=f"{len(subjects)} development participants, {len(models)} frozen parameter sets, {manifest['paths_per_cell']} paths per cell",
        inclusion='every declared participant, model, arm and path; no exclusions',
        errors='points: mean +/- 2 Monte Carlo SE; curves: baseline 10th-90th path percentiles',
        inference='descriptive fixed-parameter diagnosis; no population test or p-value',
        limits='oracle interventions use task truth; no new out-of-sample model selection claim'))
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':7, 'axes.titlesize':7, 'axes.labelsize':7, 'xtick.labelsize':6,
        'ytick.labelsize':6, 'legend.fontsize':6, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.6, 'svg.fonttype':'none', 'pdf.fonttype':42})
    files = []

    def grid(height=165):
        fig, axs = plt.subplots(len(models), len(subjects), figsize=(183/25.4, height/25.4), squeeze=False)
        for r, model in enumerate(models):
            for c, subject in enumerate(subjects):
                axs[r,c].set_title(f'{chr(97+r*len(subjects)+c)}  S{subject} / {model}', loc='left', fontweight='bold')
        return fig, axs

    def save(fig, name):
        fig.savefig(output / name, dpi=300, facecolor='white'); plt.close(fig); files.append(name)

    fig, axes = grid()
    for r, model in enumerate(models):
        for c, subject in enumerate(subjects):
            ax = axes[r,c]; data = summary[(summary.subject==subject)&(summary.model==model)].set_index('arm')
            ax.axvline(data.loc['BASE','observed_late'], color='black', lw=1., ls='--')
            for i, arm in enumerate(arms):
                row = data.loc[arm]
                ax.errorbar(row.late_accuracy, i, xerr=2*row.late_accuracy_mc_se,
                    fmt='o', color=COLORS[arm], ms=3.4, lw=1, capsize=2)
            ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms] if c==0 else ['']*len(arms))
            ax.invert_yaxis(); ax.set_xlim(0., 1.); ax.set_xticks([0.,.25,.5,.75,1.])
            if r==len(models)-1: ax.set_xlabel('Last 128 trials: accuracy')
    fig.suptitle('Correct-rule interventions at frozen parameters', fontsize=9, y=.995)
    fig.text(.5,.955,'Points: mean ± 2 MC SE; dashed black: observed participant',ha='center',fontsize=7)
    fig.subplots_adjust(left=.22, right=.985, top=.90, bottom=.07, hspace=.43, wspace=.22)
    save(fig, 'rule_intervention_accuracy.png')

    legend = [Line2D([0],[0],color=COLORS[a],ls=STYLES[a],label=LABELS[a]) for a in arms]
    legend.append(Line2D([0],[0],color='black',lw=1.3,label='Observed'))
    fig, axes = grid(176)
    for r, model in enumerate(models):
        for c, subject in enumerate(subjects):
            ax=axes[r,c]; data=curves[(curves.subject==subject)&(curves.model==model)]
            base=data[data.arm=='BASE']
            ax.fill_between(base.center_trial.to_numpy(), base.q10.to_numpy(), base.q90.to_numpy(), color=COLORS['BASE'], alpha=.12, lw=0)
            for arm in arms:
                rows=data[data.arm==arm]
                ax.plot(rows.center_trial,rows['mean'],color=COLORS[arm],ls=STYLES[arm],lw=1.)
            ax.plot(base.center_trial,base.observed,color='black',lw=1.3)
            ax.set_ylim(0.,1.02)
            if c==0: ax.set_ylabel('Block accuracy')
            if r==len(models)-1: ax.set_xlabel('Task trial (after intervention)')
    fig.suptitle('Future trajectories after the same intervention boundary',fontsize=9,y=.995)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,top=.84,bottom=.07,hspace=.42,wspace=.27)
    save(fig,'rule_intervention_curves.png')

    fig, axes = grid(176)
    for r, model in enumerate(models):
        for c, subject in enumerate(subjects):
            ax=axes[r,c]; data=paths[(paths.subject==subject)&(paths.model==model)]
            for arm in arms:
                values=np.sort(data.loc[data.arm==arm,'late_accuracy'].to_numpy())
                ax.step(np.r_[0.,values,1.], np.r_[0.,np.arange(1,len(values)+1)/len(values),1.],
                    where='post',color=COLORS[arm],ls=STYLES[arm],lw=1.)
            observed=summary.loc[(summary.subject==subject)&(summary.model==model)&(summary.arm=='BASE'),'observed_late'].iloc[0]
            ax.axvline(observed,color='black',ls='--',lw=1.)
            ax.set_xlim(0.,1.); ax.set_ylim(0,1.02)
            if c==0: ax.set_ylabel('Cumulative path fraction')
            if r==len(models)-1: ax.set_xlabel('Last 128 trials: accuracy')
    fig.suptitle('Every simulated path contributes to the distribution',fontsize=9,y=.995)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,top=.84,bottom=.07,hspace=.42,wspace=.27)
    save(fig,'rule_intervention_distributions.png')

    fig, axes=plt.subplots(1,2,figsize=(183/25.4,132/25.4))
    keys=[(s,m) for s in subjects for m in models]
    for ax,metric,title in zip(axes,['target_late_presence','target_late_contribution'],
                               ['a  Correct rule present','b  Correct rule choice contribution']):
        matrix=np.array([[summary.loc[(summary.subject==s)&(summary.model==m)&(summary.arm==a),metric].iloc[0]
                          for a in arms] for s,m in keys])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for i,j in np.ndindex(matrix.shape): ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=6,color='white' if matrix[i,j]>.55 else 'black')
        ax.set_xticks(range(len(arms)),[LABELS[a] for a in arms],rotation=55,ha='right')
        ax.set_yticks(range(len(keys)),[f'S{s} / {m}' for s,m in keys] if ax is axes[0] else ['']*len(keys))
        ax.set_title(title,loc='left',fontweight='bold'); ax.tick_params(length=0)
    fig.suptitle('Membership and use can diverge',fontsize=9,y=.995)
    fig.subplots_adjust(left=.17,right=.87,top=.91,bottom=.26,wspace=.15)
    bar=fig.colorbar(im,cax=fig.add_axes([.90,.30,.015,.56]));bar.set_label('Mean fraction')
    save(fig,'rule_membership_and_use.png')
    descriptions=[
        f"末128题正确率。每个点纳入{manifest['paths_per_cell']}条路径；误差线为两倍模拟标准误。黑色虚线为真实被试正确率。",
        '干预从原训练边界后的第一题开始。曲线为32题区块均值；灰色带为原方案路径的10%—90%分位数。',
        f"全部路径的末128题正确率经验累计分布。每条曲线使用{manifest['paths_per_cell']}条完整路径；黑色虚线为真实值。",
        '左图为正确规则在集合中的时间比例；右图为其作答贡献。两者均对全部路径和末128题取平均，缺席时贡献为零。']
    if readout_source is not None:
        audit=read(readout_source/'completion.json')
        if audit['status']!='complete' or digest(readout_source/'summary.csv')!=audit['summary_sha256']:
            raise ValueError('Incomplete or changed readout audit')
        audit_protocol=read(readout_source/'protocol.json')
        if audit_protocol['source_protocol_sha256']!=manifest['source_protocol_sha256']:
            raise ValueError('Readout audit and figure refer to different diagnostic runs')
        data=pd.read_csv(readout_source/'summary.csv')
        shutil.copy2(readout_source/'summary.csv',output/'readout_summary.csv')
        shutil.copy2(readout_source/'paths.csv',output/'readout_paths.csv')
        fig,axes=grid(165)
        levels=['learned','5.0','10.0','25.0','100.0']
        for r,model in enumerate(models):
            for c,subject in enumerate(subjects):
                ax=axes[r,c];group=data[(data.subject==subject)&(data.model==model)]
                for kind,color,marker,label in [('perceived','#76588F','o','Perceived input'),('physical','#64748B','s','Physical input')]:
                    rows=group[group.stimulus==kind].set_index('beta').loc[levels]
                    ax.errorbar(range(1,len(levels)),rows['mean'].iloc[1:],yerr=2*rows.mc_se.iloc[1:],
                        fmt=marker+'-',color=color,ms=3,lw=.8,label=label)
                    ax.errorbar(0,rows['mean'].iloc[0],yerr=2*rows.mc_se.iloc[0],fmt=marker,color=color,ms=3,lw=.8)
                truth=summary.loc[(summary.subject==subject)&(summary.model==model)&(summary.arm=='BASE'),'observed_late'].iloc[0]
                ax.axhline(truth,color='black',ls='--',lw=.8)
                ax.axvline(3.5,color='#CCCCCC',ls=':',lw=.8)
                ax.set_xticks(range(len(levels)),['Learned','5','10','25','100*']);ax.set_ylim(.45,1.)
                if c==0:ax.set_ylabel('Mean correct-response probability')
                if r==len(models)-1:ax.set_xlabel('Rule precision β')
        fig.suptitle('Readout audit on the same correct-rule-only paths',fontsize=9,y=.995)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False)
        fig.text(.5,.91,'* β = 100 exceeds the model bound of 25; diagnostic evaluation only.',ha='center',fontsize=6)
        fig.subplots_adjust(left=.09,right=.985,top=.83,bottom=.07,hspace=.42,wspace=.27)
        save(fig,'rule_precision_readout_audit.png')
        descriptions.append('固定ONLY路径的已学习状态和知觉刺激，重算不同β下的作答概率。Physical input使用对应物理刺激；这些读出计算不反馈到后续学习。β=100超出当前模型上限25，仅用于诊断。黑色虚线为真实被试末128题正确率。')
    text='<html><meta charset="utf-8"><title>H规则诊断</title><style>body{max-width:1100px;margin:36px auto;font:17px/1.65 sans-serif}img{width:100%}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}</style><h1>H模块：规则发现、保留与使用诊断</h1>'
    text+='<p>三个被试、三套冻结参数、七种对照。正确规则由任务分类信息指定；这些对照用于定位机制问题。所有被试在本轮均属于开发资料。</p>'
    text+='<p>207的部分方案同时受发现和保留限制；308共享保持率方案主要受发现限制；208独立保持率方案已稳定保留正确规则。只用正确规则仍留下预测差距，精度读出需要继续检验。</p>'
    text+='<p>ONCE：提供一次正确规则；EQUAL：干预当题将集合内规则先验设为均匀；KEEP：持续保留正确规则；ONLY：后续仅保留并使用正确规则；REJECT：重放原候选记录，允许满容量时拒绝候选。</p>'
    for file, description in zip(files,descriptions): text+=f'<p>{html.escape(description)}</p><img src="{file}">'
    text+='<p>模拟误差以完整路径为独立单位，参数与任务刺激序列固定。该误差不包含个体参数不确定性和群体差异。研究刺激序列沿用被试记录中的实际顺序。</p>'
    text+='<p>'+ ' · '.join(f'<a href="{name}">{name}</a>' for name in manifest['summary_sha256'])+'</p></html>'
    (output/'index.html').write_text(text)
    write(output/'qa_numeric.json',dict(status='passed',cells=len(summary),paths=len(paths),
        exclusions=0,source_manifest_sha256=digest(source/'manifest.json'),plot_source_sha256=digest(Path(__file__)),
        figures_sha256={name:digest(output/name) for name in files},
        format_exception='PNG only, 300 dpi discussion figures under repository policy',visual_review='pending'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--readout-source',type=Path)
    args=parser.parse_args();run(args.source,args.output,args.readout_source)


if __name__=='__main__':main()
