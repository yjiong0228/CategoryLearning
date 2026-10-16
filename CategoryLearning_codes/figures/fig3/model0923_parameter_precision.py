"""PNG review figures for independent parameter precision and rule-state checks.

Uses every candidate and completed budget in --source; --output must be new.
Original A--D labels are retained when the audit changes the numerical ranking.
Public probability/state summaries and all 1024 autonomous paths per nominee
support the 207 diagnosis. Figures are descriptive training-data diagnostics.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig3.model0923_parameter_selection import configure_style
from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load

COLORS={'NLL':'#0072B2','GEN':'#D55E00'}


def candidate_styles(frame: pd.DataFrame) -> dict:
    styles={};gray=0
    for row in frame.drop_duplicates('candidate').sort_values('letter').itertuples():
        if row.original_nll_selected:
            color,style=COLORS['NLL'],'-'
        elif row.original_gen_selected:
            color,style=COLORS['GEN'],'-'
        else:
            color,style=[('#777777','--'),('#B0B0B0',':')][gray];gray+=1
        styles[row.candidate]=(row.letter,color,style)
    return styles


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)


def panel(ax, letter: str) -> None:
    ax.text(-.15,1.08,letter,transform=ax.transAxes,fontsize=8,fontweight='bold')


def rankings(source: Path, output: Path) -> None:
    scores=pd.read_csv(source/'scores.csv');contrasts=pd.read_csv(source/'contrasts.csv')
    subjects=sorted(scores.subject.unique())
    for mode in ('scores','differences'):
        fig,axes=plt.subplots(2,2,figsize=(183/25.4,151/25.4))
        fig.subplots_adjust(left=.10,right=.98,bottom=.18,top=.91,hspace=.50,wspace=.32)
        for i,subject in enumerate(subjects):
            ax=axes.flat[i];panel(ax,'abcd'[i]);data=scores[scores.subject==subject]
            styles=candidate_styles(data);budgets=sorted(data.particles.unique());offset=1 if mode=='scores' else 0
            for candidate,(letter,color,style) in styles.items():
                rows=data[data.candidate==candidate].sort_values('particles')
                x=[budgets.index(p)+offset for p in rows.particles]
                if mode=='scores':
                    values=[rows.old_nll.iloc[0]]+rows.mean_nll.tolist()
                    errors=[rows.old_mc_se.iloc[0]]+rows.mc_se.tolist();x=[0]+x
                else:
                    c=contrasts[(contrasts.subject==subject)&(contrasts.candidate==candidate)].sort_values('particles')
                    values=c.delta_nll.tolist();errors=c.paired_mc_se.tolist()
                ax.errorbar(x,values,yerr=2*np.asarray(errors),color=color,ls=style,marker='o',ms=3.5,
                    capsize=2,lw=.9,label=letter)
            labels=(['原256粒子\n4种子'] if mode=='scores' else [])+[f'新{p}粒子\n8种子' for p in budgets]
            ax.set(title=f'被试 {subject}',xticks=range(len(labels)),xticklabels=labels,
                   ylabel='平均 NLL（越低越好）' if mode=='scores' else 'NLL − 原所选 A 的 NLL')
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.legend(ncol=len(styles),loc='best',handlelength=1.4,columnspacing=.7)
            if mode=='differences':ax.axhline(0,color='#444444',lw=.6,ls=':')
            ax.margins(x=.12)
        fig.suptitle('独立种子与粒子数复核：原候选排名是否稳定' if mode=='scores' else '相对原所选 A 的分差：计算误差是否足以改变判断',fontsize=10,y=.975)
        fig.text(.5,.10,'蓝色：原 NLL 所选；橙色：原 GEN 所选；灰色：其他原候选。字母沿用原排名。',ha='center',fontsize=6)
        fig.text(.5,.065,'误差条：±2 Monte Carlo SE'+('（候选与 A 使用配对种子）' if mode=='differences' else '（固定参数的计算波动）')+'。',ha='center',fontsize=6)
        fig.text(.5,.03,'全部使用训练段。新种子独立于原选择；新批次中的最低值仍需另行验证。',ha='center',fontsize=6)
        save(fig,output,'precision_'+mode)


def block_mean(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return frame.assign(block=(frame.trial-1)//32+1).groupby('block')[names].mean()


def mechanism(source: Path, output: Path) -> None:
    conditional=pd.read_csv(source/'conditional_state_trajectories.csv')
    autonomous=pd.read_csv(source/'autonomous_state_trajectories.csv')
    paths=pd.read_csv(source/'autonomous_paths.csv')
    contributions=pd.read_csv(source/'trial_contributions.csv')
    scores=pd.read_csv(source/'scores.csv');summary=load(source/'autonomous_summary.json')
    assert all(row['useful_rules']==[42] for row in summary['rows'] if row['subject']==207)
    subject=207;conditional=conditional[conditional.subject==subject]
    particles=int(conditional.particles.max());conditional=conditional[conditional.particles==particles]
    points=scores[scores.subject==subject]
    gen=points[points.original_gen_selected].candidate.iloc[0]
    trials=contributions[(contributions.subject==subject)&(contributions.candidate==gen)&(contributions.particles==particles)].sort_values('trial')
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,162/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.90,hspace=.48,wspace=.32)
    for model in ('NLL','GEN'):
        c=block_mean(conditional[conditional.model==model],['predicted_correct','useful_mass'])
        a=block_mean(autonomous[(autonomous.subject==subject)&(autonomous.model==model)],['predicted_correct','useful_mass'])
        for ax,field in [(axes[0,0],'predicted_correct'),(axes[0,1],'useful_mass')]:
            ax.plot(c.index,c[field],color=COLORS[model],lw=1.3)
            ax.plot(a.index,a[field],color=COLORS[model],ls='--',lw=1.3)
    observed=trials.groupby('block').correct.mean()
    axes[0,0].plot(observed.index,observed,color='#222222',marker='.',lw=1,ms=3)
    axes[0,0].set(title='条件预测与自主生成的差距',ylabel='正确率',ylim=(-.02,1.02))
    axes[0,1].set(title='任务对应规则的作答权重',ylabel='规则 42 的作答贡献',ylim=(-.02,1.02))
    for ax in axes[0]:
        ax.set_xlabel('训练区块（每块 32 试次）');ax.set_xticks([1,5,9,13,17,21])
    ax=axes[1,0];values=trials.log_advantage_original_nll.to_numpy();correct=trials.correct.to_numpy(bool)
    for label,contribution,style,color in [('全部回答',values,'-','#222222'),
        ('正确回答',np.where(correct,values,0.),'--','#777777'),('错误回答',np.where(~correct,values,0.),':','#444444')]:
        ax.plot(trials.trial,np.cumsum(contribution),ls=style,color=color,label=label,lw=1.1)
    ax.axhline(0,lw=.5,color='#AAAAAA');ax.legend(fontsize=6)
    ax.set(title='NLL 所选相对 GEN 所选的似然优势',xlabel='训练试次',ylabel='累计 log 概率差（高于0为领先）')
    ax=axes[1,1];comparison=[]
    for i,(model,initial) in enumerate([('NLL',False),('NLL',True),('GEN',False),('GEN',True)]):
        group=paths[(paths.subject==subject)&(paths.model==model)&(paths.initially_useful==initial)]
        if len(group):
            mean=group.late_accuracy.mean();low,high=np.quantile(group.late_accuracy,[.1,.9])
            ax.vlines(i,low,high,color=COLORS[model],lw=1.3)
            ax.plot(i,mean,'o',color=COLORS[model],ms=4)
            ax.text(i,1.02,f'n={len(group)}',ha='center',va='bottom',fontsize=6)
            comparison.append(dict(model=model,initially_useful=initial,paths=len(group),mean=mean,q10=low,q90=high))
    observed_late=paths[paths.subject==subject].observed_late_accuracy.iloc[0]
    ax.axhline(observed_late,color='#222222',lw=.8,ls=':')
    ax.set(title='初始规则与自主学习结果',ylabel='末128试次正确率',ylim=(-.02,1.12),xticks=range(4),
        xticklabels=['NLL\n其他初始规则','NLL\n初始规则42','GEN\n其他初始规则','GEN\n初始规则42'])
    ax.margins(x=.14)
    for i,ax in enumerate(axes.flat):panel(ax,'abcd'[i])
    fig.suptitle(f'被试 207｜似然优势来自何处（{particles}粒子 × 8种子）',fontsize=10,y=.975)
    fig.legend(handles=[Line2D([],[],color=COLORS[m],label=m+'原所选参数') for m in ('NLL','GEN')]+[
        Line2D([],[],color='#666666',label='条件预测 / 状态'),Line2D([],[],color='#666666',ls='--',label='自主路径均值'),
        Line2D([],[],color='#222222',marker='.',label='真实行为')],loc='lower center',bbox_to_anchor=(.5,.105),ncol=3)
    fig.text(.5,.068,'自主部分保留每组全部1024条路径。d：点为组均值，竖线为10%–90%分位数。',ha='center',fontsize=6)
    fig.text(.5,.043,'规则42的物理分类准确率为1；实际作答还受知觉、精度和规则权重影响。',ha='center',fontsize=6)
    fig.text(.5,.018,'初始分组是描述性关联；含规则42的小组样本量较小。全部面板使用训练段。',ha='center',fontsize=6)
    save(fig,output,'subject_207_mechanism')
    pd.DataFrame(comparison).to_csv(output/'initial_rule_groups.csv',index=False)


def probe_figure(source: Path, precision: Path, output: Path) -> None:
    """Full path distributions and paired scores for the fixed 2 x 2 probe."""
    scores=pd.read_csv(source/'scores.csv');curves=pd.read_csv(source/'curves.csv')
    paths=pd.read_csv(source/'paths.csv');observed=pd.read_csv(precision/'trial_contributions.csv')
    observed=observed[observed.subject==207].drop_duplicates('trial').sort_values('trial')
    labels={'baseline':'原参数','search':'错误促发搜索','retention':'累积保留证据','both':'二者同时'}
    colors={'baseline':'#0072B2','search':'#D55E00','retention':'#009E73','both':'#CC79A7'}
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,155/25.4))
    fig.subplots_adjust(left=.11,right=.98,bottom=.18,top=.90,hspace=.60,wspace=.35)
    for ax,objective,title in zip(axes[0],['NLL','GEN'],['逐试次似然评分','自主轨迹分布评分']):
        for i,variant in enumerate(labels):
            row=scores[(scores.variant==variant)&(scores.objective==objective)].iloc[0]
            ax.errorbar(i,row.delta_baseline,yerr=2*row.paired_delta_mc_se,color=colors[variant],
                fmt='o',ms=4,capsize=3,lw=1.)
        ax.axhline(0,color='#888888',lw=.7,ls=':')
        ax.set(title=title,ylabel=('平均 NLL' if objective=='NLL' else 'Energy score')+' − 原参数',
            xticks=range(4),xticklabels=list(labels.values()))
        ax.tick_params(axis='x',rotation=20)
    for variant,label in labels.items():
        c=curves[curves.variant==variant]
        axes[1,0].plot(c.block,c['mean'],color=colors[variant],lw=1.,label=label)
        axes[1,0].fill_between(c.block,c.q10,c.q90,color=colors[variant],alpha=.06,lw=0)
        values=np.sort(paths[paths.variant==variant].late_accuracy)
        axes[1,1].step(values,np.arange(1,len(values)+1)/len(values),where='post',color=colors[variant],lw=1.2)
    obs=observed.groupby('block').correct.mean()
    axes[1,0].plot(obs.index,obs,color='#222222',marker='.',ms=3,lw=1.)
    axes[1,0].set(title='自主生成学习曲线',xlabel='训练区块（每块 32 试次）',ylabel='正确率',ylim=(-.02,1.02),xticks=[1,5,9,13,17,21])
    axes[1,1].axvline(observed.correct.iloc[-128:].mean(),color='#222222',ls=':',lw=1.)
    axes[1,1].set(title='全部自主路径的末段分布',xlabel='末128试次正确率',ylabel='累计路径比例',xlim=(0,1),ylim=(0,1.02))
    for i,ax in enumerate(axes.flat):panel(ax,'abcd'[i])
    fig.suptitle('被试 207｜固定其他参数，改变搜索与证据保持',fontsize=10,y=.975)
    fig.legend(handles=[Line2D([],[],color=colors[v],label=l) for v,l in labels.items()]+
        [Line2D([],[],color='#222222',label='真实行为')],ncol=5,loc='lower center',bbox_to_anchor=(.5,.09),fontsize=6)
    fig.text(.5,.065,'a、b：±2配对 Monte Carlo SE；低于0表示评分改善。a：256粒子 × 8独立种子。',ha='center',fontsize=6)
    fig.text(.5,.04,'c：每组全部1024条路径的均值与10%–90%区间；d：虚线标记真实末段正确率。',ha='center',fontsize=6)
    fig.text(.5,.015,'错误增益0→0.5；证据保持系数0→1。全部使用训练段，其他参数固定，尚未重拟合。',ha='center',fontsize=6)
    save(fig,output,'subject_207_search_retention_probe')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe-source',type=Path)
    args=parser.parse_args();done=load(args.source/'completion.json')
    assert done['status']=='complete'
    if args.output.exists():raise FileExistsError('Use a new figure destination')
    args.output.mkdir(parents=True);font=configure_style()
    # PNG-only review contract; shared style keeps editable-text export settings.
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':7})
    contract=dict(conclusion='Separate numerical ranking stability from the autonomous mechanism discrepancy.',
        backend='python',archetype='quantitative grid',reuse_level='style-only inheritance; new panel design',
        width_mm=183,export='PNG only, 300 dpi scientific review',
        units='Independent complete filter seeds; complete autonomous trajectories. No population inference.',
        uncertainty='Score bars +/-2 MC SE; initial-rule groups show mean and 10th--90th percentiles.',
        selection='All candidates/budgets in completed audit; all 1024 paths per replay nominee.',
        sources=str(args.source.resolve()),source_sha256=digest(Path(__file__)),font=font,
        style_source_sha256=digest(Path(__file__).with_name('model0923_parameter_selection.py')),
        inputs_sha256={str(p.resolve()):digest(p) for source in [args.source,args.probe_source] if source
                      for p in sorted(source.glob('*.csv'))},
        versions=dict(python=platform.python_version(),**{p:version(p) for p in ['numpy','scipy','pandas','matplotlib','numba','PyYAML']}))
    (args.output/'figure_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
    shutil.copy2(__file__,args.output/'plot_source.py')
    rankings(args.source,args.output)
    if (args.source/'autonomous_summary.json').exists():mechanism(args.source,args.output)
    if args.probe_source:
        assert load(args.probe_source/'completion.json')['status']=='complete'
        probe_figure(args.probe_source,args.source,args.output)
        for name in ('scores','effects','curves','paths'):
            shutil.copy2(args.probe_source/(name+'.csv'),args.output/('probe_'+name+'.csv'))
        shutil.copy2(args.probe_source/'completion.json',args.output/'probe_provenance.json')
    for name in ('scores','contrasts','rankings','particle_drift','block_contributions','correct_error_contributions','autonomous_paths'):
        file=args.source/(name+'.csv')
        if file.exists():shutil.copy2(file,args.output/file.name)
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>个体参数精度与207机制复核</title>',
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#222}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ddd;padding:5px}</style>',
        '<h1>个体参数精度与207机制复核</h1><p>本页复核107、207、305、306的15组原最终候选。模型和参数固定，新滤波种子独立于原选择。所有评分使用训练段，标签A–D沿用原排名。</p>',
        '<h2>评分随粒子预算怎样变化</h2><img src="precision_scores.png"><p>误差条为固定参数评分的两倍Monte Carlo标准误；它只反映计算波动。粒子数有限的偏差和参数不确定性需分别检查。</p>',
        '<h2>其他候选能否与原A区分</h2><img src="precision_differences.png"><p>误差条使用候选间匹配的随机种子。负分差表示其他候选在这一批评分下优于原A；新批次最小值仍属于候选选择结果。</p>']
    if (args.output/'subject_207_mechanism.png').exists():
        parts.extend(['<h2>207：自主学习与条件状态</h2><img src="subject_207_mechanism.png">',
            '<p>上排用同一组参数比较条件预测与自主路径均值，并追踪规则42的作答权重。规则42在训练刺激的物理类别上达到完整对应，分析时才使用这一标记。</p>',
            '<p>左下分开累积正确回答和错误回答的log概率贡献。右下保留所有自主路径，按初始规则分组；小组样本量明确标注，组间差异仅作描述。</p>'])
    if args.probe_source:
        parts.extend(['<h2>207：搜索与证据保持的局部对照</h2><img src="subject_207_search_retention_probe.png">',
            '<p>以原NLL参数为起点，分别将搜索的错误增益从0改为0.5、保留证据的保持系数从0改为1，再同时改变二者。其他参数固定。四组使用相同的8个滤波种子和1024个自主路径种子。</p>',
            '<p>上排比较两个目标相对原参数的变化；下排展示自主学习曲线和末段正确率的完整经验分布。这是固定参数的局部诊断，重拟合与留出比较尚待进行。</p>'])
    parts.append('<h2>数值源表</h2><ul>')
    for file in sorted(args.output.glob('*.csv')):parts.append(f'<li><a href="{file.name}">{file.name}</a></li>')
    link=os.path.relpath(args.source.resolve(),args.output.resolve())
    parts.append(f'<li><a href="{link}/completion.json">分析记录与来源</a></li></ul></html>')
    (args.output/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    (args.output/'completion.json').write_text(json.dumps(dict(status='complete',source_completion_sha256=digest(args.source/'completion.json'),
        images=sorted(p.name for p in args.output.glob('*.png'))),indent=2))


if __name__=='__main__':main()
