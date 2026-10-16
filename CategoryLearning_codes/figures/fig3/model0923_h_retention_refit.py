"""PNG comparison of refitted retention policies and held-out generation.

Uses every registered participant and every saved path. Development and new
validation participants remain visibly distinct. All score error bars describe
Monte Carlo error at fixed parameters. Source CSVs and provenance accompany
the discussion figures; --output must be new.
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
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .model0923_parameter_selection import configure_style
from src.Bayesian_state.evaluation.model_0923_h_validation import load,digest
from src.Bayesian_state.evaluation.model_0923_h_retention_refit import aggregate_prefix_cohorts

NAMES={'FREE':'独立保持率','CUMULATIVE':'持续累积','SHARED':'共享记忆保持率'}
COLORS={'FREE':'#0072B2','CUMULATIVE':'#D55E00','SHARED':'#009E73'}
STYLES={'FREE':'-','CUMULATIVE':'--','SHARED':':'}
ORDER=[107,108,205,207,305,306,109,208,308]
GRID=[107,205,305,108,207,306,109,208,308]


def save(fig,output: Path,name: str) -> None:
    fig.savefig(output/(name+'.png'),dpi=300,facecolor='white');plt.close(fig)


def subjects(ax) -> None:
    ax.set_yticks(range(len(ORDER)),[str(s)+('（新增）' if s in ORDER[-3:] else '') for s in ORDER])
    ax.set_ylim(len(ORDER)-.5,-.5)
    ax.axhspan(5.5,8.5,color='#999999',alpha=.07,zorder=-1)
    ax.axhline(5.5,color='#AAAAAA',lw=.5)


def comparison(source: Path,output: Path) -> None:
    data=pd.read_csv(source/'contrasts.csv');data=data[data.phase=='test']
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,145/25.4))
    fig.subplots_adjust(left=.13,right=.98,bottom=.19,top=.87,wspace=.40)
    for ax,metric,title in zip(axes,['choice_nll','feature_energy'],['逐次选择预测','从起点自主生成']):
        for model,offset,marker in [('CUMULATIVE',-.13,'o'),('SHARED',.13,'s')]:
            values=data[data.model==model].set_index('subject').loc[ORDER]
            ax.errorbar(values['delta_'+metric],np.arange(len(ORDER))+offset,
                xerr=2*values[metric+'_paired_mc_se'],fmt=marker,ms=4,
                color=COLORS[model],capsize=2,lw=.9,label=NAMES[model])
        subjects(ax);ax.axvline(0,color='#444444',lw=.7,ls=':')
        ax.set(title=title,xlabel=('平均NLL' if metric=='choice_nll' else '特征Energy score')+'差值（低于0为改善）')
    axes[0].set_ylabel('被试')
    for i,ax in enumerate(axes):ax.text(-.23,1.06,'ab'[i],transform=ax.transAxes,fontweight='bold',fontsize=8)
    fig.suptitle('重拟合后的留出表现：相对独立保持率方案',fontsize=10,y=.975)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,fontsize=7)
    fig.text(.5,.055,'每点为一名被试；误差条为±2配对计算标准误差，未包含参数估计不确定性。',ha='center',fontsize=6)
    fig.text(.5,.025,'原六人属于开发资料；下方三人按编号预选，参数冻结后评分留出段。',ha='center',fontsize=6)
    save(fig,output,'retention_heldout_scores')


def trajectories(source: Path,output: Path,prefix: Path | None=None) -> None:
    data=pd.read_csv((prefix/'blocks.csv') if prefix else (source/'curves.csv'))
    if prefix:data=data[data.start=='filtered_prefix']
    fig,axes=plt.subplots(3,3,figsize=(183/25.4,200/25.4),sharey=True)
    fig.subplots_adjust(left=.08,right=.99,bottom=.16,top=.91,hspace=.55,wspace=.20)
    for i,(subject,ax) in enumerate(zip(GRID,axes.flat)):
        for model in NAMES:
            rows=data[(data.subject==subject)&(data.model==model)].sort_values('first_trial')
            x=(rows.first_trial-1)/32+1
            mean=rows.realized_mean if prefix else rows.autonomous
            low=rows.realized_q10 if prefix else rows.q10;high=rows.realized_q90 if prefix else rows.q90
            ax.plot(x,mean,color=COLORS[model],ls=STYLES[model],lw=1.)
            ax.fill_between(x,low,high,color=COLORS[model],alpha=.07,lw=0)
            if model=='FREE':
                ax.plot(x,rows.observed,color='#222222',marker='.',ms=3,lw=.8)
                if not prefix:
                    cut=rows.training_trials.iloc[0]/32+.5
                    ax.axvspan(cut,x.max()+.5,color='#555555',alpha=.07,zorder=-2)
                    ax.axvline(cut,color='#888888',lw=.6,ls=':')
        ax.set(title=f'{subject}'+('｜新增' if subject in ORDER[-3:] else ''),ylim=(-.02,1.02))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
        ax.text(-.18,1.07,'abcdefghi'[i],transform=ax.transAxes,fontweight='bold',fontsize=8)
        if i%3==0:ax.set_ylabel('正确率')
        if i>=6:ax.set_xlabel('区块（每块32试次）')
    title='给定训练状态后的自主留出轨迹' if prefix else '从任务起点生成的完整学习轨迹'
    fig.suptitle(title,fontsize=10,y=.97)
    fig.legend(handles=[Line2D([],[],color=COLORS[m],ls=STYLES[m],label=NAMES[m]) for m in NAMES]+
        [Line2D([],[],color='#222222',marker='.',label='真实行为')],loc='lower center',bbox_to_anchor=(.5,.085),ncol=4,fontsize=6)
    caption=('每组8个训练滤波，每个滤波16个未来分支；未来回答由模型自主生成。' if prefix else
        '每组全部512条路径；灰色区域为留出段，生成过程始终使用自身回答和反馈。')
    fig.text(.5,.058,caption,ha='center',fontsize=6)
    fig.text(.5,.035,'线为路径均值，色带为逐块10%–90%分位数；色带描述行为路径分布。',ha='center',fontsize=6)
    fig.text(.5,.012,'各列分别对应条件1、2、3；底行是本轮新增被试。外部刺激使用实际记录顺序。',ha='center',fontsize=6)
    save(fig,output,'retention_prefix_paths' if prefix else 'retention_autonomous_paths')


def retention_parameters(source: Path,output: Path) -> None:
    values=pd.read_csv(source/'retention_parameters.csv');differences=pd.read_csv(source/'nested_search_diagnostic.csv')
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,145/25.4))
    fig.subplots_adjust(left=.13,right=.98,bottom=.19,top=.87,wspace=.40)
    for model,offset,marker in [('FREE',-.2,'D'),('CUMULATIVE',0.,'o'),('SHARED',.2,'s')]:
        data=values[values.model==model].set_index('subject').loc[ORDER]
        axes[0].plot(data.effective_evidence_decay,np.arange(len(ORDER))+offset,marker,ms=4,color=COLORS[model],label=NAMES[model])
        if model!='FREE':
            d=differences[differences.model==model].set_index('subject').loc[ORDER]
            axes[1].errorbar(d.delta_nll_from_free,np.arange(len(ORDER))+offset-.1,
                xerr=2*d.paired_mc_se,fmt=marker,ms=4,color=COLORS[model],capsize=2,lw=.9)
    for ax in axes:subjects(ax)
    axes[0].set(title='所选参数的证据保持率',xlabel='保持率λ',xlim=(-.05,1.05));axes[0].set_ylabel('被试')
    axes[1].set(title='独立种子确认后的训练分差',xlabel='平均NLL − 独立保持率方案')
    axes[1].axvline(0,color='#444444',ls=':',lw=.7)
    for i,ax in enumerate(axes):ax.text(-.23,1.06,'ab'[i],transform=ax.transAxes,fontweight='bold',fontsize=8)
    fig.suptitle('保持率的估计与有限预算下的搜索结果',fontsize=10,y=.975)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,fontsize=6)
    fig.text(.5,.055,'右图：±2配对计算标准误差；每组8种子，305用4096粒子，其余用1024粒子。',ha='center',fontsize=6)
    fig.text(.5,.025,'自由方案能表示两个受约束方案的参数点；负分差同时提示自由方案的搜索可能不充分。',ha='center',fontsize=6)
    save(fig,output,'retention_parameters_and_search')


def late_distributions(source: Path,output: Path) -> None:
    data=pd.read_csv(source/'paths.csv')
    fig,axes=plt.subplots(3,3,figsize=(183/25.4,180/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.91,hspace=.45,wspace=.20)
    for i,(subject,ax) in enumerate(zip(GRID,axes.flat)):
        own=data[data.subject==subject]
        assert own.observed_late_accuracy.nunique()==1
        for model in NAMES:
            values=np.sort(own[own.model==model].late_accuracy.to_numpy())
            assert len(values)==512
            ax.step(np.r_[0,values,1],np.r_[0,np.arange(1,len(values)+1)/len(values),1],
                    where='post',color=COLORS[model],ls=STYLES[model],lw=1.)
        ax.axvline(own.observed_late_accuracy.iloc[0],color='#222222',ls='--',lw=.9)
        ax.set(title=f'{subject}'+('｜新增' if subject in ORDER[-3:] else ''),
               xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1])
        ax.text(-.22,1.07,'abcdefghi'[i],transform=ax.transAxes,fontweight='bold',fontsize=8)
        if i%3==0:ax.set_ylabel('累计路径比例')
        if i>=6:ax.set_xlabel('末128题正确率')
    fig.suptitle('自主路径的末段表现分布',fontsize=10,y=.97)
    fig.legend(handles=[Line2D([],[],color=COLORS[m],ls=STYLES[m],label=NAMES[m]) for m in NAMES]+
        [Line2D([],[],color='#222222',ls='--',label='真实末段正确率')],
        loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,fontsize=6)
    fig.text(.5,.05,'每条线纳入全部512条路径：纵轴表示正确率不超过横轴值的路径比例。',ha='center',fontsize=6)
    fig.text(.5,.025,'黑色虚线为被试真实值；这里描述固定刺激重放的末段表现，未估计完整任务达标概率。',ha='center',fontsize=6)
    save(fig,output,'retention_late_distributions')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--prefix-source',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    done=load(a.source/'completion.json');assert done['status']=='complete' and set(done['subjects'])==set(ORDER)
    if a.prefix_source:
        prefix_manifest=load(a.prefix_source/'comparison_manifest.json')
        assert prefix_manifest['status']=='complete'
        assert prefix_manifest['nomination_sha256']==done['nomination_sha256']
    if a.output.exists():raise FileExistsError('Use a new figure output')
    a.output.mkdir(parents=True);font=configure_style()
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':7})
    comparison(a.source,a.output);trajectories(a.source,a.output);retention_parameters(a.source,a.output)
    late_distributions(a.source,a.output)
    if a.prefix_source:
        trajectories(a.source,a.output,a.prefix_source)
    for file in a.source.glob('*.csv'):shutil.copy2(file,a.output/file.name)
    if a.prefix_source:
        for file in a.prefix_source.glob('*.csv'):shutil.copy2(file,a.output/('prefix_'+file.name))
        prefix_cohort=aggregate_prefix_cohorts(pd.read_csv(a.prefix_source/'scores.csv'),
            pd.read_csv(a.prefix_source/'model_contrasts.csv'),done['settings'])
        prefix_cohort.to_csv(a.output/'prefix_cohort_aggregate.csv',index=False)
    shutil.copy2(__file__,a.output/'plot_source.py')
    contract=dict(conclusion='Compare predictive consequences of three retention-time assumptions after refitting.',
        archetype='quantitative grid',backend='python',reuse='style-only inheritance; new scientific panels',
        width_mm=183,export='PNG only; 183 mm; 300 dpi discussion figures',font=font,
        uncertainty='Complete-filter/path Monte Carlo errors; full-path 10th--90th percentile bands; no population confidence interval.',
        data='All nine registered participants and all saved paths. Six development and three new H-study participants separated.',
        inputs_sha256={str(f.resolve()):digest(f) for s in [a.source,a.prefix_source] if s for f in s.glob('*.csv')},
        source_sha256=digest(Path(__file__)),versions=dict(python=platform.python_version(),
            **{k:version(k) for k in ['numpy','scipy','pandas','matplotlib']}))
    (a.output/'figure_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
    aggregate=pd.read_csv(a.source/'aggregate.csv').query("phase=='test'").copy()
    labels={'development':'原六人（开发）','new_validation':'新增三人'}
    aggregate['cohort']=aggregate.cohort.map(labels);aggregate['model']=aggregate.model.map(NAMES)
    table=aggregate[['cohort','model','subjects','choice_nll','delta_choice_nll','feature_energy','delta_feature_energy']].rename(
        columns=dict(cohort='被试组',model='方案',subjects='人数',choice_nll='留出NLL',delta_choice_nll='NLL差值',
                     feature_energy='自主特征Energy',delta_feature_energy='特征Energy差值')).to_html(index=False,float_format=lambda x:f'{x:.4f}')
    pooled=pd.read_csv(a.source/'pooled_search_aggregate.csv').query("phase=='test'").copy()
    pooled['cohort']=pooled.cohort.map(labels);pooled['model']=pooled.model.map(NAMES)
    pooled=pooled[['cohort','model','delta_choice_nll','delta_feature_energy']].rename(columns=dict(
        cohort='被试组',model='原方案',delta_choice_nll='相对合并参考的NLL差值',
        delta_feature_energy='相对合并参考的特征Energy差值'))
    documentation=Path(__file__).resolve().parents[3]/'src/Bayesian_state/docs/model_architecture/model_0923_h_retention_refit.md'
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>规则证据保持机制重拟合</title>',
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#222}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:5px}</style>',
        '<h1>规则证据保持机制重拟合</h1><p>三方案共享搜索、证据定义、容量处理及作答机制。独立保持率估计14项参数，持续累积和共享记忆保持率各估计13项。全部参数仅使用训练段估计，最终确认后冻结。</p>',
        '<p>107、108、205、207、305、306属于已有开发资料；109、208、308按编号预先选定。新三人与原六人分别汇总。此处比较有界搜索结果；参数恢复和完整自适应任务检验仍需另行开展。</p>',
        f'<p><a href="{os.path.relpath(documentation,a.output.resolve())}">完整方案、结果解释与复现方法</a></p>',
        '<h2>分组留出结果</h2><p>按被试等权平均；差值均相对独立保持率，负值表示改善。两个评分均为越低越好。计算误差见下图及源表。</p>',table,
        '<h2>留出评分</h2><img src="retention_heldout_scores.png">',
        '<h2>起点自主生成</h2><img src="retention_autonomous_paths.png">',
        '<h2>末段表现的完整路径分布</h2><img src="retention_late_distributions.png">',
        '<h2>保持率与训练搜索</h2><img src="retention_parameters_and_search.png">',
        '<h2>补充：合并训练候选的参考点</h2><p>每人按训练确认NLL从三个提名点中选出一个参考点，并在留出评价前冻结。自由保持率模型能表示这些点；预测复用对应提名点的全部滤波和路径。这个参考汇集三套搜索预算，用于检查候选搜索遗漏对结论的影响。上方等预算结果保留原定义。逐人源表同时标记参考点是否满足当前方案的约束；满足时，分差反映同一可表达空间内不同参数点的比较。</p>',
        pooled.to_html(index=False,float_format=lambda x:f'{x:.4f}')]
    if a.prefix_source:
        table=prefix_cohort.query("start=='filtered_prefix'").copy()
        table['cohort']=table.cohort.map(labels);table['model']=table.model.map(NAMES)
        table=table[['cohort','model','subjects','energy','delta_energy']].rename(columns=dict(
            cohort='被试组',model='方案',subjects='人数',energy='续跑曲线Energy',delta_energy='曲线Energy差值'))
        parts+=['<h2>训练状态之后的自主预测</h2><img src="retention_prefix_paths.png">',
            '<p>所有未来分支只接收未来刺激和模型自身回答对应的反馈。128个分支共享8个滤波起点，计算不确定性按完整滤波组估计。下表对正确率曲线评分，定义与上方行为特征评分分别记录。</p>',
            table.to_html(index=False,float_format=lambda x:f'{x:.4f}')]
    parts+=['<h2>逐人和分组源表</h2><ul>']
    parts += [f'<li><a href="{f.name}">{f.name}</a></li>' for f in sorted(a.output.glob('*.csv'))]
    parts += [f'<li><a href="{os.path.relpath(a.source.resolve(),a.output.resolve())}/completion.json">分析来源</a></li></ul></html>']
    (a.output/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    (a.output/'completion.json').write_text(json.dumps(dict(status='complete',images=sorted(f.name for f in a.output.glob('*.png')),
        summary_sha256=digest(a.source/'completion.json')),indent=2))


if __name__=='__main__':main()
