"""Precision-refit evidence: objectives, parameter estimates and all paths.

Python/matplotlib discussion figures, 183 mm wide, PNG only at 300 dpi.
All nine subjects, four families, finalists, seeds and autonomous paths remain
in the tables. Path traces display 32-trial realized correctness averages.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import checked_arrays, block_paths
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import read, write, digest

ORDER=['ADD1','ADD10','LOG25','LOG100']
COLORS=dict(zip(ORDER,['#68778B','#3C8798','#A985B5','#A86289']))
MARKERS=dict(zip(ORDER,['o','s','^','D']))
STYLES=dict(zip(ORDER,['-','--',':','-.']))


def run(source: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new precision figure directory')
    manifest=read(source/'manifest.json')
    width=manifest['block_width']
    for name,sha in manifest['summary_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('Summary changed')
    tables={name:pd.read_csv(source/(name+'.csv')) for name in
            ('scores','contrasts','cohort_scores','curves','paths','training_finalists','training_contrasts','candidates')}
    subjects=sorted(tables['scores'].subject.unique())
    if len(subjects)!=9:raise ValueError('This layout requires the declared nine participants')
    output.mkdir(parents=True)
    write(output/'figure_contract.json',dict(
        conclusion='Compare training-selected precision families against independent choice scores and complete autonomous learning distributions.',
        evidence=['all training finalists','independent training and later-choice scores','full-curve Energy',
                'all autonomous paths','late-accuracy distributions','estimated precision coordinates']+
                (['training-only rate profile with other parameters fixed'] if 'rate_profile.csv' in manifest['summary_sha256'] else []),
        archetype='quantitative grid',backend='python',width_mm=183,dpi=300,formats=['PNG'],
        inclusion='all nine development participants; four families; every saved seed and path',
        statistics='pooled pre-choice NLL; empirical Energy V statistic; error bars are +/-2 complete-seed/path Monte Carlo SE',
        limitations='bounded 48-candidate search per family; no independent participant validation; uncertainty excludes refitting',
        versions=dict(python=sys.version,matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__),
        adaptation='Style-only inheritance from existing project figures; scores and panels follow the new refit protocol.'))
    for name in manifest['summary_sha256']:shutil.copy2(source/name,output/name)
    shutil.copy2(source/'manifest.json',output/'source_manifest.json')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'legend.fontsize':6,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'svg.fonttype':'none','pdf.fonttype':42})
    images=[]
    def save(fig,name):
        fig.savefig(output/name,dpi=300,facecolor='white');plt.close(fig);images.append(name)
    handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],ls=STYLES[m],lw=.8,label=m) for m in ORDER]
    # Every finalist remains visible; each panel retains its own absolute NLL scale.
    fig,axes=plt.subplots(3,3,figsize=(183/25.4,155/25.4))
    for ax,s in zip(axes.flat,subjects):
        rows=tables['training_finalists'][tables['training_finalists'].subject==s]
        for i,m in enumerate(ORDER):
            group=rows[rows.model==m].sort_values('nll')
            for j,r in enumerate(group.itertuples()):
                ax.errorbar(r.nll,i+(j-.5)*.22,xerr=2*r.mc_se,fmt=MARKERS[m],ms=4,
                    color=COLORS[m],mfc=COLORS[m] if r.selected else 'white',capsize=1.5,lw=.7)
        ax.set(title=f'S{s}',yticks=range(4),yticklabels=ORDER,xlabel='Training NLL / trial')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle('Final training comparison: filled = selected; open = competing finalist\nEach point ± 2 filter-seed jackknife SE; lower NLL is better',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'training_finalists.png')

    fig,axes=plt.subplots(1,3,figsize=(183/25.4,150/25.4))
    for ax,phase,metric,title in zip(axes,['training','test','whole'],['choice_nll','choice_nll','energy'],
            ['a  Training NLL\nNew filter seeds','b  Later-choice NLL','c  Autonomous curve distribution']):
        group=tables['contrasts'][tables['contrasts'].phase==phase].set_index(['subject','model'])
        ax.axvline(0,color='#9AA1AA',lw=.7)
        for i,m in enumerate(ORDER[1:]):
            rows=group.loc[[(s,m) for s in subjects]]
            ax.errorbar(rows['delta_'+metric],np.arange(len(subjects))+(i-1)*.19,
                xerr=2*rows['delta_'+metric+'_mc_se'],fmt=MARKERS[m],color=COLORS[m],ms=3,capsize=1.5,lw=.7)
        ax.set(yticks=range(len(subjects)),yticklabels=[f'S{s}' for s in subjects],title=title,
               xlabel=('NLL / trial' if metric=='choice_nll' else 'Energy score')+' change from ADD1\nNegative favors alternative')
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5,steps=[1,2,5,10]))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    fig.legend(handles[1:],ORDER[1:],loc='upper center',ncol=3,bbox_to_anchor=(.5,.91))
    fig.suptitle('Frozen parameters: independent filters and task-start autonomous paths\nError bars: ± 2 paired Monte Carlo SE; all participants are development data',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.86));save(fig,'objective_comparison.png')

    evaluation=Path(manifest['evaluation']);auto=read(evaluation/'autonomous/evaluation.json')['rows']
    path_blocks=[]
    for m in ORDER:
        fig,axes=plt.subplots(3,3,figsize=(183/25.4,160/25.4))
        for ax,s in zip(axes.flat,subjects):
            r=next(r for r in auto if r['subject']==s and r['model']==m)
            arrays=checked_arrays(evaluation/'autonomous',r)
            generated=arrays['choices']==arrays['categories'][None,:]
            samples=block_paths(generated,0,generated.shape[1],width)
            curve=tables['curves'][(tables['curves'].subject==s)&(tables['curves'].model==m)].sort_values('trial')
            np.testing.assert_allclose(samples.mean(axis=0),curve.autonomous,atol=1e-12,rtol=0)
            ax.plot(curve.trial,samples.T,color='#B5BDC7',alpha=.12,lw=.35,zorder=1)
            ax.fill_between(curve.trial,curve.lower,curve.upper,color=COLORS[m],alpha=.13,lw=0,zorder=2)
            ax.plot(curve.trial,curve.autonomous,color=COLORS[m],lw=1.2,zorder=3)
            ax.plot(curve.trial,curve.observed,color='#17191C',lw=1,zorder=4)
            ax.plot(curve.trial,curve.conditional,color='#B7843F',ls='--',lw=.9,zorder=4)
            ax.axvline(curve.training_trials.iloc[0],color='#8A929B',ls='--',lw=.7)
            ax.set(title=f'S{s}',ylim=(0,1.04),xlabel='Trial',ylabel='Accuracy')
            for i,seed in enumerate(arrays['seeds']):
                path_blocks.extend(dict(subject=s,model=m,seed=int(seed),trial=int(t),accuracy=value)
                    for t,value in zip(curve.trial,samples[i]))
        fig.suptitle(f'{m}: all 128 autonomous paths (gray); participant (black)\nSolid color = autonomous mean; gold dashes = conditional prediction\nBand = path 10–90% range; vertical dashes = training boundary; {width}-trial blocks',fontsize=8)
        fig.tight_layout(rect=(0,0,1,.90));save(fig,f'paths_{m}.png')
    pd.DataFrame(path_blocks).to_csv(output/'path_blocks.csv',index=False)

    fig,axes=plt.subplots(3,3,figsize=(183/25.4,145/25.4))
    for ax,s in zip(axes.flat,subjects):
        group=tables['paths'][tables['paths'].subject==s]
        for m in ORDER:
            x=np.sort(group[group.model==m].late_accuracy)
            ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=COLORS[m],ls=STYLES[m],lw=1)
        ax.axvline(group.observed_late.iloc[0],color='#17191C',ls='--',lw=.8)
        ax.set(title=f'S{s}',xlim=(0,1.02),ylim=(0,1.03),xlabel='Final 128-trial accuracy',ylabel='Path fraction ≤ x')
    fig.legend(handles,ORDER,loc='upper center',ncol=4,bbox_to_anchor=(.5,.92))
    fig.suptitle('Every autonomous path: final performance distribution\nVertical dashed line = participant; a narrower distribution is useful only when calibrated',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.87));save(fig,'late_distribution.png')

    selected=tables['scores'][tables['scores'].phase=='whole'].set_index(['subject','model'])
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,100/25.4))
    for ax,field,title in zip(axes,['beta_init','effective_rate'],['a  Initial precision','b  Learning rate in the family coordinate']):
        for i,m in enumerate(ORDER):
            group=selected.loc[[(s,m) for s in subjects]]
            ax.plot(np.arange(len(subjects))+(i-1.5)*.14,group[field],MARKERS[m],color=COLORS[m],ms=4,label=m)
        ax.set(xticks=range(len(subjects)),xticklabels=subjects,title=title,xlabel='Participant',ylabel=field.replace('_',' '))
        if field=='effective_rate':
            ax.set_yscale('symlog',linthresh=.0001)
            ax.set_ylim(-.00002,20)
    fig.legend(handles,ORDER,loc='upper center',ncol=4,bbox_to_anchor=(.5,.90))
    fig.suptitle('Training-selected precision parameters\nAdditive and log-coordinate learning rates have different units; zero is retained on the rate axis',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.82));save(fig,'precision_parameters.png')
    if 'rate_profile.csv' in manifest['summary_sha256']:
        profile=pd.read_csv(source/'rate_profile.csv').sort_values('effective_rate')
        if profile.subject.nunique()!=1 or profile.particles.nunique()!=1 or profile.seeds.nunique()!=1:
            raise ValueError('Rate profile requires one subject and a common filter budget')
        fig,axes=plt.subplots(1,2,figsize=(183/25.4,95/25.4))
        for ax,field,error,title in zip(axes,['choice_nll','delta_from_reference'],['mc_se','delta_mc_se'],
                ['a  Training NLL at fixed other parameters','b  Difference from frozen learning rate']):
            ax.axvspan(0,1,color='#DDE3E8',alpha=.5,zorder=0)
            ax.axvline(1,color='#8A929B',ls='--',lw=.7)
            ax.errorbar(profile.effective_rate,profile[field],yerr=2*profile[error],
                color=COLORS['ADD10'],marker='s',ms=4,lw=.8,capsize=2)
            if field=='delta_from_reference':ax.axhline(0,color='#8A929B',lw=.7)
            ax.set_xscale('symlog',linthresh=.03)
            ax.set_xlim(-.003,12)
            ax.set(title=title,xlabel='Effective additive learning rate',ylabel='NLL / trial',
                xticks=[0,.03,.1,.3,1,3,10],xticklabels=['0','.03','.1','.3','1','3','10'])
        first=profile.iloc[0]
        fig.suptitle(f'S{int(first.subject)}: learning-rate sweep at the frozen ADD10 parameter point\n'
            f'{int(first.particles):,} particles × {int(first.seeds)} new filters; all other parameters fixed\n'
            'Shaded region = original ADD1 rate range; errors = ± 2 Monte Carlo SE',fontsize=8)
        fig.tight_layout(rect=(0,0,1,.93));save(fig,'rate_profile.png')
    content='<!doctype html><meta charset="utf-8"><title>β学习参数重拟合</title><style>body{font-family:sans-serif;max-width:1100px;margin:32px auto;line-height:1.65}img{max-width:100%}table{border-collapse:collapse}td,th{padding:6px 14px;border-bottom:1px solid #ddd;text-align:right}</style>'
    content+='<h1>β学习参数与共同H参数：等预算重拟合</h1><p>四方案每人各48个候选。ADD1：原加法学习率0—1、上限25；ADD10：加法有效学习率0—10、上限25；LOG25与LOG100：在log β上学习，上限分别为25和100。共同H参数获得相同搜索预算。九人均为开发资料。</p>'
    content+='<p>训练阶段仅使用训练段作答前的选择概率。每个方案通过独立种子复核后冻结参数；图中的后段选择评分随后逐题吸收已发生的选择，自主路径从任务起点使用自身回答产生反馈。误差线表示有限滤波或路径模拟的不确定性，不包含参数估计和人群差异。</p>'
    cohort=tables['cohort_scores'].set_index(['model','phase'])
    overview=pd.DataFrame([{'方案':m,
        '训练NLL（新种子）':cohort.loc[m,'training'].participant_mean_choice_nll,
        '后段NLL':cohort.loc[m,'test'].participant_mean_choice_nll,
        '全程自主Energy':cohort.loc[m,'whole'].participant_mean_energy} for m in ORDER])
    content+='<h2>九人等权汇总</h2><p>以下三项均以较低为好。它们分别检验逐题选择概率与完整生成分布；具体差异和模拟误差见下图。源表同时提供按题目数加权的NLL。</p>'
    content+=overview.to_html(index=False,border=0,float_format=lambda value:f'{value:.5f}')
    if 'rate_profile.csv' in manifest['summary_sha256']:
        content+='<p>学习率对照图固定ADD10所选点的其余参数，仅用训练段改变有效学习率；它用于区分学习率与共同H参数变化的作用。原先冻结的个体参数和后段评估保持原定义。</p>'
    content+=f'<p>学习曲线图保留每条自主路径，灰线为{width}题区块正确率；彩线为均值，色带为10%—90%分位数。黑线为真实被试，竖虚线标出参数拟合的训练边界。金色虚线为条件预测的区块正确作答概率。灰色路径使用模型自己的选择和反馈，条件预测使用已经发生的真实选择和反馈。路径分布的宽度、覆盖和整段评分需要一起理解。</p>'
    for name in manifest['summary_sha256']:content+=f'<a href="{name}">{name}</a> · '
    content+='<a href="path_blocks.csv">全部路径区块值</a>'
    for name in images:content+=f'<h2>{name}</h2><img src="{name}">'
    (output/'index.html').write_text(content)
    shutil.copy2(__file__,output/'plot_source.py')
    write(output/'qa_numeric.json',dict(status='pass',source_manifest_sha256=digest(source/'manifest.json'),
        plot_source_sha256=digest(Path(__file__)),
        source_rows={name:len(frame) for name,frame in tables.items()},path_blocks=len(path_blocks),
        images_sha256={name:digest(output/name) for name in images},
        export_exception='User policy: PNG only; 300 dpi discussion figures. Submission vector exports are outside this task.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
