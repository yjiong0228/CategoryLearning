"""All-subject episodic-admission diagnostics; Python, 183 mm, PNG only.

The evidence chain separates pre-choice training likelihood, autonomous curve
distributions and discovery/retention/use. Every saved path is displayed and
preserved in source tables. Intervals describe Monte Carlo error at fixed
parameters; these nine participants are development data.
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
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_0923_h_validation import read,write,digest

ORDER = ['BASE','LAST','NEW','POOL','DRAW']
COLORS = dict(zip(ORDER,['#68778B','#B49A7B','#78A7AD','#277F8E','#A16F99']))
MARKERS = dict(zip(ORDER,['o','v','s','D','^']))
LINES = dict(zip(ORDER,['-',':','--','-','-.']))


def run(source: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new episodic figure directory')
    manifest=read(source/'manifest.json')
    for name,sha in manifest['tables'].items():
        if digest(source/name)!=sha:raise ValueError('Summary table changed')
    tables={Path(name).stem:pd.read_csv(source/name) for name in manifest['tables']}
    settings=manifest['settings'];subjects=settings['subjects'];arms=settings['arms']
    if len(subjects)!=9 or arms!=ORDER:raise ValueError('This layout requires the full nine-by-five comparison')
    output.mkdir(parents=True)
    write(output/'figure_contract.json',dict(
        conclusion='Historical evidence and comparable incumbent/newcomer scores can improve autonomous learning; the gains must be assessed alongside training likelihood and remaining discovery failures.',
        evidence=['training NLL versus whole-curve Energy','correct/error contributions to training NLL',
                  'discovery, late presence and choice contribution',
                  'all 64 paths for every subject and arm','late-performance distributions'],
        archetype='quantitative grid',backend='python',width_mm=183,dpi=300,formats=['PNG'],
        inclusion='all nine development participants, all five arms and all recorded paths/filter seeds',
        statistics='paired complete-filter/path Monte Carlo SE, empirical Energy V statistic; no population significance tests',
        adaptation='style-only inheritance; new fields, preserved temporal boundaries and complete paths',
        limits='fixed ADD10 parameter points; no mechanism refit or independent participant validation',
        versions=dict(python=sys.version,matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__)))
    for name in manifest['tables']:shutil.copy2(source/name,output/name)
    shutil.copy2(source/'manifest.json',output/'source_manifest.json')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
        'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'svg.fonttype':'none','pdf.fonttype':42})
    images=[]
    def save(fig,name):
        fig.savefig(output/name,dpi=300,facecolor='white');plt.close(fig);images.append(name)
    handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],ls=LINES[m],lw=.8,label=m) for m in arms]
    if 'training' in tables:
        training=tables['training'];runs=list(training.run.unique())
        for run_name in runs:
            tr=training[training.run==run_name].set_index(['subject','model'])
            tr_subjects=[s for s in subjects if s in training[training.run==run_name].subject.values]
            tr_arms=[m for m in arms[1:] if m in tr.index.get_level_values('model')]
            fig,axes=plt.subplots(1,2,figsize=(183/25.4,145/25.4))
            contrast=tables['contrasts'].set_index(['subject','model'])
            for ax,metric,title in zip(axes,['nll','energy'],['a  Training choice likelihood','b  Autonomous curve distribution']):
                ax.axvline(0,color='#9AA1AA',lw=.7)
                for i,m in enumerate(tr_arms):
                    data=tr if metric=='nll' else contrast
                    available=[s for s in tr_subjects if (s,m) in tr.index]
                    rows=data.loc[[(s,m) for s in available]]
                    positions=[subjects.index(s)+(i-(len(tr_arms)-1)/2)*.15 for s in available]
                    ax.errorbar(rows['delta_'+metric],positions,xerr=2*rows['delta_'+metric+'_mc_se'],
                        fmt=MARKERS[m],color=COLORS[m],ms=3,capsize=1.5,lw=.7)
                ax.set(yticks=range(len(subjects)),yticklabels=[f'S{s}' for s in subjects],title=title,
                       xlabel=('NLL / trial' if metric=='nll' else 'Energy score')+' change from BASE\nNegative favors alternative')
                ax.set_ylim(len(subjects)-.4,-.6);ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            fig.legend([handles[arms.index(m)] for m in tr_arms],tr_arms,loc='upper center',ncol=4,bbox_to_anchor=(.5,.90))
            budgets=sorted(training[training.run==run_name].particles.unique())
            seeds=int(training[training.run==run_name].seeds.iloc[0])
            fig.suptitle(f'Fixed parameters: {" / ".join(map(str,budgets))} particles × {seeds} filter seeds\nError bars: ± 2 paired Monte Carlo SE; all participants are development data',fontsize=8)
            fig.tight_layout(rect=(0,0,1,.84));save(fig,'objectives_'+run_name+'.png')

    if 'training_outcomes' in tables:
        training=tables['training']
        run_name=training.assign(work=training.particles*training.seeds).groupby('run').work.mean().idxmax()
        outcomes=tables['training_outcomes'];outcomes=outcomes[outcomes.run==run_name]
        selected_arms=[m for m in ('POOL','DRAW') if m in outcomes.model.values]
        if len(selected_arms)!=2:raise ValueError('Outcome diagnostic requires POOL and DRAW')
        fig=plt.figure(figsize=(183/25.4,170/25.4));grid=fig.add_gridspec(2,2,height_ratios=[2,1])
        axes=[fig.add_subplot(grid[0,i]) for i in range(2)]
        outcome_colors={'correct':'#46899A','incorrect':'#BB8B62'}
        for ax,m,label in zip(axes,selected_arms,['a','b']):
            data=outcomes[outcomes.model==m].set_index(['subject','outcome'])
            ax.axvline(0,color='#9AA1AA',lw=.7)
            for i,outcome in enumerate(('correct','incorrect')):
                rows=data.loc[[(s,outcome) for s in subjects]]
                ax.errorbar(rows.contribution,np.arange(len(subjects))+(i-.5)*.3,
                    xerr=2*rows.contribution_mc_se,fmt='o' if i==0 else 's',ms=3,
                    color=outcome_colors[outcome],lw=.7,capsize=1.5)
            net=data.groupby('subject').contribution.sum().reindex(subjects)
            ax.plot(net,np.arange(len(subjects)),'d',color='#292D35',ms=3)
            ax.set(title=f'{label}  {m}: which responses change the score?',
                yticks=range(len(subjects)),yticklabels=[f'S{s}' for s in subjects],
                xlabel='Contribution to total NLL change / trial\nPositive = worse than BASE')
            ax.set_ylim(len(subjects)-.4,-.6);ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax=fig.add_subplot(grid[1,:]);timeline=tables['training_blocks']
        for m in selected_arms:
            rows=timeline[(timeline.run==run_name)&(timeline.subject==208)&(timeline.model==m)].sort_values('trial')
            ax.errorbar(rows.trial,rows.delta_nll,yerr=2*rows.delta_nll_mc_se,color=COLORS[m],
                marker=MARKERS[m],ls=LINES[m],ms=3,lw=.8,capsize=1.5,label=m)
        ax.axhline(0,color='#9AA1AA',lw=.7)
        ax.set(title='c  S208: timing of the training-score change',xlabel='Training trial',
               ylabel='Block NLL / trial change')
        ax.legend(ncol=2,loc='upper left')
        outcome_handles=[Line2D([],[],color=outcome_colors[o],marker='o' if o=='correct' else 's',ls='',label=o)
                         for o in ('correct','incorrect')]+[Line2D([],[],color='#292D35',marker='d',ls='',label='total')]
        fig.legend(outcome_handles,['Observed correct','Observed incorrect','Total'],loc='upper center',
                   ncol=3,bbox_to_anchor=(.5,.94))
        fig.suptitle('Autonomous gains can coexist with a higher cost for observed errors\nTop: contributions sum to the total change; error bars: ± 2 filter-seed MC SE',fontsize=8)
        fig.tight_layout(rect=(0,0,1,.90));save(fig,'error_contributions.png')

    frame=tables['scores'].set_index(['subject','model'])
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,155/25.4))
    for ax,key,title in zip(axes,['discovery','late_presence','late_contribution'],
            ['a  Ever active','b  Active in final 128 trials','c  Choice weight in final 128 trials']):
        for i,m in enumerate(arms):
            rows=frame.loc[[(s,m) for s in subjects]]
            ax.plot(rows[key],np.arange(len(subjects))+(i-2)*.15,MARKERS[m],color=COLORS[m],ms=3)
        ax.set(xlim=(-.03,1.03),yticks=range(len(subjects)),yticklabels=[f'S{s}' for s in subjects],
               title=title,xlabel='Fraction' if key!='late_contribution' else 'Mean choice weight')
        ax.set_ylim(len(subjects)-.4,-.6);ax.set_xticks([0,.5,1.])
    fig.legend(handles,arms,loc='upper center',ncol=5,bbox_to_anchor=(.5,.91))
    fig.suptitle('Task-correct rule: discovery, retention and influence on choices\n64 complete paths per cell; target identity is used only for diagnosis',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.86));save(fig,'rule_process.png')

    for m in arms:
        fig,axes=plt.subplots(3,3,figsize=(183/25.4,165/25.4))
        for ax,s in zip(axes.flat,subjects):
            curve=tables['curves'][(tables['curves'].subject==s)&(tables['curves'].model==m)].sort_values('trial')
            values=tables['path_blocks'][(tables['path_blocks'].subject==s)&(tables['path_blocks'].model==m)]
            samples=values.pivot(index='seed',columns='trial',values='accuracy').sort_index(axis=1)
            if len(samples)!=settings['paths']:raise ValueError('Incomplete path panel')
            np.testing.assert_allclose(samples.mean(axis=0),curve.autonomous,atol=1e-12,rtol=0)
            ax.plot(samples.columns,samples.to_numpy().T,color='#B5BDC7',alpha=.18,lw=.35,zorder=1)
            ax.fill_between(curve.trial,curve.lower,curve.upper,color=COLORS[m],alpha=.13,lw=0,zorder=2)
            ax.plot(curve.trial,curve.autonomous,color=COLORS[m],lw=1.2,zorder=3)
            ax.plot(curve.trial,curve.observed,color='#17191C',lw=1,zorder=4)
            ax.axvline(curve.training_trials.iloc[0],color='#8A929B',ls='--',lw=.7)
            ax.set(title=f'S{s}',ylim=(0,1.04),xlabel='Trial',ylabel='Accuracy')
        fig.suptitle(f'{m}: all 64 autonomous paths (gray); participant (black)\nColor = path mean; band = 10–90% range; vertical dashes = training boundary\nRecorded stimuli, self-generated choices and feedback; {settings["features"]["block_width"]}-trial blocks',fontsize=8)
        fig.tight_layout(rect=(0,0,1,.90));save(fig,f'paths_{m}.png')

    fig,axes=plt.subplots(3,3,figsize=(183/25.4,145/25.4))
    for ax,s in zip(axes.flat,subjects):
        group=tables['paths'][tables['paths'].subject==s]
        for m in arms:
            x=np.sort(group[group.model==m].late_accuracy)
            ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=COLORS[m],ls=LINES[m],lw=1)
        ax.axvline(group.observed_late.iloc[0],color='#17191C',ls='--',lw=.8)
        ax.set(title=f'S{s}',xlim=(0,1.02),ylim=(0,1.03),xlabel='Final 128-trial accuracy',ylabel='Path fraction ≤ x')
    fig.legend(handles,arms,loc='upper center',ncol=5,bbox_to_anchor=(.5,.92))
    fig.suptitle('Final performance distributions retain every autonomous path\nVertical dashed line = participant',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.87));save(fig,'late_distribution.png')
    labels=['BASE：原准入','LAST：一题反馈','NEW：候选回顾历史','POOL：新旧规则同口径比较','DRAW：抽取共同记忆权重']
    body='<h1>Model 0923：历史反馈与候选准入</h1><p>九名开发被试，固定ADD10参数；各方案64条自主路径。'+ '；'.join(labels)+'</p>'
    body+='<p>历史权重使用H证据保持率δ；条件3保存每题反馈前配对理解。训练选择NLL与完整自主分布分别评价。误差条为固定参数下两倍模拟标准误。</p>'
    for name in images:body+=f'<section><h2>{name.removesuffix(".png")}</h2><a href="{name}"><img src="{name}"></a></section>'
    body+='<h2>源表</h2><ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in manifest['tables'])+'</ul>'
    with (output/'index.html').open('x') as f:
        f.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>历史反馈与准入</title><style>body{max-width:1200px;margin:2em auto;font:16px sans-serif;padding:0 1em;line-height:1.6}img{width:100%;height:auto}section{margin:3em 0}</style>'+body+'</html>')
    shutil.copy2(Path(__file__),output/'plot_source.py')
    write(output/'qa_numeric.json',dict(status='pass',subjects=subjects,arms=arms,paths_per_cell=settings['paths'],
        path_blocks=len(tables['path_blocks']),images={name:digest(output/name) for name in images},
        source_tables={name:digest(output/name) for name in manifest['tables']},
        export_exception='PNG-only 300 dpi discussion figures under repository policy'))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)


if __name__=='__main__':main()
